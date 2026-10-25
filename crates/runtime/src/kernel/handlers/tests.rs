//! The handlers the test title exercises, called through the handler table.

use super::lookup;
use crate::boot::{DATA_EXPORT_SIZE, DATA_EXPORTS};
use crate::kernel::CallResult::{Block, Exit, Value};
use crate::kernel::fs::{FILE_DIRECTORY_FILE, FILE_NON_DIRECTORY_FILE};
use crate::kernel::object::Body;
use crate::kernel::spawn::SMALL_STACK;
use crate::kernel::status;
use crate::kernel::sync::WAIT_0;
use crate::kernel::testing::{kernel, run_as, threads};
use crate::kernel::thread::State;
use crate::kernel::time::BOOT_CLOCK;
use crate::kernel::{Call, CallResult, Kernel, THREAD_RETURN};

/// Guest scratch memory inside the committed heap, above anything the tests allocate.
const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;

/// Calls the handler of export `name`.
fn call(k: &mut Kernel, name: &str, args: &[u32]) -> CallResult {
    let handler = lookup(name).unwrap_or_else(|| panic!("no handler for {name}"));
    handler(
        k,
        &Call {
            args: args.to_vec(),
            esp: 0,
        },
    )
    .unwrap_or_else(|fault| panic!("{name} faulted at {:#010x}", fault.0))
}

/// A guest `LARGE_INTEGER` holding `value` at `at`.
fn large_integer(k: &Kernel, at: u32, value: i64) -> u32 {
    k.w64(at, value.cast_unsigned()).unwrap();
    at
}

/// An `ANSI_STRING` at `at` holding `text`, which follows the header.
fn ansi_string(k: &Kernel, at: u32, text: &str) -> u32 {
    let len = u16::try_from(text.len()).unwrap();
    k.w16(at, len).unwrap();
    k.w16(at + 2, len).unwrap();
    k.w32(at + 4, at + 8).unwrap();
    k.write(at + 8, text.as_bytes()).unwrap();
    at
}

/// An `OBJECT_ATTRIBUTES` at `at` naming `path`, with no root directory.
fn attributes(k: &Kernel, at: u32, path: &str) -> u32 {
    k.w32(at, 0).unwrap();
    k.w32(at + 4, at + 0x10).unwrap();
    ansi_string(k, at + 0x10, path);
    at
}

/// Creates a file of `size` zero bytes, or a directory, and returns its handle.
fn create(k: &mut Kernel, path: &str, dir: bool, size: u32) -> u32 {
    let attr = attributes(k, SCRATCH + 0x800, path);
    let (handle, iosb) = (SCRATCH + 0x900, SCRATCH + 0x908);
    let options = if dir {
        FILE_DIRECTORY_FILE
    } else {
        FILE_NON_DIRECTORY_FILE
    };
    let args = [handle, 0, attr, iosb, 0, 0, 0, 2, options];
    assert_eq!(call(k, "NtCreateFile", &args), Value(status::SUCCESS));
    let h = k.r32(handle).unwrap();
    if size > 0 {
        let data = SCRATCH + 0x1_0000;
        k.write(data, &vec![0; size as usize]).unwrap();
        let args = [h, 0, 0, 0, iosb, data, size, 0];
        assert_eq!(call(k, "NtWriteFile", &args), Value(status::SUCCESS));
    }
    h
}

fn file(k: &mut Kernel, path: &str, size: u32) -> u32 {
    create(k, path, false, size)
}

fn directory(k: &mut Kernel, path: &str) -> u32 {
    create(k, path, true, 0)
}

/// `NtQueryFullAttributesFile` of `path` into `info`.
fn full_attributes(k: &mut Kernel, path: &str, info: u32) -> CallResult {
    let attr = attributes(k, SCRATCH + 0xA00, path);
    call(k, "NtQueryFullAttributesFile", &[attr, info])
}

#[test]
fn av_saved_data_address_starts_at_zero() {
    let mut k = kernel();
    assert_eq!(call(&mut k, "AvGetSavedDataAddress", &[]), Value(0));
    call(&mut k, "AvSetSavedDataAddress", &[0x0123_4000]);
    assert_eq!(
        call(&mut k, "AvGetSavedDataAddress", &[]),
        Value(0x0123_4000)
    );
}

#[test]
fn av_capabilities_combine_the_pack_the_factory_standard_and_the_user_video_flags() {
    let mut k = kernel();
    let smc_video_mode = DATA_EXPORTS + 356 * DATA_EXPORT_SIZE;
    k.memory.commit_range(smc_video_mode, DATA_EXPORT_SIZE);
    k.eeprom[0x94..0x98].copy_from_slice(&0x00C1_0FFFu32.to_le_bytes());
    let result = SCRATCH;
    let query = |k: &mut Kernel| {
        k.w32(result, 0xDEAD_BEEF).unwrap();
        assert_eq!(
            call(k, "AvSendTVEncoderOption", &[0, 6, 0, result]),
            Value(0)
        );
        k.r32(result).unwrap()
    };
    for (mode, pack) in [(0, 3), (1, 4), (2, 5), (3, 2), (4, 6), (6, 1), (5, 0)] {
        k.w32(smc_video_mode, mode).unwrap();
        assert_eq!(
            query(&mut k),
            pack | 0x0040_0100 | 0x1_0000,
            "SMC mode {mode}"
        );
    }
    k.w32(smc_video_mode, 6).unwrap();
    k.eeprom[0x58..0x5C].copy_from_slice(&0x0080_0300u32.to_le_bytes());
    assert_eq!(
        query(&mut k),
        1 | 0x0080_0300 | 0x1_0000,
        "the factory region"
    );
    k.eeprom[0x58..0x5C].copy_from_slice(&0u32.to_le_bytes());
    let unset = if crate::kernel::time::refresh_is_50() {
        0x0080_0400
    } else {
        0x0040_0100
    };
    assert_eq!(query(&mut k), 1 | unset | 0x1_0000, "no factory region");
}

#[test]
fn av_encoder_options_other_than_6_are_accepted_without_a_result() {
    let mut k = kernel();
    k.w32(SCRATCH, 0xDEAD_BEEF).unwrap();
    assert_eq!(
        call(&mut k, "AvSendTVEncoderOption", &[0, 1, 0, SCRATCH]),
        Value(0)
    );
    assert_eq!(k.r32(SCRATCH), Ok(0xDEAD_BEEF));
}

#[test]
fn av_set_display_mode_is_recorded() {
    let mut k = kernel();
    let args = [0xFD00_0000, 0, 0x0004_0101, 0x1E, 640 * 4, 0x03C0_0000];
    assert_eq!(
        call(&mut k, "AvSetDisplayMode", &args),
        Value(status::SUCCESS)
    );
    assert_eq!(k.display_mode, args);
}

#[test]
fn pci_reads_of_the_gpu_give_its_config_dwords_and_other_devices_read_ff() {
    let mut k = kernel();
    let buf = SCRATCH;
    assert_eq!(
        call(&mut k, "HalReadWritePCISpace", &[1, 0, 0, buf, 12, 0]),
        Value(12)
    );
    assert_eq!(
        k.read(buf, 12),
        Ok(vec![
            0xDE, 0x10, 0xA0, 0x02, 0, 0, 0, 0, 0xA1, 0x00, 0x00, 0x03
        ])
    );
    assert_eq!(
        call(&mut k, "HalReadWritePCISpace", &[1, 0, 2, buf, 2, 0]),
        Value(2)
    );
    assert_eq!(k.read(buf, 2), Ok(vec![0xA0, 0x02]), "an unaligned read");
    assert_eq!(
        call(&mut k, "HalReadWritePCISpace", &[0, 1, 0, buf, 4, 0]),
        Value(4)
    );
    assert_eq!(k.read(buf, 4), Ok(vec![0xFF; 4]));
}

#[test]
fn pci_writes_are_dropped() {
    let mut k = kernel();
    let buf = SCRATCH;
    k.w32(buf, 0x1234_5678).unwrap();
    assert_eq!(
        call(&mut k, "HalReadWritePCISpace", &[1, 0, 0, buf, 4, 1]),
        Value(4)
    );
    assert_eq!(k.r32(buf), Ok(0x1234_5678), "the buffer is not touched");
    call(&mut k, "HalReadWritePCISpace", &[1, 0, 0, buf, 4, 0]);
    assert_eq!(k.r32(buf), Ok(0x02A0_10DE), "the device is unchanged");
}

#[test]
fn hal_register_shutdown_notification_returns_zero() {
    let mut k = kernel();
    assert_eq!(
        call(&mut k, "HalRegisterShutdownNotification", &[SCRATCH, 1]),
        Value(0)
    );
}

#[test]
fn ke_set_event_returns_the_previous_state_and_wakes_a_waiter() {
    let mut k = kernel();
    let ids = threads(&mut k, 1);
    let e = SCRATCH;
    call(&mut k, "KeInitializeEvent", &[e, 1, 0]);
    run_as(&mut k, ids[0]);
    assert_eq!(
        call(&mut k, "KeWaitForSingleObject", &[e, 0, 0, 0, 0]),
        Block
    );
    run_as(&mut k, 1);
    assert_eq!(call(&mut k, "KeSetEvent", &[e, 1, 0]), Value(0));
    let waiter = k.threads.get(ids[0]).unwrap();
    assert_eq!((waiter.state, waiter.resume), (State::Ready, WAIT_0));
    assert_eq!(k.r32(e + 4), Ok(0), "the waiter took the signal");
    assert_eq!(call(&mut k, "KeSetEvent", &[e, 1, 0]), Value(0));
    assert_eq!(k.r32(e + 4), Ok(1));
    assert_eq!(call(&mut k, "KeSetEvent", &[e, 1, 0]), Value(1));
}

#[test]
fn ke_set_event_adopts_an_event_the_title_built() {
    let mut k = kernel();
    let e = SCRATCH;
    k.write(e, &[0, 0, 4, 0]).unwrap();
    assert_eq!(call(&mut k, "KeSetEvent", &[e, 1, 0]), Value(0));
    assert_eq!(k.r32(e + 4), Ok(1));
    assert_eq!(call(&mut k, "KeSetEvent", &[e, 1, 0]), Value(1));
}

#[test]
fn ke_wait_for_single_object_acquires_or_times_out() {
    let mut k = kernel();
    let (e, zero) = (SCRATCH, large_integer(&k, SCRATCH + 0x40, 0));
    call(&mut k, "KeInitializeEvent", &[e, 1, 1]);
    assert_eq!(
        call(&mut k, "KeWaitForSingleObject", &[e, 0, 0, 0, zero]),
        Value(WAIT_0)
    );
    assert_eq!(k.r32(e + 4), Ok(0), "a synchronization event is cleared");
    assert_eq!(
        call(&mut k, "KeWaitForSingleObject", &[e, 0, 0, 0, zero]),
        Value(status::TIMEOUT)
    );
    let ten_ms = large_integer(&k, SCRATCH + 0x48, -100_000);
    assert_eq!(
        call(&mut k, "KeWaitForSingleObject", &[e, 0, 0, 0, ten_ms]),
        Block
    );
    k.clock.pass(100_000);
    k.service_time();
    let me = k.threads.get(1).unwrap();
    assert_eq!((me.state, me.resume), (State::Ready, status::TIMEOUT));
}

#[test]
fn ke_wait_for_single_object_on_something_that_is_not_an_object_is_invalid() {
    let mut k = kernel();
    assert_eq!(
        call(&mut k, "KeWaitForSingleObject", &[SCRATCH, 0, 0, 0, 0]),
        Value(status::INVALID_PARAMETER)
    );
    assert_eq!(
        call(&mut k, "KeWaitForSingleObject", &[0, 0, 0, 0, 0]),
        Value(status::INVALID_PARAMETER)
    );
}

#[test]
fn ke_insert_queue_dpc_runs_the_routine_on_a_small_dpc_thread_until_it_ends() {
    let mut k = kernel();
    let (d, routine, context) = (SCRATCH, 0x0001_2340, 0x55);
    assert_eq!(
        call(&mut k, "KeInsertQueueDpc", &[d, 1, 2]),
        Value(0),
        "unknown"
    );
    call(&mut k, "KeInitializeDpc", &[d, routine, context]);
    assert_eq!(call(&mut k, "KeInsertQueueDpc", &[d, 1, 2]), Value(1));
    assert_eq!(k.unstarted.len(), 1);
    let tid = k.unstarted[0];
    let thread = k.threads.get(tid).unwrap();
    assert_eq!(thread.dpc, d);
    assert_eq!(k.heap.size_of(thread.stack_block), SMALL_STACK);
    let (start, esp) = thread.start;
    assert_eq!(start, routine);
    let stack: Vec<u32> = (0..5).map(|i| k.r32(esp + 4 * i).unwrap()).collect();
    assert_eq!(stack, [THREAD_RETURN, d, context, 1, 2]);
    assert_eq!(
        call(&mut k, "KeInsertQueueDpc", &[d, 3, 4]),
        Value(0),
        "already queued"
    );
    k.end_thread(tid, None);
    assert_eq!(call(&mut k, "KeInsertQueueDpc", &[d, 3, 4]), Value(1));
}

#[test]
fn ke_initialize_interrupt_writes_the_kinterrupt() {
    let mut k = kernel();
    let at = SCRATCH;
    k.write(at, &[0xAA; 0x14]).unwrap();
    let args = [at, 0x0001_1000, 0x77, 0x33, 0x0C, 1, 0];
    assert_eq!(call(&mut k, "KeInitializeInterrupt", &args), Value(0));
    assert_eq!(k.r32(at), Ok(0x0001_1000), "routine");
    assert_eq!(k.r32(at + 4), Ok(0x77), "context");
    assert_eq!(k.r32(at + 8), Ok(0x33), "vector");
    assert_eq!(k.r32(at + 0xC), Ok(0x0C), "IRQL");
    assert_eq!(
        k.read(at + 0x10, 3),
        Ok(vec![0, 0, 1]),
        "not connected, share, mode"
    );
    assert_eq!(k.interrupts, [0u32; 0]);
}

#[test]
fn ke_connect_interrupt_registers_it_once_and_its_routine_runs_as_a_dpc() {
    let mut k = kernel();
    let at = SCRATCH;
    let args = [at, 0x0001_1000, 0x77, 0x33, 0x0C, 0, 1];
    call(&mut k, "KeInitializeInterrupt", &args);
    assert_eq!(call(&mut k, "KeConnectInterrupt", &[at]), Value(1));
    assert_eq!(k.r8(at + 0x10), Ok(1));
    assert_eq!(k.interrupts, [at]);
    assert_eq!(
        call(&mut k, "KeConnectInterrupt", &[at]),
        Value(0),
        "already connected"
    );
    assert_eq!(k.interrupts, [at]);
    assert!(k.queue_dpc(at, 0, 0));
    let (start, esp) = k.threads.get(k.unstarted[0]).unwrap().start;
    assert_eq!(start, 0x0001_1000);
    let stack: Vec<u32> = (1..5).map(|i| k.r32(esp + 4 * i).unwrap()).collect();
    assert_eq!(stack, [at, 0x77, 0, 0]);
}

#[test]
fn ps_terminate_system_thread_ends_the_thread() {
    let mut k = kernel();
    assert_eq!(call(&mut k, "PsTerminateSystemThread", &[0]), Exit);
}

#[test]
fn mm_allocate_system_memory_is_page_rounded_page_aligned_and_zeroed() {
    let mut k = kernel();
    call(&mut k, "ExAllocatePool", &[0x10]);
    let Value(a) = call(&mut k, "MmAllocateSystemMemory", &[0x1801, 4]) else {
        panic!("a value")
    };
    assert_ne!(a, 0);
    assert_eq!(a % 0x1000, 0);
    assert_eq!(k.heap.size_of(a), 0x2000);
    k.write(a, &[0xAA; 0x2000]).unwrap();
    assert_eq!(
        call(&mut k, "MmFreeSystemMemory", &[a, 0x1801]),
        Value(0x2000)
    );
    assert_eq!(k.heap.size_of(a), 0);
    assert_eq!(
        call(&mut k, "MmFreeSystemMemory", &[a, 0x1801]),
        Value(0),
        "already freed"
    );
    let Value(b) = call(&mut k, "MmAllocateSystemMemory", &[0x2000, 4]) else {
        panic!("a value")
    };
    assert_eq!(k.read(b, 0x2000), Ok(vec![0; 0x2000]));
}

#[test]
fn mm_allocate_contiguous_memory_ex_aligns_to_a_power_of_two_of_at_least_a_page() {
    let mut k = kernel();
    for (align, expected) in [
        (0, 0x1000),
        (0x10, 0x1000),
        (0x3000, 0x4000),
        (0x1_0000, 0x1_0000),
    ] {
        call(&mut k, "MmAllocateContiguousMemory", &[0x1000]);
        let args = [0x1234, 0, 0x03FF_FFFF, align, 4];
        let Value(a) = call(&mut k, "MmAllocateContiguousMemoryEx", &args) else {
            panic!("a value")
        };
        assert_ne!(a, 0);
        assert_eq!(a % expected, 0, "alignment {align:#x}");
        assert_eq!(k.heap.size_of(a), 0x2000);
    }
}

#[test]
fn mm_allocate_contiguous_memory_ex_honours_the_acceptable_addresses() {
    use crate::mem::{HEAP_BASE, HEAP_END, LOW_BASE, LOW_END, physical};
    let mut k = kernel();
    let allocate = |k: &mut Kernel, args: [u32; 5]| {
        let Value(a) = call(k, "MmAllocateContiguousMemoryEx", &args) else {
            panic!("a value")
        };
        a
    };
    // Direct3D's report page: physically under 16 MiB, so its offset fits the report method.
    let report = allocate(&mut k, [0x1000, 0, 0x00FF_FFFF, 0, 4]);
    assert!((LOW_BASE..LOW_END).contains(&report), "{report:#x}");
    assert!(physical(report) < 0x0100_0000);
    assert_eq!(crate::gpu::guest_address(physical(report)), report);
    // Above a lowest address inside the heap, aligned.
    let high = allocate(&mut k, [0x1000, 0x0200_0000, 0x03FF_FFFF, 0x4000, 4]);
    assert!((HEAP_BASE + 0x0100_0000..HEAP_END).contains(&high) && high.is_multiple_of(0x4000));
    // More than the low memory holds: from the heap after all.
    let big = allocate(&mut k, [0x10_0000, 0, 0x00FF_FFFF, 0, 4]);
    assert!((HEAP_BASE..HEAP_END).contains(&big));
    assert_eq!(
        call(&mut k, "MmQueryAllocationSize", &[report]),
        Value(0x1000)
    );
    assert_eq!(call(&mut k, "MmFreeContiguousMemory", &[report]), Value(0));
    assert_eq!(
        allocate(&mut k, [0x1000, 0, 0x00FF_FFFF, 0, 4]),
        report,
        "freed and reused"
    );
    // Bounds nothing satisfies: anywhere in the heap.
    let anywhere = allocate(&mut k, [0x1000, 0, 0, 0, 4]);
    assert!((HEAP_BASE..HEAP_END).contains(&anywhere));
}

#[test]
fn nt_allocate_virtual_memory_hands_out_user_space_addresses() {
    use crate::mem::{HEAP_BASE, heap_view};
    let mut k = kernel();
    let (base, size, info) = (SCRATCH, SCRATCH + 4, SCRATCH + 0x40);
    k.w32(base, 0).unwrap();
    k.w32(size, 0x3000).unwrap();
    let args = [base, 0, size, 0x2000, 4];
    assert_eq!(
        call(&mut k, "NtAllocateVirtualMemory", &args),
        Value(status::SUCCESS)
    );
    let block = k.r32(base).unwrap();
    // Below 0x80000000, as a console's user-space allocations are, over the heap's RAM.
    assert!(
        block < 0x8000_0000 && block.is_multiple_of(0x1000),
        "{block:#x}"
    );
    assert!(heap_view(block) >= HEAP_BASE);
    assert!(k.write(block + 0x10, &[7]).is_ok());
    assert_eq!(
        k.read(heap_view(block) + 0x10, 1),
        Ok(vec![7]),
        "one memory"
    );
    // Committing inside it, by its user address.
    k.w32(base, block + 0x1000).unwrap();
    k.w32(size, 0x1000).unwrap();
    let args = [base, 0, size, 0x1000, 4];
    assert_eq!(
        call(&mut k, "NtAllocateVirtualMemory", &args),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r32(base), Ok(block + 0x1000));
    // A commit that runs past the block's end, by its user address, conflicts.
    k.w32(base, block + 0x2000).unwrap();
    k.w32(size, 0x2000).unwrap();
    assert_eq!(
        call(&mut k, "NtAllocateVirtualMemory", &args),
        Value(status::CONFLICTING_ADDRESSES)
    );
    // Queried in the view it was handed out in.
    assert_eq!(
        call(&mut k, "NtQueryVirtualMemory", &[block + 0x20, info]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r32(info), Ok(block));
    assert_eq!(k.r32(info + 12), Ok(0x3000));
    assert_eq!(
        call(&mut k, "MmQueryAllocationSize", &[block]),
        Value(0x3000)
    );
    k.w32(base, block).unwrap();
    assert_eq!(
        call(&mut k, "NtFreeVirtualMemory", &[base, size, 0x8000]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.heap.size_of(heap_view(block)), 0);
}

#[test]
fn mm_free_contiguous_memory_frees_and_returns_zero() {
    let mut k = kernel();
    let Value(a) = call(&mut k, "MmAllocateContiguousMemory", &[0x3000]) else {
        panic!("a value")
    };
    assert_eq!(call(&mut k, "MmFreeContiguousMemory", &[a]), Value(0));
    assert_eq!(k.heap.size_of(a), 0);
    assert_eq!(call(&mut k, "MmFreeContiguousMemory", &[a]), Value(0));
}

#[test]
fn mm_claim_gpu_instance_memory_returns_a_page_aligned_block_and_writes_zero_to_skip() {
    let mut k = kernel();
    call(&mut k, "ExAllocatePool", &[0x10]);
    let skip = SCRATCH;
    k.w32(skip, 0xFFFF_FFFF).unwrap();
    let Value(a) = call(&mut k, "MmClaimGpuInstanceMemory", &[0x5000, skip]) else {
        panic!("a value")
    };
    assert_ne!(a, 0);
    assert_eq!(a % 0x1000, 0);
    assert!(k.heap.size_of(a) >= 0x5000);
    assert_eq!(k.r32(skip), Ok(0));
    let Value(b) = call(&mut k, "MmClaimGpuInstanceMemory", &[0x1000, 0]) else {
        panic!("a value")
    };
    assert_ne!(b, 0, "a null skip pointer");
}

#[test]
fn nt_free_virtual_memory_frees_the_block_and_writes_its_size_unless_decommitting() {
    let mut k = kernel();
    let (base, size) = (SCRATCH, SCRATCH + 4);
    k.w32(base, 0).unwrap();
    k.w32(size, 0x1800).unwrap();
    let args = [base, 0, size, 0x1000, 4];
    assert_eq!(
        call(&mut k, "NtAllocateVirtualMemory", &args),
        Value(status::SUCCESS)
    );
    let block = k.r32(base).unwrap();
    k.w32(size, 0).unwrap();
    assert_eq!(
        call(&mut k, "NtFreeVirtualMemory", &[base, size, 0x4000]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.heap.size_of(block), 0x2000, "decommit frees nothing");
    assert_eq!(
        call(&mut k, "NtFreeVirtualMemory", &[base, size, 0x8000]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.heap.size_of(block), 0);
    assert_eq!(k.r32(size), Ok(0x2000));
    assert_eq!(
        call(&mut k, "NtFreeVirtualMemory", &[base, size, 0x8000]),
        Value(status::INVALID_PARAMETER)
    );
}

#[test]
fn nt_create_event_writes_a_handle_to_an_event_of_the_given_type_and_state() {
    let mut k = kernel();
    let h = SCRATCH;
    assert_eq!(
        call(&mut k, "NtCreateEvent", &[h, 0, 1, 0]),
        Value(status::SUCCESS)
    );
    let first = k.r32(h).unwrap();
    let id = k.objects.handle(first).unwrap();
    assert!(matches!(
        k.objects.get(id).body,
        Body::Event {
            sync: true,
            signaled: false
        }
    ));
    let named = attributes(&k, SCRATCH + 0x100, "\\BaseNamedObjects\\e");
    assert_eq!(
        call(&mut k, "NtCreateEvent", &[h, named, 0, 1]),
        Value(status::SUCCESS)
    );
    let second = k.r32(h).unwrap();
    assert_ne!(second, first);
    let id = k.objects.handle(second).unwrap();
    assert!(matches!(
        k.objects.get(id).body,
        Body::Event {
            sync: false,
            signaled: true
        }
    ));
    assert_eq!(
        call(&mut k, "NtCreateEvent", &[h, named, 0, 1]),
        Value(status::SUCCESS),
        "the name is ignored"
    );
    assert_ne!(k.objects.handle(k.r32(h).unwrap()), Some(id));
}

#[test]
fn nt_set_event_writes_the_previous_state_and_wakes_waiters() {
    let mut k = kernel();
    let ids = threads(&mut k, 2);
    let (h, previous) = (SCRATCH, SCRATCH + 4);
    call(&mut k, "NtCreateEvent", &[h, 0, 0, 0]);
    let event = k.r32(h).unwrap();
    for &t in &ids {
        run_as(&mut k, t);
        assert_eq!(
            call(&mut k, "NtWaitForSingleObjectEx", &[event, 0, 0, 0]),
            Block
        );
    }
    run_as(&mut k, 1);
    k.w32(previous, 0xFF).unwrap();
    assert_eq!(
        call(&mut k, "NtSetEvent", &[event, previous]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r32(previous), Ok(0));
    for &t in &ids {
        let waiter = k.threads.get(t).unwrap();
        assert_eq!((waiter.state, waiter.resume), (State::Ready, WAIT_0));
    }
    assert_eq!(
        call(&mut k, "NtSetEvent", &[event, previous]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r32(previous), Ok(1));
    assert_eq!(
        call(&mut k, "NtSetEvent", &[event, 0]),
        Value(status::SUCCESS),
        "no previous-state pointer"
    );
}

#[test]
fn nt_set_event_on_a_bad_handle_is_invalid() {
    let mut k = kernel();
    assert_eq!(
        call(&mut k, "NtSetEvent", &[0x1234, 0]),
        Value(status::INVALID_HANDLE)
    );
}

#[test]
fn nt_wait_for_single_object_ex_acquires_times_out_or_rejects_a_bad_handle() {
    let mut k = kernel();
    let h = SCRATCH;
    let zero = large_integer(&k, SCRATCH + 8, 0);
    call(&mut k, "NtCreateEvent", &[h, 0, 1, 1]);
    let event = k.r32(h).unwrap();
    assert_eq!(
        call(&mut k, "NtWaitForSingleObjectEx", &[event, 1, 1, zero]),
        Value(WAIT_0)
    );
    assert_eq!(
        call(&mut k, "NtWaitForSingleObjectEx", &[event, 1, 1, zero]),
        Value(status::TIMEOUT),
        "the synchronization event was cleared"
    );
    assert_eq!(
        call(&mut k, "NtWaitForSingleObjectEx", &[0x1234, 1, 1, zero]),
        Value(status::INVALID_HANDLE)
    );
}

#[test]
fn nt_wait_for_single_object_ex_on_a_thread_waits_until_it_ends() {
    let mut k = kernel();
    let (h, id) = (SCRATCH, SCRATCH + 4);
    let args = [h, 0, 0, 0, id, 0, 0, 0, 0, 0x0001_0000];
    call(&mut k, "PsCreateSystemThreadEx", &args);
    let (thread, tid) = (k.r32(h).unwrap(), k.r32(id).unwrap());
    assert_eq!(
        call(&mut k, "NtWaitForSingleObjectEx", &[thread, 0, 0, 0]),
        Block
    );
    k.end_thread(tid, None);
    let me = k.threads.get(1).unwrap();
    assert_eq!((me.state, me.resume), (State::Ready, WAIT_0));
}

#[test]
fn nt_query_directory_file_returns_one_matching_entry_per_call() {
    let mut k = kernel();
    let dir = directory(&mut k, "E:\\dir");
    file(&mut k, "E:\\dir\\b.txt", 3000);
    file(&mut k, "E:\\dir\\A.dat", 0);
    file(&mut k, "E:\\dir\\c.TXT", 0);
    let (iosb, buf) = (SCRATCH, SCRATCH + 0x100);
    let pattern = ansi_string(&k, SCRATCH + 0x40, "*.txt");
    let args =
        |len, class, pattern, restart| [dir, 0, 0, 0, iosb, buf, len, class, pattern, restart];
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(0x100, 1, pattern, 0)),
        Value(status::SUCCESS)
    );
    let mut expected = vec![0, 0, 0, 0, 0, 0, 0, 0];
    for _ in 0..4 {
        expected.extend_from_slice(&BOOT_CLOCK.to_le_bytes());
    }
    expected.extend_from_slice(&3000u64.to_le_bytes());
    expected.extend_from_slice(&4096u64.to_le_bytes());
    expected.extend_from_slice(&0x80u32.to_le_bytes());
    expected.extend_from_slice(&5u32.to_le_bytes());
    expected.extend_from_slice(b"b.txt");
    assert_eq!(k.read(buf, 0x45), Ok(expected));
    assert_eq!(k.r32(iosb), Ok(status::SUCCESS));
    assert_eq!(k.r32(iosb + 4), Ok(0x45));
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(0x100, 1, 0, 0)),
        Value(status::SUCCESS),
        "the pattern is kept"
    );
    assert_eq!(k.r32(buf + 4), Ok(1), "the index");
    assert_eq!(k.read(buf + 0x40, 5), Ok(b"c.TXT".to_vec()));
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(0x100, 1, 0, 0)),
        Value(status::NO_MORE_FILES)
    );
    assert_eq!(k.r32(iosb), Ok(status::NO_MORE_FILES));
}

#[test]
fn nt_query_directory_file_restarts_with_names_short_buffers_and_no_matches() {
    let mut k = kernel();
    let dir = directory(&mut k, "E:\\dir");
    file(&mut k, "E:\\dir\\b", 0);
    directory(&mut k, "E:\\dir\\Alpha");
    let (iosb, buf) = (SCRATCH, SCRATCH + 0x100);
    let args =
        |len, class, pattern, restart| [dir, 0, 0, 0, iosb, buf, len, class, pattern, restart];
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(16, 12, 0, 1)),
        Value(status::BUFFER_OVERFLOW)
    );
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(17, 12, 0, 0)),
        Value(status::SUCCESS),
        "the entry is not skipped"
    );
    assert_eq!(
        k.read(buf, 17),
        Ok([&[0u8; 8][..], &[5, 0, 0, 0], b"Alpha"].concat())
    );
    let none = ansi_string(&k, SCRATCH + 0x40, "q*");
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(0x100, 12, none, 1)),
        Value(status::NO_SUCH_FILE)
    );
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args(0x100, 3, 0, 1)),
        Value(status::INVALID_INFO_CLASS)
    );
}

#[test]
fn nt_query_directory_file_on_a_file_is_invalid() {
    let mut k = kernel();
    let f = file(&mut k, "E:\\f", 0);
    let args = [f, 0, 0, 0, SCRATCH, SCRATCH + 0x100, 0x100, 1, 0, 0];
    assert_eq!(
        call(&mut k, "NtQueryDirectoryFile", &args),
        Value(status::INVALID_PARAMETER)
    );
}

#[test]
fn nt_query_full_attributes_file_reports_network_open_information_by_name() {
    let mut k = kernel();
    file(&mut k, "E:\\f.bin", 5000);
    directory(&mut k, "E:\\dir");
    let info = SCRATCH;
    assert_eq!(
        full_attributes(&mut k, "E:\\F.BIN", info),
        Value(status::SUCCESS)
    );
    let mut expected = Vec::new();
    for _ in 0..4 {
        expected.extend_from_slice(&BOOT_CLOCK.to_le_bytes());
    }
    expected.extend_from_slice(&6144u64.to_le_bytes());
    expected.extend_from_slice(&5000u64.to_le_bytes());
    expected.extend_from_slice(&0x80u32.to_le_bytes());
    expected.extend_from_slice(&[0; 4]);
    assert_eq!(k.read(info, 56), Ok(expected));
    assert_eq!(
        full_attributes(&mut k, "E:\\dir", info),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r32(info + 48), Ok(0x10));
    assert_eq!(
        full_attributes(&mut k, "E:\\missing", info),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
}

#[test]
fn nt_set_information_file_moves_the_position_and_resizes() {
    let mut k = kernel();
    let f = file(&mut k, "E:\\f", 10);
    let (iosb, buf) = (SCRATCH, SCRATCH + 0x100);
    k.w64(buf, 0x1234).unwrap();
    assert_eq!(
        call(&mut k, "NtSetInformationFile", &[f, iosb, buf, 8, 14]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r32(iosb), Ok(status::SUCCESS));
    let query = [f, iosb, SCRATCH + 0x200, 8, 14];
    call(&mut k, "NtQueryInformationFile", &query);
    assert_eq!(k.r64(SCRATCH + 0x200), Ok(0x1234));
    k.w64(buf, 100).unwrap();
    assert_eq!(
        call(&mut k, "NtSetInformationFile", &[f, iosb, buf, 8, 20]),
        Value(status::SUCCESS)
    );
    full_attributes(&mut k, "E:\\f", SCRATCH + 0x200);
    assert_eq!(k.r64(SCRATCH + 0x200 + 40), Ok(100));
    assert_eq!(
        call(&mut k, "NtSetInformationFile", &[f, iosb, buf, 40, 4]),
        Value(status::SUCCESS),
        "other classes do nothing"
    );
    assert_eq!(
        call(&mut k, "NtSetInformationFile", &[0x1234, iosb, buf, 8, 14]),
        Value(status::INVALID_HANDLE)
    );
}

#[test]
fn nt_set_information_file_renames_on_the_same_volume() {
    let mut k = kernel();
    let f = file(&mut k, "E:\\a", 0);
    file(&mut k, "E:\\taken", 0);
    let (iosb, buf) = (SCRATCH, SCRATCH + 0x100);
    let rename = |k: &mut Kernel, to: &str, replace: u8| {
        k.w8(buf, replace).unwrap();
        k.w32(buf + 4, 0).unwrap();
        ansi_string(k, buf + 8, to);
        call(k, "NtSetInformationFile", &[f, iosb, buf, 0x40, 10])
    };
    assert_eq!(rename(&mut k, "E:\\b", 0), Value(status::SUCCESS));
    let info = SCRATCH + 0x200;
    assert_eq!(
        full_attributes(&mut k, "E:\\b", info),
        Value(status::SUCCESS)
    );
    assert_eq!(
        full_attributes(&mut k, "E:\\a", info),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
    assert_eq!(
        rename(&mut k, "E:\\taken", 0),
        Value(status::OBJECT_NAME_COLLISION)
    );
    assert_eq!(rename(&mut k, "E:\\taken", 1), Value(status::SUCCESS));
    assert_eq!(
        rename(&mut k, "\\Device\\Harddisk0\\Partition2\\b", 0),
        Value(status::NOT_SAME_DEVICE)
    );
    assert_eq!(k.r32(iosb), Ok(status::NOT_SAME_DEVICE));
}

#[test]
fn nt_set_information_file_deletes_when_the_disposition_byte_is_set() {
    let mut k = kernel();
    let f = file(&mut k, "E:\\f", 0);
    let (iosb, buf) = (SCRATCH, SCRATCH + 0x100);
    let info = SCRATCH + 0x200;
    k.w8(buf, 0).unwrap();
    assert_eq!(
        call(&mut k, "NtSetInformationFile", &[f, iosb, buf, 1, 13]),
        Value(status::SUCCESS)
    );
    assert_eq!(
        full_attributes(&mut k, "E:\\f", info),
        Value(status::SUCCESS)
    );
    k.w8(buf, 1).unwrap();
    assert_eq!(
        call(&mut k, "NtSetInformationFile", &[f, iosb, buf, 1, 13]),
        Value(status::SUCCESS)
    );
    assert_eq!(
        full_attributes(&mut k, "E:\\f", info),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
}

#[test]
fn rtl_time_to_time_fields_splits_nt_time_into_calendar_fields() {
    let mut k = kernel();
    let (time, fields) = (SCRATCH, SCRATCH + 0x10);
    let mut split = |t: u64| {
        k.w64(time, t).unwrap();
        assert_eq!(
            call(&mut k, "RtlTimeToTimeFields", &[time, fields]),
            Value(0)
        );
        (0..8)
            .map(|i| k.r16(fields + 2 * i).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(split(0), [1601, 1, 1, 0, 0, 0, 0, 1], "a Monday");
    let day = 864_000_000_000;
    let clock = 13 * 36_000_000_000 + 45 * 600_000_000 + 30 * 10_000_000 + 250 * 10_000 + 9_999;
    assert_eq!(
        split(BOOT_CLOCK + day + clock),
        [2004, 1, 2, 13, 45, 30, 250, 5],
        "a Friday"
    );
    assert_eq!(
        split(BOOT_CLOCK + 59 * day),
        [2004, 2, 29, 0, 0, 0, 0, 0],
        "a Sunday, leap day"
    );
    assert_eq!(
        split(BOOT_CLOCK + 366 * day - 1),
        [2004, 12, 31, 23, 59, 59, 999, 5]
    );
}

#[test]
fn ob_reference_object_by_handle_takes_the_current_thread_pseudo_handle() {
    let mut k = kernel();
    let ids = threads(&mut k, 1);
    run_as(&mut k, ids[0]);
    let out = SCRATCH;
    assert_eq!(
        call(&mut k, "ObReferenceObjectByHandle", &[0xFFFF_FFFE, 0, out]),
        Value(status::SUCCESS)
    );
    let Value(kthread) = call(&mut k, "KeGetCurrentThread", &[]) else {
        panic!("a value")
    };
    assert_ne!(kthread, 0);
    assert_eq!(k.r32(out), Ok(kthread), "the running thread's KTHREAD");
    assert_eq!(
        call(&mut k, "ObReferenceObjectByHandle", &[0x1234, 0, out]),
        Value(status::INVALID_HANDLE)
    );
}

#[test]
fn ke_set_base_priority_thread_sets_the_priority_the_scheduler_runs_by() {
    let mut k = kernel();
    let ids = threads(&mut k, 1);
    let worker = k.threads.get(ids[0]).unwrap().kthread;
    let Value(me) = call(&mut k, "KeGetCurrentThread", &[]) else {
        panic!("a value")
    };
    assert_eq!(
        k.threads.next_runnable(1),
        Some(ids[0]),
        "equals take turns"
    );
    // The boot thread raises its own priority: it keeps the CPU, and the worker waits.
    assert_eq!(call(&mut k, "KeSetBasePriorityThread", &[me, 2]), Value(0));
    assert_eq!(k.threads.get(1).unwrap().priority, 10);
    assert_eq!(k.threads.next_runnable(1), Some(1));
    assert!(!k.threads.preempted(1));
    // Raised above it, the worker takes the CPU at once.
    call(&mut k, "KeSetPriorityThread", &[worker, 12]);
    assert!(k.threads.preempted(1));
    assert_eq!(k.threads.next_runnable(1), Some(ids[0]));
    assert_eq!(call(&mut k, "KeQueryBasePriorityThread", &[me]), Value(2));
}
