//! Tests of the handlers copied from the author's original anybox-tk code, with the author's
//! permission: IRPs, queues, APCs, Nt objects and Ob. The IRP completion and symbolic link
//! tests are ported from anybox-tk's own; the rest check what each export does here.

use super::lookup;
use crate::kernel::CallResult::{Block, Value};
use crate::kernel::fs::FILE_NON_DIRECTORY_FILE;
use crate::kernel::status;
use crate::kernel::sync::WAIT_0;
use crate::kernel::testing::{kernel, run_as, threads};
use crate::kernel::thread::State;
use crate::kernel::{Call, CallResult, Kernel};

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

fn word(k: &Kernel, address: u32) -> u32 {
    k.r32(address).unwrap()
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

/// The routine and the first `n` stdcall arguments of the thread created last.
fn last_started(k: &Kernel, n: u32) -> (u32, u32, Vec<u32>) {
    let tid = *k.unstarted.last().expect("a thread was started");
    let (routine, esp) = k.threads.get(tid).unwrap().start;
    let args = (0..n).map(|i| word(k, esp + 4 + 4 * i)).collect();
    (tid, routine, args)
}

// IRP completion, ported from anybox-tk's `irp_tests.rs`. Expected values follow the Xbox
// kernel's `io.h` layouts and `IopfCompleteRequest`'s documented walk.

const R0: u32 = 0x0001_0400;
const R1: u32 = 0x0001_0500;
const IRP: u32 = SCRATCH + 0x800;
const IOSB: u32 = SCRATCH + 0x300;
/// The IRP's three stack locations, lowest first, and the end of the packet.
const L0: u32 = IRP + 0x68;
const L1: u32 = L0 + 0x24;
const L2: u32 = L1 + 0x24;
const END: u32 = L2 + 0x24;
const ALL: u8 = 0xE0;

/// Builds a three-location IRP whose lowest location is current, with completion routines on
/// the two lowest, and completes it with `result`.
fn completed(result: u32, control: u8) -> Kernel {
    let mut k = kernel();
    k.w16(IRP, 6).unwrap();
    k.w8(IRP + 0x18, 3).unwrap();
    k.w8(IRP + 0x19, 1).unwrap();
    k.w32(IRP + 0x10, result).unwrap();
    k.w32(IRP + 0x14, 0x55).unwrap();
    k.w32(IRP + 0x1C, IOSB).unwrap();
    k.w32(IRP + 0x5C, L0).unwrap();
    for (location, routine, context, device) in [(L0, R0, 0xAA, 0x1000), (L1, R1, 0xBB, 0x1111)] {
        k.w8(location + 3, control).unwrap();
        k.w32(location + 0x14, device).unwrap();
        k.w32(location + 0x1C, routine).unwrap();
        k.w32(location + 0x20, context).unwrap();
    }
    k.w32(L2 + 0x14, 0x2222).unwrap();
    assert_eq!(call(&mut k, "IofCompleteRequest", &[IRP, 0]), Value(0));
    k
}

#[test]
fn completion_walks_up_the_stack_running_each_routine_in_turn() {
    let mut k = completed(status::SUCCESS, ALL);
    // The first routine runs with its caller's location current and that caller's device.
    let (first, routine, args) = last_started(&k, 3);
    assert_eq!((routine, args), (R0, vec![0x1111, IRP, 0xAA]));
    assert_eq!(word(&k, IRP + 0x5C), L1);
    assert_eq!(word(&k, IOSB), 0, "not finished while a routine runs");
    k.end_thread(first, Some(status::SUCCESS));
    // The second gets the next one up; there is no device above the last location's caller.
    let (second, routine, args) = last_started(&k, 3);
    assert_eq!((routine, args), (R1, vec![0x2222, IRP, 0xBB]));
    assert_eq!(word(&k, IRP + 0x5C), L2);
    k.end_thread(second, Some(status::SUCCESS));
    // Then the request is finished: status block filled in, locations passed cleared.
    assert_eq!(word(&k, IRP + 0x5C), END);
    assert_eq!(k.r8(IRP + 0x19), Ok(4));
    assert_eq!(
        (word(&k, IOSB), word(&k, IOSB + 4)),
        (status::SUCCESS, 0x55)
    );
    assert_eq!(word(&k, L0 + 0x1C), 0);
    assert_eq!(word(&k, L1 + 0x1C), 0);
}

#[test]
fn more_processing_required_keeps_the_request() {
    let mut k = completed(status::SUCCESS, ALL);
    let (first, ..) = last_started(&k, 0);
    let started = k.unstarted.len();
    k.end_thread(first, Some(status::MORE_PROCESSING_REQUIRED));
    assert_eq!(
        k.unstarted.len(),
        started,
        "the second routine does not run"
    );
    assert_eq!(
        word(&k, IOSB),
        0,
        "the caller's status block is not filled in"
    );
    assert_eq!(
        word(&k, IRP + 0x5C),
        L1,
        "the request stays at the routine's caller"
    );
}

#[test]
fn routines_run_only_for_the_status_they_ask_for() {
    // Success-only routines are skipped for an error, but the request still finishes.
    let k = completed(status::UNSUCCESSFUL, 0x40);
    assert_eq!(k.unstarted, Vec::<u32>::new());
    assert_eq!(word(&k, IOSB), status::UNSUCCESSFUL);
    assert_eq!(word(&k, IRP + 0x5C), END);
}

#[test]
fn device_control_request_has_the_xbox_layout() {
    let mut k = kernel();
    let device = SCRATCH + 0xC00;
    // A device object that asks for two stack locations.
    k.w8(device + 0x1E, 2).unwrap();
    let args = [0x1234, device, 0xAAAA_0000, 7, 0xBBBB_0000, 9, 0, 0, 0];
    let Value(irp) = call(&mut k, "IoBuildDeviceIoControlRequest", &args) else {
        panic!("a value")
    };
    assert_ne!(irp, 0);
    assert_eq!(k.r16(irp), Ok(6));
    // sizeof(IRP) is 0x68, a stack location 0x24.
    assert_eq!(k.r16(irp + 2), Ok(0x68 + 2 * 0x24));
    assert_eq!(k.r8(irp + 0x18), Ok(2));
    assert_eq!(k.r8(irp + 0x19), Ok(3));
    assert_eq!(word(&k, irp + 0x30), 0xBBBB_0000, "UserBuffer");
    assert_eq!(word(&k, irp + 0x50), k.current_kthread(), "Thread");
    assert_eq!(
        word(&k, irp + 0x5C),
        irp + 0x68 + 2 * 0x24,
        "CurrentStackLocation"
    );
    // The location the driver gets: major function, then OutputBufferLength, InputBuffer,
    // InputBufferLength and IoControlCode.
    let next = irp + 0x68 + 0x24;
    assert_eq!(k.r8(next), Ok(0x0A));
    let parameters: Vec<u32> = (0..4).map(|i| word(&k, next + 4 + 4 * i)).collect();
    assert_eq!(parameters, [9, 0xAAAA_0000, 7, 0x1234]);
}

/// A device at `SCRATCH + 0x1000` with one stack location, whose driver at `SCRATCH + 0x1100`
/// handles `IRP_MJ_DEVICE_CONTROL` with `routine` and has the start-I/O routine `start_io`.
fn device(k: &Kernel, routine: u32, start_io: u32) -> u32 {
    let (device, driver) = (SCRATCH + 0x1000, SCRATCH + 0x1100);
    k.w32(device + 8, driver).unwrap();
    k.w8(device + 0x1E, 1).unwrap();
    k.w32(driver, start_io).unwrap();
    k.w32(driver + 0x0C + 4 * 0x0A, routine).unwrap();
    device
}

#[test]
fn iof_call_driver_starts_the_dispatch_routine_or_rejects_the_request() {
    let mut k = kernel();
    let routine = 0x0001_2000;
    let dev = device(&k, routine, 0);
    let iosb = SCRATCH + 0x40;
    let args = [0x99, dev, 0, 0, 0, 0, 0, 0, iosb];
    let Value(irp) = call(&mut k, "IoBuildDeviceIoControlRequest", &args) else {
        panic!("a value")
    };
    assert_eq!(
        call(&mut k, "IofCallDriver", &[dev, irp]),
        Value(status::PENDING)
    );
    let (_, started, args) = last_started(&k, 2);
    assert_eq!((started, args), (routine, vec![dev, irp]));
    assert_eq!(word(&k, irp + 0x5C), irp + 0x68, "the driver's location");
    assert_eq!(word(&k, irp + 0x68 + 0x14), dev);
    assert_eq!(k.r8(irp + 0x1A), Ok(1), "PendingReturned");

    // A major function the driver has no routine for.
    k.w8(dev + 0x1E, 1).unwrap();
    let args = [SCRATCH + 0x60, dev, 0, 0, 0, iosb];
    let Value(read) = call(&mut k, "IoBuildAsynchronousFsdRequest", &args) else {
        panic!("a value")
    };
    assert_eq!(
        call(&mut k, "IofCallDriver", &[dev, read]),
        Value(status::INVALID_DEVICE_REQUEST)
    );
    assert_eq!(word(&k, iosb), status::INVALID_DEVICE_REQUEST);
}

#[test]
fn a_synchronous_request_waits_for_the_driver_to_complete_it() {
    let mut k = kernel();
    let dev = device(&k, 0x0001_2000, 0);
    let returned = SCRATCH + 0x40;
    let args = [0x99, dev, 0, 0, SCRATCH + 0x80, 16, returned, 0];
    assert_eq!(
        call(&mut k, "IoSynchronousDeviceIoControlRequest", &args),
        Block
    );
    let (driver, _, args) = last_started(&k, 2);
    let irp = args[1];
    run_as(&mut k, driver);
    k.w32(irp + 0x10, status::SUCCESS).unwrap();
    k.w32(irp + 0x14, 7).unwrap();
    assert_eq!(call(&mut k, "IofCompleteRequest", &[irp, 0]), Value(0));
    let me = k.threads.get(1).unwrap();
    assert_eq!((me.state, me.resume), (State::Ready, status::SUCCESS));
    assert_eq!(word(&k, returned), 7);
    assert!(k.sync_irps.is_empty());
}

#[test]
fn io_invalid_device_request_completes_the_request_as_invalid() {
    let mut k = kernel();
    let dev = device(&k, 0, 0);
    let (event, iosb) = (SCRATCH + 0x20, SCRATCH + 0x40);
    call(&mut k, "KeInitializeEvent", &[event, 0, 0]);
    let args = [3, dev, SCRATCH + 0x80, 4, 0, event, iosb];
    let Value(irp) = call(&mut k, "IoBuildSynchronousFsdRequest", &args) else {
        panic!("a value")
    };
    assert_eq!(k.r8(irp + 0x68), Ok(3), "MajorFunction");
    assert_eq!(word(&k, irp + 0x68 + 4), 4, "Length");
    assert_eq!(
        call(&mut k, "IoInvalidDeviceRequest", &[dev, irp]),
        Value(status::INVALID_DEVICE_REQUEST)
    );
    assert_eq!(word(&k, iosb), status::INVALID_DEVICE_REQUEST);
    assert_eq!(word(&k, event + 4), 1, "the caller's event is set");
    call(&mut k, "IoMarkIrpMustComplete", &[irp]);
    assert_eq!(word(&k, irp + 4), 0x8000_0000);
    assert_eq!(call(&mut k, "IoFreeIrp", &[irp]), Value(0));
}

#[test]
fn io_allocate_and_initialize_irp_leave_no_location_current() {
    let mut k = kernel();
    let Value(irp) = call(&mut k, "IoAllocateIrp", &[2]) else {
        panic!("a value")
    };
    assert_eq!(k.r16(irp + 2), Ok(0x68 + 2 * 0x24));
    assert_eq!(k.r8(irp + 0x19), Ok(3));
    assert_eq!(word(&k, irp + 0x5C), irp + 0x68 + 2 * 0x24);
    assert_eq!(word(&k, irp + 8), irp + 8, "an empty thread list");
    call(&mut k, "IoInitializeIrp", &[SCRATCH, 0x68 + 0x24, 1]);
    assert_eq!(k.r8(SCRATCH + 0x18), Ok(1));
    assert_eq!(word(&k, SCRATCH + 0x5C), SCRATCH + 0x68 + 0x24);
}

#[test]
fn start_packet_starts_an_idle_device_and_queues_on_a_busy_one() {
    let mut k = kernel();
    let start_io = 0x0001_3000;
    let dev = device(&k, 0, start_io);
    call(&mut k, "KeInitializeDeviceQueue", &[dev + 0x28]);
    let (a, b) = (SCRATCH + 0x2000, SCRATCH + 0x2100);
    call(&mut k, "IoStartPacket", &[dev, a, 0]);
    assert_eq!(word(&k, dev + 0x10), a, "CurrentIrp");
    assert_eq!(last_started(&k, 2).1, start_io);
    let started = k.unstarted.len();
    call(&mut k, "IoStartPacket", &[dev, b, 0]);
    assert_eq!(k.unstarted.len(), started, "queued behind the busy device");
    call(&mut k, "IoStartNextPacket", &[dev, 0]);
    assert_eq!(word(&k, dev + 0x10), b);
    assert_eq!(last_started(&k, 2).2, [dev, b]);
    call(&mut k, "IoStartNextPacket", &[dev, 0]);
    assert_eq!(word(&k, dev + 0x10), 0, "nothing left");
    assert_eq!(k.r8(dev + 0x28 + 4), Ok(0), "the queue is idle");
}

#[test]
fn start_next_packet_by_key_takes_the_first_key_at_least_the_one_given() {
    let mut k = kernel();
    let dev = device(&k, 0, 0);
    call(&mut k, "KeInitializeDeviceQueue", &[dev + 0x28]);
    let key = SCRATCH + 0x40;
    let irps = [SCRATCH + 0x2000, SCRATCH + 0x2100, SCRATCH + 0x2200];
    for (irp, value) in irps.into_iter().zip([0, 30, 10]) {
        k.w32(key, value).unwrap();
        call(&mut k, "IoStartPacket", &[dev, irp, key]);
    }
    call(&mut k, "IoStartNextPacketByKey", &[dev, 0, 20]);
    assert_eq!(word(&k, dev + 0x10), irps[1]);
    call(&mut k, "IoStartNextPacketByKey", &[dev, 0, 99]);
    assert_eq!(word(&k, dev + 0x10), irps[2], "none is: the first");
}

#[test]
fn share_access_counts_openers_and_refuses_conflicts() {
    let mut k = kernel();
    let (first, second, share) = (SCRATCH, SCRATCH + 0x40, SCRATCH + 0x80);
    // Read access, sharing read only.
    call(&mut k, "IoSetShareAccess", &[1, 1, first, share]);
    assert_eq!(k.read(share, 7), Ok(vec![1, 1, 0, 0, 1, 0, 0]));
    assert_eq!(
        call(&mut k, "IoCheckShareAccess", &[2, 3, second, share, 1]),
        Value(status::SHARING_VIOLATION),
        "writing is not shared"
    );
    assert_eq!(
        call(&mut k, "IoCheckShareAccess", &[1, 1, second, share, 1]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.read(share, 7), Ok(vec![2, 2, 0, 0, 2, 0, 0]));
    call(&mut k, "IoRemoveShareAccess", &[second, share]);
    assert_eq!(k.read(share, 7), Ok(vec![1, 1, 0, 0, 1, 0, 0]));
}

#[test]
fn io_query_information_answers_on_a_referenced_file_object() {
    let mut k = kernel();
    let attr = attributes(&k, SCRATCH + 0x800, "E:\\query.bin");
    let (handle, iosb) = (SCRATCH, SCRATCH + 8);
    let args = [handle, 0, attr, iosb, 0, 0, 0, 2, FILE_NON_DIRECTORY_FILE];
    assert_eq!(call(&mut k, "NtCreateFile", &args), Value(status::SUCCESS));
    let h = word(&k, handle);
    let position = large_integer(&k, SCRATCH + 0x20, 0x1234);
    let args = [h, iosb, position, 8, 14];
    call(&mut k, "NtSetInformationFile", &args);
    call(&mut k, "ObReferenceObjectByHandle", &[h, 0, SCRATCH + 0x30]);
    let object = word(&k, SCRATCH + 0x30);
    let (buffer, returned) = (SCRATCH + 0x40, SCRATCH + 0x50);
    assert_eq!(
        call(
            &mut k,
            "IoQueryFileInformation",
            &[object, 14, 8, buffer, returned]
        ),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r64(buffer), Ok(0x1234));
    assert_eq!(word(&k, returned), 8);
    assert_eq!(
        call(
            &mut k,
            "IoQueryVolumeInformation",
            &[object, 4, 4, buffer, 0]
        ),
        Value(status::INFO_LENGTH_MISMATCH)
    );
    assert_eq!(
        call(
            &mut k,
            "IoQueryFileInformation",
            &[SCRATCH, 14, 8, buffer, 0]
        ),
        Value(status::INVALID_PARAMETER),
        "not a file object"
    );
}

#[test]
fn io_delete_device_forgets_its_name() {
    let mut k = kernel();
    let name = ansi_string(&k, SCRATCH, "\\Device\\Gone");
    let out = SCRATCH + 0x40;
    call(&mut k, "IoCreateDevice", &[0, 8, name, 0x22, 0, out]);
    let device = word(&k, out);
    let found = SCRATCH + 0x50;
    assert_eq!(
        call(&mut k, "ObReferenceObjectByName", &[name, 0, 0, 0, found]),
        Value(status::SUCCESS)
    );
    assert_eq!(word(&k, found), device);
    call(&mut k, "IoDeleteDevice", &[device]);
    assert_eq!(
        call(&mut k, "ObReferenceObjectByName", &[name, 0, 0, 0, found]),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
}

#[test]
fn kernel_queues_hand_entries_to_waiters_first() {
    let mut k = kernel();
    let ids = threads(&mut k, 1);
    let queue = SCRATCH;
    call(&mut k, "KeInitializeQueue", &[queue, 0]);
    assert_eq!(k.r8(queue), Ok(4));
    assert_eq!(word(&k, queue + 0x1C), 1, "the maximum count is at least 1");
    run_as(&mut k, ids[0]);
    assert_eq!(call(&mut k, "KeRemoveQueue", &[queue, 0, 0]), Block);
    run_as(&mut k, 1);
    assert_eq!(call(&mut k, "KeInsertQueue", &[queue, 0x100]), Value(0));
    let waiter = k.threads.get(ids[0]).unwrap();
    assert_eq!((waiter.state, waiter.resume), (State::Ready, 0x100));
    call(&mut k, "KeInsertQueue", &[queue, 0x200]);
    assert_eq!(call(&mut k, "KeInsertHeadQueue", &[queue, 0x300]), Value(1));
    assert_eq!(call(&mut k, "KeRemoveQueue", &[queue, 0, 0]), Value(0x300));
    assert_eq!(call(&mut k, "KeRemoveQueue", &[queue, 0, 0]), Value(0x200));
    let zero = large_integer(&k, SCRATCH + 0x40, 0);
    assert_eq!(
        call(&mut k, "KeRemoveQueue", &[queue, 0, zero]),
        Value(status::TIMEOUT)
    );
    let soon = large_integer(&k, SCRATCH + 0x40, -100);
    assert_eq!(call(&mut k, "KeRemoveQueue", &[queue, 0, soon]), Block);
    k.clock.pass(100);
    k.service_time();
    let me = k.threads.get(1).unwrap();
    assert_eq!((me.state, me.resume), (State::Ready, status::TIMEOUT));
}

#[test]
fn ke_rundown_queue_returns_the_entries_as_a_ring() {
    let mut k = kernel();
    let queue = SCRATCH;
    call(&mut k, "KeInitializeQueue", &[queue, 4]);
    assert_eq!(call(&mut k, "KeRundownQueue", &[queue]), Value(0));
    let entries = [SCRATCH + 0x100, SCRATCH + 0x110, SCRATCH + 0x120];
    for entry in entries {
        call(&mut k, "KeInsertQueue", &[queue, entry]);
    }
    assert_eq!(call(&mut k, "KeRundownQueue", &[queue]), Value(entries[0]));
    assert_eq!(word(&k, entries[0]), entries[1]);
    assert_eq!(word(&k, entries[2]), entries[0]);
    assert_eq!(word(&k, entries[0] + 4), entries[2]);
    assert_eq!(
        call(&mut k, "KeRundownQueue", &[queue]),
        Value(0),
        "emptied"
    );
}

#[test]
fn device_queues_sort_by_key_and_go_idle_when_empty() {
    let mut k = kernel();
    let queue = SCRATCH;
    call(&mut k, "KeInitializeDeviceQueue", &[queue]);
    assert_eq!(k.r16(queue), Ok(0x14));
    let [tail, twenty, ten, thirty] = [0x100, 0x120, 0x140, 0x160].map(|o| SCRATCH + o);
    assert_eq!(
        call(&mut k, "KeInsertDeviceQueue", &[queue, tail]),
        Value(0),
        "an idle queue becomes busy and keeps nothing"
    );
    assert_eq!(
        call(&mut k, "KeInsertByKeyDeviceQueue", &[queue, twenty, 20]),
        Value(1)
    );
    call(&mut k, "KeInsertByKeyDeviceQueue", &[queue, ten, 10]);
    call(&mut k, "KeInsertByKeyDeviceQueue", &[queue, thirty, 30]);
    assert_eq!(
        call(&mut k, "KeRemoveEntryDeviceQueue", &[queue, thirty]),
        Value(1)
    );
    assert_eq!(
        call(&mut k, "KeRemoveEntryDeviceQueue", &[queue, thirty]),
        Value(0),
        "no longer queued"
    );
    assert_eq!(
        call(&mut k, "KeRemoveByKeyDeviceQueue", &[queue, 15]),
        Value(twenty)
    );
    assert_eq!(call(&mut k, "KeRemoveDeviceQueue", &[queue]), Value(ten));
    assert_eq!(call(&mut k, "KeRemoveDeviceQueue", &[queue]), Value(0));
    assert_eq!(k.r8(queue + 4), Ok(0), "idle again");
}

#[test]
fn ke_insert_queue_apc_runs_the_kernel_routine_on_the_apc_fields() {
    let mut k = kernel();
    let apc = SCRATCH;
    let (kernel_routine, normal) = (0x0001_4000, 0x0001_5000);
    let args = [apc, 0x8000_1000, kernel_routine, 0, normal, 1, 0x77];
    call(&mut k, "KeInitializeApc", &args);
    assert_eq!(k.r16(apc), Ok(0x12));
    assert_eq!(k.r8(apc + 2), Ok(1), "user mode");
    assert_eq!(word(&k, apc + 0x1C), 0x77);
    assert_eq!(call(&mut k, "KeInsertQueueApc", &[apc, 5, 6, 0]), Value(1));
    let (_, routine, args) = last_started(&k, 5);
    assert_eq!(routine, kernel_routine);
    assert_eq!(args, [apc, apc + 0x18, apc + 0x1C, apc + 0x20, apc + 0x24]);
    assert_eq!((word(&k, apc + 0x20), word(&k, apc + 0x24)), (5, 6));
    assert_eq!(
        call(&mut k, "KeInsertQueueApc", &[apc, 5, 6, 0]),
        Value(0),
        "already queued"
    );
    // Without a kernel routine, the normal one runs.
    let other = SCRATCH + 0x40;
    call(
        &mut k,
        "KeInitializeApc",
        &[other, 0, 0, 0, normal, 1, 0x88],
    );
    call(&mut k, "KeInsertQueueApc", &[other, 7, 8, 0]);
    let (_, routine, args) = last_started(&k, 3);
    assert_eq!((routine, args), (normal, vec![0x88, 7, 8]));
}

#[test]
fn nt_queue_apc_thread_runs_the_routine_with_its_arguments() {
    let mut k = kernel();
    let routine = 0x0001_6000;
    assert_eq!(
        call(&mut k, "NtQueueApcThread", &[0x1234, routine, 1, 2, 3]),
        Value(status::INVALID_HANDLE)
    );
    assert_eq!(
        call(&mut k, "NtQueueApcThread", &[0xFFFF_FFFE, routine, 1, 2, 3]),
        Value(status::SUCCESS)
    );
    let (_, started, args) = last_started(&k, 3);
    assert_eq!((started, args), (routine, vec![1, 2, 3]));
}

#[test]
fn nt_user_io_apc_dispatcher_calls_the_routine_and_waits_for_it() {
    let mut k = kernel();
    let (routine, iosb) = (0x0001_7000, SCRATCH);
    k.w32(iosb, status::END_OF_FILE).unwrap();
    k.w32(iosb + 4, 99).unwrap();
    assert_eq!(
        call(&mut k, "NtUserIoApcDispatcher", &[routine, iosb, 0]),
        Block
    );
    let (tid, started, args) = last_started(&k, 3);
    assert_eq!((started, args), (routine, vec![38, 0, iosb]));
    k.end_thread(tid, None);
    let me = k.threads.get(1).unwrap();
    assert_eq!((me.state, me.resume), (State::Ready, WAIT_0));
}

#[test]
fn nt_timers_fire_their_apc_and_report_their_state() {
    let mut k = kernel();
    let (handle, info, previous) = (SCRATCH, SCRATCH + 0x10, SCRATCH + 0x20);
    assert_eq!(
        call(&mut k, "NtCreateTimer", &[handle, 0, 1]),
        Value(status::SUCCESS)
    );
    let h = word(&k, handle);
    let due = large_integer(&k, SCRATCH + 0x30, -1000);
    let routine = 0x0001_8000;
    let args = [h, due, routine, 1, 0x44, 0, 0, previous];
    assert_eq!(call(&mut k, "NtSetTimerEx", &args), Value(status::SUCCESS));
    assert_eq!(k.r8(previous), Ok(0));
    call(&mut k, "NtQueryTimer", &[h, info]);
    assert_eq!((k.r64(info), k.r8(info + 8)), (Ok(1000), Ok(0)));
    k.clock.pass(1000);
    k.service_time();
    call(&mut k, "NtQueryTimer", &[h, info]);
    assert_eq!((k.r64(info), k.r8(info + 8)), (Ok(0), Ok(1)));
    let (_, started, args) = last_started(&k, 1);
    assert_eq!((started, args), (routine, vec![0x44]));
    assert_eq!(
        call(&mut k, "NtCancelTimer", &[h, previous]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.r8(previous), Ok(1), "it was signalled");
    assert_eq!(
        call(&mut k, "NtQueryTimer", &[0x1234, info]),
        Value(status::INVALID_HANDLE)
    );
}

#[test]
fn nt_query_mutant_and_semaphore_report_their_state() {
    let mut k = kernel();
    let (handle, info) = (SCRATCH, SCRATCH + 0x10);
    call(&mut k, "NtCreateMutant", &[handle, 0, 1]);
    let m = word(&k, handle);
    assert_eq!(
        call(&mut k, "NtQueryMutant", &[m, info]),
        Value(status::SUCCESS)
    );
    assert_eq!((word(&k, info), k.r8(info + 4)), (0, Ok(1)), "owned once");
    call(&mut k, "NtReleaseMutant", &[m, 0]);
    call(&mut k, "NtQueryMutant", &[m, info]);
    assert_eq!((word(&k, info), k.r8(info + 4)), (1, Ok(0)), "free");
    call(&mut k, "NtCreateSemaphore", &[handle, 0, 2, 5]);
    let s = word(&k, handle);
    assert_eq!(
        call(&mut k, "NtQuerySemaphore", &[s, info]),
        Value(status::SUCCESS)
    );
    assert_eq!((word(&k, info), word(&k, info + 4)), (2, 5));
    assert_eq!(
        call(&mut k, "NtQuerySemaphore", &[m, info]),
        Value(status::INVALID_HANDLE)
    );
}

#[test]
fn io_completion_ports_queue_packets_or_hand_them_to_a_waiter() {
    let mut k = kernel();
    let ids = threads(&mut k, 1);
    let handle = SCRATCH;
    assert_eq!(
        call(&mut k, "NtCreateIoCompletion", &[handle, 0, 0, 0]),
        Value(status::SUCCESS)
    );
    let port = word(&k, handle);
    let (key, context, iosb) = (SCRATCH + 0x10, SCRATCH + 0x14, SCRATCH + 0x18);
    call(&mut k, "NtSetIoCompletion", &[port, 1, 2, 3, 4]);
    call(&mut k, "NtQueryIoCompletion", &[port, SCRATCH + 0x20]);
    assert_eq!(word(&k, SCRATCH + 0x20), 1);
    assert_eq!(
        call(
            &mut k,
            "NtRemoveIoCompletion",
            &[port, key, context, iosb, 0]
        ),
        Value(status::SUCCESS)
    );
    let packet: Vec<u32> = [key, context, iosb, iosb + 4]
        .iter()
        .map(|&a| word(&k, a))
        .collect();
    assert_eq!(packet, [1, 2, 3, 4]);
    run_as(&mut k, ids[0]);
    assert_eq!(
        call(
            &mut k,
            "NtRemoveIoCompletion",
            &[port, key, context, iosb, 0]
        ),
        Block
    );
    run_as(&mut k, 1);
    // Through the port object, as `IoSetIoCompletion` takes it.
    call(
        &mut k,
        "ObReferenceObjectByHandle",
        &[port, 0, SCRATCH + 0x30],
    );
    let object = word(&k, SCRATCH + 0x30);
    call(&mut k, "IoSetIoCompletion", &[object, 5, 6, 7, 8]);
    let waiter = k.threads.get(ids[0]).unwrap();
    assert_eq!(
        (waiter.state, waiter.resume),
        (State::Ready, status::SUCCESS)
    );
    assert_eq!(word(&k, key), 5);
    assert_eq!(word(&k, iosb + 4), 8);
    call(&mut k, "NtQueryIoCompletion", &[port, SCRATCH + 0x20]);
    assert_eq!(word(&k, SCRATCH + 0x20), 0, "nothing was queued");
    let zero = large_integer(&k, SCRATCH + 0x40, 0);
    assert_eq!(
        call(
            &mut k,
            "NtRemoveIoCompletion",
            &[port, key, context, iosb, zero]
        ),
        Value(status::TIMEOUT)
    );
}

/// Reads the entry `NtQueryDirectoryObject` wrote at `buffer`: its name.
fn directory_entry(k: &Kernel, buffer: u32) -> String {
    k.ansi_string(buffer).unwrap()
}

#[test]
fn object_directories_list_what_is_named_inside_them() {
    let mut k = kernel();
    let (handle, buffer, context, returned) =
        (SCRATCH, SCRATCH + 0x100, SCRATCH + 0x10, SCRATCH + 0x14);
    let missing = attributes(&k, SCRATCH + 0x200, "\\Nowhere");
    assert_eq!(
        call(&mut k, "NtOpenDirectoryObject", &[handle, missing]),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
    let dos = attributes(&k, SCRATCH + 0x200, "\\??");
    assert_eq!(
        call(&mut k, "NtOpenDirectoryObject", &[handle, dos]),
        Value(status::SUCCESS)
    );
    let dir = word(&k, handle);
    let query = |restart| [dir, buffer, 0x40, restart, context, returned];
    assert_eq!(
        call(&mut k, "NtQueryDirectoryObject", &query(1)),
        Value(status::SUCCESS)
    );
    let first = directory_entry(&k, buffer);
    assert_eq!(word(&k, returned), 8 + 2 + 1);
    let mut names = vec![first];
    while call(&mut k, "NtQueryDirectoryObject", &query(0)) == Value(status::SUCCESS) {
        names.push(directory_entry(&k, buffer));
    }
    assert!(names.contains(&"D:".to_owned()) && names.contains(&"E:".to_owned()));

    let mine = attributes(&k, SCRATCH + 0x200, "\\Mine");
    assert_eq!(
        call(&mut k, "NtCreateDirectoryObject", &[handle, mine]),
        Value(status::SUCCESS)
    );
    let dir = word(&k, handle);
    let lock = attributes(&k, SCRATCH + 0x200, "\\Mine\\Lock");
    call(&mut k, "NtCreateSemaphore", &[SCRATCH + 4, lock, 0, 1]);
    let query = [dir, buffer, 0x40, 1, context, returned];
    assert_eq!(
        call(&mut k, "NtQueryDirectoryObject", &query),
        Value(status::SUCCESS)
    );
    assert_eq!(directory_entry(&k, buffer), "Lock");
    assert_eq!(
        call(
            &mut k,
            "NtQueryDirectoryObject",
            &[dir, buffer, 8, 1, context, 0]
        ),
        Value(status::BUFFER_TOO_SMALL)
    );
    assert_eq!(
        call(
            &mut k,
            "NtQueryDirectoryObject",
            &[dir, buffer, 0x40, 0, context, 0]
        ),
        Value(status::NO_MORE_ENTRIES)
    );
    let mine = attributes(&k, SCRATCH + 0x200, "\\Mine");
    assert_eq!(
        call(&mut k, "NtOpenDirectoryObject", &[handle, mine]),
        Value(status::SUCCESS),
        "a directory created earlier opens"
    );
}

/// Ported from anybox-tk's `file_tests.rs`: links made, opened as objects, queried, followed and
/// deleted.
#[test]
fn symbolic_links_can_be_created_queried_used_and_deleted() {
    let mut k = kernel();
    let link = ansi_string(&k, SCRATCH, "\\??\\G:");
    let target = ansi_string(&k, SCRATCH + 0x20, "\\Device\\Harddisk0\\Partition1");
    assert_eq!(
        call(&mut k, "IoCreateSymbolicLink", &[link, target]),
        Value(status::SUCCESS)
    );
    assert_eq!(
        call(&mut k, "IoCreateSymbolicLink", &[link, target]),
        Value(status::OBJECT_NAME_COLLISION)
    );
    let path = attributes(&k, SCRATCH + 0x100, "\\??\\G:\\linked.bin");
    let (handle, iosb) = (SCRATCH + 0x300, SCRATCH + 0x308);
    let create = [handle, 0, path, iosb, 0, 0, 0, 2, FILE_NON_DIRECTORY_FILE];
    assert_eq!(
        call(&mut k, "NtCreateFile", &create),
        Value(status::SUCCESS),
        "through the new link"
    );
    let name = attributes(&k, SCRATCH + 0x200, "\\??\\G:");
    assert_eq!(
        call(&mut k, "NtOpenSymbolicLinkObject", &[SCRATCH + 0x330, name]),
        Value(status::SUCCESS)
    );
    // An empty ANSI_STRING with room for the link target.
    let out = SCRATCH + 0x400;
    k.w16(out, 0).unwrap();
    k.w16(out + 2, 64).unwrap();
    k.w32(out + 4, SCRATCH + 0x500).unwrap();
    let h = word(&k, SCRATCH + 0x330);
    assert_eq!(
        call(
            &mut k,
            "NtQuerySymbolicLinkObject",
            &[h, out, SCRATCH + 0x340]
        ),
        Value(status::SUCCESS)
    );
    let expected = b"\\Device\\Harddisk0\\Partition1";
    assert_eq!(word(&k, SCRATCH + 0x340), 28);
    assert_eq!(k.read(SCRATCH + 0x500, 28), Ok(expected.to_vec()));
    assert_eq!(k.r16(out), Ok(28));
    k.w16(out + 2, 4).unwrap();
    assert_eq!(
        call(&mut k, "NtQuerySymbolicLinkObject", &[h, out, 0]),
        Value(status::BUFFER_TOO_SMALL)
    );
    assert_eq!(
        call(&mut k, "IoDeleteSymbolicLink", &[link]),
        Value(status::SUCCESS)
    );
    let open = [handle, 0, path, iosb, 0, 0, 0, 1, FILE_NON_DIRECTORY_FILE];
    assert_ne!(
        call(&mut k, "NtCreateFile", &open),
        Value(status::SUCCESS),
        "the link is gone"
    );
    assert_eq!(
        call(&mut k, "NtOpenSymbolicLinkObject", &[SCRATCH + 0x330, name]),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
    assert_eq!(
        call(&mut k, "IoDeleteSymbolicLink", &[link]),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
}

#[test]
fn scatter_and_gather_move_the_data_page_by_page() {
    let mut k = kernel();
    let attr = attributes(&k, SCRATCH + 0x800, "E:\\pages.bin");
    let (handle, iosb) = (SCRATCH, SCRATCH + 8);
    let args = [handle, 0, attr, iosb, 0, 0, 0, 2, FILE_NON_DIRECTORY_FILE];
    call(&mut k, "NtCreateFile", &args);
    let h = word(&k, handle);
    let pages = [SCRATCH + 0x1_0000, SCRATCH + 0x2_0000];
    k.write(pages[0], &[0xAA; 0x1000]).unwrap();
    k.write(pages[1], &[0xBB; 0x10]).unwrap();
    let segments = SCRATCH + 0x100;
    for (i, page) in (0..).zip(pages) {
        k.w64(segments + 8 * i, u64::from(page)).unwrap();
    }
    let zero = large_integer(&k, SCRATCH + 0x20, 0);
    let args = [h, 0, 0, 0, iosb, segments, 0x1010, zero];
    assert_eq!(
        call(&mut k, "NtWriteFileGather", &args),
        Value(status::SUCCESS)
    );
    assert_eq!(word(&k, iosb + 4), 0x1010);

    let back = [SCRATCH + 0x3_0000, SCRATCH + 0x4_0000];
    for (i, page) in (0..).zip(back) {
        k.w64(segments + 8 * i, u64::from(page)).unwrap();
    }
    let args = [h, 0, 0, 0, iosb, segments, 0x1010, zero];
    assert_eq!(
        call(&mut k, "NtReadFileScatter", &args),
        Value(status::SUCCESS)
    );
    assert_eq!(k.read(back[0], 0x1000), Ok(vec![0xAA; 0x1000]));
    assert_eq!(k.read(back[1], 0x10), Ok(vec![0xBB; 0x10]));
    assert_eq!(k.r8(back[1] + 0x10), Ok(0), "nothing past the length");
    assert_eq!(
        call(
            &mut k,
            "NtReadFileScatter",
            &args.map(|a| if a == h { 0x1234 } else { a })
        ),
        Value(status::INVALID_HANDLE)
    );
    // From the file position, now at the end.
    let args = [h, 0, 0, 0, iosb, segments, 0x10, 0];
    assert_eq!(
        call(&mut k, "NtReadFileScatter", &args),
        Value(status::END_OF_FILE)
    );
}

#[test]
fn signal_and_wait_signals_one_object_then_waits_on_another() {
    let mut k = kernel();
    let (handle, other) = (SCRATCH, SCRATCH + 4);
    call(&mut k, "NtCreateEvent", &[handle, 0, 1, 0]);
    call(&mut k, "NtCreateSemaphore", &[other, 0, 0, 1]);
    let (event, semaphore) = (word(&k, handle), word(&k, other));
    assert_eq!(
        call(
            &mut k,
            "NtSignalAndWaitForSingleObjectEx",
            &[semaphore, event, 0, 0, 0]
        ),
        Block,
        "the event is not set"
    );
    let ids = threads(&mut k, 1);
    run_as(&mut k, ids[0]);
    assert_eq!(
        call(
            &mut k,
            "NtSignalAndWaitForSingleObjectEx",
            &[event, semaphore, 0, 0, 0]
        ),
        Value(WAIT_0),
        "the semaphore was released by the first call"
    );
    let first = k.threads.get(1).unwrap();
    assert_eq!((first.state, first.resume), (State::Ready, WAIT_0));
    assert_eq!(
        call(
            &mut k,
            "NtSignalAndWaitForSingleObjectEx",
            &[semaphore, event, 0, 0, 0]
        ),
        Block,
        "the first thread's wake consumed the synchronization event"
    );
    run_as(&mut k, 1);
    assert_eq!(
        call(
            &mut k,
            "NtSignalAndWaitForSingleObjectEx",
            &[semaphore, event, 0, 0, 0]
        ),
        Value(status::SEMAPHORE_LIMIT_EXCEEDED)
    );
}

#[test]
fn ob_objects_are_created_named_opened_and_referenced() {
    let mut k = kernel();
    let out = SCRATCH;
    assert_eq!(
        call(&mut k, "ObCreateObject", &[0x8000_2000, 0, 0x20, out]),
        Value(status::SUCCESS)
    );
    let body = word(&k, out);
    assert_eq!(k.read(body, 0x20), Ok(vec![0; 0x20]));
    let name = attributes(&k, SCRATCH + 0x100, "\\Thing");
    let handle = SCRATCH + 4;
    assert_eq!(
        call(&mut k, "ObInsertObject", &[body, name, 0, handle]),
        Value(status::SUCCESS)
    );
    let first = word(&k, handle);
    assert_eq!(k.object_of_handle(first), k.objects.at(body));
    call(&mut k, "ObCreateObject", &[0x8000_2000, 0, 4, out]);
    let second = word(&k, out);
    assert_eq!(
        call(&mut k, "ObInsertObject", &[second, name, 0, handle]),
        Value(status::OBJECT_NAME_COLLISION)
    );
    assert_eq!(
        call(&mut k, "ObOpenObjectByName", &[name, 0, 0, handle]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.object_of_handle(word(&k, handle)), k.objects.at(body));
    assert_eq!(
        call(&mut k, "ObOpenObjectByPointer", &[body, 0, handle]),
        Value(status::SUCCESS)
    );
    assert_eq!(k.object_of_handle(word(&k, handle)), k.objects.at(body));
    assert_eq!(
        call(
            &mut k,
            "ObOpenObjectByPointer",
            &[SCRATCH + 0x900, 0, handle]
        ),
        Value(status::INVALID_PARAMETER)
    );
    let text = ansi_string(&k, SCRATCH + 0x200, "\\thing");
    assert_eq!(
        call(&mut k, "ObReferenceObjectByName", &[text, 0, 0, 0, out]),
        Value(status::SUCCESS)
    );
    assert_eq!(word(&k, out), body, "names are case-insensitive");
    let missing = attributes(&k, SCRATCH + 0x300, "\\Other");
    assert_eq!(
        call(&mut k, "ObOpenObjectByName", &[missing, 0, 0, handle]),
        Value(status::OBJECT_NAME_NOT_FOUND)
    );
}

#[test]
fn mm_dbg_write_check_returns_mapped_addresses_only() {
    let mut k = kernel();
    assert_eq!(
        call(&mut k, "MmDbgWriteCheck", &[SCRATCH, 0]),
        Value(SCRATCH)
    );
    assert_eq!(call(&mut k, "MmDbgWriteCheck", &[0x7000_0000, 0]), Value(0));
}

#[test]
fn io_queue_thread_irp_does_nothing() {
    let mut k = kernel();
    assert_eq!(call(&mut k, "IoQueueThreadIrp", &[SCRATCH]), Value(0));
}
