//! I/O manager exports on request packets (IRPs): building them, passing them to drivers,
//! completing them, the device start-I/O queue, and share access on file objects.
//!
//! Copied from the author's original anybox-tk code, with the author's permission, and adapted
//! to this runtime's kernel.
//!
//! The drivers are the title's own: devices it created with `IoCreateDevice`. A driver routine is
//! guest code, so `IofCallDriver` starts it on a thread of its own and returns `STATUS_PENDING`,
//! which callers handle by waiting for the request's completion, as they must for any driver
//! that completes later. Completion routines run the same way, one after the other, and one that
//! returns `STATUS_MORE_PROCESSING_REQUIRED` stops the completion.
//!
//! Layouts follow the Xbox kernel's `io.h`: an IRP is 0x68 bytes followed by its stack locations
//! of 0x24 bytes each, the current one named at offset 0x5C.

use super::queue::{
    first_entry, first_entry_by_key, init_list, insert_device_queue, remove_device_entry,
};
use super::value;
use crate::kernel::object::{Body, WaitOn};
use crate::kernel::thread::FollowUp;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// The IRP header's size, and a stack location's.
const IRP_SIZE: u32 = 0x68;
const STACK_LOCATION_SIZE: u32 = 0x24;
/// `IO_TYPE_IRP`.
const IO_TYPE_IRP: u16 = 6;
/// IRP offsets.
const IRP_FLAGS: u32 = 0x04;
const IRP_THREAD_LIST: u32 = 0x08;
const IRP_IO_STATUS: u32 = 0x10;
const IRP_STACK_COUNT: u32 = 0x18;
const IRP_CURRENT_LOCATION: u32 = 0x19;
const IRP_PENDING_RETURNED: u32 = 0x1A;
const IRP_USER_IOSB: u32 = 0x1C;
const IRP_USER_EVENT: u32 = 0x20;
const IRP_USER_BUFFER: u32 = 0x30;
const IRP_DEVICE_QUEUE_ENTRY: u32 = 0x3C;
const IRP_THREAD: u32 = 0x50;
const IRP_CURRENT_STACK: u32 = 0x5C;
/// Stack location offsets: function codes, control flags, parameters, then the device, and the
/// completion routine with its context.
const STACK_MAJOR: u32 = 0x00;
const STACK_CONTROL: u32 = 0x03;
const STACK_PARAMETERS: u32 = 0x04;
const STACK_DEVICE: u32 = 0x14;
const STACK_COMPLETION: u32 = 0x1C;
const STACK_CONTEXT: u32 = 0x20;
/// `Control` bits saying when a completion routine runs.
const SL_INVOKE_ON_CANCEL: u8 = 0x20;
const SL_INVOKE_ON_SUCCESS: u8 = 0x40;
const SL_INVOKE_ON_ERROR: u8 = 0x80;
/// `IRP_MJ_DEVICE_CONTROL` and `IRP_MJ_INTERNAL_DEVICE_CONTROL`.
const IRP_MJ_DEVICE_CONTROL: u8 = 0x0A;
const IRP_MJ_INTERNAL_DEVICE_CONTROL: u8 = 0x0B;
/// `DEVICE_OBJECT` offsets: the driver, the current IRP, the extension, the stack size, the
/// device queue.
const DEVICE_DRIVER: u32 = 0x08;
const DEVICE_CURRENT_IRP: u32 = 0x10;
const DEVICE_EXTENSION: u32 = 0x18;
const DEVICE_STACK_SIZE: u32 = 0x1E;
const DEVICE_QUEUE: u32 = 0x28;
/// `DRIVER_OBJECT` offsets: the start-I/O routine and the dispatch routine table.
const DRIVER_START_IO: u32 = 0x00;
const DRIVER_MAJOR_FUNCTION: u32 = 0x0C;
/// The flag `IoMarkIrpMustComplete` sets.
const IRP_MUST_COMPLETE: u32 = 0x8000_0000;

/// Fills in an IRP of `size` bytes with `stack_size` stack locations, none current yet.
fn initialize(k: &Kernel, irp: u32, size: u32, stack_size: u8) -> KResult<()> {
    k.write(irp, &vec![0; size as usize])?;
    k.w16(irp, IO_TYPE_IRP)?;
    k.w16(irp + 2, u16::try_from(size).unwrap_or(u16::MAX))?;
    k.w8(irp + IRP_STACK_COUNT, stack_size)?;
    k.w8(irp + IRP_CURRENT_LOCATION, stack_size.wrapping_add(1))?;
    init_list(k, irp + IRP_THREAD_LIST)?;
    // One past the last location: the first `IofCallDriver` moves to the last one.
    k.w32(
        irp + IRP_CURRENT_STACK,
        irp + IRP_SIZE + u32::from(stack_size) * STACK_LOCATION_SIZE,
    )
}

/// Allocates an IRP with `stack_size` stack locations; zero when out of memory.
fn allocate(k: &mut Kernel, stack_size: u8) -> KResult<u32> {
    let size = IRP_SIZE + u32::from(stack_size) * STACK_LOCATION_SIZE;
    let irp = k.heap.alloc(size);
    if irp != 0 {
        initialize(k, irp, size, stack_size)?;
    }
    Ok(irp)
}

/// `IoAllocateIrp(StackSize)`.
pub fn allocate_irp(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(allocate(k, c.arg(0) as u8)?)
}

/// `IoInitializeIrp(Irp, PacketSize, StackSize)`.
pub fn initialize_irp(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    initialize(k, c.arg(0), c.arg(1) & 0xFFFF, c.arg(2) as u8)?;
    value(0)
}

/// `IoFreeIrp(Irp)`.
pub fn free_irp(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    k.heap.free(c.arg(0));
    value(0)
}

/// The stack location the next driver gets: the one before the current one.
fn next_stack(k: &Kernel, irp: u32) -> KResult<u32> {
    Ok(k.r32(irp + IRP_CURRENT_STACK)? - STACK_LOCATION_SIZE)
}

/// A request to `device` with its next stack location filled by `fill`: the caller's thread,
/// status block and event, and the user buffer. Zero when out of memory.
fn build_request(
    k: &mut Kernel,
    device: u32,
    (iosb, event, buffer): (u32, u32, u32),
    fill: impl FnOnce(&Kernel, u32) -> KResult<()>,
) -> KResult<u32> {
    let stack_size = k.r8(device + DEVICE_STACK_SIZE)?.max(1);
    let irp = allocate(k, stack_size)?;
    if irp == 0 {
        return Ok(0);
    }
    k.w32(irp + IRP_USER_IOSB, iosb)?;
    k.w32(irp + IRP_USER_EVENT, event)?;
    k.w32(irp + IRP_USER_BUFFER, buffer)?;
    k.w32(irp + IRP_THREAD, k.current_kthread())?;
    let stack = next_stack(k, irp)?;
    fill(k, stack)?;
    Ok(irp)
}

/// Fills a read or write stack location: `MajorFunction`, then length and offset.
fn fill_fsd(k: &Kernel, stack: u32, major: u8, length: u32, offset_at: u32) -> KResult<()> {
    k.w8(stack + STACK_MAJOR, major)?;
    k.w32(stack + STACK_PARAMETERS, length)?;
    let offset = if offset_at == 0 { 0 } else { k.r64(offset_at)? };
    k.w64(stack + STACK_PARAMETERS + 8, offset)
}

/// `IoBuildAsynchronousFsdRequest(MajorFunction, DeviceObject, Buffer, Length, StartingOffset,
/// IoStatusBlock)`.
pub fn build_asynchronous_fsd_request(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (major, device, buffer, length, offset) =
        (c.arg(0) as u8, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let irp = build_request(k, device, (c.arg(5), 0, buffer), |k, stack| {
        fill_fsd(k, stack, major, length, offset)
    })?;
    value(irp)
}

/// `IoBuildSynchronousFsdRequest(MajorFunction, DeviceObject, Buffer, Length, StartingOffset,
/// Event, IoStatusBlock)`.
pub fn build_synchronous_fsd_request(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (major, device, buffer, length, offset) =
        (c.arg(0) as u8, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let irp = build_request(k, device, (c.arg(6), c.arg(5), buffer), |k, stack| {
        fill_fsd(k, stack, major, length, offset)
    })?;
    value(irp)
}

/// Fills a device control stack location.
fn fill_device_control(
    k: &Kernel,
    stack: u32,
    internal: bool,
    (code, input, input_len, output_len): (u32, u32, u32, u32),
) -> KResult<()> {
    let major = if internal {
        IRP_MJ_INTERNAL_DEVICE_CONTROL
    } else {
        IRP_MJ_DEVICE_CONTROL
    };
    k.w8(stack + STACK_MAJOR, major)?;
    // OutputBufferLength, InputBuffer, InputBufferLength, IoControlCode.
    k.w32(stack + STACK_PARAMETERS, output_len)?;
    k.w32(stack + STACK_PARAMETERS + 4, input)?;
    k.w32(stack + STACK_PARAMETERS + 8, input_len)?;
    k.w32(stack + STACK_PARAMETERS + 12, code)
}

/// `IoBuildDeviceIoControlRequest(IoControlCode, DeviceObject, InputBuffer, InputBufferLength,
/// OutputBuffer, OutputBufferLength, InternalDeviceIoControl, Event, IoStatusBlock)`.
pub fn build_device_io_control_request(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let parameters = (c.arg(0), c.arg(2), c.arg(3), c.arg(5));
    let (device, output, internal) = (c.arg(1), c.arg(4), c.arg(6) & 0xFF != 0);
    let irp = build_request(k, device, (c.arg(8), c.arg(7), output), |k, stack| {
        fill_device_control(k, stack, internal, parameters)
    })?;
    value(irp)
}

/// Passes `irp` to `device`'s driver: moves to the next stack location and starts the driver's
/// routine for it. A device without a routine for the request completes it as invalid at once.
/// Returns the status the call returns.
fn call_driver(k: &mut Kernel, device: u32, irp: u32) -> KResult<u32> {
    let stack = next_stack(k, irp)?;
    let location = k.r8(irp + IRP_CURRENT_LOCATION)?;
    k.w8(irp + IRP_CURRENT_LOCATION, location.wrapping_sub(1))?;
    k.w32(irp + IRP_CURRENT_STACK, stack)?;
    k.w32(stack + STACK_DEVICE, device)?;
    let major = k.r8(stack + STACK_MAJOR)?;
    let driver = k.r32(device + DEVICE_DRIVER)?;
    let routine = if driver == 0 {
        0
    } else {
        k.r32(driver + DRIVER_MAJOR_FUNCTION + 4 * u32::from(major))?
    };
    if routine == 0 {
        set_io_status(k, irp, status::INVALID_DEVICE_REQUEST, 0)?;
        complete_request(k, irp)?;
        return Ok(status::INVALID_DEVICE_REQUEST);
    }
    k.run_routine(routine, &[device, irp]);
    k.w8(irp + IRP_PENDING_RETURNED, 1)?;
    Ok(status::PENDING)
}

fn set_io_status(k: &Kernel, irp: u32, result: u32, information: u32) -> KResult<()> {
    k.w32(irp + IRP_IO_STATUS, result)?;
    k.w32(irp + IRP_IO_STATUS + 4, information)
}

/// Fastcall `IofCallDriver(DeviceObject, Irp)`.
pub fn call_driver_export(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(call_driver(k, c.arg(0), c.arg(1))?)
}

/// Completes `irp`, as `IopfCompleteRequest` does: passes up the stack locations from the
/// current one, running each completion routine that asks to run for the request's status; then
/// copies the status to the caller's block, sets the caller's event and wakes a thread the
/// kernel parked on it.
///
/// A completion routine is guest code, so it runs on a thread of its own and the walk resumes
/// when it returns ([`resume_completion`]): a routine that returns
/// `STATUS_MORE_PROCESSING_REQUIRED` keeps the request, and the walk stops there.
fn complete_request(k: &mut Kernel, irp: u32) -> KResult<()> {
    let result = k.r32(irp + IRP_IO_STATUS)?;
    let wanted = if result == status::CANCELLED {
        SL_INVOKE_ON_CANCEL
    } else if result & 0x8000_0000 == 0 {
        SL_INVOKE_ON_SUCCESS
    } else {
        SL_INVOKE_ON_ERROR
    };
    let count = k.r8(irp + IRP_STACK_COUNT)?;
    let last = irp + IRP_SIZE + u32::from(count) * STACK_LOCATION_SIZE;
    let mut stack = k.r32(irp + IRP_CURRENT_STACK)?;
    while stack < last {
        // The kernel moves up a location before it calls the routine, so the routine sees its
        // caller's location as the current one, and gets the caller's device (none past the
        // last location). The location passed is cleared.
        let routine = k.r32(stack + STACK_COMPLETION)?;
        let control = k.r8(stack + STACK_CONTROL)?;
        let context = k.r32(stack + STACK_CONTEXT)?;
        k.write(stack, &[0; STACK_LOCATION_SIZE as usize])?;
        stack += STACK_LOCATION_SIZE;
        k.w32(irp + IRP_CURRENT_STACK, stack)?;
        let location = k.r8(irp + IRP_CURRENT_LOCATION)?;
        k.w8(irp + IRP_CURRENT_LOCATION, location.wrapping_add(1))?;
        if routine == 0 || control & wanted == 0 {
            continue;
        }
        let device = if stack < last {
            k.r32(stack + STACK_DEVICE)?
        } else {
            0
        };
        if let Some(tid) = k.run_routine(routine, &[device, irp, context]) {
            if let Some(thread) = k.threads.get_mut(tid) {
                thread.follow_up = Some(FollowUp::IrpCompletion { irp });
            }
            return Ok(());
        }
    }
    finish_request(k, irp)
}

/// Carries on completing `irp` once a completion routine has returned `result` (`None` when the
/// thread ended without returning): unless it kept the request, the walk goes on.
///
/// # Errors
///
/// A fault when the IRP cannot be read or written.
pub(crate) fn resume_completion(k: &mut Kernel, irp: u32, result: Option<u32>) -> KResult<()> {
    if result == Some(status::MORE_PROCESSING_REQUIRED) {
        return Ok(());
    }
    complete_request(k, irp)
}

/// The end of a completion: the caller's status block and event, and a thread waiting for the
/// packet inside the kernel.
fn finish_request(k: &mut Kernel, irp: u32) -> KResult<()> {
    let result = k.r32(irp + IRP_IO_STATUS)?;
    let information = k.r32(irp + IRP_IO_STATUS + 4)?;
    let iosb = k.r32(irp + IRP_USER_IOSB)?;
    if iosb != 0 {
        k.w32(iosb, result)?;
        k.w32(iosb + 4, information)?;
    }
    let event = k.r32(irp + IRP_USER_EVENT)?;
    if event != 0
        && let Some(id) = k.object_at(event)?
    {
        k.set_event(id);
    }
    if let Some((returned, waiter)) = k.sync_irps.remove(&irp) {
        k.w32_opt(returned, information)?;
        k.wake(waiter, result);
        k.heap.free(irp);
    }
    Ok(())
}

/// Fastcall `IofCompleteRequest(Irp, PriorityBoost)`.
pub fn complete_request_export(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    complete_request(k, c.arg(0))?;
    value(0)
}

/// `IoInvalidDeviceRequest(DeviceObject, Irp)`: a driver's routine for requests it does not
/// handle.
pub fn invalid_device_request(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let irp = c.arg(1);
    set_io_status(k, irp, status::INVALID_DEVICE_REQUEST, 0)?;
    complete_request(k, irp)?;
    value(status::INVALID_DEVICE_REQUEST)
}

/// `IoMarkIrpMustComplete(Irp)`.
pub fn mark_irp_must_complete(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let irp = c.arg(0);
    let flags = k.r32(irp + IRP_FLAGS)?;
    k.w32(irp + IRP_FLAGS, flags | IRP_MUST_COMPLETE)?;
    value(0)
}

/// Sends `irp` to `device` and parks the caller until it completes; the call then returns the
/// request's status, and the bytes transferred go to `returned`.
fn send_and_wait(k: &mut Kernel, device: u32, irp: u32, returned: u32) -> KResult<CallResult> {
    let result = call_driver(k, device, irp)?;
    if result != status::PENDING {
        // Completed at once: the IRP is done with.
        let information = k.r32(irp + IRP_IO_STATUS + 4)?;
        k.w32_opt(returned, information)?;
        k.heap.free(irp);
        return value(result);
    }
    k.sync_irps.insert(irp, (returned, k.current));
    k.block(WaitOn::Explicit, None);
    Ok(CallResult::Block)
}

/// `IoSynchronousDeviceIoControlRequest(IoControlCode, DeviceObject, InputBuffer,
/// InputBufferLength, OutputBuffer, OutputBufferLength, ReturnedOutputBufferLength,
/// InternalDeviceIoControl)`.
pub fn synchronous_device_io_control_request(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let parameters = (c.arg(0), c.arg(2), c.arg(3), c.arg(5));
    let (device, output, returned, internal) = (c.arg(1), c.arg(4), c.arg(6), c.arg(7) & 0xFF != 0);
    let irp = build_request(k, device, (0, 0, output), |k, stack| {
        fill_device_control(k, stack, internal, parameters)
    })?;
    if irp == 0 {
        return value(status::NO_MEMORY);
    }
    send_and_wait(k, device, irp, returned)
}

/// `IoSynchronousFsdRequest(MajorFunction, DeviceObject, Buffer, Length, StartingOffset)`.
pub fn synchronous_fsd_request(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (major, device, buffer, length, offset) =
        (c.arg(0) as u8, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    let irp = build_request(k, device, (0, 0, buffer), |k, stack| {
        fill_fsd(k, stack, major, length, offset)
    })?;
    if irp == 0 {
        return value(status::NO_MEMORY);
    }
    send_and_wait(k, device, irp, 0)
}

/// Makes `irp` the device's current request and starts the driver's start-I/O routine for it.
fn start_io(k: &mut Kernel, device: u32, irp: u32) -> KResult<()> {
    k.w32(device + DEVICE_CURRENT_IRP, irp)?;
    if irp == 0 {
        return Ok(());
    }
    let driver = k.r32(device + DEVICE_DRIVER)?;
    let routine = if driver == 0 {
        0
    } else {
        k.r32(driver + DRIVER_START_IO)?
    };
    if routine != 0 {
        k.run_routine(routine, &[device, irp]);
    }
    Ok(())
}

/// `IoStartPacket(DeviceObject, Irp, *Key)`: queues the request on a busy device, or starts it.
pub fn start_packet(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (device, irp, key_at) = (c.arg(0), c.arg(1), c.arg(2));
    let key = if key_at == 0 {
        None
    } else {
        Some(k.r32(key_at)?)
    };
    let queued = insert_device_queue(k, device + DEVICE_QUEUE, irp + IRP_DEVICE_QUEUE_ENTRY, key)?;
    if !queued {
        start_io(k, device, irp)?;
    }
    value(0)
}

/// The IRP whose device queue entry is `entry`, or zero.
fn irp_of_entry(entry: u32) -> u32 {
    if entry == 0 {
        0
    } else {
        entry - IRP_DEVICE_QUEUE_ENTRY
    }
}

/// `IoStartNextPacket(DeviceObject, Cancelable)`: starts the next queued request, if any.
pub fn start_next_packet(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let device = c.arg(0);
    let entry = remove_device_entry(k, device + DEVICE_QUEUE, first_entry)?;
    start_io(k, device, irp_of_entry(entry))?;
    value(0)
}

/// `IoStartNextPacketByKey(DeviceObject, Cancelable, Key)`: starts the next queued request
/// whose key is at least `Key`, or else the first.
pub fn start_next_packet_by_key(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (device, key) = (c.arg(0), c.arg(2));
    let entry = remove_device_entry(k, device + DEVICE_QUEUE, |k, head| {
        first_entry_by_key(k, head, key)
    })?;
    start_io(k, device, irp_of_entry(entry))?;
    value(0)
}

/// Access rights and share modes.
const FILE_READ_DATA: u32 = 0x01;
const FILE_WRITE_DATA: u32 = 0x02;
const FILE_APPEND_DATA: u32 = 0x04;
const FILE_EXECUTE: u32 = 0x20;
const DELETE: u32 = 0x1_0000;
const FILE_SHARE_READ: u32 = 1;
const FILE_SHARE_WRITE: u32 = 2;
const FILE_SHARE_DELETE: u32 = 4;
/// The access bits in a `FILE_OBJECT`'s byte at offset 2: read, write, delete access, then the
/// shared read, write and delete.
const FO_ACCESS: u32 = 2;

/// The access asked for: read, write, delete.
fn access_of(desired: u32) -> [bool; 3] {
    [
        desired & (FILE_READ_DATA | FILE_EXECUTE) != 0,
        desired & (FILE_WRITE_DATA | FILE_APPEND_DATA) != 0,
        desired & DELETE != 0,
    ]
}

/// The sharing allowed: read, write, delete.
fn sharing_of(share: u32) -> [bool; 3] {
    [
        share & FILE_SHARE_READ != 0,
        share & FILE_SHARE_WRITE != 0,
        share & FILE_SHARE_DELETE != 0,
    ]
}

/// Records the access and sharing in the file object's bits.
fn mark_file_object(k: &Kernel, object: u32, access: [bool; 3], shared: [bool; 3]) -> KResult<()> {
    let mut bits = k.r8(object + FO_ACCESS)? & 0x81;
    for (index, on) in access.iter().chain(&shared).enumerate() {
        if *on {
            bits |= 2 << index;
        }
    }
    k.w8(object + FO_ACCESS, bits)
}

/// A `SHARE_ACCESS`: open count, readers, writers, deleters, shared read, write and delete.
fn read_share(k: &Kernel, share: u32) -> KResult<[u8; 7]> {
    let mut counts = [0; 7];
    k.read_into(share, &mut counts)?;
    Ok(counts)
}

/// `IoCheckShareAccess(DesiredAccess, DesiredShareAccess, FileObject, ShareAccess, Update)`.
pub fn check_share_access(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (access, shared, object, share, update) = (
        access_of(c.arg(0)),
        sharing_of(c.arg(1)),
        c.arg(2),
        c.arg(3),
        c.arg(4) & 0xFF != 0,
    );
    mark_file_object(k, object, access, [false; 3])?;
    if !access.contains(&true) {
        return value(status::SUCCESS);
    }
    let mut counts = read_share(k, share)?;
    let open = counts[0];
    let violation = (0..3).any(|kind| {
        (access[kind] && counts[4 + kind] < open) || (counts[1 + kind] != 0 && !shared[kind])
    });
    if violation {
        return value(status::SHARING_VIOLATION);
    }
    mark_file_object(k, object, access, shared)?;
    if update {
        counts[0] = counts[0].wrapping_add(1);
        for kind in 0..3 {
            counts[1 + kind] = counts[1 + kind].wrapping_add(u8::from(access[kind]));
            counts[4 + kind] = counts[4 + kind].wrapping_add(u8::from(shared[kind]));
        }
        k.write(share, &counts)?;
    }
    value(status::SUCCESS)
}

/// `IoSetShareAccess(DesiredAccess, DesiredShareAccess, FileObject, ShareAccess)`: the first
/// opener's access and sharing.
pub fn set_share_access(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (access, shared, object, share) = (
        access_of(c.arg(0)),
        sharing_of(c.arg(1)),
        c.arg(2),
        c.arg(3),
    );
    if !access.contains(&true) {
        mark_file_object(k, object, access, [false; 3])?;
        k.write(share, &[0; 7])?;
        return value(0);
    }
    mark_file_object(k, object, access, shared)?;
    let mut counts = [1, 0, 0, 0, 0, 0, 0];
    for kind in 0..3 {
        counts[1 + kind] = u8::from(access[kind]);
        counts[4 + kind] = u8::from(shared[kind]);
    }
    k.write(share, &counts)?;
    value(0)
}

/// `IoRemoveShareAccess(FileObject, ShareAccess)`: takes back what the file object's opening
/// counted.
pub fn remove_share_access(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (object, share) = (c.arg(0), c.arg(1));
    let bits = k.r8(object + FO_ACCESS)?;
    let has = |index: usize| bits & (2 << index) != 0;
    if !(has(0) || has(1) || has(2)) {
        return value(0);
    }
    let mut counts = read_share(k, share)?;
    counts[0] = counts[0].saturating_sub(1);
    for kind in 0..3 {
        counts[1 + kind] = counts[1 + kind].saturating_sub(u8::from(has(kind)));
        counts[4 + kind] = counts[4 + kind].saturating_sub(u8::from(has(3 + kind)));
    }
    k.write(share, &counts)?;
    value(0)
}

/// Answers a file or volume query on the file object at `object` (a pointer
/// `ObReferenceObjectByHandle` gave out) into `buffer` of `capacity` bytes, storing the length at
/// `returned` (when given); returns the status.
fn query_file_object(
    k: &mut Kernel,
    object: u32,
    (class, volume): (u32, bool),
    buffer: u32,
    capacity: u32,
    returned: u32,
) -> KResult<u32> {
    let Some(id) = k
        .objects
        .at(object)
        .filter(|&id| matches!(k.objects.get(id).body, Body::File(_)))
    else {
        return Ok(status::INVALID_PARAMETER);
    };
    let info = if volume {
        super::io::volume_information(k, id, class)
    } else {
        super::io::file_information(k, id, class)
    };
    let Some(info) = info else {
        return Ok(status::INVALID_INFO_CLASS);
    };
    if (capacity as usize) < info.len() {
        return Ok(status::INFO_LENGTH_MISMATCH);
    }
    k.write(buffer, &info)?;
    k.w32_opt(returned, u32::try_from(info.len()).unwrap_or(0))?;
    Ok(status::SUCCESS)
}

/// `IoQueryFileInformation(FileObject, FileInformationClass, Length, FileInformation,
/// ReturnedLength)`.
pub fn query_file_information(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (object, class, length, buffer, returned) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    value(query_file_object(
        k,
        object,
        (class, false),
        buffer,
        length,
        returned,
    )?)
}

/// `IoQueryVolumeInformation(FileObject, FsInformationClass, Length, FsInformation,
/// ReturnedLength)`.
pub fn query_volume_information(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let (object, class, length, buffer, returned) =
        (c.arg(0), c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    value(query_file_object(
        k,
        object,
        (class, true),
        buffer,
        length,
        returned,
    )?)
}

/// `IoSetIoCompletion(IoCompletion, KeyContext, ApcContext, IoStatus, IoStatusInformation)`:
/// `NtSetIoCompletion` on the port object itself.
pub fn set_io_completion(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = k.objects.at(c.arg(0)) else {
        return value(status::INVALID_PARAMETER);
    };
    let packet = [c.arg(1), c.arg(2), c.arg(3), c.arg(4)];
    value(super::ntobjects::set_io_completion_object(k, id, packet)?)
}

/// `IoDeleteDevice(DeviceObject)`: forgets the device's name and frees it with its extension.
pub fn delete_device(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let device = c.arg(0);
    if let Some(id) = k.objects.at(device)
        && matches!(k.objects.get(id).body, Body::Device { .. })
    {
        k.objects.unname(id);
    }
    let extension = k.r32(device + DEVICE_EXTENSION)?;
    k.heap.free(extension);
    k.heap.free(device);
    value(0)
}
