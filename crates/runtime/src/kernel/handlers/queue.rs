//! Kernel queues (`KQUEUE`) and device queues (`KDEVICE_QUEUE`).
//!
//! Copied from the author's original anybox-tk code (its `dispatcher.rs`), with the author's
//! permission, and adapted to this runtime's kernel.
//!
//! A kernel queue is an object bound to the guest structure's address, holding its entries; a
//! thread waiting in `KeRemoveQueue` is handed the next entry inserted. A device queue lives only
//! in guest memory: its busy flag and its list of entries.

use std::collections::VecDeque;

use super::value;
use crate::kernel::object::{Body, ObjId, WaitOn};
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `DISPATCHER_HEADER` object types.
const QUEUE_OBJECT: u8 = 4;
const DEVICE_QUEUE_OBJECT: u8 = 0x14;

/// `KQUEUE` offsets: the entry list, the current and maximum counts, the thread list.
const QUEUE_ENTRY_LIST: u32 = 0x10;
const QUEUE_CURRENT_COUNT: u32 = 0x18;
const QUEUE_MAXIMUM_COUNT: u32 = 0x1C;
const QUEUE_THREAD_LIST: u32 = 0x20;

/// `KDEVICE_QUEUE` offsets: busy flag and list head. A `KDEVICE_QUEUE_ENTRY` is its list link,
/// the sort key at 8 and the inserted flag at 12.
const DEVICE_QUEUE_BUSY: u32 = 0x04;
const DEVICE_QUEUE_LIST: u32 = 0x08;
pub(super) const ENTRY_SORT_KEY: u32 = 0x08;
const ENTRY_INSERTED: u32 = 0x0C;

/// Makes the `LIST_ENTRY` at `head` an empty list.
pub(super) fn init_list(k: &Kernel, head: u32) -> KResult<()> {
    k.w32(head, head)?;
    k.w32(head + 4, head)
}

/// Links `entry` into a list just before `next` (at the tail when `next` is the head).
fn link_before(k: &Kernel, next: u32, entry: u32) -> KResult<()> {
    let previous = k.r32(next + 4)?;
    k.w32(entry, next)?;
    k.w32(entry + 4, previous)?;
    k.w32(previous, entry)?;
    k.w32(next + 4, entry)
}

/// Unlinks `entry` from its list.
fn unlink(k: &Kernel, entry: u32) -> KResult<()> {
    let (next, previous) = (k.r32(entry)?, k.r32(entry + 4)?);
    k.w32(previous, next)?;
    k.w32(next + 4, previous)
}

/// `KeInitializeQueue(Queue, Count)`.
pub fn initialize_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let at = c.arg(0);
    k.write(at, &[QUEUE_OBJECT, 0, 10, 0])?;
    k.w32(at + 4, 0)?;
    init_list(k, at + 8)?;
    init_list(k, at + QUEUE_ENTRY_LIST)?;
    k.w32(at + QUEUE_CURRENT_COUNT, 0)?;
    k.w32(at + QUEUE_MAXIMUM_COUNT, c.arg(1).max(1))?;
    init_list(k, at + QUEUE_THREAD_LIST)?;
    k.objects.add_at(
        at,
        Body::Queue {
            entries: VecDeque::new(),
        },
    );
    value(0)
}

/// The queue object at `address`, made when a title built the queue without the kernel.
fn queue_at(k: &mut Kernel, address: u32) -> ObjId {
    match k.objects.at(address) {
        Some(id) if matches!(k.objects.get(id).body, Body::Queue { .. }) => id,
        _ => k.objects.add_at(
            address,
            Body::Queue {
                entries: VecDeque::new(),
            },
        ),
    }
}

/// Gives `entry` to the oldest thread waiting on queue `id`, or keeps it, at the front with
/// `head`. Returns how many entries the queue held before.
fn insert_queue(k: &mut Kernel, id: ObjId, entry: u32, head: bool) -> u32 {
    if let Some(&waiter) = k.takers(id).first() {
        k.wake(waiter, entry);
        return 0;
    }
    let Body::Queue { entries } = &mut k.objects.get_mut(id).body else {
        return 0;
    };
    let previous = u32::try_from(entries.len()).unwrap_or(u32::MAX);
    if head {
        entries.push_front(entry);
    } else {
        entries.push_back(entry);
    }
    previous
}

/// `KeInsertQueue(Queue, Entry)`.
pub fn insert_queue_tail(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let id = queue_at(k, c.arg(0));
    value(insert_queue(k, id, c.arg(1), false))
}

/// `KeInsertHeadQueue(Queue, Entry)`.
pub fn insert_head_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let id = queue_at(k, c.arg(0));
    value(insert_queue(k, id, c.arg(1), true))
}

/// `KeRemoveQueue(Queue, WaitMode, *Timeout)`: the oldest entry, waiting for one if there is
/// none. A wait that times out returns `STATUS_TIMEOUT` in place of an entry.
pub fn remove_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let id = queue_at(k, c.arg(0));
    if let Body::Queue { entries } = &mut k.objects.get_mut(id).body
        && let Some(entry) = entries.pop_front()
    {
        return value(entry);
    }
    let deadline = k.deadline(c.arg(2))?;
    if deadline.is_some_and(|d| d <= k.clock.now()) {
        return value(status::TIMEOUT);
    }
    k.block(WaitOn::Take(id), deadline);
    Ok(CallResult::Block)
}

/// `KeRundownQueue(Queue)`: empties the queue and returns its entries as a list without a head,
/// linked in order and around, by the first one; null when it was empty.
pub fn rundown_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let id = queue_at(k, c.arg(0));
    let entries: Vec<u32> = match &mut k.objects.get_mut(id).body {
        Body::Queue { entries } => entries.drain(..).collect(),
        _ => Vec::new(),
    };
    let Some(&first) = entries.first() else {
        return value(0);
    };
    let n = entries.len();
    for (index, &entry) in entries.iter().enumerate() {
        k.w32(entry, entries[(index + 1) % n])?;
        k.w32(entry + 4, entries[(index + n - 1) % n])?;
    }
    value(first)
}

/// `KeInitializeDeviceQueue(DeviceQueue)`.
pub fn initialize_device_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let queue = c.arg(0);
    k.w16(queue, u16::from(DEVICE_QUEUE_OBJECT))?;
    k.w8(queue + 2, 0x10)?;
    k.w8(queue + DEVICE_QUEUE_BUSY, 0)?;
    init_list(k, queue + DEVICE_QUEUE_LIST)?;
    value(0)
}

/// Inserts `entry` into a busy device queue, before the first entry whose key is greater than
/// `key` (at the tail without one). An idle queue becomes busy instead and keeps nothing.
/// Returns whether the entry was queued.
pub(super) fn insert_device_queue(
    k: &Kernel,
    queue: u32,
    entry: u32,
    key: Option<u32>,
) -> KResult<bool> {
    if k.r8(queue + DEVICE_QUEUE_BUSY)? == 0 {
        k.w8(queue + DEVICE_QUEUE_BUSY, 1)?;
        k.w8(entry + ENTRY_INSERTED, 0)?;
        return Ok(false);
    }
    let head = queue + DEVICE_QUEUE_LIST;
    let mut next = head;
    if let Some(key) = key {
        k.w32(entry + ENTRY_SORT_KEY, key)?;
        let mut at = k.r32(head)?;
        while at != head {
            if k.r32(at + ENTRY_SORT_KEY)? > key {
                next = at;
                break;
            }
            at = k.r32(at)?;
        }
    }
    link_before(k, next, entry)?;
    k.w8(entry + ENTRY_INSERTED, 1)?;
    Ok(true)
}

/// `KeInsertDeviceQueue(DeviceQueue, DeviceQueueEntry)`.
pub fn insert_device_queue_tail(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let queued = insert_device_queue(k, c.arg(0), c.arg(1), None)?;
    value(u32::from(queued))
}

/// `KeInsertByKeyDeviceQueue(DeviceQueue, DeviceQueueEntry, SortKey)`.
pub fn insert_by_key_device_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let queued = insert_device_queue(k, c.arg(0), c.arg(1), Some(c.arg(2)))?;
    value(u32::from(queued))
}

/// Takes the entry `choose` picks (given the list head) out of a device queue; an empty queue
/// goes idle and gives nothing.
pub(super) fn remove_device_entry(
    k: &Kernel,
    queue: u32,
    choose: impl Fn(&Kernel, u32) -> KResult<u32>,
) -> KResult<u32> {
    let head = queue + DEVICE_QUEUE_LIST;
    if k.r32(head)? == head {
        k.w8(queue + DEVICE_QUEUE_BUSY, 0)?;
        return Ok(0);
    }
    let entry = choose(k, head)?;
    unlink(k, entry)?;
    k.w8(entry + ENTRY_INSERTED, 0)?;
    Ok(entry)
}

/// The first entry of the list at `head`.
pub(super) fn first_entry(k: &Kernel, head: u32) -> KResult<u32> {
    k.r32(head)
}

/// The first entry of the list at `head` whose key is at least `key`, or else the first entry.
pub(super) fn first_entry_by_key(k: &Kernel, head: u32, key: u32) -> KResult<u32> {
    let first = k.r32(head)?;
    let mut at = first;
    while at != head {
        if k.r32(at + ENTRY_SORT_KEY)? >= key {
            return Ok(at);
        }
        at = k.r32(at)?;
    }
    Ok(first)
}

/// `KeRemoveDeviceQueue(DeviceQueue)`: the first entry.
pub fn remove_device_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    value(remove_device_entry(k, c.arg(0), first_entry)?)
}

/// `KeRemoveByKeyDeviceQueue(DeviceQueue, SortKey)`: the first entry whose key is at least
/// `SortKey`, or the first entry when none is.
pub fn remove_by_key_device_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let key = c.arg(1);
    value(remove_device_entry(k, c.arg(0), |k, head| {
        first_entry_by_key(k, head, key)
    })?)
}

/// `KeRemoveEntryDeviceQueue(DeviceQueue, DeviceQueueEntry)`: returns whether the entry was
/// queued.
pub fn remove_entry_device_queue(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let entry = c.arg(1);
    if k.r8(entry + ENTRY_INSERTED)? == 0 {
        return value(0);
    }
    unlink(k, entry)?;
    k.w8(entry + ENTRY_INSERTED, 0)?;
    value(1)
}
