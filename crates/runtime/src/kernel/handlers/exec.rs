//! Executive exports: reader/writer locks, and the console's refurbishment record.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission, and adapted
//! to this crate's handler interface and its waits (spec 8.3).

use super::ke::release_semaphore_object;
use super::value;
use crate::kernel::object::Body;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `ERWLOCK` offsets: the counts, then the event writers wait on and the semaphore readers
/// wait on.
const LOCK_COUNT: u32 = 0x00;
const WRITERS_WAITING: u32 = 0x04;
const READERS_WAITING: u32 = 0x08;
const READERS_ENTRY: u32 = 0x0C;
const WRITER_EVENT: u32 = 0x10;
const READER_SEMAPHORE: u32 = 0x20;

/// Writes a dispatcher header: type, size in dwords, signal state, and an empty wait list.
fn write_header(k: &Kernel, at: u32, kind: u8, size: u8, signal: u32) -> KResult<()> {
    k.write(at, &[kind, 0, size, 0])?;
    k.w32(at + 4, signal)?;
    k.w32(at + 8, at + 8)?;
    k.w32(at + 12, at + 8)
}

/// `ExInitializeReadWriteLock(ReadWriteLock)`.
pub fn initialize_read_write_lock(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let lock = c.arg(0);
    k.w32(lock + LOCK_COUNT, u32::MAX)?;
    for field in [WRITERS_WAITING, READERS_WAITING, READERS_ENTRY] {
        k.w32(lock + field, 0)?;
    }
    // A synchronization event, not signaled, and a semaphore at zero with no real limit.
    write_header(k, lock + WRITER_EVENT, 1, 4, 0)?;
    k.objects.add_at(
        lock + WRITER_EVENT,
        Body::Event {
            sync: true,
            signaled: false,
        },
    );
    write_header(k, lock + READER_SEMAPHORE, 5, 5, 0)?;
    k.w32(lock + READER_SEMAPHORE + 0x10, i32::MAX.cast_unsigned())?;
    k.objects.add_at(
        lock + READER_SEMAPHORE,
        Body::Semaphore {
            count: 0,
            limit: i32::MAX,
        },
    );
    value(0)
}

/// Adds `delta` to the dword at `address`; returns the new value.
fn add(k: &Kernel, address: u32, delta: i32) -> KResult<u32> {
    let value = k.r32(address)?.wrapping_add_signed(delta);
    k.w32(address, value)?;
    Ok(value)
}

/// Waits on the lock's event or semaphore at `address`.
fn wait_at(k: &mut Kernel, address: u32) -> KResult<CallResult> {
    let Some(id) = k.objects.at(address) else {
        return Ok(CallResult::Unsupported(
            "a read/write lock that was never initialized".to_owned(),
        ));
    };
    k.wait_for(vec![id], false, 0)
}

/// `ExAcquireReadWriteLockExclusive(ReadWriteLock)`: a writer waits while anyone holds the lock.
pub fn acquire_read_write_lock_exclusive(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let lock = c.arg(0);
    if add(k, lock + LOCK_COUNT, 1)? != 0 {
        add(k, lock + WRITERS_WAITING, 1)?;
        return wait_at(k, lock + WRITER_EVENT);
    }
    value(0)
}

/// `ExAcquireReadWriteLockShared(ReadWriteLock)`: a reader waits while a writer holds the lock
/// or one is waiting for it.
pub fn acquire_read_write_lock_shared(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let lock = c.arg(0);
    let readers = k.r32(lock + READERS_ENTRY)?;
    let writers_waiting = k.r32(lock + WRITERS_WAITING)?;
    let must_wait = readers == 0 || writers_waiting != 0;
    if add(k, lock + LOCK_COUNT, 1)? != 0 && must_wait {
        add(k, lock + READERS_WAITING, 1)?;
        return wait_at(k, lock + READER_SEMAPHORE);
    }
    add(k, lock + READERS_ENTRY, 1)?;
    value(0)
}

/// `ExReleaseReadWriteLock(ReadWriteLock)`: the last reader out lets a writer in; a writer out
/// lets the waiting readers in, or else the next writer.
pub fn release_read_write_lock(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let lock = c.arg(0);
    if add(k, lock + LOCK_COUNT, -1)? == u32::MAX {
        k.w32(lock + READERS_ENTRY, 0)?;
        return value(0);
    }
    if k.r32(lock + READERS_ENTRY)? == 0 {
        let waiting = k.r32(lock + READERS_WAITING)?;
        if waiting != 0 {
            k.w32(lock + READERS_ENTRY, waiting)?;
            k.w32(lock + READERS_WAITING, 0)?;
            if let Some(id) = k.objects.at(lock + READER_SEMAPHORE) {
                release_semaphore_object(k, id, waiting.cast_signed());
            }
            return value(0);
        }
    } else if add(k, lock + READERS_ENTRY, -1)? != 0 {
        return value(0);
    }
    add(k, lock + WRITERS_WAITING, -1)?;
    if let Some(id) = k.objects.at(lock + WRITER_EVENT) {
        k.set_event(id);
    }
    value(0)
}

/// `ExReadWriteRefurbInfo(RefurbInfo, ValueLength, DoWrite)`: the hard disk's refurbishment
/// record (signature, power cycles, first boot time). A console without one reads zeros;
/// writing it changes nothing kept.
pub fn read_write_refurb_info(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    /// `XBOX_REFURB_INFO`: three fields, 16 bytes.
    const REFURB_INFO_LEN: u32 = 16;
    let (info, len, write) = (c.arg(0), c.arg(1), c.arg(2) & 0xFF != 0);
    if info == 0 || len != REFURB_INFO_LEN {
        return value(status::INVALID_PARAMETER);
    }
    if !write {
        k.write(info, &[0; REFURB_INFO_LEN as usize])?;
    }
    value(status::SUCCESS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::handlers::lookup;
    use crate::kernel::testing::{kernel, run_as, threads};
    use crate::kernel::thread::State;

    /// Guest scratch memory inside the committed heap.
    const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;
    const LOCK: u32 = SCRATCH;

    fn call(k: &mut Kernel, name: &str, args: &[u32]) -> CallResult {
        let handler = lookup(name).unwrap_or_else(|| panic!("no handler for {name}"));
        handler(
            k,
            &Call {
                args: args.to_vec(),
                esp: 0,
            },
        )
        .unwrap()
    }

    fn state(k: &Kernel, tid: u32) -> State {
        k.threads.get(tid).unwrap().state
    }

    #[test]
    fn readers_share_and_a_writer_waits_for_the_last_one() {
        let mut k = kernel();
        let ids = threads(&mut k, 2);
        call(&mut k, "ExInitializeReadWriteLock", &[LOCK]);
        assert_eq!(k.r32(LOCK + LOCK_COUNT), Ok(u32::MAX));
        assert_eq!(k.r8(LOCK + WRITER_EVENT), Ok(1));
        assert_eq!(k.r8(LOCK + READER_SEMAPHORE), Ok(5));
        // Two readers at once.
        let shared = "ExAcquireReadWriteLockShared";
        assert_eq!(call(&mut k, shared, &[LOCK]), CallResult::Value(0));
        run_as(&mut k, ids[0]);
        assert_eq!(call(&mut k, shared, &[LOCK]), CallResult::Value(0));
        assert_eq!(k.r32(LOCK + READERS_ENTRY), Ok(2));
        // A writer waits for both.
        run_as(&mut k, ids[1]);
        let exclusive = "ExAcquireReadWriteLockExclusive";
        assert_eq!(call(&mut k, exclusive, &[LOCK]), CallResult::Block);
        let release = "ExReleaseReadWriteLock";
        run_as(&mut k, ids[0]);
        call(&mut k, release, &[LOCK]);
        assert_eq!(state(&k, ids[1]), State::Waiting);
        run_as(&mut k, 1);
        call(&mut k, release, &[LOCK]);
        assert_eq!(state(&k, ids[1]), State::Ready, "the last reader let it in");
        assert_eq!(k.r32(LOCK + WRITERS_WAITING), Ok(0));
        assert_eq!(k.r32(LOCK + LOCK_COUNT), Ok(0), "the writer holds it");
    }

    #[test]
    fn a_writer_out_lets_every_waiting_reader_in() {
        let mut k = kernel();
        let ids = threads(&mut k, 2);
        call(&mut k, "ExInitializeReadWriteLock", &[LOCK]);
        assert_eq!(
            call(&mut k, "ExAcquireReadWriteLockExclusive", &[LOCK]),
            CallResult::Value(0)
        );
        for &t in &ids {
            run_as(&mut k, t);
            assert_eq!(
                call(&mut k, "ExAcquireReadWriteLockShared", &[LOCK]),
                CallResult::Block
            );
        }
        assert_eq!(k.r32(LOCK + READERS_WAITING), Ok(2));
        run_as(&mut k, 1);
        call(&mut k, "ExReleaseReadWriteLock", &[LOCK]);
        assert!(ids.iter().all(|&t| state(&k, t) == State::Ready));
        assert_eq!(k.r32(LOCK + READERS_ENTRY), Ok(2));
        assert_eq!(k.r32(LOCK + READERS_WAITING), Ok(0));
        // Both readers out: the lock is free again.
        for &t in &ids {
            run_as(&mut k, t);
            call(&mut k, "ExReleaseReadWriteLock", &[LOCK]);
        }
        assert_eq!(k.r32(LOCK + LOCK_COUNT), Ok(u32::MAX));
        assert_eq!(k.r32(LOCK + READERS_ENTRY), Ok(0));
    }

    #[test]
    fn a_lock_never_initialized_cannot_be_waited_on() {
        let mut k = kernel();
        k.w32(LOCK + LOCK_COUNT, 0).unwrap();
        assert!(matches!(
            call(&mut k, "ExAcquireReadWriteLockExclusive", &[LOCK]),
            CallResult::Unsupported(_)
        ));
    }

    #[test]
    fn the_refurbishment_record_reads_as_zeros() {
        let mut k = kernel();
        let info = SCRATCH + 0x100;
        k.write(info, &[0xEE; 16]).unwrap();
        let refurb = "ExReadWriteRefurbInfo";
        assert_eq!(
            call(&mut k, refurb, &[info, 16, 0]),
            CallResult::Value(status::SUCCESS)
        );
        assert_eq!(k.read(info, 16).unwrap(), [0; 16]);
        assert_eq!(
            call(&mut k, refurb, &[info, 12, 0]),
            CallResult::Value(status::INVALID_PARAMETER)
        );
        assert_eq!(
            call(&mut k, refurb, &[info, 16, 1]),
            CallResult::Value(status::SUCCESS)
        );
    }
}
