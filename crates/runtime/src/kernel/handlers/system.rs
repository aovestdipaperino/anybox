//! x86 port string I/O and the file system cache's size.
//!
//! Copied from the author's original `anybox-tk` code, with the author's permission, and adapted
//! to this crate's handler interface.

use super::value;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// The file system cache's size, in pages.
const DEFAULT_CACHE_PAGES: u32 = 64;
const MAXIMUM_CACHE_PAGES: u32 = 2048;

/// `FscGetCacheSize()`.
pub fn get_cache_size(k: &mut Kernel, _: &Call) -> KResult<CallResult> {
    value(k.cache_pages.unwrap_or(DEFAULT_CACHE_PAGES))
}

/// `FscSetCacheSize(NumberOfCachePages)`: files are not cached here, but the size reads back.
pub fn set_cache_size(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let pages = c.arg(0);
    if pages > MAXIMUM_CACHE_PAGES {
        return value(status::INVALID_PARAMETER);
    }
    k.cache_pages = Some(pages);
    value(status::SUCCESS)
}

/// `READ_PORT_BUFFER_*(Port, Buffer, Count)`: no device answers on the ports titles reach, so
/// reads float high.
fn read_port_buffer(k: &Kernel, c: &Call, width: u32) -> KResult<CallResult> {
    k.write(c.arg(1), &vec![0xFF; (width * c.arg(2)) as usize])?;
    value(0)
}

/// `READ_PORT_BUFFER_UCHAR(Port, Buffer, Count)`.
pub fn read_port_buffer_uchar(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    read_port_buffer(k, c, 1)
}

/// `READ_PORT_BUFFER_USHORT(Port, Buffer, Count)`.
pub fn read_port_buffer_ushort(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    read_port_buffer(k, c, 2)
}

/// `READ_PORT_BUFFER_ULONG(Port, Buffer, Count)`.
pub fn read_port_buffer_ulong(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    read_port_buffer(k, c, 4)
}

#[cfg(test)]
mod tests {
    use crate::kernel::handlers::lookup;
    use crate::kernel::testing::kernel;
    use crate::kernel::{Call, CallResult, Kernel, status};

    const SCRATCH: u32 = crate::mem::HEAP_BASE + 0x2F_0000;

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

    #[test]
    fn the_cache_size_reads_back_within_the_limit() {
        let mut k = kernel();
        assert_eq!(call(&mut k, "FscGetCacheSize", &[]), CallResult::Value(64));
        assert_eq!(
            call(&mut k, "FscSetCacheSize", &[300]),
            CallResult::Value(status::SUCCESS)
        );
        assert_eq!(call(&mut k, "FscGetCacheSize", &[]), CallResult::Value(300));
        assert_eq!(
            call(&mut k, "FscSetCacheSize", &[2049]),
            CallResult::Value(status::INVALID_PARAMETER)
        );
        assert_eq!(call(&mut k, "FscGetCacheSize", &[]), CallResult::Value(300));
        assert_eq!(
            call(&mut k, "FscInvalidateIdleBlocks", &[]),
            CallResult::Value(0)
        );
    }

    #[test]
    fn port_reads_float_high_and_writes_go_nowhere() {
        let mut k = kernel();
        let buffer = SCRATCH;
        k.write(buffer, &[0; 16]).unwrap();
        call(&mut k, "READ_PORT_BUFFER_USHORT", &[0x1F0, buffer, 3]);
        assert_eq!(
            k.read(buffer, 8).unwrap(),
            [0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0, 0]
        );
        call(&mut k, "READ_PORT_BUFFER_ULONG", &[0x1F0, buffer, 2]);
        assert_eq!(k.r64(buffer), Ok(u64::MAX));
        call(&mut k, "READ_PORT_BUFFER_UCHAR", &[0x1F0, buffer + 8, 1]);
        assert_eq!(k.read(buffer + 8, 2).unwrap(), [0xFF, 0]);
        for name in [
            "WRITE_PORT_BUFFER_UCHAR",
            "WRITE_PORT_BUFFER_USHORT",
            "WRITE_PORT_BUFFER_ULONG",
        ] {
            assert_eq!(
                call(&mut k, name, &[0x1F0, buffer, 4]),
                CallResult::Value(0)
            );
        }
    }
}
