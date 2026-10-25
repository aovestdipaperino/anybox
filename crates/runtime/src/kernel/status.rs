//! NTSTATUS values (spec 8.2).

pub const SUCCESS: u32 = 0;
pub const TIMEOUT: u32 = 0x102;
pub const PENDING: u32 = 0x103;
pub const OBJECT_NAME_EXISTS: u32 = 0x4000_0000;
pub const BUFFER_OVERFLOW: u32 = 0x8000_0005;
pub const NO_MORE_FILES: u32 = 0x8000_0006;
pub const NO_MORE_ENTRIES: u32 = 0x8000_001A;
pub const UNSUCCESSFUL: u32 = 0xC000_0001;
pub const INVALID_INFO_CLASS: u32 = 0xC000_0003;
pub const INFO_LENGTH_MISMATCH: u32 = 0xC000_0004;
pub const INVALID_HANDLE: u32 = 0xC000_0008;
pub const INVALID_PARAMETER: u32 = 0xC000_000D;
pub const NO_SUCH_FILE: u32 = 0xC000_000F;
pub const INVALID_DEVICE_REQUEST: u32 = 0xC000_0010;
pub const END_OF_FILE: u32 = 0xC000_0011;
pub const MORE_PROCESSING_REQUIRED: u32 = 0xC000_0016;
pub const NO_MEMORY: u32 = 0xC000_0017;
pub const CONFLICTING_ADDRESSES: u32 = 0xC000_0018;
pub const ACCESS_DENIED: u32 = 0xC000_0022;
pub const BUFFER_TOO_SMALL: u32 = 0xC000_0023;
pub const OBJECT_TYPE_MISMATCH: u32 = 0xC000_0024;
pub const OBJECT_NAME_NOT_FOUND: u32 = 0xC000_0034;
pub const OBJECT_NAME_COLLISION: u32 = 0xC000_0035;
pub const OBJECT_PATH_NOT_FOUND: u32 = 0xC000_003A;
pub const SHARING_VIOLATION: u32 = 0xC000_0043;
pub const MUTANT_NOT_OWNED: u32 = 0xC000_0046;
pub const SEMAPHORE_LIMIT_EXCEEDED: u32 = 0xC000_0047;
pub const MEDIA_WRITE_PROTECTED: u32 = 0xC000_00A2;
pub const FILE_IS_A_DIRECTORY: u32 = 0xC000_00BA;
pub const NOT_SAME_DEVICE: u32 = 0xC000_00D4;
pub const DIRECTORY_NOT_EMPTY: u32 = 0xC000_0101;
pub const NOT_A_DIRECTORY: u32 = 0xC000_0103;
pub const CANCELLED: u32 = 0xC000_0120;

/// `RtlNtStatusToDosError` (spec 8.2).
#[must_use]
pub fn to_dos_error(status: u32) -> u32 {
    match status {
        SUCCESS => 0,
        NO_SUCH_FILE | OBJECT_NAME_NOT_FOUND => 2,
        ACCESS_DENIED => 5,
        INVALID_HANDLE => 6,
        NO_MEMORY => 8,
        END_OF_FILE => 38,
        INVALID_PARAMETER => 87,
        _ => 317,
    }
}

/// Whether a status is an error (severity bits `11`).
#[must_use]
pub fn is_error(status: u32) -> bool {
    status >> 30 == 3
}
