use super::*;

const BASE: u32 = 0x1_0000;
const ENTRY: u32 = 0x1_1010;
const THUNK: u32 = 0x1_1000;

fn put_u32(buf: &mut [u8], offset: usize, value: u32) {
    buf[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u16(buf: &mut [u8], offset: usize, value: u16) {
    buf[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Builds a one-section XBE whose `.text` holds a two-entry thunk table.
fn sample(entry_key: u32, thunk_key: u32) -> Vec<u8> {
    let mut d = vec![0u8; 0x1100];
    d[..4].copy_from_slice(b"XBEH");
    put_u32(&mut d, 0x104, BASE);
    put_u32(&mut d, 0x108, 0x1000);
    put_u32(&mut d, 0x10C, 0x2000);
    put_u32(&mut d, 0x110, 0x178);
    put_u32(&mut d, 0x118, BASE + 0x200);
    put_u32(&mut d, 0x11C, 1);
    put_u32(&mut d, 0x120, BASE + 0x400);
    put_u32(&mut d, 0x128, ENTRY ^ entry_key);
    put_u32(&mut d, 0x12C, BASE + 0x700);
    put_u32(&mut d, 0x158, THUNK ^ thunk_key);
    put_u32(&mut d, 0x160, 1);
    put_u32(&mut d, 0x164, BASE + 0x600);
    // Certificate: title ID "UB-034", name, region, version.
    put_u32(&mut d, 0x208, 0x5542_0022);
    for (i, unit) in "Test Game".encode_utf16().enumerate() {
        put_u16(&mut d, 0x20C + i * 2, unit);
    }
    put_u32(&mut d, 0x2A0, 1);
    put_u32(&mut d, 0x2AC, 0x100);
    // Section header for .text.
    put_u32(&mut d, 0x400, 0x16);
    put_u32(&mut d, 0x404, THUNK);
    put_u32(&mut d, 0x408, 0x200);
    put_u32(&mut d, 0x40C, 0x1000);
    put_u32(&mut d, 0x410, 0x100);
    put_u32(&mut d, 0x414, BASE + 0x500);
    d[0x500..0x506].copy_from_slice(b".text\0");
    // Library version XAPILIB 1.0.5849.
    d[0x600..0x607].copy_from_slice(b"XAPILIB");
    put_u16(&mut d, 0x608, 1);
    put_u16(&mut d, 0x60C, 5849);
    // Kernel thunk table: ordinals 1 and 156, then terminator.
    put_u32(&mut d, 0x1000, 0x8000_0001);
    put_u32(&mut d, 0x1004, 0x8000_009C);
    d
}

fn retail() -> Vec<u8> {
    sample(RETAIL_ENTRY_KEY, RETAIL_THUNK_KEY)
}

#[test]
fn parses_retail_header() {
    let xbe = Xbe::parse(retail()).unwrap();
    assert_eq!(xbe.build_kind(), BuildKind::Retail);
    assert_eq!(xbe.base_address(), BASE);
    assert_eq!(xbe.headers_size(), 0x1000);
    assert_eq!(xbe.image_size(), 0x2000);
    assert_eq!(xbe.entry_point(), ENTRY);
    assert_eq!(xbe.kernel_thunk_address(), THUNK);
    assert_eq!(xbe.tls_address(), BASE + 0x700);
}

#[test]
fn parses_certificate() {
    let xbe = Xbe::parse(retail()).unwrap();
    let cert = xbe.certificate();
    assert_eq!(cert.title_name(), "Test Game");
    assert_eq!(cert.title_id(), 0x5542_0022);
    assert_eq!(cert.title_id_display(), "UB-034");
    assert_eq!(cert.game_region(), 1);
    assert_eq!(cert.version(), 0x100);
}

#[test]
fn parses_sections_and_libraries() {
    let xbe = Xbe::parse(retail()).unwrap();
    let text = &xbe.sections()[0];
    assert_eq!(text.name(), ".text");
    assert_eq!(text.virtual_address(), THUNK);
    assert_eq!(text.virtual_size(), 0x200);
    assert_eq!(text.raw_address(), 0x1000);
    assert_eq!(text.raw_size(), 0x100);
    let flags = text.flags();
    assert!(flags.preload() && flags.executable() && flags.head_page_read_only());
    assert!(!flags.writable() && !flags.inserted_file() && !flags.tail_page_read_only());
    assert_eq!(xbe.section_data(text).len(), 0x100);
    assert_eq!(xbe.libraries()[0].to_string(), "XAPILIB 1.0.5849");
}

#[test]
fn reads_kernel_imports() {
    let xbe = Xbe::parse(retail()).unwrap();
    assert_eq!(xbe.kernel_imports().unwrap(), [1, 156]);
}

#[test]
fn reads_virtual_addresses_in_headers_and_sections() {
    let xbe = Xbe::parse(retail()).unwrap();
    assert_eq!(xbe.read_virtual(BASE, 4).unwrap(), b"XBEH");
    assert_eq!(
        xbe.read_virtual(THUNK, 4).unwrap(),
        0x8000_0001u32.to_le_bytes()
    );
    assert!(xbe.read_virtual(THUNK + 0xFE, 4).is_none());
    assert!(xbe.read_virtual(0x9000_0000, 1).is_none());
}

#[test]
fn detects_debug_keys() {
    let xbe = Xbe::parse(sample(DEBUG_ENTRY_KEY, DEBUG_THUNK_KEY)).unwrap();
    assert_eq!(xbe.build_kind(), BuildKind::Debug);
    assert_eq!(xbe.entry_point(), ENTRY);
    assert_eq!(xbe.kernel_thunk_address(), THUNK);
}

#[test]
fn unknown_keys_are_rejected() {
    assert!(Xbe::parse(sample(0, 0)).unwrap_err().is_unknown_keys());
}

#[test]
fn bad_magic_is_rejected() {
    let mut d = retail();
    d[0] = b'M';
    let err = Xbe::parse(d).unwrap_err();
    assert!(err.is_bad_magic());
    assert!(err.to_string().contains("XBEH"));
}

#[test]
fn truncated_header_is_corrupt() {
    let mut d = retail();
    d.truncate(0x100);
    assert!(Xbe::parse(d).unwrap_err().is_corrupt());
}

#[test]
fn section_past_end_of_file_is_corrupt() {
    let mut d = retail();
    put_u32(&mut d, 0x410, 0x1000);
    assert!(Xbe::parse(d).unwrap_err().is_corrupt());
}

#[test]
fn import_by_name_is_corrupt() {
    let mut d = retail();
    put_u32(&mut d, 0x1000, 1);
    let xbe = Xbe::parse(d).unwrap();
    assert!(xbe.kernel_imports().unwrap_err().is_corrupt());
}
