use std::io::Cursor;

use super::*;

const PADDING: u8 = 0xFF;

/// Builds a directory table whose records are chained through right links.
fn dir_table(records: &[(&str, u32, u32, u8)]) -> Vec<u8> {
    let mut table = Vec::new();
    for (index, (name, sector, size, attributes)) in records.iter().enumerate() {
        let start = table.len();
        let len = (DIR_ENTRY_HEADER + name.len() + 3) & !3;
        let right = if index + 1 < records.len() {
            u16::try_from((start + len) / 4).unwrap()
        } else {
            0
        };
        table.extend_from_slice(&0u16.to_le_bytes());
        table.extend_from_slice(&right.to_le_bytes());
        table.extend_from_slice(&sector.to_le_bytes());
        table.extend_from_slice(&size.to_le_bytes());
        table.push(*attributes);
        table.push(u8::try_from(name.len()).unwrap());
        table.extend_from_slice(name.as_bytes());
        table.resize(start + len, PADDING);
    }
    table
}

/// Builds an image with `default.xbe` and `Media/intro.xmv` at `partition`.
fn image(partition: usize) -> Vec<u8> {
    let sector = |n: usize| partition + n * SECTOR_LEN;
    let media = dir_table(&[("intro.xmv", 41, 3, 0x20)]);
    let media_len = u32::try_from(media.len()).unwrap();
    let root = dir_table(&[("Media", 35, media_len, 0x10), ("default.xbe", 40, 5, 0x20)]);
    let mut img = vec![0u8; sector(42)];
    let vd = sector(32);
    img[vd..vd + MAGIC.len()].copy_from_slice(MAGIC);
    img[vd + 20..vd + 24].copy_from_slice(&33u32.to_le_bytes());
    img[vd + 24..vd + 28].copy_from_slice(&u32::try_from(root.len()).unwrap().to_le_bytes());
    img[vd + TRAILING_MAGIC_OFFSET..vd + SECTOR_LEN].copy_from_slice(MAGIC);
    img[sector(33)..sector(33) + root.len()].copy_from_slice(&root);
    img[sector(35)..sector(35) + media.len()].copy_from_slice(&media);
    img[sector(40)..sector(40) + 5].copy_from_slice(b"HELLO");
    img[sector(41)..sector(41) + 3].copy_from_slice(b"abc");
    img
}

fn names(entries: &[Entry]) -> Vec<&str> {
    entries.iter().map(Entry::name).collect()
}

#[test]
fn opens_plain_xiso_and_lists_root_sorted() {
    let mut iso = XIso::open(Cursor::new(image(0))).unwrap();
    assert_eq!(iso.partition_offset(), 0);
    let root = iso.root().clone();
    assert!(root.is_dir());
    let entries = iso.read_dir(&root).unwrap();
    assert_eq!(names(&entries), ["default.xbe", "Media"]);
    assert_eq!(entries[0].size(), 5);
    assert_eq!(entries[0].sector(), 40);
    assert!(entries[1].is_dir());
}

#[test]
fn opens_redump_image_at_partition_offset() {
    let iso = XIso::open(Cursor::new(image(0x0208_0000))).unwrap();
    assert_eq!(iso.partition_offset(), 0x0208_0000);
}

#[test]
fn lookup_is_case_insensitive_and_nested() {
    let mut iso = XIso::open(Cursor::new(image(0))).unwrap();
    let xbe = iso.lookup("DEFAULT.XBE").unwrap().unwrap();
    assert_eq!(iso.read_file(&xbe).unwrap(), b"HELLO");
    let movie = iso.lookup("\\media\\Intro.XMV").unwrap().unwrap();
    assert_eq!(iso.read_file(&movie).unwrap(), b"abc");
    let same = iso.lookup("Media/intro.xmv").unwrap().unwrap();
    assert_eq!(same, movie);
}

#[test]
fn lookup_of_missing_paths_is_none() {
    let mut iso = XIso::open(Cursor::new(image(0))).unwrap();
    assert!(iso.lookup("nope.txt").unwrap().is_none());
    assert!(iso.lookup("default.xbe/inner").unwrap().is_none());
    assert!(iso.lookup("").unwrap().unwrap().is_dir());
}

#[test]
fn rejects_images_without_volume_descriptor() {
    let err = XIso::open(Cursor::new(vec![0u8; 0x20000])).unwrap_err();
    assert!(err.is_not_xdvdfs());
    assert!(err.to_string().contains("XDVDFS"));
}

#[test]
fn reading_a_directory_as_a_file_fails() {
    let mut iso = XIso::open(Cursor::new(image(0))).unwrap();
    let media = iso.lookup("Media").unwrap().unwrap();
    assert!(iso.read_file(&media).unwrap_err().is_a_directory());
    let xbe = iso.lookup("default.xbe").unwrap().unwrap();
    assert!(iso.read_dir(&xbe).unwrap_err().is_not_a_directory());
}

#[test]
fn truncated_extent_is_corrupt() {
    let mut img = image(0);
    img.truncate(41 * SECTOR_LEN + 1);
    let mut iso = XIso::open(Cursor::new(img)).unwrap();
    let movie = iso.lookup("Media/intro.xmv").unwrap().unwrap();
    assert!(iso.read_file(&movie).unwrap_err().is_corrupt());
}

#[test]
fn cyclic_directory_tree_is_corrupt() {
    let mut img = image(0);
    // Point the second root record's right link at itself.
    let root = 33 * SECTOR_LEN;
    let first_len = (DIR_ENTRY_HEADER + "Media".len() + 3) & !3;
    let self_link = u16::try_from(first_len / 4).unwrap().to_le_bytes();
    img[root + first_len + 2..root + first_len + 4].copy_from_slice(&self_link);
    let mut iso = XIso::open(Cursor::new(img)).unwrap();
    let root = iso.root().clone();
    assert!(iso.read_dir(&root).unwrap_err().is_corrupt());
}

#[test]
fn read_at_returns_ranges_and_zero_past_end() {
    let mut iso = XIso::open(Cursor::new(image(0))).unwrap();
    let xbe = iso.lookup("default.xbe").unwrap().unwrap(); // "HELLO"
    let mut buf = [0u8; 3];
    assert_eq!(iso.read_at(&xbe, 0, &mut buf).unwrap(), 3);
    assert_eq!(&buf, b"HEL");
    assert_eq!(
        iso.read_at(&xbe, 3, &mut buf).unwrap(),
        2,
        "short read at the end"
    );
    assert_eq!(&buf[..2], b"LO");
    assert_eq!(iso.read_at(&xbe, 5, &mut buf).unwrap(), 0);
    assert_eq!(iso.read_at(&xbe, 99, &mut buf).unwrap(), 0);
    assert_eq!(iso.read_at(&xbe, 1, &mut []).unwrap(), 0);
    let media = iso.lookup("Media").unwrap().unwrap();
    assert!(
        iso.read_at(&media, 0, &mut buf)
            .unwrap_err()
            .is_a_directory()
    );
}
