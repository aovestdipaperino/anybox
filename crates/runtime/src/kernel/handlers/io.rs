//! I/O exports: files, directories, volumes, device controls, symbolic links and devices (spec
//! 8.9, 8.10).

use super::nt::attributes_name;
use super::value;
use crate::kernel::fs::{
    Entry, FileSystem, Listing, Location, OpenFile, PARTITIONS, RAW_DISK_SECTORS, wildcard,
};
use crate::kernel::object::{Body, ObjId};
use crate::kernel::time::BOOT_CLOCK;
use crate::kernel::{Call, CallResult, KResult, Kernel, status};

/// `FILE_USE_FILE_POINTER_POSITION`: an offset that means "the file position".
const USE_FILE_POSITION: u64 = 0xFFFF_FFFF_FFFF_FFFE;

/// `IOCTL_SCSI_PASS_THROUGH_DIRECT`, `FSCTL_DISMOUNT_VOLUME`, `IOCTL_DISK_GET_DRIVE_GEOMETRY`,
/// `IOCTL_DISK_GET_PARTITION_INFO`.
const IOCTL_SCSI_PASS_THROUGH_DIRECT: u32 = 0x4D014;
const FSCTL_DISMOUNT_VOLUME: u32 = 0x9_0020;
const IOCTL_DISK_GET_DRIVE_GEOMETRY: u32 = 0x7_0000;
const IOCTL_DISK_GET_PARTITION_INFO: u32 = 0x7_4004;

/// Writes an `IO_STATUS_BLOCK`: status, information.
fn complete(k: &Kernel, iosb: u32, status: u32, information: u32) -> KResult<()> {
    if iosb != 0 {
        k.w32(iosb, status)?;
        k.w32(iosb + 4, information)?;
    }
    Ok(())
}

/// The open file a handle names.
fn file(k: &Kernel, handle: u32) -> Option<ObjId> {
    k.objects
        .handle(handle)
        .filter(|&id| matches!(k.objects.get(id).body, Body::File(_)))
}

/// The open file behind an object id.
fn opened(k: &mut Kernel, id: ObjId) -> &mut OpenFile {
    match &mut k.objects.get_mut(id).body {
        Body::File(file) => file,
        _ => unreachable!("the object is a file"),
    }
}

/// Opens a file and writes its handle and status block.
fn create(
    k: &mut Kernel,
    handle_out: u32,
    attributes: u32,
    iosb: u32,
    disposition: u32,
    options: u32,
) -> KResult<CallResult> {
    let Some(path) = attributes_name(k, attributes)? else {
        complete(k, iosb, status::OBJECT_NAME_NOT_FOUND, 0)?;
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    match k.fs.open(&path, disposition, options) {
        Ok((file, information)) => {
            let id = k.objects.add(Body::File(Box::new(file)));
            let handle = k.objects.open(id);
            k.w32(handle_out, handle)?;
            complete(k, iosb, status::SUCCESS, information)?;
            value(status::SUCCESS)
        }
        Err(code) => {
            complete(k, iosb, code, 0)?;
            value(code)
        }
    }
}

/// `NtCreateFile(*h, access, attr, *iosb, *alloc, attrs, share, disposition, options)`.
pub fn create_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    create(k, c.arg(0), c.arg(2), c.arg(3), c.arg(7), c.arg(8))
}

/// `IoCreateFile`: `NtCreateFile` with one more argument.
pub fn io_create_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    create(k, c.arg(0), c.arg(2), c.arg(3), c.arg(7), c.arg(8))
}

/// `NtOpenFile(*h, access, attr, *iosb, share, options)`.
pub fn open(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    create(k, c.arg(0), c.arg(2), c.arg(3), 1, c.arg(5))
}

/// The offset a read or write starts at: `*offset`, or the file position.
fn start_offset(k: &Kernel, position: u64, offset_ptr: u32) -> KResult<u64> {
    if offset_ptr == 0 {
        return Ok(position);
    }
    let offset = k.r64(offset_ptr)?;
    Ok(if offset == USE_FILE_POSITION {
        position
    } else {
        offset
    })
}

/// After a read or write: sets the event and starts the APC routine `(context, iosb, 0)`.
fn signal_completion(k: &mut Kernel, event: u32, apc: u32, context: u32, iosb: u32) {
    if event != 0
        && let Some(id) = k.object_of_handle(event)
    {
        k.set_event(id);
    }
    if apc != 0 {
        k.run_routine(apc, &[context, iosb, 0]);
    }
}

/// `NtReadFile(h, event, apc, ctx, *iosb, buf, len, *offset)`.
pub fn read_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let (location, position) = {
        let f = opened(k, id);
        (f.location, f.position)
    };
    let offset = start_offset(k, position, c.arg(7))?;
    let len = c.arg(6);
    let code = match k.fs.read(location, offset, len) {
        Ok(data) if data.is_empty() && len > 0 => status::END_OF_FILE,
        Ok(data) => {
            k.write(c.arg(5), &data)?;
            let n = u32::try_from(data.len()).unwrap_or(0);
            opened(k, id).position = offset + u64::from(n);
            complete(k, c.arg(4), status::SUCCESS, n)?;
            signal_completion(k, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
            return value(status::SUCCESS);
        }
        Err(code) => code,
    };
    complete(k, c.arg(4), code, 0)?;
    signal_completion(k, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    value(code)
}

/// `NtWriteFile(h, event, apc, ctx, *iosb, buf, len, *offset)`.
pub fn write_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let (location, position) = {
        let f = opened(k, id);
        (f.location, f.position)
    };
    let offset = start_offset(k, position, c.arg(7))?;
    let data = k.read(c.arg(5), c.arg(6))?;
    let (code, n) = match k.fs.write(location, offset, &data) {
        Ok(n) => {
            opened(k, id).position = offset + n as u64;
            (status::SUCCESS, u32::try_from(n).unwrap_or(0))
        }
        Err(code) => (code, 0),
    };
    complete(k, c.arg(4), code, n)?;
    signal_completion(k, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    value(code)
}

/// The pages a scatter/gather transfer names: `FILE_SEGMENT_ELEMENT`s, 64-bit pointers to whole
/// pages, as many as `length` needs. Copied from the author's original anybox-tk code, with the
/// author's permission.
fn segments(k: &Kernel, array: u32, length: u32) -> KResult<Vec<(u32, u32)>> {
    const PAGE: u32 = 0x1000;
    let mut pieces = Vec::new();
    let mut left = length;
    let mut index = 0;
    while left > 0 {
        let page = k.r32(array + 8 * index)?;
        let len = left.min(PAGE);
        pieces.push((page, len));
        left -= len;
        index += 1;
    }
    Ok(pieces)
}

/// `NtReadFileScatter(h, event, apc, ctx, *iosb, segments, len, *offset)`: as `NtReadFile`,
/// into separate pages. Copied from the author's original anybox-tk code, with the author's
/// permission.
pub fn read_file_scatter(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let (location, position) = {
        let f = opened(k, id);
        (f.location, f.position)
    };
    let pieces = segments(k, c.arg(5), c.arg(6))?;
    let offset = start_offset(k, position, c.arg(7))?;
    let len = c.arg(6);
    let code = match k.fs.read(location, offset, len) {
        Ok(data) if data.is_empty() && len > 0 => status::END_OF_FILE,
        Ok(data) => {
            let mut rest = data.as_slice();
            for (page, piece) in pieces {
                let (head, tail) = rest.split_at((piece as usize).min(rest.len()));
                k.write(page, head)?;
                rest = tail;
            }
            let n = u32::try_from(data.len()).unwrap_or(0);
            opened(k, id).position = offset + u64::from(n);
            complete(k, c.arg(4), status::SUCCESS, n)?;
            signal_completion(k, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
            return value(status::SUCCESS);
        }
        Err(code) => code,
    };
    complete(k, c.arg(4), code, 0)?;
    signal_completion(k, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    value(code)
}

/// `NtWriteFileGather(h, event, apc, ctx, *iosb, segments, len, *offset)`: as `NtWriteFile`,
/// from separate pages. Copied from the author's original anybox-tk code, with the author's
/// permission.
pub fn write_file_gather(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let (location, position) = {
        let f = opened(k, id);
        (f.location, f.position)
    };
    let offset = start_offset(k, position, c.arg(7))?;
    let mut data = Vec::with_capacity(c.arg(6) as usize);
    for (page, piece) in segments(k, c.arg(5), c.arg(6))? {
        data.extend(k.read(page, piece)?);
    }
    let (code, n) = match k.fs.write(location, offset, &data) {
        Ok(n) => {
            opened(k, id).position = offset + n as u64;
            (status::SUCCESS, u32::try_from(n).unwrap_or(0))
        }
        Err(code) => (code, 0),
    };
    complete(k, c.arg(4), code, n)?;
    signal_completion(k, c.arg(1), c.arg(2), c.arg(3), c.arg(4));
    value(code)
}

/// The 56-byte `FILE_NETWORK_OPEN_INFORMATION`: four times, allocation, size, attributes, 4 zero
/// bytes.
fn network_open_info(entry: &Entry) -> Vec<u8> {
    let mut out = Vec::with_capacity(56);
    for _ in 0..4 {
        out.extend_from_slice(&BOOT_CLOCK.to_le_bytes());
    }
    out.extend_from_slice(&entry.allocation().to_le_bytes());
    out.extend_from_slice(&entry.size.to_le_bytes());
    out.extend_from_slice(&entry.attributes.to_le_bytes());
    out.extend_from_slice(&[0; 4]);
    out
}

/// The bytes of information class `class` about the open file `id`; `None` for a class not
/// answered.
pub(super) fn file_information(k: &mut Kernel, id: ObjId, class: u32) -> Option<Vec<u8>> {
    let (location, position) = {
        let f = opened(k, id);
        (f.location, f.position)
    };
    let entry = k.fs.stat(location);
    Some(match class {
        // FileBasicInformation: four times, attributes.
        4 => {
            let mut out = Vec::with_capacity(40);
            for _ in 0..4 {
                out.extend_from_slice(&BOOT_CLOCK.to_le_bytes());
            }
            out.extend_from_slice(&entry.attributes.to_le_bytes());
            out.extend_from_slice(&[0; 4]);
            out
        }
        // FileStandardInformation: allocation, size, links, delete pending, directory.
        5 => {
            let mut out = Vec::with_capacity(24);
            out.extend_from_slice(&entry.allocation().to_le_bytes());
            out.extend_from_slice(&entry.size.to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
            out.extend_from_slice(&[0, u8::from(entry.is_dir), 0, 0]);
            out
        }
        // FilePositionInformation.
        14 => position.to_le_bytes().to_vec(),
        // FileNetworkOpenInformation.
        34 => network_open_info(&entry),
        _ => return None,
    })
}

/// `NtQueryInformationFile(h, *iosb, buf, len, class)`.
pub fn query_information_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let Some(bytes) = file_information(k, id, c.arg(4)) else {
        complete(k, c.arg(1), status::INVALID_INFO_CLASS, 0)?;
        return value(status::INVALID_INFO_CLASS);
    };
    let len = c.arg(3) as usize;
    if len < bytes.len() {
        complete(k, c.arg(1), status::INFO_LENGTH_MISMATCH, 0)?;
        return value(status::INFO_LENGTH_MISMATCH);
    }
    k.write(c.arg(2), &bytes)?;
    complete(
        k,
        c.arg(1),
        status::SUCCESS,
        u32::try_from(bytes.len()).unwrap_or(0),
    )?;
    value(status::SUCCESS)
}

/// `NtSetInformationFile(h, *iosb, buf, len, class)`.
pub fn set_information_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let location = opened(k, id).location;
    let buf = c.arg(2);
    let code = match c.arg(4) {
        14 => {
            opened(k, id).position = k.r64(buf)?;
            status::SUCCESS
        }
        13 if k.r8(buf)? != 0 => k.fs.delete(location),
        20 => {
            let size = k.r64(buf)?;
            k.fs.set_size(location, size)
        }
        10 => {
            let replace = k.r8(buf)? != 0;
            let root = k.r32(buf + 4)?;
            let mut name = k.ansi_string(buf + 8)?;
            if root != 0
                && let Some(base) = file(k, root).map(|r| opened(k, r).path.clone())
            {
                name = format!("{}\\{name}", base.trim_end_matches('\\'));
            }
            k.fs.rename(location, &name, replace)
        }
        _ => status::SUCCESS,
    };
    complete(k, c.arg(1), code, 0)?;
    value(code)
}

/// The bytes of volume information class `class` about the volume of the open file `id`;
/// `None` for a class not answered.
pub(super) fn volume_information(k: &mut Kernel, id: ObjId, class: u32) -> Option<Vec<u8>> {
    let location = opened(k, id).location;
    Some(match class {
        // FileFsSizeInformation: total and free units, sectors per unit, bytes per sector.
        3 => {
            let (total, free, sectors, bytes) = k.fs.volume_size(location);
            let mut out = Vec::with_capacity(24);
            out.extend_from_slice(&total.to_le_bytes());
            out.extend_from_slice(&free.to_le_bytes());
            out.extend_from_slice(&sectors.to_le_bytes());
            out.extend_from_slice(&bytes.to_le_bytes());
            out
        }
        // FileFsDeviceInformation: device type, characteristics.
        4 => {
            let (kind, characteristics) = FileSystem::device_info(location);
            [kind.to_le_bytes(), characteristics.to_le_bytes()].concat()
        }
        _ => return None,
    })
}

/// `NtQueryVolumeInformationFile(h, *iosb, buf, len, class)`.
pub fn query_volume_information_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let Some(bytes) = volume_information(k, id, c.arg(4)) else {
        complete(k, c.arg(1), status::INVALID_INFO_CLASS, 0)?;
        return value(status::INVALID_INFO_CLASS);
    };
    if (c.arg(3) as usize) < bytes.len() {
        complete(k, c.arg(1), status::INFO_LENGTH_MISMATCH, 0)?;
        return value(status::INFO_LENGTH_MISMATCH);
    }
    k.write(c.arg(2), &bytes)?;
    complete(
        k,
        c.arg(1),
        status::SUCCESS,
        u32::try_from(bytes.len()).unwrap_or(0),
    )?;
    value(status::SUCCESS)
}

/// `NtQueryDirectoryFile(h, event, apc, ctx, *iosb, buf, len, class, *pattern, restart)`: one
/// entry per call.
pub fn query_directory_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let iosb = c.arg(4);
    let location = opened(k, id).location;
    let mut first = false;
    if c.arg(9) != 0 || opened(k, id).listing.is_none() {
        let pattern = if c.arg(8) == 0 {
            "*".to_owned()
        } else {
            k.ansi_string(c.arg(8))?
        };
        let pattern = if pattern.is_empty() {
            "*".to_owned()
        } else {
            pattern
        };
        let entries = match k.fs.list(location) {
            Ok(entries) => entries,
            Err(code) => {
                complete(k, iosb, code, 0)?;
                return value(code);
            }
        };
        let entries = entries
            .into_iter()
            .filter(|e| wildcard(&pattern, &e.name))
            .collect();
        opened(k, id).listing = Some(Listing { entries, next: 0 });
        first = true;
    }
    let (entry, index) = {
        let listing = opened(k, id)
            .listing
            .as_ref()
            .expect("the listing was just made");
        (listing.entries.get(listing.next).cloned(), listing.next)
    };
    let Some(entry) = entry else {
        let code = if first {
            status::NO_SUCH_FILE
        } else {
            status::NO_MORE_FILES
        };
        complete(k, iosb, code, 0)?;
        return value(code);
    };
    let name = entry.name.as_bytes();
    let name_len = u32::try_from(name.len()).unwrap_or(0);
    let index = u32::try_from(index).unwrap_or(0);
    let mut out = Vec::new();
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&index.to_le_bytes());
    match c.arg(7) {
        // FileDirectoryInformation.
        1 => {
            for _ in 0..4 {
                out.extend_from_slice(&BOOT_CLOCK.to_le_bytes());
            }
            out.extend_from_slice(&entry.size.to_le_bytes());
            out.extend_from_slice(&entry.allocation().to_le_bytes());
            out.extend_from_slice(&entry.attributes.to_le_bytes());
        }
        // FileNamesInformation.
        12 => {}
        _ => {
            complete(k, iosb, status::INVALID_INFO_CLASS, 0)?;
            return value(status::INVALID_INFO_CLASS);
        }
    }
    out.extend_from_slice(&name_len.to_le_bytes());
    out.extend_from_slice(name);
    if (c.arg(6) as usize) < out.len() {
        complete(k, iosb, status::BUFFER_OVERFLOW, 0)?;
        return value(status::BUFFER_OVERFLOW);
    }
    k.write(c.arg(5), &out)?;
    if let Some(listing) = opened(k, id).listing.as_mut() {
        listing.next += 1;
    }
    complete(
        k,
        iosb,
        status::SUCCESS,
        u32::try_from(out.len()).unwrap_or(0),
    )?;
    value(status::SUCCESS)
}

/// `NtQueryFullAttributesFile(attr, *info)`: network-open information by name.
pub fn query_full_attributes_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(path) = attributes_name(k, c.arg(0))? else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    match k.fs.open(&path, 1, 0) {
        Ok((file, _)) => {
            let entry = k.fs.stat(file.location);
            k.write(c.arg(1), &network_open_info(&entry))?;
            value(status::SUCCESS)
        }
        Err(code) => value(code),
    }
}

/// `NtDeleteFile(attr)`.
pub fn delete_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(path) = attributes_name(k, c.arg(0))? else {
        return value(status::OBJECT_NAME_NOT_FOUND);
    };
    match k.fs.open(&path, 1, 0) {
        Ok((file, _)) => value(k.fs.delete(file.location)),
        Err(code) => value(code),
    }
}

/// `NtFlushBuffersFile(h, *iosb)`: persists the saves.
pub fn flush_buffers_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    if file(k, c.arg(0)).is_none() {
        return value(status::INVALID_HANDLE);
    }
    k.fs.persist();
    complete(k, c.arg(1), status::SUCCESS, 0)?;
    value(status::SUCCESS)
}

/// The partition a hard-disk handle is on.
fn partition(location: Location) -> Option<usize> {
    match location {
        Location::Raw(n) | Location::Fs { part: n, .. } => Some(n),
        _ => None,
    }
}

/// `NtDeviceIoControlFile` and `NtFsControlFile(h, event, apc, ctx, *iosb, code, in, in_len,
/// out, out_len)`.
pub fn device_io_control_file(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let Some(id) = file(k, c.arg(0)) else {
        return value(status::INVALID_HANDLE);
    };
    let location = opened(k, id).location;
    let (code, input, output, out_len) = (c.arg(5), c.arg(6), c.arg(8), c.arg(9));
    let iosb = c.arg(4);
    match code {
        IOCTL_SCSI_PASS_THROUGH_DIRECT if FileSystem::is_disc(location) => {
            let cdb = k.read(input + 28, 3)?;
            if cdb[0] != 0x5A || cdb[2] & 0x3F != 0x3E {
                return Ok(CallResult::Unsupported(format!(
                    "the SCSI command {:#04x}",
                    cdb[0]
                )));
            }
            // MODE SENSE (10) of the DVD authentication page: partition area, CDF valid,
            // authenticated.
            let mut page = [0u8; 28];
            page[1] = 26;
            page[8] = 0x3E;
            page[9] = 0x12;
            page[10] = 1;
            page[11] = 1;
            page[12] = 1;
            k.w8(input + 2, 0)?;
            let len = k.r32(input + 12)?.min(28) as usize;
            let buffer = k.r32(input + 20)?;
            k.write(buffer, &page[..len])?;
        }
        FSCTL_DISMOUNT_VOLUME if !FileSystem::is_disc(location) => {}
        IOCTL_DISK_GET_DRIVE_GEOMETRY if !FileSystem::is_disc(location) && out_len >= 24 => {
            k.w64(output, RAW_DISK_SECTORS)?;
            k.w32(output + 8, 12)?;
            k.w32(output + 12, 1)?;
            k.w32(output + 16, 1)?;
            k.w32(output + 20, 512)?;
        }
        IOCTL_DISK_GET_PARTITION_INFO
            if out_len >= 32 && partition(location).is_some_and(|n| n > 0) =>
        {
            let n = partition(location).unwrap_or(0);
            let (start, size) = PARTITIONS[n];
            k.write(output, &[0; 32])?;
            k.w64(output, start * 512)?;
            k.w64(output + 8, size * 512)?;
            k.w32(output + 20, u32::try_from(n).unwrap_or(0))?;
            k.w8(output + 26, 1)?;
        }
        _ => {
            let path = opened(k, id).path.clone();
            return Ok(CallResult::Unsupported(format!(
                "control code {code:#x} on {path} (handle {:#x}), input {input:#x}+{:#x}, output {output:#x}+{out_len:#x}",
                c.arg(0),
                c.arg(7)
            )));
        }
    }
    complete(k, iosb, status::SUCCESS, 0)?;
    signal_completion(k, c.arg(1), c.arg(2), c.arg(3), iosb);
    value(status::SUCCESS)
}

/// `IoCreateSymbolicLink(*name, *target)`, both `ANSI_STRING`s.
pub fn create_symbolic_link(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let name = k.ansi_string(c.arg(0))?;
    let target = k.ansi_string(c.arg(1))?;
    value(k.fs.add_link(&name, &target))
}

pub fn delete_symbolic_link(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let name = k.ansi_string(c.arg(0))?;
    value(k.fs.delete_link(&name))
}

/// `IoCreateDevice(driver, ext_size, *name, type, exclusive, **out)`: a `DEVICE_OBJECT` and its
/// extension.
pub fn create_device(k: &mut Kernel, c: &Call) -> KResult<CallResult> {
    let device = k.heap.alloc(0x60);
    let extension = k.heap.alloc(c.arg(1).max(1));
    if device == 0 || extension == 0 {
        k.heap.free(device);
        k.heap.free(extension);
        return value(status::NO_MEMORY);
    }
    k.w16(device, 3)?;
    k.w16(device + 2, 0x60)?;
    k.w32(device + 8, c.arg(0))?;
    k.w32(device + 0xC, device)?;
    k.w32(device + 0x14, 0x80)?;
    k.w32(device + 0x18, extension)?;
    k.w8(device + 0x1C, c.arg(3) as u8)?;
    k.w8(device + 0x1E, 1)?;
    if c.arg(2) != 0 {
        let name = k.ansi_string(c.arg(2))?;
        let id = k
            .objects
            .add_at(device, Body::Device { name: name.clone() });
        k.objects.set_name(id, &name);
    }
    k.w32(c.arg(5), device)?;
    value(status::SUCCESS)
}
