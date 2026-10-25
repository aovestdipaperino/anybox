//! Booting the title: loading the XBE, the data exports, the KPCR, TLS, the boot stack, the
//! import binding, and the kernel state. Spec 6.3, 6.4.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

use crate::kernel::heap::Heap;
use crate::kernel::object::{Body, Objects};
use crate::kernel::thread::{State, Thread, Threads};
use crate::kernel::time::Clock;
use crate::kernel::{Kernel, TRAP_BASE, Trace, fs, ordinals};
use crate::mem::{HEAP_BASE, HEAP_END, LOW_BASE, LOW_END, Memory};

/// The boot thread's stack bottom (spec 6.2).
pub const STACK_BOTTOM: u32 = 0x7FF0_0000;
/// The boot thread's stack top.
pub const STACK_TOP: u32 = 0x8000_0000;
/// The boot thread's TIB stack base: the FPU save area sits 0x210 below the top.
pub const BOOT_TIB_STACK_BASE: u32 = STACK_TOP - 0x210;
/// The KPCR's guest address (FS points here).
pub const KPCR: u32 = 0x8004_0000;
/// The boot thread's KTHREAD.
pub const BOOT_KTHREAD: u32 = 0x8004_0400;
/// The KTHREAD blocks of later threads.
pub const KTHREAD_BLOCKS: u32 = 0x8006_0000;
/// The number of KTHREAD slots.
pub const KTHREAD_SLOTS: u32 = 256;
/// The size of a KTHREAD slot.
pub const KTHREAD_SIZE: u32 = 0x400;
/// The kernel data export area.
pub const DATA_EXPORTS: u32 = 0x8002_0000;
/// The size of one data export slot.
pub const DATA_EXPORT_SIZE: u32 = 0x100;
/// The page-0 mirror.
pub const PAGE0_MIRROR: u32 = 0x8000_0000;

/// Writes an `ANSI_STRING` export at `slot`: length, maximum = length + 1, buffer pointer =
/// slot + 8, then the text at slot + 8 (spec 6.3).
fn write_ansi_string(memory: &Memory, slot: u32, text: &[u8]) {
    let len = u16::try_from(text.len()).unwrap_or(u16::MAX);
    let mut buf = Vec::with_capacity(8 + text.len());
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(&(len + 1).to_le_bytes());
    buf.extend_from_slice(&(slot + 8).to_le_bytes());
    buf.extend_from_slice(text);
    memory.write(slot, &buf);
}

/// Writes the data exports (spec 6.3, table).
fn write_data_exports(memory: &Memory) {
    let slot = |ordinal: u32| DATA_EXPORTS + ordinal * DATA_EXPORT_SIZE;
    let version: Vec<u8> = [1u16, 0, 5838, 1]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    memory.write(slot(324), &version);
    write_ansi_string(memory, slot(326), b"\\Device\\CdRom0\\default.xbe");
    write_ansi_string(memory, slot(41), b"XBOX HARDDISK");
    write_ansi_string(memory, slot(42), b"0000000000000000");
    memory.write(slot(40), &3u32.to_le_bytes());
    memory.write(slot(89), &[1]);
    memory.write(slot(356), &6u32.to_le_bytes());
    memory.write(slot(157), &10000u32.to_le_bytes());
}

/// Commits and copies the header block and every preloaded section to its virtual address (spec
/// 6.3 step 1).
fn load_sections(memory: &Memory, xbe: &xbe::Xbe) {
    let base = xbe.base_address();
    let headers = xbe.headers_size();
    memory.commit_range(base, headers);
    if let Some(data) = xbe.read_virtual(base, headers as usize) {
        memory.write(base, data);
    }
    for section in xbe.sections().iter().filter(|s| s.flags().preload()) {
        let address = section.virtual_address();
        // The start is rarely page aligned, so the range can reach one page more than its size
        // suggests; committed pages are zero-filled, which zeroes the rest of the section.
        memory.commit_range(address, section.virtual_size().max(section.raw_size()));
        if !memory.write(address, xbe.section_data(section)) {
            eprintln!("anyxbox_boot: could not load the section at {address:#010x}");
        }
    }
}

/// Binds every kernel import: the data slot address for a data export, else the ordinal's trap
/// address (spec 6.3 step 3).
fn bind_imports(memory: &Memory, xbe: &xbe::Xbe) -> Result<(), String> {
    let imports = xbe.kernel_imports().map_err(|e| format!("{e:?}"))?;
    let thunk = xbe.kernel_thunk_address();
    for (index, &ordinal) in imports.iter().enumerate() {
        let export = ordinals::export(ordinal).ok_or_else(|| {
            format!("the title imports kernel ordinal {ordinal}, which does not exist")
        })?;
        let address = if export.data {
            DATA_EXPORTS + ordinal * DATA_EXPORT_SIZE
        } else {
            TRAP_BASE + ordinal * 4
        };
        let at = thunk + u32::try_from(index).unwrap_or(0) * 4;
        memory.commit(at);
        memory.write(at, &address.to_le_bytes());
    }
    Ok(())
}

/// The XBE's TLS template and zero-fill size, if it has a TLS directory.
fn tls_template(xbe: &xbe::Xbe) -> Option<(Vec<u8>, u32)> {
    let directory = xbe
        .read_virtual(xbe.tls_address(), 20)
        .filter(|_| xbe.tls_address() != 0)?;
    let field = |i: usize| u32::from_le_bytes(directory[i..i + 4].try_into().unwrap_or_default());
    let (start, end, zero_fill) = (field(0), field(4), field(16));
    let len = end.checked_sub(start)? as usize;
    let template = if len == 0 {
        Vec::new()
    } else {
        xbe.read_virtual(start, len)?.to_vec()
    };
    Some((template, zero_fill))
}

/// Places the boot thread's TLS block below the TIB stack base (spec 6.3 step 5): its first
/// dword points to the next one, then the template, then zeros. Returns its address.
fn boot_tls(memory: &Memory, template: &[u8], zero_fill: u32) -> u32 {
    let template_len = u32::try_from(template.len()).unwrap_or(0);
    let total = (template_len + zero_fill).next_multiple_of(4);
    let block = BOOT_TIB_STACK_BASE - (total + 8);
    memory.write(block, &(block + 4).to_le_bytes());
    memory.write(block + 4, template);
    memory.write(block + 4 + template_len, &vec![0; zero_fill as usize]);
    block
}

/// Sets up the KPCR and the boot KTHREAD (spec 6.3 step 6).
fn setup_kpcr(memory: &Memory, tls: u32) {
    memory.commit(KPCR);
    let mut kpcr = vec![0u8; 0x1000];
    let mut put = |at: usize, v: u32| kpcr[at..at + 4].copy_from_slice(&v.to_le_bytes());
    put(0x00, 0xFFFF_FFFF);
    put(0x04, BOOT_TIB_STACK_BASE);
    put(0x08, STACK_BOTTOM);
    put(0x18, KPCR);
    put(0x1C, KPCR);
    put(0x20, KPCR + 0x28);
    put(0x28, BOOT_KTHREAD);
    memory.write(KPCR, &kpcr);
    let mut kthread = vec![0u8; KTHREAD_SIZE as usize];
    kthread[0x28..0x2C].copy_from_slice(&tls.to_le_bytes());
    memory.write(BOOT_KTHREAD, &kthread);
}

/// Where the disc pack is: `ANYXBOX_PACK`, else `../Resources/disc.pack` next to the executable
/// when it exists, else `disc.pack` beside the executable (spec 12.4).
fn pack_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ANYXBOX_PACK") {
        return Some(path.into());
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    let bundled = dir.join("../Resources/disc.pack");
    Some(if bundled.exists() {
        bundled
    } else {
        dir.join("disc.pack")
    })
}

/// Where `E:` is kept: `ANYXBOX_SAVE`, else `~/.anyxbox/TTTTTTTT.save` (spec 8.9).
fn save_path(title_id: u32) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("ANYXBOX_SAVE") {
        return Some(path.into());
    }
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".anyxbox")
            .join(format!("{title_id:08x}.save")),
    )
}

/// The kernel call log `ANYXBOX_TRACE` asks for: `all`, any other value, or none (spec 8.1).
fn trace_from_env() -> Trace {
    match std::env::var("ANYXBOX_TRACE").as_deref() {
        Ok("all") => Trace::All,
        Ok(_) => Trace::On,
        Err(_) => Trace::Off,
    }
}

/// Boots the kernel state (spec 6.3, 6.4).
///
/// # Errors
///
/// A message when the title imports an ordinal the kernel does not have, or the disc pack
/// cannot be opened.
pub fn boot(memory: Memory, xbe: &xbe::Xbe) -> Result<Kernel, String> {
    load_sections(&memory, xbe);
    memory.commit_range(DATA_EXPORTS, ordinals::COUNT * DATA_EXPORT_SIZE);
    write_data_exports(&memory);
    bind_imports(&memory, xbe)?;
    memory.commit_range(STACK_BOTTOM, STACK_TOP - STACK_BOTTOM);
    memory.commit(PAGE0_MIRROR);
    memory.commit_range(KTHREAD_BLOCKS, KTHREAD_SLOTS * KTHREAD_SIZE);
    let (template, zero_fill) = tls_template(xbe).unwrap_or_default();
    let tls = if xbe.tls_address() == 0 {
        0
    } else {
        boot_tls(&memory, &template, zero_fill)
    };
    setup_kpcr(&memory, tls);

    let heap = Heap::new(memory.heap(), HEAP_BASE, HEAP_END - HEAP_BASE);
    let low_heap = Heap::new(memory.low(), LOW_BASE, LOW_END - LOW_BASE);
    let mut objects = Objects::new();
    let object = objects.add_at(BOOT_KTHREAD, Body::Thread { tid: 1 });
    let boot_thread = Thread {
        id: 1,
        state: State::Ready,
        suspend_count: 0,
        priority: 8,
        base_priority: 8,
        stack_block: 0,
        stack_base: BOOT_TIB_STACK_BASE,
        stack_limit: STACK_BOTTOM,
        kthread: BOOT_KTHREAD,
        slot: None,
        tls,
        exception_list: u32::MAX,
        resume: 0,
        wait: None,
        object,
        dpc: 0,
        follow_up: None,
        park: crate::kernel::thread::Park::Start,
        start: (
            xbe.entry_point(),
            if tls == 0 { BOOT_TIB_STACK_BASE } else { tls },
        ),
    };

    let mut file_system = fs::FileSystem::new();
    if let Some(path) = pack_path() {
        match pack::Pack::open(&path) {
            Ok(pack) => file_system.mount_disc(pack),
            Err(error) => return Err(format!("{}: {error}", path.display())),
        }
    }
    if let Some(path) = save_path(xbe.certificate().title_id()) {
        match file_system.keep_saves(&path) {
            Ok(()) => eprintln!("anyxbox: saves in {}", path.display()),
            Err(error) => eprintln!("anyxbox: {}: {error}; saves are not kept", path.display()),
        }
    }

    let sections = xbe
        .sections()
        .iter()
        .map(|s| (s.virtual_address(), xbe.section_data(s).to_vec()))
        .collect();
    let mut clock = Clock::new();
    clock.follow_host();
    let mut kernel = Kernel {
        memory,
        heap,
        low_heap,
        objects,
        threads: Threads::new(boot_thread, KTHREAD_SLOTS),
        current: 1,
        clock,
        dpcs: HashMap::new(),
        fs: file_system,
        eeprom: crate::kernel::handlers::default_eeprom(),
        cache_pages: None,
        tls_template: template,
        tls_zero_fill: zero_fill,
        sections,
        unstarted: Vec::new(),
        calls: 0,
        av_saved_data: 0,
        display_mode: [0; 6],
        trace: trace_from_env(),
        reported: HashSet::new(),
        interrupts: Vec::new(),
        section_waiters: HashMap::new(),
        hooks: crate::kernel::hooks::Hooks::default(),
        sound: crate::kernel::sound::Sound::default(),
        gpu: crate::gpu::Gpu::new(),
        next_vblank: 0,
        fps_report: std::env::var_os("ANYXBOX_TRACE_FPS").map(|_| (0, 0)),
        frame_sink: crate::render::start(crate::host::window_enabled()),
        sync_irps: HashMap::new(),
    };
    if kernel.frame_sink.is_some() {
        kernel.gpu.record_frames(true);
    }
    crate::kernel::hooks::install(&mut kernel, xbe)?;
    kernel.publish_time();
    Ok(kernel)
}
