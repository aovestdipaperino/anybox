//! Kernel objects, handles, names and waits. Spec 8.3.

use serde::{Deserialize, Serialize};

use std::collections::{HashMap, VecDeque};

use super::fs::OpenFile;

/// An index into the object table.
pub type ObjId = usize;

/// The current-thread pseudo-handle.
pub const CURRENT_THREAD: u32 = 0xFFFF_FFFE;

/// What an object is, with its state.
#[derive(Debug, Serialize, Deserialize)]
pub enum Body {
    /// A guest thread; signalled once it is done.
    Thread {
        tid: u32,
    },
    /// An event: `sync` events (and timers) are cleared by a satisfied wait.
    Event {
        sync: bool,
        signaled: bool,
    },
    Timer(Timer),
    /// A mutant: owner thread id (0 when free) and its acquisition count.
    Mutant {
        owner: u32,
        count: u32,
    },
    Semaphore {
        count: i32,
        limit: i32,
    },
    /// The wait queue of a critical section; its state is the guest structure (spec 5.4).
    CriticalSection,
    File(Box<OpenFile>),
    SymbolicLink(String),
    /// A device a title created with `IoCreateDevice`.
    Device {
        name: String,
    },
    Directory(String),
    /// An I/O completion port's queued packets (key, context, status, information), and where
    /// each thread waiting in `NtRemoveIoCompletion` wants one (key, context, status block).
    IoCompletion {
        packets: VecDeque<[u32; 4]>,
        receivers: HashMap<u32, [u32; 3]>,
    },
    /// A kernel queue's entries (`KeInitializeQueue`).
    Queue {
        entries: VecDeque<u32>,
    },
    /// A title-defined object (`ObCreateObject`) or a pointer made on demand.
    Other,
}

/// A timer's state.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Timer {
    pub sync: bool,
    pub signaled: bool,
    /// The due time in clock ticks, when armed.
    pub due: Option<u64>,
    /// The period in ticks, 0 for one-shot.
    pub period: u64,
    /// The DPC to queue at expiry, or 0.
    pub dpc: u32,
    /// The APC routine and its context to run at expiry.
    pub apc: Option<(u32, u32)>,
}

/// A kernel object.
#[derive(Debug, Serialize, Deserialize)]
pub struct Object {
    pub body: Body,
    /// The guest address the object is bound to, or 0.
    pub address: u32,
    /// The full path in the namespace, if named.
    pub name: Option<String>,
}

/// The object table.
#[derive(Debug, Serialize, Deserialize)]
pub struct Objects {
    table: Vec<Object>,
    handles: HashMap<u32, ObjId>,
    next_handle: u32,
    by_address: HashMap<u32, ObjId>,
    /// The namespace, keyed by lower-case full path.
    names: HashMap<String, ObjId>,
    /// The timers, which time services on every kernel contact: the table also holds every file
    /// a title ever opened.
    timers: Vec<ObjId>,
}

impl Default for Objects {
    fn default() -> Self {
        Self::new()
    }
}

impl Objects {
    /// An empty table; handles start at 0x80.
    #[must_use]
    pub fn new() -> Self {
        Self {
            table: Vec::new(),
            handles: HashMap::new(),
            next_handle: 0x80,
            by_address: HashMap::new(),
            names: HashMap::new(),
            timers: Vec::new(),
        }
    }

    /// Adds an object, unbound and unnamed.
    pub fn add(&mut self, body: Body) -> ObjId {
        if matches!(body, Body::Timer(_)) {
            self.timers.push(self.table.len());
        }
        self.table.push(Object {
            body,
            address: 0,
            name: None,
        });
        self.table.len() - 1
    }

    /// Adds an object bound to `address`.
    pub fn add_at(&mut self, address: u32, body: Body) -> ObjId {
        let id = self.add(body);
        self.bind(address, id);
        id
    }

    /// Opens a new handle to an object.
    pub fn open(&mut self, id: ObjId) -> u32 {
        let handle = self.next_handle;
        self.next_handle += 4;
        self.handles.insert(handle, id);
        handle
    }

    /// Closes a handle; the object lives on. Returns the object it named.
    pub fn close(&mut self, handle: u32) -> Option<ObjId> {
        self.handles.remove(&handle)
    }

    /// The object a handle names.
    #[must_use]
    pub fn handle(&self, handle: u32) -> Option<ObjId> {
        self.handles.get(&handle).copied()
    }

    /// The number of open handles.
    #[must_use]
    pub fn handle_count(&self) -> usize {
        self.handles.len()
    }

    /// The object bound to `address`.
    #[must_use]
    pub fn at(&self, address: u32) -> Option<ObjId> {
        self.by_address.get(&address).copied()
    }

    /// Binds an object to `address`, replacing earlier bindings of either side.
    pub fn bind(&mut self, address: u32, id: ObjId) {
        let old = self.table[id].address;
        if old != 0 && self.by_address.get(&old) == Some(&id) {
            self.by_address.remove(&old);
        }
        if let Some(previous) = self.by_address.insert(address, id)
            && previous != id
        {
            self.table[previous].address = 0;
        }
        self.table[id].address = address;
    }

    /// The object with a full path name (case-insensitive).
    #[must_use]
    pub fn named(&self, name: &str) -> Option<ObjId> {
        self.names.get(&name.to_lowercase()).copied()
    }

    /// Gives an object a name.
    pub fn set_name(&mut self, id: ObjId, name: &str) {
        self.names.insert(name.to_lowercase(), id);
        self.table[id].name = Some(name.to_owned());
    }

    /// Takes an object's name out of the namespace.
    pub fn unname(&mut self, id: ObjId) {
        if let Some(name) = self.table[id].name.take() {
            self.names.remove(&name.to_lowercase());
        }
    }

    /// The last component of every name directly inside directory `dir` (case-insensitive),
    /// as given.
    #[must_use]
    pub fn names_in(&self, dir: &str) -> Vec<String> {
        self.names
            .values()
            .filter_map(|&id| self.table[id].name.as_deref())
            .filter_map(|name| {
                let (parent, last) = name.rsplit_once('\\')?;
                let parent = if parent.is_empty() { "\\" } else { parent };
                parent.eq_ignore_ascii_case(dir).then(|| last.to_owned())
            })
            .collect()
    }

    /// The object with `id`.
    #[must_use]
    pub fn get(&self, id: ObjId) -> &Object {
        &self.table[id]
    }

    /// The object with `id`, mutably.
    pub fn get_mut(&mut self, id: ObjId) -> &mut Object {
        &mut self.table[id]
    }

    /// Every object, with its id.
    pub fn iter(&self) -> impl Iterator<Item = (ObjId, &Object)> {
        self.table.iter().enumerate()
    }

    /// Every timer, with its id. An object's body never changes kind, so the timers are the
    /// objects added as timers.
    pub fn timers(&self) -> impl Iterator<Item = (ObjId, &Timer)> {
        self.timers
            .iter()
            .filter_map(|&id| match &self.table[id].body {
                Body::Timer(timer) => Some((id, timer)),
                _ => None,
            })
    }

    /// Every object, with its id, mutably.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ObjId, &mut Object)> {
        self.table.iter_mut().enumerate()
    }
}

/// What a waiting thread waits for.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaitOn {
    /// Objects: any of them (resuming with `WAIT_0 + index`), or all of them (resuming with
    /// `WAIT_0`). A single-object wait is "any" over one object.
    Objects { ids: Vec<ObjId>, all: bool },
    /// Only the deadline: a delay; resumes with `SUCCESS`.
    Delay,
    /// Woken by the kernel itself (critical sections, completions, thread joins); the deadline,
    /// if any, still applies.
    Explicit,
    /// Handed an entry of a kernel queue or a packet of a completion port, by the kernel; the
    /// deadline, if any, still applies.
    Take(ObjId),
}

/// A thread's wait.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wait {
    pub on: WaitOn,
    /// The clock tick at which the wait times out.
    pub deadline: Option<u64>,
    /// The global wait sequence: waiters are served oldest first.
    pub seq: u64,
}
