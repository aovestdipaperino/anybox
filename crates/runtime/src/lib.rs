//! The clean-room Xbox runtime: kernel HLE, GPU model, renderer and host layer.
//!
//! This crate builds the static library `libruntime.a` that the `anybox` tool links with the
//! lifted title code. The C ABI is the contract in `crates/anybox-core/src/glue.rs` and
//! `crates/anybox-core/src/layout.rs`; the behaviour is specified in `RUNTIME-CLEAN-ROOM.md`.

pub mod abi;
pub mod boot;
pub mod fault;
pub mod gpu;
pub mod hibernate;
pub mod host;
pub mod kernel;
pub mod mem;
pub mod render;
