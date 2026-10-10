//! `llama-core` — the logic core of the Llama macOS app, migrated from Swift.
//!
//! This crate holds pure and near-pure logic only. It never touches AppKit,
//! SwiftUI, UserDefaults, or Metal; those stay in the Swift host and are
//! injected through `llama-core-ffi`. See `MIGRACAO_RUST_REVIEW.md` for the
//! migration plan and the FFI contract.

pub mod config;
pub mod engine;
pub mod events;
pub mod hf;
pub mod models;
pub mod net;
pub mod pathutil;
