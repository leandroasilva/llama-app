//! Port of `Llama/Engine/LlamaBinaries.swift`: locating the `llama` executable
//! and classifying whether the app may update it.
//!
//! The shared-path model is preserved: the managed curl-install path wins, then
//! the unmanaged dirs in order, with a Homebrew Cellar install classified as
//! `brew` (the app may use it but never modifies it). Home is injected so the
//! resolution is testable without touching the real machine.

use super::version::LlamaVersion;

/// The app-managed install location, relative to the user's home directory
/// (matches `install.sh`'s layout). The real binary lives in `~/.llama-app`;
/// `install.sh` also drops a `~/.local/bin/llama` symlink onto PATH, but the
/// app points at the real file.
const MANAGED_RELATIVE: &str = ".llama-app/llama";

/// Unmanaged locations to probe when the app hasn't installed its own binary.
/// Covers the Homebrew bin dirs (Apple Silicon and Intel).
pub const UNMANAGED_DIRS: [&str; 2] = ["/opt/homebrew/bin", "/usr/local/bin"];

/// Where the in-use binary comes from. Only `managed` is the app's to update;
/// `brew` and `external` are both used as-is and never modified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Managed,
    Brew,
    External,
}

impl Origin {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Managed => "managed",
            Self::Brew => "brew",
            Self::External => "external",
        }
    }
}

/// The curl-install path the app manages. Mirrors
/// `LlamaBinaries.managedPath` (`(NSHomeDirectory() as NSString).appendingPathComponent`).
pub fn managed_path(home: &str) -> String {
    join(home, MANAGED_RELATIVE)
}

/// Joins a base and a relative path with a single `/`, ignoring a trailing
/// slash on the base (like Foundation's `appendingPathComponent`).
fn join(base: &str, relative: &str) -> String {
    if base == "/" {
        return format!("/{}", relative.trim_start_matches('/'));
    }
    let base = base.trim_end_matches('/');
    if base.is_empty() {
        relative.to_string()
    } else {
        format!("{base}/{relative}")
    }
}

/// Whether `path` is executable (has any execute bit set). Mirrors
/// `FileManager.isExecutableFile(atPath:)`.
pub fn is_executable_file(path: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    std::fs::metadata(path)
        .map(|m| m.is_file() && (m.permissions().mode() & 0o111) != 0)
        .unwrap_or(false)
}

/// Whether the binary at `path` is a Homebrew install. Follows symlinks and
/// checks for Homebrew's Cellar layout (`bin/llama` is a symlink into
/// `../Cellar/llama.cpp/...`) rather than the bin dir alone. Mirrors
/// `LlamaBinaries.isHomebrew`.
pub fn is_homebrew(path: &str) -> bool {
    std::fs::canonicalize(path)
        .map(|p| p.to_string_lossy().contains("/Cellar/"))
        .unwrap_or(false)
}

/// The version constants of the engine policy. Pinned per app release; see the
/// Swift original for the floor/multi-host rationale.
pub fn target_version() -> LlamaVersion {
    LlamaVersion { build: 11429 }
}
pub fn floor_version() -> LlamaVersion {
    LlamaVersion { build: 9831 }
}
pub fn multi_host_version() -> LlamaVersion {
    LlamaVersion { build: 11104 }
}

/// Resolves the active `llama` binary: the managed path wins, then the
/// unmanaged locations in order. Nil when none is found, which is what sends
/// the app down the install flow. Mirrors `LlamaBinaries.resolve`, with the
/// home directory and the unmanaged search order injected.
pub fn resolve(
    home: &str,
    unmanaged_dirs: &[String],
    ignore_unmanaged: bool,
) -> Option<(String, Origin)> {
    let managed = managed_path(home);
    if is_executable_file(&managed) {
        return Some((managed, Origin::Managed));
    }

    if ignore_unmanaged {
        return None;
    }

    for dir in unmanaged_dirs {
        let path = join(dir, "llama");
        if is_executable_file(&path) {
            let origin = if is_homebrew(&path) {
                Origin::Brew
            } else {
                Origin::External
            };
            return Some((path, origin));
        }
    }

    None
}

/// Reads the version reported by the binary at `path`, or nil if it can't be
/// run or its output can't be parsed. Runs `<path> version` -- which just prints
/// the build and exits, no model load. Both streams are captured into one string
/// because which stream carries the version has changed across builds; the
/// parser finds the build number either way. Mirrors `LlamaBinaries.readVersion`.
pub fn read_version(path: &str) -> Option<LlamaVersion> {
    use std::process::Stdio;
    let out = std::process::Command::new(path)
        .arg("version")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    LlamaVersion::parse(&text)
}
