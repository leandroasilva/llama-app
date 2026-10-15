//! Port of `Llama/Engine/LlamaInstaller.swift`: the leaf steps of installing
//! and updating the app-managed `llama` CLI. The async orchestration (downloads,
//! `install(version:staged:)`) stays in Swift for now; every piece that can be
//! verified without the network lives here.
//!
//! Only the managed path is ever written; a Homebrew or other unmanaged install
//! is never touched (see `super::binaries`).

use super::binaries::{is_executable_file, read_version};
use super::partial_writer::sha256_hex;
use super::version::LlamaVersion;
use crate::net::response::blob_hash;

/// Root of the prebuilt-binary bucket. `…/resolve/latest` returns the current
/// version tag (e.g. `b9553`); `…/resolve/<ver>/<arch>/<os>/…` are artifacts.
pub const BUCKET_BASE: &str = "https://huggingface.co/buckets/ggml-org/install.sh/resolve";

/// We only ship prebuilt Apple Silicon (Metal) macOS binaries.
pub const ARCH: &str = "aarch64";
pub const OS: &str = "macos";

/// The on-PATH symlink `install.sh` creates so `llama` works in a terminal.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InstallError {
    UnsupportedHardware(String),
    DownloadFailed(String),
    DecompressFailed(String),
    InstallFailed(String),
}

impl InstallError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::UnsupportedHardware(_) => "unsupportedHardware",
            Self::DownloadFailed(_) => "downloadFailed",
            Self::DecompressFailed(_) => "decompressFailed",
            Self::InstallFailed(_) => "installFailed",
        }
    }

    /// Mirrors the Swift `LocalizedError.errorDescription`.
    pub fn message(&self) -> String {
        match self {
            Self::UnsupportedHardware(detail) => {
                format!("No prebuilt llama binary for this Mac ({detail}).")
            }
            Self::DownloadFailed(detail) => format!("Download failed: {detail}"),
            Self::DecompressFailed(detail) => format!("Couldn't unpack the llama binary: {detail}"),
            Self::InstallFailed(detail) => format!("Couldn't install the llama binary: {detail}"),
        }
    }
}

/// Maps this Mac's CPU to a Metal bucket config (`m1`…`m5`, `a18`), mirroring
/// `install.sh`. The brand string (from `sysctlbyname("machdep.cpu.brand_string")`)
/// is injected; only the relevant "Apple MX"/"Apple A18" span is kept,
/// lowercased. Mirrors `LlamaInstaller.metalConfig`.
pub fn metal_config(brand: &str) -> Result<String, InstallError> {
    let re = regex::Regex::new("Apple (M[1-5]|A18)").expect("static regex");
    let m = re
        .find(brand)
        .ok_or_else(|| InstallError::UnsupportedHardware(brand.to_string()))?;
    if m.as_str().len() < 6 {
        return Err(InstallError::UnsupportedHardware(brand.to_string()));
    }
    Ok(m.as_str()[6..].to_lowercase())
}

/// The zstd-compressed binary artifact URL for a version + Metal config.
pub fn artifact_url(version: &str, config: &str) -> String {
    format!("{BUCKET_BASE}/{version}/{ARCH}/{OS}/metal/{config}/llama-app.zst")
}

/// The standalone `unzstd` helper URL (macOS ships no zstd).
pub fn unzstd_url(version: &str) -> String {
    format!("{BUCKET_BASE}/{version}/{ARCH}/{OS}/unzstd")
}

/// Downloads `url` to bytes, gating on a 2xx status. Mirrors
/// `LlamaInstaller.download` minus the temp file (the caller writes it where it
/// wants it). Headers (User-Agent / token-on-HF) come from the `HttpClient`.
///
/// Integrity: when Hugging Face advertises the blob's SHA256 (`X-Linked-Etag`,
/// read via the shared [`blob_hash`] rule), the downloaded bytes must match it.
/// A mismatch aborts the install — the artifact is decompressed and then run, so
/// a corrupted or substituted download must never reach the user.
pub fn download(
    client: &dyn crate::net::client::HttpClient,
    url: &str,
    token: Option<&str>,
) -> Result<Vec<u8>, InstallError> {
    let resp = client
        .get_json(url, token)
        .map_err(|e| InstallError::DownloadFailed(e.to_string()))?;
    if !(200..300).contains(&resp.status) {
        return Err(InstallError::DownloadFailed(format!(
            "{url} (HTTP {})",
            resp.status
        )));
    }
    if let Some(expected) = blob_hash(&resp) {
        let computed = sha256_hex(&resp.body);
        if !super::download::expected_hash_matches(&computed, &expected) {
            return Err(InstallError::DownloadFailed(format!(
                "{url} failed its integrity check (SHA256 mismatch)"
            )));
        }
    }
    Ok(resp.body)
}

/// Decompresses `zst` through the `unzstd` helper (stdin → stdout) and
/// atomically swaps the result into `dest`. The temp output lives next to
/// `dest` so the final `rename(2)` is a same-volume atomic replace -- a crash
/// never leaves a half-written binary in place. Mirrors
/// `LlamaInstaller.decompress`.
pub fn decompress(unzstd: &str, zst: &str, dest: &str) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Stdio;

    let install_dir = parent_dir(dest);
    std::fs::create_dir_all(&install_dir)
        .map_err(|e| InstallError::DecompressFailed(e.to_string()))?;

    let tmp = format!("{}/llama.tmp-{}", install_dir, unique_suffix());
    // Mirror Swift's `createFile(atPath:contents: nil)`: an empty file we reopen
    // for writing as unzstd's stdout.
    let saw_error: Result<(), InstallError> = (|| {
        std::fs::File::create(&tmp)
            .map_err(|e| InstallError::DecompressFailed(format!("couldn't create {tmp}: {e}")))?;

        let stdin = std::fs::File::open(zst)
            .map_err(|e| InstallError::DecompressFailed(format!("couldn't open {zst}: {e}")))?;
        let stdout = std::fs::File::create(&tmp)
            .map_err(|e| InstallError::DecompressFailed(format!("couldn't open {tmp}: {e}")))?;

        let out = std::process::Command::new(unzstd)
            .stdin(stdin)
            .stdout(stdout)
            .stderr(Stdio::piped())
            .output()
            .map_err(|e| InstallError::DecompressFailed(format!("couldn't run {unzstd}: {e}")))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
            return Err(InstallError::DecompressFailed(format!(
                "unzstd exited {}: {}",
                out.status,
                prefix(&stderr, 200)
            )));
        }

        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| InstallError::InstallFailed(e.to_string()))?;
        std::fs::rename(&tmp, dest).map_err(|e| InstallError::InstallFailed(format!("rename: {e}")))
    })();

    if saw_error.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    saw_error
}

/// Copies `src` onto the destination volume, then atomically renames it into
/// place as an executable. Mirrors `LlamaInstaller.atomicInstall`.
pub fn atomic_install(src: &str, dest: &str) -> Result<(), InstallError> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::create_dir_all(parent_dir(dest))
        .map_err(|e| InstallError::InstallFailed(e.to_string()))?;

    let tmp = format!("{dest}.tmp-{}", unique_suffix());
    let saw_error: Result<(), InstallError> = (|| {
        std::fs::copy(src, &tmp).map_err(|e| InstallError::InstallFailed(format!("copy: {e}")))?;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| InstallError::InstallFailed(e.to_string()))?;
        std::fs::rename(&tmp, dest)
            .map_err(|e| InstallError::InstallFailed(format!("rename {dest}: {e}")))
    })();

    if saw_error.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    saw_error
}

/// Points `~/.local/bin/llama` (derived from `home`) at `llama_path` so the CLI
/// is available in a terminal. Best-effort, like the Swift original, which
/// never clobbers a real file the user may have placed there -- only absent or
/// symlink entries are (re)written. Returns the outcome as a string for the
/// caller to log; a failure here doesn't break the app.
pub enum PathSymlinkOutcome {
    Linked,
    LeftExisting,
    Failed(String),
}

/// Mirrors `LlamaInstaller.linkOnPath` with the home directory injected.
pub fn ensure_path_symlink(home: &str, llama_path: &str) -> PathSymlinkOutcome {
    let symlink_dir = format!("{}/.local/bin", home.trim_end_matches('/'));
    let symlink_path = format!("{symlink_dir}/llama");
    if let Err(e) = std::fs::create_dir_all(&symlink_dir) {
        return PathSymlinkOutcome::Failed(e.to_string());
    }

    let is_link = std::fs::symlink_metadata(&symlink_path)
        .map(|m| m.file_type().is_symlink())
        .unwrap_or(false);
    if is_link {
        // Replace our (or any) existing symlink.
        let _ = std::fs::remove_file(&symlink_path);
    } else if std::fs::symlink_metadata(&symlink_path).is_ok() {
        // A real file is already there -- leave it for the user.
        return PathSymlinkOutcome::LeftExisting;
    }

    match std::os::unix::fs::symlink(llama_path, &symlink_path) {
        Ok(()) => PathSymlinkOutcome::Linked,
        Err(e) => PathSymlinkOutcome::Failed(e.to_string()),
    }
}

/// Outcome of a staged-binary promotion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PromoteOutcome {
    /// No staged binary present.
    NothingStaged,
    /// A staged binary was present but didn't match `target`; discarded.
    DiscardedStale,
    /// The rename failed; the staged binary was removed.
    Failed(String),
    /// Promoted into the live path and the PATH symlink refreshed.
    Promoted,
}

/// Promotes a staged binary into the live path if it matches `target`, else
/// discards it (the pin moved on since it was staged). Runs a `version`
/// subprocess, so call off the main thread. Mirrors
/// `LlamaInstaller.promoteStaged`. `home` is injected for the PATH symlink.
pub fn promote_staged(
    staged: &str,
    live: &str,
    target: LlamaVersion,
    home: &str,
) -> PromoteOutcome {
    if !is_executable_file(staged) {
        return PromoteOutcome::NothingStaged;
    }

    if read_version(staged) != Some(target) {
        let _ = std::fs::remove_file(staged);
        return PromoteOutcome::DiscardedStale;
    }

    // The real flow pre-creates the install dir before staging; make the
    // promotion self-contained rather than depending on that order.
    if let Err(e) = std::fs::create_dir_all(parent_dir(live)) {
        let _ = std::fs::remove_file(staged);
        return PromoteOutcome::Failed(e.to_string());
    }

    match std::fs::rename(staged, live) {
        Ok(()) => {
            let _ = ensure_path_symlink(home, live);
            PromoteOutcome::Promoted
        }
        Err(e) => {
            let _ = std::fs::remove_file(staged);
            PromoteOutcome::Failed(e.to_string())
        }
    }
}

fn parent_dir(path: &str) -> String {
    match path.rfind('/') {
        Some(i) if i > 0 => path[..i].to_string(),
        _ => ".".to_string(),
    }
}

fn prefix(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// A best-effort unique suffix for temp names (mirrors Swift's `UUID()` for the
/// "no collisions" property, not for cryptographic strength).
fn unique_suffix() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{}", std::process::id(), nanos)
}
