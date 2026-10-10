//! Port of the per-file `.partial` download state — `Llama/Downloads/PartialWriter.swift`,
//! `ModelManager.openPartialWriter` and `HFCache.feedHasher/hex`. Streams bytes into a
//! staging file while keeping a running SHA256 and a byte counter, so at completion the
//! digest is ready without a second full-file pass.
//!
//! The Swift original threads this through `URLSession`'s delegate queue on a
//! per-task `PartialWriter` guarded by `WriterTable`. Here the streaming type is a
//! plain `struct`; the host (Swift) still owns the transfer loop and calls the leaf
//! pieces — the 1 MB-chunk prefix re-hash on resume ([`feed_sha256`]), the digest
//! ([`sha256_hex`]) — through the FFI hooks. `PartialWriter` itself exists so the
//! incremental-or-equals-once hashing equivalence is provable on both sides against
//! the shared fixture.

use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Re-hash chunk size, matching `HFCache.feedHasher` (1 MB). Bounded memory on a
/// resume: a many-GB `.partial` prefix is read in 1 MB blocks, not loaded whole.
pub const CHUNK_SIZE: usize = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartialError {
    /// Couldn't open the file / stat it / feed its prefix (see [`feed_sha256`]).
    OpenFailed(String),
    /// A byte append / truncate failed mid-stream.
    WriteFailed(String),
}

impl PartialError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::OpenFailed(_) => "openFailed",
            Self::WriteFailed(_) => "writeFailed",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::OpenFailed(detail) => format!("Couldn't open the staging file: {detail}"),
            Self::WriteFailed(detail) => format!("Couldn't write to the staging file: {detail}"),
        }
    }
}

/// Lowercase hex of the first 32 bytes — the form HF names blobs by. Matches
/// `HFCache.hex`.
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA256 of an in-memory slice, lowercase hex. The one-pass reference for the
/// streaming equivalence.
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Feeds an entire `.partial` file into the running hash in 1 MB chunks — the
/// resume-time re-hash of the on-disk prefix, and equivalently the digest of a
/// completed file. Mirrors `HFCache.feedHasher` + a finalize.
pub fn feed_sha256(path: &str) -> Result<String, PartialError> {
    let mut hasher = Sha256::new();
    feed_path(Path::new(path), &mut hasher)?;
    Ok(hex(&hasher.finalize()))
}

fn feed_path(path: &Path, hasher: &mut Sha256) -> Result<(), PartialError> {
    let mut file = File::open(path)
        .map_err(|e| PartialError::OpenFailed(format!("{}: {e}", path.display())))?;
    let mut buf = vec![0u8; CHUNK_SIZE];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| PartialError::OpenFailed(format!("reading {}: {e}", path.display())))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(())
}

/// Streaming `.partial` state: an always-append file handle, the running hash
/// over the bytes present on disk, the byte counter, the full remote size once
/// known, and the HF-advertised blob hash when available.
pub struct PartialWriter {
    path: PathBuf,
    file: File,
    hasher: Sha256,
    bytes_written: i64,
    total_expected: i64,
    expected_blob_hash: Option<String>,
}

impl PartialWriter {
    /// Opens (creating if absent) the partial at `path`, seeks to the end, and —
    /// when the file already holds bytes — re-hashes the existing prefix in 1 MB
    /// chunks so the running hash resumes where the previous session left off.
    /// Mirrors `ModelManager.openPartialWriter` minus the Swift-only context
    /// (model / cache dir / plan).
    pub fn open(path: &str, expected_blob_hash: Option<String>) -> Result<Self, PartialError> {
        let path = PathBuf::from(path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                PartialError::OpenFailed(format!("creating {}: {e}", parent.display()))
            })?;
        }
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)
            .map_err(|e| PartialError::OpenFailed(format!("{}: {e}", path.display())))?;
        let existing = file.metadata().map(|m| m.len() as i64).unwrap_or(0);

        let mut hasher = Sha256::new();
        if existing > 0 {
            feed_path(&path, &mut hasher)?;
        }

        Ok(Self {
            path,
            file,
            hasher,
            bytes_written: existing,
            total_expected: 0,
            expected_blob_hash,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn bytes_written(&self) -> i64 {
        self.bytes_written
    }

    pub fn total_expected(&self) -> i64 {
        self.total_expected
    }

    pub fn set_total_expected(&mut self, size: i64) {
        self.total_expected = size;
    }

    pub fn expected_blob_hash(&self) -> Option<&str> {
        self.expected_blob_hash.as_deref()
    }

    /// Runs `data` into the file handle, the running hash, and the byte counter
    /// — the `didReceive data:` path in the URLSession delegate.
    pub fn append(&mut self, data: &[u8]) -> Result<(), PartialError> {
        self.file
            .write_all(data)
            .map_err(|e| PartialError::WriteFailed(e.to_string()))?;
        self.hasher.update(data);
        self.bytes_written += data.len() as i64;
        Ok(())
    }

    /// The server answered 200 despite our Range request — restart the file:
    /// truncate to zero, reset the running hash, reset the byte counter. Mirrors
    /// the `status == 200` branch of the response delegate. Append mode keeps
    /// subsequent writes at offset 0.
    pub fn restart(&mut self) -> Result<(), PartialError> {
        self.file
            .set_len(0)
            .map_err(|e| PartialError::WriteFailed(format!("truncate: {e}")))?;
        self.hasher = Sha256::new();
        self.bytes_written = 0;
        Ok(())
    }

    /// Digest over the bytes on disk so far, lowercase hex — ready at completion
    /// without a second full-file pass.
    pub fn digest_hex(&self) -> String {
        hex(&self.hasher.clone().finalize())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);

    fn temp_partial(label: &str) -> std::path::PathBuf {
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        std::env::temp_dir().join(format!("llama-core-pw-{label}-{}-{n}", std::process::id()))
    }

    fn digest_bytes(bytes: &[u8]) -> String {
        // One-pass reference digest over the exact file bytes.
        hex(&Sha256::digest(bytes))
    }

    #[test]
    fn fresh_writer_hashes_streamed_bytes() {
        let p = temp_partial("fresh");
        let mut w = PartialWriter::open(p.to_str().unwrap(), None).unwrap();
        assert_eq!(w.bytes_written(), 0);
        assert_eq!(w.digest_hex(), digest_bytes(b""));

        w.append(b"abc").unwrap();
        assert_eq!(w.bytes_written(), 3);
        assert_eq!(w.digest_hex(), digest_bytes(b"abc"));
        assert_eq!(
            w.digest_hex(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        w.append(b"def").unwrap();
        assert_eq!(w.bytes_written(), 6);
        assert_eq!(w.digest_hex(), digest_bytes(b"abcdef"));
        assert_eq!(
            w.digest_hex(),
            "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
        );

        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn reopen_rehashes_existing_prefix() {
        let p = temp_partial("resume");
        std::fs::write(&p, b"prefix-resume-").unwrap();

        // Simulate the previous session: open (re-hashes on-disk prefix)…
        let mut w = PartialWriter::open(p.to_str().unwrap(), Some("sha".into())).unwrap();
        assert_eq!(w.bytes_written(), 14);
        assert_eq!(w.digest_hex(), digest_bytes(b"prefix-resume-"));
        assert_eq!(w.expected_blob_hash(), Some("sha"));

        // …continue appending what arrived after the Range request.
        w.append(b"tail").unwrap();
        assert_eq!(w.bytes_written(), 18);
        assert_eq!(w.digest_hex(), digest_bytes(b"prefix-resume-tail"));

        // One-pass over the assembled file equals the incremental digest.
        assert_eq!(w.digest_hex(), feed_sha256(p.to_str().unwrap()).unwrap());

        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn restart_truncates_hash_and_counter() {
        let p = temp_partial("restart");
        let mut w = PartialWriter::open(p.to_str().unwrap(), None).unwrap();
        w.append(b"stale-from-ignored-range").unwrap();
        assert_eq!(w.bytes_written(), 24);

        w.restart().unwrap();
        assert_eq!(w.bytes_written(), 0);
        assert_eq!(w.digest_hex(), digest_bytes(b""));
        // File on disk is empty for the fresh stream.
        assert_eq!(w.digest_hex(), feed_sha256(p.to_str().unwrap()).unwrap());

        w.append(b"fresh").unwrap();
        assert_eq!(w.digest_hex(), digest_bytes(b"fresh"));

        std::fs::remove_file(&p).unwrap();
    }

    #[test]
    fn missing_file_feeds_back_open_failed() {
        let missing = temp_partial("missing");
        let err = feed_sha256(missing.to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), "openFailed");
        assert!(err.message().contains("staging file"));
    }
}
