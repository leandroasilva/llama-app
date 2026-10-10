//! Fase 5 equivalence harness (Download engine leaves + partial writer). Shares
//! `Fixtures/download.json` with the Swift `LlamaTests` target; the pinned
//! SHA-256 vectors make the incremental-vs-once streaming equivalence provable
//! from both sides of the FFI.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;

use llama_core::engine::download::{
    self, expected_hash_matches, is_too_small, rate_bytes_per_second, refresh_progress,
    retry_delay_seconds, transient_decision, user_message, DownloadError, TransientDecision,
};
use llama_core::engine::partial_writer::{self, PartialWriter};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Fixtures")
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = fixtures_dir().join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// A unique temp dir, removed on drop. (No tempfile crate is in the workspace.)
struct TestDir(PathBuf);

static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

impl TestDir {
    fn new(_label: &str) -> Self {
        let n = TEST_DIR_SEQ.fetch_add(1, Ordering::SeqCst);
        let path =
            std::env::temp_dir().join(format!("llama-core-phase5-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        TestDir(path)
    }
    fn root(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// MARK: - fixture shapes

#[derive(Deserialize)]
struct FullSizeCase {
    name: String,
    status: i64,
    headers: Vec<Vec<String>>,
    size: i64,
}

#[derive(Deserialize)]
struct MessageCase {
    name: String,
    status: i64,
    message: String,
}

#[derive(Deserialize)]
struct TransientCodeCase {
    name: String,
    code: i64,
    #[serde(rename = "transient")]
    is_transient: bool,
}

#[derive(Deserialize)]
struct RetryDelayCase {
    name: String,
    attempts: i64,
    base: f64,
    delay: f64,
}

#[derive(Deserialize)]
struct DecisionCase {
    name: String,
    #[serde(rename = "isNetworkAvailable")]
    is_network_available: bool,
    attempts: i64,
    #[serde(rename = "maxAttempts")]
    max_attempts: i64,
    outcome: String,
    #[serde(rename = "resumeOnReconnect")]
    resume_on_reconnect: Option<bool>,
}

#[derive(Deserialize)]
struct RateCase {
    name: String,
    samples: Vec<Vec<f64>>,
    window: f64,
    rate: Option<i64>,
}

#[derive(Deserialize)]
struct ProgressCase {
    name: String,
    #[serde(rename = "totalUnitCount")]
    total_unit_count: i64,
    #[serde(rename = "completedFilesBytes")]
    completed_files_bytes: i64,
    #[serde(rename = "activeBytes")]
    active_bytes: i64,
    #[serde(rename = "expectedActiveBytes")]
    expected_active_bytes: i64,
    #[serde(rename = "totalUnitCountAfter")]
    total_after: i64,
    #[serde(rename = "completedUnitCountAfter")]
    completed_after: i64,
}

#[derive(Deserialize)]
struct ErrorMessageCase {
    name: String,
    kind: String,
    reason: Option<String>,
    required: Option<String>,
    available: Option<String>,
    description: String,
    recovery: String,
}

#[derive(Deserialize)]
struct ShaVectorCase {
    input: String,
    hex: String,
}

#[derive(Deserialize)]
struct HashVerifyCase {
    name: String,
    computed: String,
    expected: String,
    matches: bool,
}

#[derive(Deserialize)]
struct TooSmallCase {
    name: String,
    size: i64,
    #[serde(rename = "tooSmall")]
    too_small: bool,
}

#[derive(Deserialize)]
struct DownloadFixture {
    #[serde(rename = "fullSizeCases")]
    full_size: Vec<FullSizeCase>,
    #[serde(rename = "userMessageCases")]
    user_message: Vec<MessageCase>,
    #[serde(rename = "transientCodes")]
    transient_codes: Vec<TransientCodeCase>,
    #[serde(rename = "retryDelays")]
    retry_delays: Vec<RetryDelayCase>,
    #[serde(rename = "transientDecisions")]
    decisions: Vec<DecisionCase>,
    #[serde(rename = "rateCases")]
    rates: Vec<RateCase>,
    #[serde(rename = "progressCases")]
    progress: Vec<ProgressCase>,
    #[serde(rename = "errorMessageCases")]
    error_messages: Vec<ErrorMessageCase>,
    #[serde(rename = "sha256Vectors")]
    sha256_vectors: Vec<ShaVectorCase>,
    #[serde(rename = "hashVerifyCases")]
    hash_verify: Vec<HashVerifyCase>,
    #[serde(rename = "tooSmallCases")]
    too_small: Vec<TooSmallCase>,
}

fn headers(c: &FullSizeCase) -> Vec<(String, String)> {
    c.headers
        .iter()
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect()
}

// MARK: - response leaves

#[test]
fn full_size_and_user_message_fixture() {
    let f: DownloadFixture = load("download.json");

    for c in &f.full_size {
        assert_eq!(
            download::extract_full_size(c.status, &headers(c)),
            c.size,
            "size case {:?}",
            c.name
        );
    }

    for c in &f.user_message {
        assert_eq!(user_message(c.status), c.message, "message {:?}", c.name);
    }
}

#[test]
fn transient_and_retry_fixture() {
    let f: DownloadFixture = load("download.json");

    for c in &f.transient_codes {
        assert_eq!(
            download::is_transient_error(c.code),
            c.is_transient,
            "code {:?}",
            c.name
        );
    }

    for c in &f.retry_delays {
        assert_eq!(
            retry_delay_seconds(c.attempts, c.base),
            c.delay,
            "retry {:?}",
            c.name
        );
    }

    for c in &f.decisions {
        let expected = match c.outcome.as_str() {
            "retry" => TransientDecision::Retry,
            "park" => TransientDecision::Park {
                resume_on_reconnect: c
                    .resume_on_reconnect
                    .expect("park cases carry resumeOnReconnect"),
            },
            other => panic!("unknown outcome {other:?}"),
        };
        assert_eq!(
            transient_decision(c.is_network_available, c.attempts, c.max_attempts),
            expected,
            "decision {:?}",
            c.name
        );
    }
}

#[test]
fn rate_and_progress_fixture() {
    let f: DownloadFixture = load("download.json");

    for c in &f.rates {
        let samples: Vec<(f64, i64)> = c.samples.iter().map(|s| (s[0], s[1] as i64)).collect();
        assert_eq!(
            rate_bytes_per_second(&samples, c.window),
            c.rate,
            "rate {:?}",
            c.name
        );
    }

    for c in &f.progress {
        let s = refresh_progress(
            c.total_unit_count,
            c.completed_files_bytes,
            c.active_bytes,
            c.expected_active_bytes,
        );
        assert_eq!(s.total_unit_count, c.total_after, "progress {:?}", c.name);
        assert_eq!(
            s.completed_unit_count, c.completed_after,
            "progress {:?}",
            c.name
        );
    }
}

#[test]
fn error_message_fixture() {
    let f: DownloadFixture = load("download.json");

    for c in &f.error_messages {
        let e = match c.kind.as_str() {
            "notCompatible" => DownloadError::NotCompatible(c.reason.clone().unwrap()),
            "notEnoughDiskSpace" => DownloadError::NotEnoughDiskSpace {
                required: c.required.clone().unwrap(),
                available: c.available.clone().unwrap(),
            },
            other => panic!("unknown error kind {other:?}"),
        };
        assert_eq!(e.description(), c.description, "error {:?}", c.name);
        assert_eq!(e.recovery_suggestion(), c.recovery, "error {:?}", c.name);
    }
}

#[test]
fn hash_and_too_small_fixture() {
    let f: DownloadFixture = load("download.json");

    for c in &f.sha256_vectors {
        assert_eq!(
            partial_writer::sha256_hex(c.input.as_bytes()),
            c.hex,
            "vector {:?}",
            c.input
        );
    }

    for c in &f.hash_verify {
        assert_eq!(
            expected_hash_matches(&c.computed, &c.expected),
            c.matches,
            "verify {:?}",
            c.name
        );
    }

    for c in &f.too_small {
        assert_eq!(is_too_small(c.size), c.too_small, "too small {:?}", c.name);
    }
}

// MARK: - partial writer streaming equality

#[test]
fn streamed_digest_matches_one_pass_and_vector() {
    let f: DownloadFixture = load("download.json");
    let td = TestDir::new("pw");
    let path = td.root().join("model.gguf.partial");

    let expected_hex = f
        .sha256_vectors
        .iter()
        .find(|c| c.input == "abcdef")
        .map(|c| c.hex.as_str())
        .expect("fixture has abcdef vector");

    // Streamed, chunk-by-chunk: "abc" then "def".
    let mut w = PartialWriter::open(path.to_str().unwrap(), Some("blob-hash".into())).unwrap();
    w.append(b"abc").unwrap();
    w.append(b"def").unwrap();
    assert_eq!(w.bytes_written(), 6);
    assert_eq!(w.digest_hex(), expected_hex);

    // One-pass over the exact same bytes == streamed.
    assert_eq!(w.digest_hex(), partial_writer::sha256_hex(b"abcdef"));

    // Finished file == what the resume-time re-hash would produce.
    assert_eq!(
        w.digest_hex(),
        partial_writer::feed_sha256(path.to_str().unwrap()).unwrap()
    );

    // Reopen: the "resume" path re-hashes the on-disk prefix and continues.
    let mut w2 = PartialWriter::open(path.to_str().unwrap(), None).unwrap();
    assert_eq!(w2.bytes_written(), 6);
    assert_eq!(w2.digest_hex(), expected_hex);
    w2.append(b"g").unwrap();
    assert_eq!(w2.digest_hex(), partial_writer::sha256_hex(b"abcdefg"));
}
