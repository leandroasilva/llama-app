//! Port of the download engine's pure logic (`Llama/Downloads/ModelManager+Transfer.swift`,
//! `ActiveDownload.swift`, `DownloadError.swift`). Everything that parses a response,
//! classifies an error, or formats a message without touching the network, the disk
//! or the UI lives here; the `URLSession`/`NWPathMonitor` orchestration stays in Swift.
//!
//! Concurrency note: this is the phase that pays the migration's motivation. The pure
//! decisions below (retry vs. park, transient vs. actionable, progress monotonicity)
//! are the ones a buggy concurrent rewrite gets wrong, so they're pinned against the
//! shared `Fixtures/download.json` corpus from both sides.

/// Total retry budget per file URL. The per-URL counters live on the Swift
/// `ActiveDownload` entry and die with it (`ModelManager.maxRetryAttempts`).
pub const MAX_RETRY_ATTEMPTS: i64 = 3;

/// Base exponential-backoff delay in seconds; doubles each attempt: 2s, 4s, 8s
/// (`ModelManager.baseRetryDelay`).
pub const BASE_RETRY_DELAY: f64 = 2.0;

/// Sanity floor for a finalized `.partial`: reject obviously broken downloads
/// (error pages, empty files). We don't require an exact size match — expected
/// sizes can drift when HF re-uploads (`finalizeTask`'s `minThreshold`).
pub const MIN_FINAL_SIZE: i64 = 1_000_000;

/// `URLSession` error codes treated as transient/resumable network failures, as
/// opposed to actionable errors (bad certs, malformed URLs) that warrant an
/// alert. Mirrors `ModelManager+Transfer.transientNetworkErrorCodes`.
pub const TRANSIENT_ERROR_TIMEOUT: i64 = -1001; // NSURLErrorTimedOut
pub const TRANSIENT_ERROR_CANNOT_CONNECT_TO_HOST: i64 = -1004; // NSURLErrorCannotConnectToHost
pub const TRANSIENT_ERROR_NETWORK_CONNECTION_LOST: i64 = -1005; // NSURLErrorNetworkConnectionLost
pub const TRANSIENT_ERROR_NOT_CONNECTED_TO_INTERNET: i64 = -1009; // NSURLErrorNotConnectedToInternet
pub const TRANSIENT_ERROR_DNS_LOOKUP_FAILED: i64 = -1003; // NSURLErrorDNSLookupFailed

/// Whether an `NSError.code` is a transient, resumable network failure. Mirrors
/// `ModelManager.isTransientNetworkError(_:)`.
pub fn is_transient_error(code: i64) -> bool {
    matches!(
        code,
        TRANSIENT_ERROR_TIMEOUT
            | TRANSIENT_ERROR_CANNOT_CONNECT_TO_HOST
            | TRANSIENT_ERROR_NETWORK_CONNECTION_LOST
            | TRANSIENT_ERROR_NOT_CONNECTED_TO_INTERNET
            | TRANSIENT_ERROR_DNS_LOOKUP_FAILED
    )
}

/// Case-insensitive header lookup (URLSession's `value(forHTTPHeaderField:)` is
/// case-insensitive too).
fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

/// Extracts the full (not just remaining) size of the remote file from a
/// response. For 206 we parse `Content-Range: bytes X-Y/Z` (Z may be `*` when
/// the total is unknown); for anything else we fall back to `Content-Length`.
/// Returns 0 when neither header is present / parseable. Mirrors
/// `ModelManager+Transfer.extractFullSize(from:)` — note it runs for any status,
/// so a 200 `Content-Length` counts.
pub fn extract_full_size(status: i64, headers: &[(String, String)]) -> i64 {
    if status == 206 {
        if let Some(cr) = header_value(headers, "Content-Range") {
            if let Some(slash) = cr.find('/') {
                let total = cr[slash + 1..].trim();
                // Format: "bytes X-Y/Z"; Z = "*" means unknown total.
                if total != "*" {
                    if let Ok(size) = total.parse::<i64>() {
                        return size;
                    }
                }
            }
        }
    }
    header_value(headers, "Content-Length")
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or_default()
}

/// Maps HTTP status codes to user-facing guidance. We deliberately don't claim a
/// specific cause (rate limit vs. gated repo vs. CDN outage) — Hugging Face uses
/// these codes for several reasons, so we hedge with "usually" and point the user
/// at the most common remedy. Mirrors `userMessage(forHTTPStatus:)`.
pub fn user_message(status: i64) -> String {
    match status {
        401 => "Hugging Face requires authentication for this download. Set a Hugging Face token in Settings and try again.".to_string(),
        403 | 429 => format!(
            "Hugging Face refused the download (HTTP {status}). This usually means a rate limit — try again in a few minutes, or set a Hugging Face token in Settings to lift the limit."
        ),
        404 => "Hugging Face returned 404 for this file. The repo may have moved or been removed.".to_string(),
        500..=599 => format!(
            "Hugging Face is temporarily unavailable (HTTP {status}). Try again in a few minutes."
        ),
        _ => format!("Download failed with HTTP {status}."),
    }
}

/// Exponential-backoff delay for a retry: `base · 2^attempts` (2s, 4s, 8s).
/// Mirrors `scheduleRetry`'s `baseRetryDelay * pow(2.0, Double(attempts))`.
pub fn retry_delay_seconds(attempts: i64, base: f64) -> f64 {
    base * 2f64.powi(attempts as i32)
}

/// What a transient failure does next. Mirrors the decision in
/// `ModelManager.handleTransientFailure(model:url:modelId:)`:
/// - path up + attempts left: genuine blip → retry (keeps the download live),
///   consuming an attempt.
/// - path down: don't consume an attempt (wifi may just be reassociating) — park
///   the whole model as a paused row and auto-resume when the path is restored.
/// - attempts exhausted while online: park as a paused row (no modal) the user
///   can resume by hand, with a fresh budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransientDecision {
    Retry,
    Park { resume_on_reconnect: bool },
}

pub fn transient_decision(
    is_network_available: bool,
    attempts: i64,
    max_attempts: i64,
) -> TransientDecision {
    if is_network_available && attempts < max_attempts {
        TransientDecision::Retry
    } else {
        TransientDecision::Park {
            resume_on_reconnect: !is_network_available,
        }
    }
}

/// The numeric pair `ActiveDownload.refreshProgress` computes and stuffs into
/// a Foundation `Progress`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProgressSum {
    pub total_unit_count: i64,
    pub completed_unit_count: i64,
}

/// Refreshes an aggregate `Progress` from caller-supplied byte sums across
/// active tasks. `active_bytes` is the total bytes currently on disk across all
/// in-flight `.partial` files; `expected_active_bytes` is the sum of each task's
/// known total size (0 before the response arrives). Mirrors
/// `ActiveDownload.refreshProgress(activeBytes:expectedActiveBytes:)`, including
/// the "don't shrink `totalUnitCount`" rule (`max(totalUnitCount, …)`, seeded
/// from the HF resolve's aggregated byte count) and the `max(…, 1)` floor.
pub fn refresh_progress(
    current_total_unit_count: i64,
    completed_files_bytes: i64,
    active_bytes: i64,
    expected_active_bytes: i64,
) -> ProgressSum {
    let total_completed = completed_files_bytes + active_bytes;
    let total_expected =
        current_total_unit_count.max(completed_files_bytes + expected_active_bytes);
    ProgressSum {
        total_unit_count: total_expected.max(1),
        completed_unit_count: total_completed,
    }
}

/// Current transfer rate in bytes/sec averaged over a trailing `window`.
/// Returns nil until the window holds at least a second of data — a shorter span
/// makes the first readouts wildly noisy. Clamped: a retry can rewind the byte
/// count (truncated `.partial`), and a negative rate must never reach the UI.
/// Samples are `(time, completed_bytes)` pairs in append order. Mirrors
/// `ActiveDownload.bytesPerSecond`.
pub fn rate_bytes_per_second(samples: &[(f64, i64)], _window: f64) -> Option<i64> {
    let (first_time, first_bytes) = *samples.first()?;
    let (last_time, last_bytes) = *samples.last()?;
    let span = last_time - first_time;
    if span < 1.0 {
        return None;
    }
    let rate = (last_bytes as f64 - first_bytes as f64) / span;
    Some(rate.max(0.0) as i64)
}

/// Retains only samples within `window` of `now`. Mirrors the
/// `rateSamples.removeAll { now.timeIntervalSince($0.time) > Self.rateWindow }`
/// in `ActiveDownload.refreshProgress`.
pub fn prune_rate_samples(now: f64, samples: &mut Vec<(f64, i64)>, window: f64) {
    samples.retain(|(t, _)| now - *t <= window);
}

/// Pre-download errors, mirroring `DownloadError`'s `errorDescription` and
/// `recoverySuggestion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadError {
    NotCompatible(String),
    NotEnoughDiskSpace { required: String, available: String },
}

impl DownloadError {
    pub fn description(&self) -> &'static str {
        match self {
            Self::NotCompatible(_) => "Model Not Compatible",
            Self::NotEnoughDiskSpace { .. } => "Not Enough Disk Space",
        }
    }

    pub fn recovery_suggestion(&self) -> String {
        match self {
            Self::NotCompatible(reason) => {
                format!("This model {reason}. Choose a smaller model or upgrade your system memory.")
            }
            Self::NotEnoughDiskSpace { required, available } => format!(
                "This model requires {required} of free space, but only {available} is available. Please free up some space."
            ),
        }
    }
}

/// Whether a computed digest matches the blob hash HF advertised
/// (`X-Linked-Etag`). When no expected hash exists the computed digest is trusted
/// instead — that choice stays with the caller.
pub fn expected_hash_matches(computed: &str, expected: &str) -> bool {
    computed == expected
}

/// Whether a finalized `.partial` file size is too small to be a real download
/// (see [`MIN_FINAL_SIZE`]).
pub fn is_too_small(size: i64) -> bool {
    size <= MIN_FINAL_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn extract_full_size_prefers_content_range_on_206() {
        assert_eq!(
            extract_full_size(206, &h(&[("Content-Range", "bytes 302-90210/100000")])),
            100_000
        );
        // Star total falls through to Content-Length.
        assert_eq!(
            extract_full_size(206, &h(&[("Content-Range", "bytes 0-99/*")])),
            0
        );
        assert_eq!(
            extract_full_size(
                206,
                &h(&[("Content-Range", "bytes 0-99/*"), ("content-length", "500")])
            ),
            500
        );
    }

    #[test]
    fn extract_full_size_falls_back_to_content_length() {
        assert_eq!(
            extract_full_size(200, &h(&[("Content-Length", "12345")])),
            12345
        );
        assert_eq!(extract_full_size(200, &h(&[])), 0);
        assert_eq!(
            extract_full_size(404, &h(&[("Content-Length", "garbage")])),
            0
        );
    }

    #[test]
    fn extract_full_size_header_lookup_is_case_insensitive() {
        assert_eq!(
            extract_full_size(206, &h(&[("content-range", "bytes 0-99/777")])),
            777
        );
    }

    #[test]
    fn retry_delay_doubles() {
        assert_eq!(retry_delay_seconds(0, BASE_RETRY_DELAY), 2.0);
        assert_eq!(retry_delay_seconds(1, BASE_RETRY_DELAY), 4.0);
        assert_eq!(retry_delay_seconds(2, BASE_RETRY_DELAY), 8.0);
    }

    #[test]
    fn transient_decision_matrix() {
        assert_eq!(transient_decision(true, 0, 3), TransientDecision::Retry);
        assert_eq!(transient_decision(true, 2, 3), TransientDecision::Retry);
        // Offline parks with auto-resume and does not consume the budget.
        assert_eq!(
            transient_decision(false, 0, 3),
            TransientDecision::Park {
                resume_on_reconnect: true
            }
        );
        // Exhausted while online parks but never auto-resumes.
        assert_eq!(
            transient_decision(true, 3, 3),
            TransientDecision::Park {
                resume_on_reconnect: false
            }
        );
    }

    #[test]
    fn progress_monotonic_and_floored() {
        let s = refresh_progress(100, 0, 37, 100);
        assert_eq!(s.total_unit_count, 100);
        assert_eq!(s.completed_unit_count, 37);

        // Content-Length arriving late grows total; never shrinks.
        let s = refresh_progress(50, 0, 10, 500);
        assert_eq!(s.total_unit_count, 500);
        assert_eq!(s.completed_unit_count, 10);

        let s = refresh_progress(900, 800, 0, 0);
        assert_eq!(s.total_unit_count, 900);

        // Zero total floors to 1 (Foundation Progress can't be 0).
        let s = refresh_progress(0, 0, 0, 0);
        assert_eq!(s.total_unit_count, 1);
        assert_eq!(s.completed_unit_count, 0);
    }

    #[test]
    fn rate_requires_a_second_and_clamps_negative() {
        assert_eq!(rate_bytes_per_second(&[], 5.0), None);
        assert_eq!(rate_bytes_per_second(&[(0.0, 0), (0.5, 100)], 5.0), None);
        assert_eq!(
            rate_bytes_per_second(&[(0.0, 0), (1.0, 500), (2.0, 3000)], 5.0),
            Some(1500)
        );
        // Retry rewound the byte count: negative rate clamps to 0.
        assert_eq!(
            rate_bytes_per_second(&[(0.0, 9000), (2.0, 1000)], 5.0),
            Some(0)
        );
    }

    #[test]
    fn prune_drops_stale_samples() {
        let mut samples = vec![(-1.0, 10), (4.0, 500), (5.0, 900)];
        prune_rate_samples(5.0, &mut samples, 5.0);
        // t=-1 is 6s old (> window): dropped; a sample exactly at the window edge
        // survives (Swift's predicate is strictly `>`).
        assert_eq!(samples, vec![(4.0, 500), (5.0, 900)]);
    }

    #[test]
    fn download_error_messages() {
        assert_eq!(
            DownloadError::NotCompatible("requires 32 GB".into()).description(),
            "Model Not Compatible"
        );
        assert_eq!(
            DownloadError::NotCompatible("requires 32 GB".into()).recovery_suggestion(),
            "This model requires 32 GB. Choose a smaller model or upgrade your system memory."
        );
        let e = DownloadError::NotEnoughDiskSpace {
            required: "15.0 GB".into(),
            available: "3.2 GB".into(),
        };
        assert_eq!(e.description(), "Not Enough Disk Space");
        assert_eq!(
            e.recovery_suggestion(),
            "This model requires 15.0 GB of free space, but only 3.2 GB is available. Please free up some space."
        );
    }

    #[test]
    fn transient_codes_fixed_set() {
        for code in [
            TRANSIENT_ERROR_TIMEOUT,
            TRANSIENT_ERROR_CANNOT_CONNECT_TO_HOST,
            TRANSIENT_ERROR_NETWORK_CONNECTION_LOST,
            TRANSIENT_ERROR_NOT_CONNECTED_TO_INTERNET,
            TRANSIENT_ERROR_DNS_LOOKUP_FAILED,
        ] {
            assert!(is_transient_error(code), "{code}");
        }
        assert!(!is_transient_error(-999)); // NSURLErrorCancelled
        assert!(!is_transient_error(-1200)); // SSL
        assert!(!is_transient_error(0));
    }

    #[test]
    fn too_small_threshold_inclusive() {
        assert!(is_too_small(0));
        assert!(is_too_small(999_999));
        assert!(is_too_small(1_000_000)); // <= minThreshold
        assert!(!is_too_small(1_000_001));
        assert!(!is_too_small(700_000_000));
    }
}
