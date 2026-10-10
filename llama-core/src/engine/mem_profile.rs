//! Port of `Llama/Server/MemProfile.swift` + `MemProfileRunner` + `MemProfileCache`
//! minus the probe itself (running `LlamaServer.shared`'s LLM stays in Swift).
//!
//! The memory model: resident bytes are roughly `a + b·ctx`, two unknowns fit
//! from two observations at different context sizes. `fit` derives the exact
//! M-series model from the llama `fit-params` utility's slope/intercept,
//! `parse_fitp_output` reads that utility's output, and the cache persists a
//! profile per model (hashed id) so the pairing is reused across launches.

use serde::{Deserialize, Serialize};

/// A fitted memory profile for a model at a context size. Encoded with camelCase
/// keys to stay byte-compatible with the JSON `MemProfileCache` already wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemProfile {
    #[serde(rename = "ctxBytesPer1kTokens")]
    pub ctx_bytes_per_1k_tokens: i64,
    #[serde(rename = "residentBytes")]
    pub resident_bytes: i64,
    #[serde(rename = "schemaVersion")]
    pub schema_version: i64,
}

impl MemProfile {
    pub fn new(ctx_bytes_per_1k_tokens: i64, resident_bytes: i64) -> Self {
        Self {
            ctx_bytes_per_1k_tokens,
            resident_bytes,
            schema_version: 2,
        }
    }
}

/// Fits `resident ≈ a + b·ctx` from two observations and returns it as the
/// `(b_per_1k, a)` pair, or nil when the fit is degenerate (non-positive slope,
/// i.e. growth would shrink with context). Mirrors `MemProfileRunner.fit`.
pub fn fit(lo_mib: i64, hi_mib: i64, ctx_lo: i64, ctx_hi: i64) -> Option<MemProfile> {
    let dc = (ctx_hi - ctx_lo) as f64;
    let b_per_token = (hi_mib - lo_mib) as f64 / dc;
    let a_mib = lo_mib as f64 - b_per_token * ctx_lo as f64;

    if b_per_token < 0.0 {
        return None;
    }

    let a_bytes = (a_mib.max(0.0) * 1_048_576.0) as i64;
    let b_bytes_per_1k = ((b_per_token * 1000.0 * 1_048_576.0) as i64).max(1);
    Some(MemProfile::new(b_bytes_per_1k, a_bytes))
}

/// Sums the `fit-params` estimates. Each usable row has exactly four
/// whitespace-separated tokens -- `N_ctx`, `KV (MB)`, `total (MB)`, `/ (MB)`, or
/// `MTLHV`/`MLU0`+kv/total/(MB) -- the last three all numeric. Mirrors
/// `MemProfileRunner.parseFitpOutput`: rows are *added* (the utility prints one
/// row per layer it scheduled), so the total across levels is what matches the
/// whole-model profile. Returns nil when no usable row exists.
pub fn parse_fitp_output(output: &str) -> Option<i64> {
    let mut total: i64 = 0;
    let mut rows: i64 = 0;

    for line in output.split('\n') {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() != 4 {
            continue;
        }
        if tokens[0].parse::<i64>().is_ok() {
            continue;
        }
        let m: i64 = match tokens[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let c: i64 = match tokens[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let cp: i64 = match tokens[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        total += m + c + cp;
        rows += 1;
    }

    (rows > 0).then_some(total)
}

/// The per-model cache file name: the first 16 bytes of the SHA-256 digest of
/// the model id, hex-encoded (32 chars, no dashes).
pub fn cache_file_name(model_id: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(model_id.as_bytes());
    digest.iter().take(16).map(|b| format!("{b:02x}")).collect()
}

/// The on-disk path for a model's profile in `cache_dir`.
pub fn cache_path(cache_dir: &str, model_id: &str) -> String {
    let cache_dir = cache_dir.trim_end_matches('/');
    format!("{cache_dir}/{}.json", cache_file_name(model_id))
}

/// Loads the cached profile for `model_id`. Reading a stale scheme (a missing
/// field) yields nil rather than a partial profile, mirroring Swift's
/// `keyNotFound` invalidation.
pub fn cache_get(cache_dir: &str, model_id: &str) -> Option<MemProfile> {
    let path = cache_path(cache_dir, model_id);
    let data = std::fs::read(&path).ok()?;
    serde_json::from_slice(&data).ok()
}

/// Writes the profile atomically (temp + rename, same volume) so a crash never
/// leaves a half-written JSON next to a model id.
pub fn cache_set(cache_dir: &str, model_id: &str, profile: &MemProfile) -> Result<(), String> {
    let data = serde_json::to_vec(profile).map_err(|e| e.to_string())?;
    let path = cache_path(cache_dir, model_id);
    let parent = match path.rfind('/') {
        Some(i) => &path[..i],
        None => ".",
    };
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;

    let tmp = format!(
        "{path}.tmp-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    std::fs::write(&tmp, &data).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_file_name_is_hex_sha256_prefix() {
        assert_eq!(
            cache_file_name("unsloth/Tiny-GGUF:Q4_K_M"),
            "26fce773f312d183395cda0d5faa7fd8"
        );
        assert_eq!(
            cache_file_name("org/Model:Q4_0"),
            "5de1d60b53c5e6d063a5ae9421de7b42"
        );
        assert_eq!(cache_file_name("org/Model:Q4_0").len(), 32);
        assert!(cache_file_name("org/Model:Q4_0")
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn cache_roundtrip_uses_camelcase_keys() {
        let dir = std::env::temp_dir().join(format!("memcache-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = "org/Model:Q8_0";
        assert!(cache_get(dir.to_str().unwrap(), model).is_none());

        let p = MemProfile::new(62_000, 1_234_567_890);
        cache_set(dir.to_str().unwrap(), model, &p).unwrap();
        assert_eq!(cache_get(dir.to_str().unwrap(), model), Some(p));

        let data = std::fs::read(cache_path(dir.to_str().unwrap(), model)).unwrap();
        let text = String::from_utf8(data).unwrap();
        assert!(text.contains("\"ctxBytesPer1kTokens\""), "{text}");
        assert!(text.contains("\"residentBytes\""), "{text}");
        assert!(text.contains("\"schemaVersion\""), "{text}");

        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn stale_scheme_reads_as_missing() {
        let dir = std::env::temp_dir().join(format!("memcache-old-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = "org/Model:b5803";
        std::fs::write(
            cache_path(dir.to_str().unwrap(), model),
            r#"{"ctxBytesPer1kTokens":60000,"residentBytes":1000000}"#, // no schemaVersion
        )
        .unwrap();
        // Partial-scheme data decodes to nil rather than a corrupt profile.
        assert!(cache_get(dir.to_str().unwrap(), model).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
