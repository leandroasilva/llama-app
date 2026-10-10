//! Port of `Llama/HF/HFRepoParser.swift`.
//!
//! Parses HuggingFace repo directory names and GGUF filenames into structured
//! components. (Display parsing of a model's name, params and tags lives in
//! `ModelIdParser`, which works off the id and so covers every model alike.)

use regex::Regex;
use std::path::Path;
use std::sync::OnceLock;

/// Namespace mirroring the Swift `enum HFRepoParser` (no instances).
pub struct HFRepoParser;

fn split_shard_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Same pattern as Swift: `-(\d{5})-of-(\d{5})\.gguf$`, case-insensitive.
    // Rust's `\d` is Unicode `\p{Nd}`, matching ICU's default in NSRegularExpression.
    RE.get_or_init(|| Regex::new(r"(?i)-(\d{5})-of-(\d{5})\.gguf$").expect("valid regex"))
}

impl HFRepoParser {
    /// Splits a repo dir name (e.g. `models--bartowski--Llama-3.2-1B-Instruct-GGUF`)
    /// into its org and repo. Returns `None` if the format is unrecognized.
    pub fn parse(repo_dir: &str) -> Option<(String, String)> {
        if !repo_dir.starts_with("models--") {
            return None;
        }

        // Swift's `components(separatedBy: "--")` keeps empty components, as does `split`.
        let parts: Vec<&str> = repo_dir.split("--").collect();
        if parts.len() < 3 {
            return None;
        }

        let org = parts[1];
        let repo = parts[2..].join("--");
        if org.is_empty() || repo.is_empty() {
            return None;
        }
        Some((org.to_string(), repo))
    }

    /// Extracts quantization from a GGUF filename.
    /// e.g. `Llama-3.2-1B-Instruct-Q4_K_M.gguf` -> `Q4_K_M`
    /// e.g. `model-00001-of-00003.gguf` -> `None` (split shard, no quant in name)
    pub fn parse_quant(filename: &str) -> Option<String> {
        // Remove `.gguf` extension (Unicode-lowercased check, like Swift).
        let mut base = filename.to_string();
        if base.to_lowercase().ends_with(".gguf") {
            base.truncate(base.len() - ".gguf".len());
        }

        // For split shards, extract quant from the base name (before the shard suffix).
        if Self::is_split_shard(filename) {
            let shard_base = Self::split_shard_base_name(filename)?;
            let name_only = Path::new(&shard_base)
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or(shard_base);
            return Self::parse_quant(&format!("{name_only}.gguf"));
        }

        // Quantization is the last segment after the final `-`.
        let last_dash = base.rfind('-')?;
        let candidate = &base[last_dash + 1..];

        // Validate it looks like a quantization label (Q, F, IQ, BF, ...).
        let upper = candidate.to_uppercase();
        let quant_prefixes = ["Q", "F", "IQ", "BF"];
        if quant_prefixes.iter().any(|p| upper.starts_with(p)) {
            return Some(candidate.to_string());
        }
        None
    }

    /// Whether the filename matches the split-shard pattern.
    pub fn is_split_shard(filename: &str) -> bool {
        split_shard_re().is_match(filename)
    }

    /// Whether this is the first shard of a split model.
    pub fn is_first_shard(filename: &str) -> bool {
        match split_shard_re().captures(filename) {
            Some(c) => c.get(1).map(|m| m.as_str() == "00001").unwrap_or(false),
            None => false,
        }
    }

    /// The declared shard count of a split-shard filename.
    pub fn shard_total(filename: &str) -> Option<i64> {
        let caps = split_shard_re().captures(filename)?;
        caps.get(2)?.as_str().parse::<i64>().ok()
    }

    /// The base name for a split shard (everything before the shard pattern).
    pub fn split_shard_base_name(filename: &str) -> Option<String> {
        let m = split_shard_re().find(filename)?;
        let base = &filename[..m.start()];
        Some(base.strip_suffix('-').unwrap_or(base).to_string())
    }
}
