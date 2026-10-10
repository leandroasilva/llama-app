//! Port of `Llama/HF/GGUFQuant.swift`.
//!
//! Quant naming lives in two forms, and this module owns both plus the
//! boundary between them:
//!
//! - *label* — HF-style, keeps prefixes like Unsloth's `UD-` (`UD-Q4_K_XL`).
//! - *tag* — llama.cpp-canonical (`Q4_K_XL`); what model ids are built from.
//!
//! `parse_label` mirrors `parseGGUFQuantLabel` from huggingface.js; the
//! canonical/bit-width helpers mirror llama.cpp's own derivations.

use crate::hf::HFRepoParser;
use crate::pathutil::last_path_component;
use regex::Regex;
use std::sync::OnceLock;

/// Canonical label alternatives, in huggingface.js declaration order. The order
/// is load-bearing: regex alternation is leftmost-match-first.
pub const QUANTS: [&str; 48] = [
    "F32",
    "F16",
    "Q4_0",
    "Q4_1",
    "Q4_1_SOME_F16",
    "Q4_2",
    "Q4_3",
    "Q8_0",
    "Q5_0",
    "Q5_1",
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "IQ2_XXS",
    "IQ2_XS",
    "Q2_K_S",
    "IQ3_XS",
    "IQ3_XXS",
    "IQ1_S",
    "IQ4_NL",
    "IQ3_S",
    "IQ3_M",
    "IQ2_S",
    "IQ2_M",
    "IQ4_XS",
    "IQ1_M",
    "BF16",
    "Q4_0_4_4",
    "Q4_0_4_8",
    "Q4_0_8_8",
    "TQ1_0",
    "TQ2_0",
    // MXFP4 (bare) is our extension: listed after MXFP4_MOE to preserve order.
    "MXFP4_MOE",
    "MXFP4",
    "NVFP4",
    "Q1_0",
    "Q2_K_XL",
    "Q3_K_XL",
    "Q4_K_XL",
    "Q5_K_XL",
    "Q6_K_XL",
    "Q8_K_XL",
];

fn label_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let pattern = format!("(UD-)?({})(_[A-Z]+)?", QUANTS.join("|"));
        Regex::new(&pattern).expect("valid quant label regex")
    })
}

fn shard_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"-[0-9]{5}-of-[0-9]{5}$").expect("valid shard regex"))
}

fn trailing_tag_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[-.][A-Za-z0-9_]+$").expect("valid trailing tag regex"))
}

/// Returns the last quant label match in `path` (HF's `.at(-1)` semantics) or
/// `None`. The result is uppercased and includes any `UD-` prefix and trailing
/// `_<SUFFIX>`.
pub fn parse_label(path: &str) -> Option<String> {
    let upper = path.to_uppercase();
    let m = label_regex().find_iter(&upper).last()?;
    Some(upper[m.start()..m.end()].to_string())
}

/// The one path → quant-tag derivation for building model ids.
pub fn tag_for_path(path: &str, requested: Option<&str>) -> String {
    let filename = last_path_component(path);
    let label = parse_label(path)
        .or_else(|| HFRepoParser::parse_quant(filename))
        .or_else(|| requested.map(|s| s.to_string()));
    match label {
        Some(l) => canonical_tag(&l),
        None => "unknown".to_string(),
    }
}

/// The quant's bit width, as llama.cpp's `extract_quant_bits` derives it.
pub fn quant_bits_for_path(path: &str) -> i64 {
    if !path.to_lowercase().ends_with(".gguf") {
        return 0;
    }
    let mut prefix = path[..path.len() - ".gguf".len()].to_string();

    if let Some(m) = shard_regex().find(&prefix) {
        prefix.truncate(m.start());
    }

    match trailing_tag_regex().find(&prefix) {
        Some(m) => {
            // Drop the leading separator (`-` or `.`), which is one ASCII byte.
            quant_bits_for_tag(&prefix[m.start() + 1..m.end()])
        }
        None => 0,
    }
}

/// `quant_bits_for_path` for an already-isolated tag (`"Q8_0"` -> 8).
pub fn quant_bits_for_tag(tag: &str) -> i64 {
    let start = tag.find(|c: char| c.is_ascii_digit());
    match start {
        None => 0,
        Some(s) => {
            let digits: String = tag[s..]
                .chars()
                .take_while(|c| c.is_ascii_digit())
                .collect();
            digits.parse::<i64>().unwrap_or(0)
        }
    }
}

/// Convenience: true iff `a` and `b` match case-insensitively.
pub fn matches(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase()
}

/// Canonicalizes a quant tag the way llama.cpp does: keep only the last
/// `-`/`.`-delimited token, uppercased — `UD-Q4_K_XL` -> `Q4_K_XL`.
pub fn canonical_tag(label: &str) -> String {
    let upper = label.to_uppercase();
    let mut last_sep: Option<usize> = None;
    for (i, c) in upper.char_indices() {
        if c == '-' || c == '.' {
            last_sep = Some(i);
        }
    }
    if let Some(i) = last_sep {
        let sep_len = upper[i..].chars().next().map(char::len_utf8).unwrap_or(1);
        let token = &upper[i + sep_len..];
        if !token.is_empty()
            && token
                .chars()
                .all(|c| c.is_alphabetic() || c.is_numeric() || c == '_')
        {
            return token.to_string();
        }
    }
    upper
}
