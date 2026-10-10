//! Port of `Llama/Models/ModelIdParser.swift`.
//!
//! Model ids look like `org/Name-7B-Q4_K_M:32k` or `Name-7B-Q4_K_M:4k`.
//! This splits them into org / name / parameter count / quant / trailing tags.

use regex::Regex;
use std::sync::OnceLock;

pub const DEFAULT_ORG: &str = "ggml-org";
const IGNORED_SEGMENTS: [&str; 2] = ["GGUF", "GGML"];

/// Parsed model id parts. `org` is the literal org (may equal
/// `DEFAULT_ORG`); use `display_org` for the UI-facing value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Parsed {
    pub org: Option<String>,
    pub name: String,
    pub params: Option<String>,
    pub quant: Option<String>,
    pub tags: Vec<String>,
}

impl Parsed {
    /// The org to show, or `None` when it's the default.
    pub fn display_org(&self) -> Option<&str> {
        match &self.org {
            Some(o) if o == DEFAULT_ORG => None,
            other => other.as_deref(),
        }
    }

    /// Stable key for dedup: `org|name|params|quant` (defaults blanked).
    pub fn display_key(&self) -> String {
        let org = self.display_org().unwrap_or("");
        let params = self.params.as_deref().unwrap_or("");
        let quant = self.quant.as_deref().unwrap_or("");
        [org, &self.name, params, quant].join("|")
    }

    /// Human-readable name, optionally with tags.
    pub fn plain_name(&self, show_tags: bool) -> String {
        let mut parts: Vec<String> = Vec::new();
        let prefix = self
            .display_org()
            .map(|o| format!("{o}/"))
            .unwrap_or_default();
        parts.push(format!("{prefix}{}", self.name));
        if let Some(p) = &self.params {
            parts.push(p.clone());
        }
        if show_tags {
            parts.extend(self.tags.iter().cloned());
        }
        if let Some(q) = &self.quant {
            parts.push(q.clone());
        }
        parts.join(" ")
    }
}

fn params_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[Ee]?\d+(\.\d+)?[BbMmKkTt]$").expect("valid params regex"))
}

fn activated_params_regex() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^[Aa]\d+(\.\d+)?[BbMmKkTt]$").expect("valid activated params regex")
    })
}

fn is_ignored(segment: &str) -> bool {
    let u = segment.to_uppercase();
    IGNORED_SEGMENTS.contains(&u.as_str())
}

/// Splits a model id into its parts.
pub fn parse(id: &str) -> Parsed {
    let mut rest = id.to_string();
    let mut org: Option<String> = None;
    let mut quant: Option<String> = None;

    if let Some(ci) = rest.rfind(':') {
        let tag = &rest[ci + 1..];
        if !tag.eq_ignore_ascii_case("unknown") {
            quant = Some(tag.to_string());
        }
        rest = rest[..ci].to_string();
    }

    if let Some(si) = rest.find('/') {
        let org_part = &rest[..si];
        if !org_part.is_empty() {
            org = Some(org_part.to_string());
        }
        rest = rest[si + 1..].to_string();
    }

    let segments: Vec<String> = rest
        .split('-')
        .filter(|s| !s.is_empty())
        .filter(|s| !is_ignored(s))
        .map(|s| s.to_string())
        .collect();

    let mut params_idx: Option<usize> = None;
    let mut activated_idx: Option<usize> = None;
    let mut params: Option<String> = None;
    for (idx, seg) in segments.iter().enumerate() {
        if params_idx.is_none() && params_regex().is_match(seg) {
            params_idx = Some(idx);
            params = Some(seg.to_uppercase());
        } else if params_idx.is_some()
            && activated_idx.is_none()
            && activated_params_regex().is_match(seg)
        {
            activated_idx = Some(idx);
            params = Some(format!(
                "{}-{}",
                params.as_deref().unwrap_or(""),
                seg.to_uppercase()
            ));
        }
    }

    let pivot = params_idx.unwrap_or(segments.len());
    let name = segments[..pivot].join("-");

    let mut tags: Vec<String> = Vec::new();
    if let Some(pi) = params_idx {
        for (idx, seg) in segments.iter().enumerate().skip(pi + 1) {
            if Some(idx) != activated_idx {
                tags.push(seg.clone());
            }
        }
    }

    Parsed {
        org,
        name,
        params,
        quant,
        tags,
    }
}
