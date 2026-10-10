//! Port of the selection logic in `Llama/Catalog/Catalog.swift`.
//!
//! `fetchFeatured` mixes a network fetch with pure filtering; the fetch stays in
//! the transport layer and this module carries the pure part: decoding the
//! `featured` slice into one suggestion per fitting size of each family. The
//! device memory budget (`Model.deviceMemoryBudgetMb`) and `systemMemoryMb` are
//! injected.

use crate::models::compat;
use serde::Deserialize;

/// A downloadable quant of a size. Mirrors `Catalog.Build`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Build {
    pub quant: Option<String>,
    pub size: Option<String>,
    pub repo: String,
}

/// A parameter tier within a family. Mirrors `Catalog.Size`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Size {
    #[serde(default)]
    pub vision: bool,
    pub max_mem_gb: Option<u64>,
    pub builds: Vec<Build>,
}

/// A named release line. Mirrors `Catalog.Family`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Family {
    pub brand: String,
    #[serde(default)]
    pub featured: bool,
    pub max_mem_gb: Option<u64>,
    pub sizes: Vec<Size>,
}

/// One resolved catalog pick. Mirrors `Catalog.Suggestion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suggestion {
    pub brand: String,
    pub repo: String,
    pub quant: Option<String>,
    pub size_label: Option<String>,
    pub has_vision: bool,
}

/// Parses catalog size strings like "5.0 GB", "806 MB", "12.1 GB" into bytes,
/// using decimal units (1 GB = 1e9). Returns 0 when missing or unparseable.
/// Mirrors `Catalog.parseBytes`.
pub fn parse_bytes(label: Option<&str>) -> i64 {
    let label = match label {
        Some(l) => l.trim(),
        None => return 0,
    };
    if label.is_empty() {
        return 0;
    }
    let parts: Vec<&str> = label.split_whitespace().collect();
    let value: f64 = match parts.first().and_then(|s| s.parse().ok()) {
        Some(v) => v,
        None => return 0,
    };
    let unit = parts
        .get(1)
        .map(|s| s.to_uppercase())
        .unwrap_or_else(|| "GB".to_string());
    let multiplier: f64 = match unit.as_str() {
        "GB" | "G" => 1_000_000_000.0,
        "MB" | "M" => 1_000_000.0,
        "KB" | "K" => 1_000.0,
        _ => 1_000_000_000.0,
    };
    (value * multiplier) as i64
}

/// The catalog's featured families, filtered and resolved to one suggestion per
/// fitting size (in catalog order). Mirrors the pure tail of `fetchFeatured`.
pub fn featured_suggestions(
    families: &[Family],
    budget_mb: f64,
    system_memory_mb: u64,
) -> Vec<Suggestion> {
    families
        .iter()
        .filter(|f| f.featured)
        .filter(|f| f.max_mem_gb.map(|c| system_memory_mb <= c * 1024) != Some(false))
        .flat_map(|f| picks(f, budget_mb, system_memory_mb))
        .collect()
}

/// Picks the builds to suggest for a family: one per size that fits this Mac's
/// memory budget, in catalog order. Mirrors `Catalog.picks(for:budgetMb:systemMemoryMb:)`.
pub fn picks(family: &Family, budget_mb: f64, system_memory_mb: u64) -> Vec<Suggestion> {
    let mut out = Vec::new();
    for size in &family.sizes {
        if size.max_mem_gb.map(|c| system_memory_mb <= c * 1024) == Some(false) {
            continue;
        }
        let fitting: Vec<&Build> = size
            .builds
            .iter()
            .filter(|b| {
                compat::estimated_weight_fits(Some(parse_bytes(b.size.as_deref())), budget_mb)
            })
            .collect();
        // Best quant = biggest fitting download of this size; ties resolved to
        // the later entry (Swift `max(by:)`).
        let mut best: Option<&Build> = None;
        for b in &fitting {
            match best {
                None => best = Some(b),
                Some(prev) => {
                    if parse_bytes(b.size.as_deref()) >= parse_bytes(prev.size.as_deref()) {
                        best = Some(b);
                    }
                }
            }
        }
        if let Some(best) = best {
            out.push(Suggestion {
                brand: family.brand.clone(),
                repo: best.repo.clone(),
                quant: best.quant.clone(),
                size_label: best.size.clone(),
                has_vision: size.vision,
            });
        }
    }
    out
}
