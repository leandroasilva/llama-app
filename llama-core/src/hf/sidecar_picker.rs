//! Port of `Llama/HF/SidecarPicker.swift`.
//!
//! Given a set of downloaded files, pick the best sibling to run alongside a
//! main model: an mmproj projector or a draft-head sidecar.

use crate::hf::GGUFQuant;
use crate::models::DraftHead;
use crate::pathutil::{dir_components, last_path_component};

/// Recognized draft-head sidecar filename prefixes.
pub const DRAFT_HEAD_PREFIXES: [&str; 4] = ["mtp-", "dflash-", "eagle3-", "dspark-"];

/// Whether `path` names an mmproj projector: basename contains `mmproj` and
/// ends in `.gguf`.
pub fn is_mmproj(path: &str) -> bool {
    let name = last_path_component(path).to_lowercase();
    name.contains("mmproj") && name.ends_with(".gguf")
}

/// Whether `path` names a recognized draft-head sidecar.
pub fn is_draft_head(path: &str) -> bool {
    let name = last_path_component(path).to_lowercase();
    name.ends_with(".gguf") && DRAFT_HEAD_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Picks the best projector among `names` for the model at `main_path`, or
/// `None`. "Best" is the deepest candidate still under the main model's folder,
/// preferring one whose quant matches.
pub fn mmproj(among: &[String], main_path: &str) -> Option<String> {
    best_sibling(among, main_path, None, &|p| is_mmproj(p))
}

/// Picks the best draft head among `names` for the model at `main_path`,
/// honoring `DraftHead::PREFERENCE_ORDER`; `tag` nudges toward a matching quant.
pub fn draft_head(among: &[String], main_path: &str, tag: Option<&str>) -> Option<String> {
    for kind in DraftHead::PREFERENCE_ORDER {
        let picked = best_sibling(among, main_path, tag, &|p| {
            DraftHead::from_sidecar_path(p) == Some(kind)
        });
        if picked.is_some() {
            return picked;
        }
    }
    None
}

/// The shared sibling picker. Ranks candidates by folder depth, then by quant
/// match with the main model, then by quant bit-width closeness.
pub fn best_sibling(
    names: &[String],
    main_path: &str,
    tag: Option<&str>,
    is_candidate: &dyn Fn(&str) -> bool,
) -> Option<String> {
    let main_dirs = dir_components(main_path);
    let main_bits = match tag {
        Some(t) => quant_bits_for_tag(t),
        None => GGUFQuant::quant_bits_for_path(main_path),
    };
    let tag_upper = tag.map(|t| t.to_uppercase());

    let mut sorted: Vec<&String> = names.iter().collect();
    sorted.sort();

    let mut best: Option<(String, i64, bool, i64)> = None;
    for path in sorted {
        if !is_candidate(path) {
            continue;
        }
        let dirs = dir_components(path);
        if dirs.len() > main_dirs.len() {
            continue;
        }
        if !main_dirs.starts_with(&dirs) {
            continue;
        }

        let depth = dirs.len() as i64;
        let diff = (GGUFQuant::quant_bits_for_path(path) - main_bits).abs();
        let exact = match &tag_upper {
            Some(t) => path
                .to_uppercase()
                .contains(&format!("-{t}.")),
            None => false,
        };

        if let Some((_, best_depth, best_exact, best_diff)) = &best {
            let better = depth > *best_depth
                || (depth == *best_depth && exact && !*best_exact)
                || (depth == *best_depth && exact == *best_exact && diff < *best_diff);
            if !better {
                continue;
            }
        }
        best = Some((path.clone(), depth, exact, diff));
    }
    best.map(|b| b.0)
}

fn quant_bits_for_tag(tag: &str) -> i64 {
    GGUFQuant::quant_bits_for_tag(tag)
}
