//! Port of the id/sort helpers in `Llama/Models/Model.swift` (the pure parts;
//! the struct itself stays in Swift, which owns URLs and Codable).

use crate::models::ModelIdParser;
use std::cmp::Ordering;

/// The stable id base for an org/repo pair: `"{org}/{repo}"`.
pub fn id_base(org: &str, repo: &str) -> String {
    format!("{org}/{repo}")
}

/// `id_base` for a combined `{org}/{repo}` string — the string verbatim.
pub fn id_base_org_slash_repo(org_slash_repo: &str) -> String {
    org_slash_repo.to_string()
}

/// The stable model id: `"{org}/{repo}:{tag}"`.
pub fn make_id(org: &str, repo: &str, tag: &str) -> String {
    format!("{}:{}", id_base(org, repo), tag)
}

/// `make_id` for a combined `{org}/{repo}` string.
pub fn make_id_org_slash_repo(org_slash_repo: &str, tag: &str) -> String {
    format!("{}:{}", id_base_org_slash_repo(org_slash_repo), tag)
}

/// The pre-colon portion of `id` (`"unsloth/GLM-4.7-Flash-GGUF"`).
pub fn id_base_of(id: &str) -> String {
    match id.rfind(':') {
        Some(i) => id[..i].to_string(),
        None => id.to_string(),
    }
}

/// The row's rendering used as a sort key.
pub fn sort_key(id: &str) -> String {
    let parsed = ModelIdParser::parse(id);
    match parsed.display_org() {
        Some(org) => format!("{org}/{}", parsed.name),
        None => parsed.name,
    }
}

/// Sort order: by displayed name (case-insensitively), then by id for
/// stability.
pub fn display_order(lhs_id: &str, rhs_id: &str) -> Ordering {
    let key_order = sort_key(lhs_id)
        .to_lowercase()
        .cmp(&sort_key(rhs_id).to_lowercase());
    if key_order != Ordering::Equal {
        return key_order;
    }
    lhs_id.cmp(rhs_id)
}
