//! The HF API lookup, the catalog fetch and the metadata HEADs -- the parts of
//! Fase 3 that actually hit the network. Each takes an `HttpClient`, so the
//! logic is exercised against a local server (or a canned client) in tests.

use std::collections::HashMap;

use serde::Deserialize;

use super::catalog::Family;
use super::client::HttpClient;
use super::hf_repo_resolver::{ResolveError, Sibling};
use super::response::{file_metadata, FileMetadata};

#[derive(Deserialize)]
struct ModelInfoResponse {
    siblings: Option<Vec<Sibling>>,
}

/// The `?blobs=true` sibling list for a repo. Maps HTTP status to the Swift
/// `ResolveError` cases (401/403 → gated, 404 → repoNotFound, else network).
/// A decode failure or absent `siblings` yields an empty list, exactly like
/// `fetchSiblings`.
pub fn fetch_siblings(
    client: &dyn HttpClient,
    repo: &str,
    token: Option<&str>,
) -> Result<Vec<Sibling>, ResolveError> {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let resp = client
        .get_json(&url, token)
        .map_err(|_| ResolveError::NetworkUnavailable(repo.to_string()))?;

    match resp.status {
        200..=299 => {}
        401 | 403 => return Err(ResolveError::Gated(repo.to_string())),
        404 => return Err(ResolveError::RepoNotFound(repo.to_string())),
        _ => return Err(ResolveError::NetworkUnavailable(repo.to_string())),
    }

    let decoded: ModelInfoResponse =
        serde_json::from_slice(&resp.body).unwrap_or(ModelInfoResponse { siblings: None });
    Ok(decoded.siblings.unwrap_or_default())
}

/// Fetch and decode the published catalog's family list. Returns `Err` on any
/// non-2xx status or a decode failure so `fetchFeatured` can return an empty
/// list (Discover simply doesn't appear).
pub fn fetch_catalog(client: &dyn HttpClient, url: &str) -> Result<Vec<Family>, String> {
    let resp = client.get_json(url, None).map_err(|e| e.to_string())?;
    if !(200..300).contains(&resp.status) {
        return Err(format!("catalog returned {}", resp.status));
    }
    serde_json::from_slice(&resp.body).map_err(|e| e.to_string())
}

/// HEAD every URL and harvest blob/commit metadata from 2xx/3xx responses.
/// Mirrors `HFCache.fetchFileMetadata` (which skips failures and only records
/// successful entries).
pub fn fetch_file_metadata(
    client: &dyn HttpClient,
    urls: &[String],
    token: Option<&str>,
) -> HashMap<String, FileMetadata> {
    let mut results = HashMap::new();
    for url in urls {
        if let Ok(resp) = client.head(url, token) {
            if let Some(meta) = file_metadata(&resp) {
                results.insert(url.clone(), meta);
            }
        }
    }
    results
}
