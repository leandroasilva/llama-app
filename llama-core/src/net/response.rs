//! HTTP response primitives shared by the HF API lookup, the metadata HEADs and
//! (later) the download engine.
//!
//! The extractors mirror the header handling inline in `HFCache.fetchFileMetadata`
//! and the cross-host redirect rule in `SameHostRedirectDelegate`.

/// A minimal HTTP response: the transport (reqwest today) fills this in, and
/// the pure logic below reads headers case-insensitively.
#[derive(Debug, Clone, Default)]
pub struct HttpResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpResponse {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn is_success_or_redirect(&self) -> bool {
        (200..400).contains(&self.status)
    }
}

/// The two values `fetchFileMetadata` harvests from a HEAD response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileMetadata {
    pub blob_hash: Option<String>,
    pub commit_hash: Option<String>,
}

/// Blob hash from `X-Linked-Etag` (preferred), else a SHA256-looking `ETag`.
/// `X-Linked-Etag` is trimmed of surrounding quotes; `ETag` additionally drops
/// a weak `W/` prefix and is only accepted when it is 64 hex characters, so a
/// non-content ETag can't masquerade as a blob hash.
pub fn blob_hash(headers: &HttpResponse) -> Option<String> {
    if let Some(etag) = headers.header("X-Linked-Etag") {
        return Some(etag.trim_matches('"').to_string());
    }
    if let Some(etag) = headers.header("ETag") {
        let cleaned = etag.replace("W/", "").trim_matches('"').to_string();
        if cleaned.len() == 64 && cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(cleaned);
        }
    }
    None
}

/// The repo commit from `X-Repo-Commit`.
pub fn commit_hash(headers: &HttpResponse) -> Option<String> {
    headers.header("X-Repo-Commit").map(str::to_string)
}

/// Harvest `FileMetadata` from a HEAD response, gated on a 2xx/3xx status.
pub fn file_metadata(response: &HttpResponse) -> Option<FileMetadata> {
    if !response.is_success_or_redirect() {
        return None;
    }
    Some(FileMetadata {
        blob_hash: blob_hash(response),
        commit_hash: commit_hash(response),
    })
}

/// Whether a redirect from `from_host` to `to_host` should be followed. Only
/// same-host redirects are followed (e.g. http → https); a cross-host hop
/// (HF → CDN) is stopped so the original 3xx response -- with its `X-Linked-Etag`
/// / `X-Repo-Commit` headers -- is what the caller sees. Mirrors
/// `SameHostRedirectDelegate`.
pub fn same_host_redirect(from_host: Option<&str>, to_host: Option<&str>) -> bool {
    from_host == to_host
}
