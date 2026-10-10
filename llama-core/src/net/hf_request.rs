//! Port of `Llama/HF/HFRequest.swift`.
//!
//! The single place that decides which headers go out. The user agent is
//! runtime-derived (`AppInfo.userAgent`) so it is injected here: build it with
//! `user_agent(short_version, os_major, os_minor)` and pass it to `make`.

/// The app's outbound User-Agent. Mirrors `AppInfo.userAgent`.
pub fn user_agent(short_version: &str, os_major: i64, os_minor: i64) -> String {
    format!("llama-app/{short_version} (macOS {os_major}.{os_minor}; arm64)")
}

/// The host of `url`, or `None` when it is not a parseable absolute URL. A
/// minimal parser: scheme://authority/... with the authority running to the
/// first `/`, `?` or `#`, userinfo stripped, port kept off the host (Swift's
/// `url.host` excludes the port).
fn host(url: &str) -> Option<&str> {
    let after_scheme = url.split_once("://")?.1;
    let authority = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let no_userinfo = authority.rsplit('@').next().unwrap_or(authority);
    if no_userinfo.is_empty() {
        return None;
    }
    // Strip the port: `host:port` -> `host` (IPv6 brackets left untouched, as
    // HF is never an IPv6 literal here).
    Some(no_userinfo.split(':').next().unwrap_or(no_userinfo))
}

/// Whether `url`'s host is `huggingface.co` or a subdomain of it. Matches the
/// host exactly *or* as a subdomain -- a bare `hasSuffix` would also accept
/// `evilhuggingface.co`, and this check is the only thing keeping the token off
/// a third-party host.
pub fn is_huggingface_host(url: &str) -> bool {
    match host(url) {
        Some(h) => h == "huggingface.co" || h.ends_with(".huggingface.co"),
        None => false,
    }
}

/// The request headers for `url`: always the app's User-Agent, plus
/// `Authorization: Bearer {token}` when `url` is on Hugging Face. Mirrors
/// `HFRequest.make`.
pub fn make(url: &str, token: Option<&str>, user_agent: &str) -> Vec<(String, String)> {
    let mut headers = vec![("User-Agent".to_string(), user_agent.to_string())];
    if let Some(token) = token {
        if is_huggingface_host(url) {
            headers.push(("Authorization".to_string(), format!("Bearer {token}")));
        }
    }
    headers
}
