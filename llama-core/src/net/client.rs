//! Blocking HTTP transport for the HF API lookup and the metadata HEADs.
//!
//! The real client is `reqwest` (native-tls, matching the app's documented TLS
//! choice). Everything above it depends on the `HttpClient` trait, so tests can
//! drive the selection/fetch logic against a hand-rolled local server (or a
//! canned response) without touching the network.

use super::hf_request;
use super::response::HttpResponse;

/// A transport failure (DNS, TLS, connection, timeout). Maps to
/// `HFRepoResolver.ResolveError.networkUnavailable` at the call site.
#[derive(Debug, Clone)]
pub struct NetError(pub String);

impl std::fmt::Display for NetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for NetError {}

/// A blocking HTTP client. Both verbs carry the HF headers (User-Agent + token
/// on HF hosts) and follow only same-host redirects.
pub trait HttpClient: Send + Sync {
    fn head(&self, url: &str, token: Option<&str>) -> Result<HttpResponse, NetError>;
    fn get_json(&self, url: &str, token: Option<&str>) -> Result<HttpResponse, NetError>;
}

/// The production client: `reqwest` with a same-host redirect policy.
pub struct ReqwestClient {
    client: reqwest::blocking::Client,
    user_agent: String,
}

impl ReqwestClient {
    /// Builds a client whose redirect policy follows only same-host hops, so a
    /// cross-host HF → CDN redirect is stopped and the original 3xx (with its
    /// metadata headers) is returned.
    pub fn new(user_agent: String) -> Result<Self, NetError> {
        let client = reqwest::blocking::Client::builder()
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                let from = attempt.previous().last().and_then(|u| u.host_str());
                let to = attempt.url().host_str();
                if super::response::same_host_redirect(from, to) {
                    attempt.follow()
                } else {
                    attempt.stop()
                }
            }))
            .build()
            .map_err(|e| NetError(e.to_string()))?;
        Ok(Self { client, user_agent })
    }

    fn request(
        &self,
        method: reqwest::Method,
        url: &str,
        token: Option<&str>,
        accept_json: bool,
    ) -> Result<HttpResponse, NetError> {
        let mut req = self.client.request(method, url);
        for (k, v) in hf_request::make(url, token, &self.user_agent) {
            req = req.header(k, v);
        }
        if accept_json {
            req = req.header("Accept", "application/json");
        }
        let resp = req.send().map_err(|e| NetError(e.to_string()))?;

        let status = resp.status().as_u16();
        let headers = resp
            .headers()
            .iter()
            .map(|(k, v)| {
                (
                    k.as_str().to_string(),
                    v.to_str().unwrap_or_default().to_string(),
                )
            })
            .collect();
        let body = resp.bytes().map_err(|e| NetError(e.to_string()))?.to_vec();
        Ok(HttpResponse {
            status,
            headers,
            body,
        })
    }
}

impl HttpClient for ReqwestClient {
    fn head(&self, url: &str, token: Option<&str>) -> Result<HttpResponse, NetError> {
        self.request(reqwest::Method::HEAD, url, token, false)
    }

    fn get_json(&self, url: &str, token: Option<&str>) -> Result<HttpResponse, NetError> {
        self.request(reqwest::Method::GET, url, token, true)
    }
}
