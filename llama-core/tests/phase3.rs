//! Fase 3 equivalence harness (network + remote selection). Shares `Fixtures/*`
//! with the Swift `LlamaTests` target. Network is hermetic: the transport is
//! exercised against a local `std::net` HTTP server, and the pure selection /
//! header rules are pinned by fixtures.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread;

use serde::Deserialize;

use llama_core::net::catalog::{self, Family};
use llama_core::net::client::{HttpClient, NetError, ReqwestClient};
use llama_core::net::hf_api::{fetch_catalog, fetch_file_metadata, fetch_siblings};
use llama_core::net::hf_repo_resolver::{resolve_from_siblings, ResolveError, Sibling};
use llama_core::net::hf_request;
use llama_core::net::response::{
    blob_hash, commit_hash, file_metadata, same_host_redirect, HttpResponse,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Fixtures")
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = fixtures_dir().join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

// MARK: - Catalog

#[derive(Deserialize)]
struct ParseBytesCase {
    label: Option<String>,
    bytes: i64,
}

#[derive(Deserialize)]
struct SuggestionExp {
    brand: String,
    repo: String,
    quant: Option<String>,
    #[serde(rename = "sizeLabel")]
    size_label: Option<String>,
    #[serde(rename = "hasVision")]
    has_vision: bool,
}

#[derive(Deserialize)]
struct CatalogCase {
    #[serde(rename = "systemMemoryMb")]
    system_memory_mb: u64,
    #[serde(rename = "budgetMb")]
    budget_mb: f64,
    expected: Vec<SuggestionExp>,
}

#[derive(Deserialize)]
struct CatalogFixture {
    families: Vec<Family>,
    #[serde(rename = "parseBytes")]
    parse_bytes: Vec<ParseBytesCase>,
    cases: Vec<CatalogCase>,
}

#[test]
fn catalog_fixtures() {
    let f: CatalogFixture = load("catalog.json");

    for c in &f.parse_bytes {
        assert_eq!(
            catalog::parse_bytes(c.label.as_deref()),
            c.bytes,
            "parseBytes({:?})",
            c.label
        );
    }

    for case in &f.cases {
        let got = catalog::featured_suggestions(&f.families, case.budget_mb, case.system_memory_mb);
        assert_eq!(got.len(), case.expected.len(), "{}", "case len");
        for (g, e) in got.iter().zip(case.expected.iter()) {
            assert_eq!(g.brand, e.brand);
            assert_eq!(g.repo, e.repo);
            assert_eq!(g.quant, e.quant);
            assert_eq!(g.size_label, e.size_label);
            assert_eq!(g.has_vision, e.has_vision);
        }
    }
}

// MARK: - HFRepoResolver

#[derive(Deserialize)]
struct ResolvedExp {
    #[serde(rename = "modelId")]
    model_id: String,
    #[serde(rename = "mainUrl")]
    main_url: String,
    #[serde(rename = "additionalParts")]
    additional_parts: Vec<String>,
    #[serde(rename = "mmprojUrl")]
    mmproj_url: Option<String>,
    #[serde(rename = "draftUrl")]
    draft_url: Option<String>,
    #[serde(rename = "approximateBytes")]
    approximate_bytes: i64,
}

#[derive(Deserialize)]
struct ResolverCase {
    name: String,
    repo: String,
    #[serde(rename = "requestedQuant")]
    requested_quant: Option<String>,
    #[serde(rename = "budgetMb")]
    budget_mb: f64,
    siblings: Vec<Sibling>,
    expected: Option<ResolvedExp>,
    #[serde(rename = "expectedError")]
    expected_error: Option<String>,
    #[serde(rename = "errorShard")]
    error_shard: Option<String>,
    #[serde(rename = "errorQuant")]
    error_quant: Option<String>,
}

#[derive(Deserialize)]
struct ResolverFixture {
    cases: Vec<ResolverCase>,
}

#[test]
fn hf_repo_resolver_fixtures() {
    let f: ResolverFixture = load("hf_repo_resolver.json");
    for c in &f.cases {
        let result = resolve_from_siblings(
            &c.repo,
            c.requested_quant.as_deref(),
            &c.siblings,
            c.budget_mb,
        );
        match (&c.expected, &c.expected_error) {
            (Some(exp), None) => {
                let got = result.unwrap_or_else(|e| panic!("{}: unexpected {e:?}", c.name));
                assert_eq!(got.model_id, exp.model_id, "{} modelId", c.name);
                assert_eq!(got.main_url, exp.main_url, "{} mainUrl", c.name);
                assert_eq!(
                    got.additional_parts, exp.additional_parts,
                    "{} parts",
                    c.name
                );
                assert_eq!(got.mmproj_url, exp.mmproj_url, "{} mmproj", c.name);
                assert_eq!(got.draft_url, exp.draft_url, "{} draft", c.name);
                assert_eq!(
                    got.approximate_bytes, exp.approximate_bytes,
                    "{} bytes",
                    c.name
                );
            }
            (None, Some(kind)) => {
                let err = result
                    .err()
                    .unwrap_or_else(|| panic!("{}: expected error", c.name));
                assert_eq!(err.kind(), kind, "{} error kind", c.name);
                if let (ResolveError::ShardMissing { shard, .. }, Some(want)) =
                    (&err, &c.error_shard)
                {
                    assert_eq!(shard, want, "{} shard", c.name);
                }
                if let (ResolveError::QuantNotFound { quant, .. }, Some(want)) =
                    (&err, &c.error_quant)
                {
                    assert_eq!(quant, want, "{} quant", c.name);
                }
            }
            _ => panic!(
                "{}: fixture must set exactly one of expected/expectedError",
                c.name
            ),
        }
    }
}

// MARK: - HFRequest + response helpers

#[derive(Deserialize)]
struct UserAgentCase {
    #[serde(rename = "shortVersion")]
    short_version: String,
    #[serde(rename = "osMajor")]
    os_major: i64,
    #[serde(rename = "osMinor")]
    os_minor: i64,
    expected: String,
}

#[derive(Deserialize)]
struct RequestCase {
    name: String,
    url: String,
    token: Option<String>,
    expected: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct BlobHashCase {
    name: String,
    #[serde(rename = "xLinkedEtag")]
    x_linked_etag: Option<String>,
    etag: Option<String>,
    expected: Option<String>,
}

#[derive(Deserialize)]
struct CommitHashCase {
    #[serde(rename = "xRepoCommit")]
    x_repo_commit: Option<String>,
    expected: Option<String>,
}

#[derive(Deserialize)]
struct RedirectCase {
    from: Option<String>,
    to: Option<String>,
    expected: bool,
}

#[derive(Deserialize)]
struct HostCase {
    url: String,
    expected: bool,
}

#[derive(Deserialize)]
struct HttpFixture {
    #[serde(rename = "testUserAgent")]
    test_user_agent: String,
    #[serde(rename = "userAgent")]
    user_agent: Vec<UserAgentCase>,
    requests: Vec<RequestCase>,
    #[serde(rename = "blobHash")]
    blob_hash: Vec<BlobHashCase>,
    #[serde(rename = "commitHash")]
    commit_hash: Vec<CommitHashCase>,
    redirect: Vec<RedirectCase>,
    hosts: Vec<HostCase>,
}

fn response_with(headers: &[(&str, &str)]) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect(),
        body: Vec::new(),
    }
}

#[test]
fn hf_http_fixtures() {
    let f: HttpFixture = load("hf_http.json");

    for c in &f.user_agent {
        assert_eq!(
            hf_request::user_agent(&c.short_version, c.os_major, c.os_minor),
            c.expected,
            "userAgent {:?}",
            c.short_version
        );
    }

    for c in &f.requests {
        let got = hf_request::make(&c.url, c.token.as_deref(), &f.test_user_agent);
        assert_eq!(got, c.expected, "request {}", c.name);
    }

    for c in &f.blob_hash {
        let mut headers: Vec<(&str, &str)> = Vec::new();
        if let Some(v) = &c.x_linked_etag {
            headers.push(("X-Linked-Etag", v));
        }
        if let Some(v) = &c.etag {
            headers.push(("ETag", v));
        }
        let resp = response_with(&headers);
        assert_eq!(blob_hash(&resp), c.expected, "blobHash {}", c.name);
    }

    for c in &f.commit_hash {
        let mut headers: Vec<(&str, &str)> = Vec::new();
        if let Some(v) = &c.x_repo_commit {
            headers.push(("X-Repo-Commit", v));
        }
        let resp = response_with(&headers);
        assert_eq!(commit_hash(&resp), c.expected);
    }

    for c in &f.redirect {
        assert_eq!(
            same_host_redirect(c.from.as_deref(), c.to.as_deref()),
            c.expected,
            "redirect {:?} -> {:?}",
            c.from,
            c.to
        );
    }

    for c in &f.hosts {
        assert_eq!(
            hf_request::is_huggingface_host(&c.url),
            c.expected,
            "host {}",
            c.url
        );
    }
}

// MARK: - fetch_siblings status mapping (canned transport)

struct CannedClient {
    responses: HashMap<String, HttpResponse>,
}

impl HttpClient for CannedClient {
    fn head(&self, url: &str, _token: Option<&str>) -> Result<HttpResponse, NetError> {
        self.responses
            .get(url)
            .cloned()
            .ok_or_else(|| NetError("no canned HEAD".to_string()))
    }
    fn get_json(&self, url: &str, _token: Option<&str>) -> Result<HttpResponse, NetError> {
        self.responses
            .get(url)
            .cloned()
            .ok_or_else(|| NetError("no canned GET".to_string()))
    }
}

fn canned(repo: &str, status: u16, body: &str) -> CannedClient {
    let url = format!("https://huggingface.co/api/models/{repo}?blobs=true");
    let mut responses = HashMap::new();
    responses.insert(
        url,
        HttpResponse {
            status,
            headers: Vec::new(),
            body: body.as_bytes().to_vec(),
        },
    );
    CannedClient { responses }
}

#[test]
fn fetch_siblings_decodes_and_maps_status() {
    let repo = "unsloth/Tiny-GGUF";
    let client = canned(
        repo,
        200,
        r#"{"siblings":[{"rfilename":"a-Q4_K_M.gguf","size":123},{"rfilename":"b.gguf"}]}"#,
    );
    let siblings = fetch_siblings(&client, repo, None).unwrap();
    assert_eq!(siblings.len(), 2);
    assert_eq!(siblings[0].rfilename, "a-Q4_K_M.gguf");
    assert_eq!(siblings[0].size, Some(123));
    assert_eq!(siblings[1].size, None);

    let gated = fetch_siblings(&canned(repo, 401, "{}"), repo, None).unwrap_err();
    assert_eq!(gated.kind(), "gated");
    let forbidden = fetch_siblings(&canned(repo, 403, "{}"), repo, None).unwrap_err();
    assert_eq!(forbidden.kind(), "gated");
    let missing = fetch_siblings(&canned(repo, 404, "{}"), repo, None).unwrap_err();
    assert_eq!(missing.kind(), "repoNotFound");
    let boom = fetch_siblings(&canned(repo, 500, "{}"), repo, None).unwrap_err();
    assert_eq!(boom.kind(), "networkUnavailable");

    // A decode failure yields an empty sibling list, like the Swift `try?`.
    let broken = fetch_siblings(&canned(repo, 200, "not json"), repo, None).unwrap();
    assert!(broken.is_empty());
}

// MARK: - Transport against a local HTTP server

type Handler = Arc<dyn Fn(&str, &str) -> (u16, Vec<(String, String)>, Vec<u8>) + Send + Sync>;

fn spawn_server(handler: Handler) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let handler = handler.clone();
            thread::spawn(move || serve(stream, handler));
        }
    });
    format!("http://127.0.0.1:{}", addr.port())
}

fn serve(mut stream: std::net::TcpStream, handler: Handler) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 1024];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => return,
        }
    }
    let request = String::from_utf8_lossy(&buf);
    let mut parts = request.lines().next().unwrap_or("").split_whitespace();
    let method = parts.next().unwrap_or("GET").to_string();
    let path = parts.next().unwrap_or("/").to_string();

    let (status, headers, body) = handler(&method, &path);
    let reason = match status {
        200 => "OK",
        302 => "Found",
        404 => "Not Found",
        _ => "Status",
    };
    let mut resp = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n",
        body.len()
    );
    for (k, v) in headers {
        resp.push_str(&format!("{k}: {v}\r\n"));
    }
    resp.push_str("\r\n");
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.write_all(&body);
}

#[test]
fn transport_against_local_server() {
    let handler: Handler = Arc::new(|method, path| {
        if method == "HEAD" && path.starts_with("/meta") {
            (
                200,
                vec![
                    ("X-Linked-Etag".to_string(), "\"blobhash\"".to_string()),
                    ("X-Repo-Commit".to_string(), "c0ffee".to_string()),
                ],
                Vec::new(),
            )
        } else if path.starts_with("/same") {
            (
                302,
                vec![("Location".to_string(), "/meta".to_string())],
                Vec::new(),
            )
        } else if path.starts_with("/cross") {
            (
                302,
                vec![("Location".to_string(), "http://localhost/meta".to_string())],
                Vec::new(),
            )
        } else if path.starts_with("/catalog.json") {
            (
                200,
                vec![("Content-Type".to_string(), "application/json".to_string())],
                br#"[{"brand":"Gemma","featured":true,"sizes":[]}]"#.to_vec(),
            )
        } else {
            (404, Vec::new(), Vec::new())
        }
    });
    let base = spawn_server(handler);
    let client = ReqwestClient::new("test-agent/1 (macOS; arm64)".to_string()).unwrap();

    let meta = client.head(&format!("{base}/meta"), None).unwrap();
    assert_eq!(meta.status, 200);
    assert_eq!(blob_hash(&meta).as_deref(), Some("blobhash"));
    assert_eq!(commit_hash(&meta).as_deref(), Some("c0ffee"));
    let fm = file_metadata(&meta).unwrap();
    assert_eq!(fm.commit_hash.as_deref(), Some("c0ffee"));

    // Same-host redirect (relative Location) is followed.
    let same = client.head(&format!("{base}/same"), None).unwrap();
    assert_eq!(same.status, 200);

    // Cross-host redirect is stopped, preserving the 302 (with its headers).
    let cross = client.head(&format!("{base}/cross"), None).unwrap();
    assert_eq!(cross.status, 302);

    // fetch_file_metadata records only successful HEADs, keyed by URL.
    let url = format!("{base}/meta");
    let map = fetch_file_metadata(&client, std::slice::from_ref(&url), None);
    assert_eq!(
        map.get(&url).unwrap().blob_hash.as_deref(),
        Some("blobhash")
    );

    // The catalog GET decodes through the real transport.
    let families = fetch_catalog(&client, &format!("{base}/catalog.json")).unwrap();
    assert_eq!(families.len(), 1);
    assert_eq!(families[0].brand, "Gemma");
}
