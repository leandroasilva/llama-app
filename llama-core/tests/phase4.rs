//! Fase 4 equivalence harness (Engine: LlamaBinaries, LlamaInstaller leaves,
//! MemProfile fit/parse/cache). Shares `Fixtures/engine.json` with the Swift
//! `LlamaTests` target; fake shell binaries under `Fixtures/engine/` stand in
//! for the real `llama`/`unzstd` subprocesses so nothing touches the network or
//! the real machine.

use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::Deserialize;

use llama_core::engine::binaries::{self, Origin};
use llama_core::engine::installer::{
    artifact_url, atomic_install, decompress, download, ensure_path_symlink, metal_config,
    promote_staged, unzstd_url, InstallError, PathSymlinkOutcome, PromoteOutcome,
};
use llama_core::engine::mem_profile::{self, fit, parse_fitp_output, MemProfile};
use llama_core::net::client::{HttpClient, NetError};
use llama_core::net::response::HttpResponse;

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

/// A unique temp dir, removed on drop. (No tempfile crate is in the workspace.)
struct TestDir(PathBuf);

static TEST_DIR_SEQ: AtomicU64 = AtomicU64::new(0);

impl TestDir {
    fn new(_label: &str) -> Self {
        let n = TEST_DIR_SEQ.fetch_add(1, Ordering::SeqCst);
        let path =
            std::env::temp_dir().join(format!("llama-core-phase4-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        TestDir(path)
    }
    fn root(&self) -> &Path {
        &self.0
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Copies a fixture script and makes it executable.
fn install_script(from: &str, to: &Path) -> PathBuf {
    std::fs::create_dir_all(to.parent().unwrap()).unwrap();
    std::fs::copy(fixtures_dir().join("engine").join(from), to).unwrap();
    let mut perms = std::fs::metadata(to).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(to, perms).unwrap();
    to.to_path_buf()
}

fn make_exec(path: &Path) {
    let mut perms = std::fs::metadata(path).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms).unwrap();
}

// MARK: - managedPath + metalConfig + URLs

#[derive(Deserialize)]
struct ManagedPathCase {
    home: String,
    path: String,
}

#[derive(Deserialize)]
struct MetalConfigCase {
    brand: String,
    config: String,
}

#[derive(Deserialize)]
struct UrlCase {
    version: String,
    config: Option<String>,
    url: String,
}

#[derive(Deserialize)]
struct EngineFixture {
    #[serde(rename = "managedPath")]
    managed_path: Vec<ManagedPathCase>,
    #[serde(rename = "metalConfig")]
    metal_config: Vec<MetalConfigCase>,
    #[serde(rename = "artifactUrls")]
    artifact_urls: Vec<UrlCase>,
    #[serde(rename = "unzstdUrls")]
    unzstd_urls: Vec<UrlCase>,
}

#[test]
fn managed_path_fixture() {
    let f: EngineFixture = load("engine.json");
    for c in &f.managed_path {
        assert_eq!(binaries::managed_path(&c.home), c.path, "home={:?}", c.home);
    }
}

#[test]
fn metal_config_fixture() {
    let f: EngineFixture = load("engine.json");
    for c in &f.metal_config {
        match metal_config(&c.brand) {
            Ok(config) => assert_eq!(config, c.config, "brand={:?}", c.brand),
            Err(e) => {
                assert_eq!(c.config, "unsupportedHardware", "brand={:?}", c.brand);
                assert_eq!(e.kind(), "unsupportedHardware");
                assert!(e.message().contains(&c.brand));
            }
        }
    }
}

#[test]
fn artifact_urls_fixture() {
    let f: EngineFixture = load("engine.json");
    for c in &f.artifact_urls {
        assert_eq!(
            artifact_url(&c.version, c.config.as_deref().unwrap()),
            c.url
        );
    }
    for c in &f.unzstd_urls {
        assert_eq!(unzstd_url(&c.version), c.url);
    }
}

// MARK: - resolve (managed → unmanaged, brew vs external)

#[derive(Deserialize)]
struct ManagedSpec {
    exists: bool,
    exec: bool,
}

#[derive(Deserialize)]
struct UnmanagedSpec {
    dir: String,
    kind: String,
}

#[derive(Deserialize)]
struct ResolveExpected {
    origin: String,
    path: Option<String>,
}

#[derive(Deserialize)]
struct ResolveCase {
    name: String,
    home: String,
    #[serde(rename = "unmanagedDirs")]
    unmanaged_dirs: Vec<String>,
    managed: ManagedSpec,
    unmanaged: Vec<UnmanagedSpec>,
    #[serde(rename = "ignoreUnmanaged")]
    ignore_unmanaged: bool,
    expected: ResolveExpected,
}

#[derive(Deserialize)]
struct ResolveFixture {
    #[serde(rename = "resolveCases")]
    cases: Vec<ResolveCase>,
}

fn build_resolve_tree(root: &Path, case: &ResolveCase) {
    let home = PathBuf::from(case.home.replace("<ROOT>", &root.to_string_lossy()));

    if case.managed.exists {
        let managed = home.join(".llama-app").join("llama");
        std::fs::create_dir_all(managed.parent().unwrap()).unwrap();
        std::fs::write(&managed, "managed-placeholder").unwrap();
        if case.managed.exec {
            make_exec(&managed);
        }
    }

    for spec in &case.unmanaged {
        let dir = PathBuf::from(spec.dir.replace("<ROOT>", &root.to_string_lossy()));
        std::fs::create_dir_all(&dir).unwrap();
        match spec.kind.as_str() {
            "none" => {}
            "external" => {
                let bin = dir.join("llama");
                std::fs::write(&bin, "external-placeholder").unwrap();
                make_exec(&bin);
            }
            "brew" => {
                let cellar_target = dir.join("../Cellar/llama.cpp/1.0.0/bin/llama");
                std::fs::create_dir_all(cellar_target.parent().unwrap()).unwrap();
                std::fs::write(&cellar_target, "brew-placeholder").unwrap();
                make_exec(&cellar_target);
                symlink("../Cellar/llama.cpp/1.0.0/bin/llama", dir.join("llama")).unwrap();
            }
            other => panic!("unknown unmanaged kind {other:?}"),
        }
    }
}

#[test]
fn resolve_fixture() {
    let f: ResolveFixture = load("engine.json");
    let td = TestDir::new("resolve");
    let root = td.root();

    for (i, case) in f.cases.iter().enumerate() {
        let case_root = root.join(format!("case-{i}"));
        build_resolve_tree(&case_root, case);
        let home = case.home.replace("<ROOT>", &case_root.to_string_lossy());
        let dirs: Vec<String> = case
            .unmanaged_dirs
            .iter()
            .map(|d| d.replace("<ROOT>", &case_root.to_string_lossy()))
            .collect();

        let got = binaries::resolve(&home, &dirs, case.ignore_unmanaged);
        let path;
        let origin_str = match &got {
            Some((p, o)) => {
                path = p.clone();
                match o {
                    Origin::Managed => "managed",
                    Origin::Brew => "brew",
                    Origin::External => "external",
                }
            }
            None => {
                path = String::new();
                "none"
            }
        };

        assert_eq!(origin_str, case.expected.origin, "{}", case.name);
        match &case.expected.path {
            Some(expected_path) => {
                let expected = expected_path.replace("<ROOT>", &case_root.to_string_lossy());
                assert_eq!(path, expected, "{}", case.name);
            }
            None => {
                assert!(got.is_none(), "{}", case.name);
            }
        }
    }
}

#[test]
fn resolve_brew_detection_is_symlink_based() {
    // The unmanaged bin is only classified brew when it *resolves* into a Cellar
    // dir; a same-named binary in a plain dir is external regardless of the dir.
    let td = TestDir::new("brew");
    let root = td.root();

    let plain = root.join("opt").join("homebrew").join("bin");
    std::fs::create_dir_all(&plain).unwrap();
    std::fs::write(plain.join("llama"), "x").unwrap();
    make_exec(&plain.join("llama"));

    let (_, origin) = binaries::resolve(
        &root.join("home").to_string_lossy(),
        &[plain.to_string_lossy().into_owned()],
        false,
    )
    .expect("plain bin should resolve");
    assert_eq!(origin, Origin::External);
}

// MARK: - read_version

#[test]
fn read_version_parses_real_formats() {
    let td = TestDir::new("version");
    let root = td.root();

    let ok = install_script("fake-llama-ok", &root.join("ok/llama"));
    let stderr = install_script("fake-llama-stderr", &root.join("stderr/llama"));
    let fail = install_script("fake-llama-fail", &root.join("fail/llama"));

    assert_eq!(
        binaries::read_version(&ok.to_string_lossy()),
        Some(binaries::target_version())
    );
    assert_eq!(
        binaries::read_version(&stderr.to_string_lossy()),
        Some(llama_core::engine::LlamaVersion { build: 11429 })
    );
    assert_eq!(binaries::read_version(&fail.to_string_lossy()), None);
    assert_eq!(
        binaries::read_version(&root.join("missing").to_string_lossy()),
        None
    );
}

// MARK: - installer leaves (download, decompress, atomic_install, symlink, promote)

/// A canned transport for `download` (mirrors the Fase 3 local-server client).
struct StubClient {
    status: u16,
    body: Vec<u8>,
}

impl HttpClient for StubClient {
    fn head(&self, _url: &str, _token: Option<&str>) -> Result<HttpResponse, NetError> {
        unreachable!("download uses GET")
    }
    fn get_json(&self, _url: &str, _token: Option<&str>) -> Result<HttpResponse, NetError> {
        Ok(HttpResponse {
            status: self.status,
            headers: Default::default(),
            body: self.body.clone(),
        })
    }
}

#[test]
fn download_statuses() {
    let ok = StubClient {
        status: 200,
        body: b"artifact-bytes".to_vec(),
    };
    assert_eq!(
        download(&ok, "https://example.com/llama-app.zst", None).unwrap(),
        b"artifact-bytes"
    );

    let redirect = StubClient {
        status: 302,
        body: vec![],
    };
    match download(&redirect, "https://example.com/resolve", Some("tok")) {
        Err(InstallError::DownloadFailed(m)) => {
            assert!(m.contains("301") || m.contains("302"), "{m}");
        }
        other => panic!("expected DownloadFailed, got {other:?}"),
    }

    let gone = StubClient {
        status: 404,
        body: vec![],
    };
    assert!(matches!(
        download(&gone, "https://example.com/resolve", None),
        Err(InstallError::DownloadFailed(_))
    ));
}

#[test]
fn decompress_handles_success_and_failure() {
    let td = TestDir::new("unzstd");
    let root = td.root();

    let payload = b"DEFLATED bytes that fake-unzstd just cats through".to_vec();
    let zst = root.join("llama-app.zst");
    std::fs::write(&zst, &payload).unwrap();
    let dest = root.join("out").join("llama");
    let unzstd_ok = install_script("fake-unzstd", &root.join("tools/unzstd"));

    decompress(
        &unzstd_ok.to_string_lossy(),
        &zst.to_string_lossy(),
        &dest.to_string_lossy(),
    )
    .unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), payload);
    assert_eq!(
        std::fs::metadata(&dest).unwrap().permissions().mode() & 0o111,
        0o111,
        "decompressed binary must be executable"
    );

    // Failing unzstd: error classified, dest untouched, temp cleaned up.
    let dest2 = root.join("out2").join("llama");
    let bad = install_script("fake-unzstd-fail", &root.join("tools/unzstd-bad"));
    let err = decompress(
        &bad.to_string_lossy(),
        &zst.to_string_lossy(),
        &dest2.to_string_lossy(),
    )
    .unwrap_err();
    assert_eq!(err.kind(), "decompressFailed");
    assert!(err.message().contains("oops"));
    assert!(!dest2.exists());
    let leftover: Vec<_> = std::fs::read_dir(root.join("out2"))
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();
    assert!(leftover.is_empty(), "temp file must be cleaned up");

    // Missing source: classified decompressFailed, not a panic.
    let err = decompress(
        &unzstd_ok.to_string_lossy(),
        &root.join("missing.zst").to_string_lossy(),
        &root.join("out3/llama").to_string_lossy(),
    )
    .unwrap_err();
    assert_eq!(err.kind(), "decompressFailed");
}

#[test]
fn atomic_install_replaces_and_chmods() {
    let td = TestDir::new("atomic");
    let root = td.root();

    let src = root.join("staged/llama");
    std::fs::create_dir_all(src.parent().unwrap()).unwrap();
    std::fs::write(&src, "staged-bytes").unwrap();

    let dest = root.join("deeply/nested/dest/llama");
    atomic_install(&src.to_string_lossy(), &dest.to_string_lossy()).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"staged-bytes");
    assert_eq!(
        std::fs::metadata(&dest).unwrap().permissions().mode() & 0o111,
        0o111
    );

    // Overwrite an existing executable.
    atomic_install(&src.to_string_lossy(), &dest.to_string_lossy()).unwrap();
    assert_eq!(std::fs::read(&dest).unwrap(), b"staged-bytes");

    // Missing src: InstallFailed, dest untouched.
    let missing = root.join("nope");
    let err = atomic_install(
        &missing.to_string_lossy(),
        &root.join("x/llama").to_string_lossy(),
    )
    .unwrap_err();
    assert_eq!(err.kind(), "installFailed");
}

#[test]
fn ensure_path_symlink_linked_replaced_left() {
    let td = TestDir::new("symlink");
    let root = td.root();
    let home = root.join("home");

    // Fresh: creates the dir + symlink to the live path.
    let live = root.join("App Support/llama/llama");
    std::fs::create_dir_all(live.parent().unwrap()).unwrap();
    std::fs::write(&live, "x").unwrap();
    assert!(matches!(
        ensure_path_symlink(&home.to_string_lossy(), &live.to_string_lossy()),
        PathSymlinkOutcome::Linked
    ));
    let linked = std::fs::read_link(home.join(".local/bin/llama")).unwrap();
    assert_eq!(linked, live);

    // Existing symlink (pointing elsewhere) is replaced.
    let other = root.join("other/llama");
    std::fs::create_dir_all(other.parent().unwrap()).unwrap();
    std::fs::write(&other, "x").unwrap();
    let link = home.join(".local/bin/llama");
    std::fs::remove_file(&link).unwrap();
    symlink(&other, &link).unwrap();
    assert!(matches!(
        ensure_path_symlink(&home.to_string_lossy(), &live.to_string_lossy()),
        PathSymlinkOutcome::Linked
    ));
    assert_eq!(std::fs::read_link(&link).unwrap(), live);

    // A real file on PATH is left alone.
    std::fs::remove_file(&link).unwrap();
    std::fs::write(&link, "user-file").unwrap();
    assert!(matches!(
        ensure_path_symlink(&home.to_string_lossy(), &live.to_string_lossy()),
        PathSymlinkOutcome::LeftExisting
    ));
    assert_eq!(std::fs::read_to_string(&link).unwrap(), "user-file");
}

#[test]
fn promote_staged_paths() {
    let td = TestDir::new("promote");
    let root = td.root();
    let live = root.join("live").join("llama");
    let home = root.join("home");

    // Nothing staged.
    assert_eq!(
        promote_staged(
            &root.join("missing").to_string_lossy(),
            &live.to_string_lossy(),
            binaries::target_version(),
            &home.to_string_lossy()
        ),
        PromoteOutcome::NothingStaged
    );

    // Staged but wrong version: discarded, live untouched.
    let staged_old = install_script("fake-llama-ok", &root.join("staged/llama"));
    assert_eq!(
        promote_staged(
            &staged_old.to_string_lossy(),
            &live.to_string_lossy(),
            binaries::floor_version(),
            &home.to_string_lossy()
        ),
        PromoteOutcome::DiscardedStale
    );
    assert!(!staged_old.exists());
    assert!(!live.exists());

    // Staged, matching target and valid: promoted + PATH symlink refreshed.
    let staged_ok = install_script("fake-llama-ok", &root.join("staged2/llama"));
    assert_eq!(
        promote_staged(
            &staged_ok.to_string_lossy(),
            &live.to_string_lossy(),
            binaries::target_version(),
            &home.to_string_lossy()
        ),
        PromoteOutcome::Promoted
    );
    assert_eq!(
        std::fs::metadata(&live).unwrap().permissions().mode() & 0o111,
        0o111
    );
    let link = std::fs::read_link(home.join(".local/bin/llama")).unwrap();
    assert_eq!(link, live);

    // Staged that can't be renamed (dest is a dir): fails, staged removed.
    let staged_last = install_script("fake-llama-ok", &root.join("staged3/llama"));
    let dir_dest = root.join("live-as-dir").join("llama");
    std::fs::create_dir_all(&dir_dest).unwrap();
    match promote_staged(
        &staged_last.to_string_lossy(),
        &dir_dest.to_string_lossy(),
        binaries::target_version(),
        &home.to_string_lossy(),
    ) {
        PromoteOutcome::Failed(_) => {}
        other => panic!("expected Failed, got {other:?}"),
    }
    assert!(!staged_last.exists());
}

// MARK: - mem profile: fit + parse + cache

#[derive(Deserialize)]
struct ProfileExp {
    #[serde(rename = "ctxBytesPer1kTokens")]
    ctx_bytes_per_1k_tokens: i64,
    #[serde(rename = "residentBytes")]
    resident_bytes: i64,
    #[serde(rename = "schemaVersion")]
    schema_version: i64,
}

#[derive(Deserialize)]
struct FitCase {
    #[serde(rename = "loMib")]
    lo_mib: i64,
    #[serde(rename = "hiMib")]
    hi_mib: i64,
    #[serde(rename = "ctxLo")]
    ctx_lo: i64,
    #[serde(rename = "ctxHi")]
    ctx_hi: i64,
    profile: Option<ProfileExp>,
}

#[derive(Deserialize)]
struct ParseCase {
    name: String,
    output: String,
    sum: Option<i64>,
}

#[derive(Deserialize)]
struct HashCase {
    #[serde(rename = "modelId")]
    model_id: String,
    hash: String,
}

#[derive(Deserialize)]
struct MemFixture {
    #[serde(rename = "fitParams")]
    fit_params: Vec<FitCase>,
    #[serde(rename = "fitpParse")]
    fitp_parse: Vec<ParseCase>,
    #[serde(rename = "memProfileHash")]
    mem_profile_hash: Vec<HashCase>,
}

#[test]
fn fit_fixture() {
    let f: MemFixture = load("engine.json");
    for c in &f.fit_params {
        let got = fit(c.lo_mib, c.hi_mib, c.ctx_lo, c.ctx_hi);
        match &c.profile {
            Some(expected) => {
                let g = got.expect("expected a profile");
                assert_eq!(g.ctx_bytes_per_1k_tokens, expected.ctx_bytes_per_1k_tokens);
                assert_eq!(g.resident_bytes, expected.resident_bytes);
                assert_eq!(g.schema_version, expected.schema_version);
            }
            None => assert!(got.is_none()),
        }
    }
}

#[test]
fn fitp_parse_fixture() {
    let f: MemFixture = load("engine.json");
    for c in &f.fitp_parse {
        assert_eq!(parse_fitp_output(&c.output), c.sum, "case {:?}", c.name);
    }
}

#[test]
fn mem_profile_cache_uses_hashed_names() {
    let f: MemFixture = load("engine.json");
    for c in &f.mem_profile_hash {
        assert_eq!(mem_profile::cache_file_name(&c.model_id), c.hash);
    }

    let td = TestDir::new("memcache");
    let root = td.root();
    let dir = root.join("Caches").to_string_lossy().into_owned();
    let model = "unsloth/Tiny-GGUF:Q4_K_M";

    // Miss on a fresh dir.
    assert!(mem_profile::cache_get(&dir, model).is_none());

    // Round-trip.
    let p = MemProfile::new(62_000, 505_413_632);
    mem_profile::cache_set(&dir, model, &p).unwrap();
    assert_eq!(mem_profile::cache_get(&dir, model), Some(p));

    // Corrupt entry reads as a miss.
    let path = mem_profile::cache_path(&dir, model);
    std::fs::write(&path, "not json").unwrap();
    assert!(mem_profile::cache_get(&dir, model).is_none());
}
