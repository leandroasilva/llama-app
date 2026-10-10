//! Fase 2 equivalence harness (filesystem I/O). Shares `Fixtures/*` with the
//! Swift `LlamaTests` target; both sides must reproduce the pinned expectations.

use llama_core::hf::{GGUFMetadata, HFCache, SideloadedEntry};
use llama_core::models::ResolvedPaths;
use serde::Deserialize;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

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

// MARK: - GGUFMetadata

#[derive(Deserialize)]
struct CtxExp {
    parsed: bool,
    length: Option<i64>,
}

#[derive(Deserialize)]
struct BoolExp {
    parsed: bool,
    value: Option<bool>,
}

#[derive(Deserialize)]
struct MetadataCase {
    file: String,
    #[serde(rename = "contextLength")]
    context_length: CtxExp,
    #[serde(rename = "hasEmbeddedMTPHead")]
    has_embedded_mtp_head: BoolExp,
    #[serde(rename = "isDecisionModel")]
    is_decision_model: BoolExp,
}

#[derive(Deserialize)]
struct MetadataFixture {
    files: Vec<MetadataCase>,
}

#[test]
fn gguf_metadata_fixtures() {
    let f: MetadataFixture = load("gguf_metadata.json");
    for c in &f.files {
        let path = fixtures_dir().join(&c.file);
        let path = path.to_str().unwrap();

        let got = GGUFMetadata::context_length(path);
        if c.context_length.parsed {
            assert_eq!(
                got,
                Some(c.context_length.length),
                "contextLength {}",
                c.file
            );
        } else {
            assert_eq!(got, None, "contextLength {} should be unparseable", c.file);
        }

        let got = GGUFMetadata::has_embedded_mtp_head(path);
        if c.has_embedded_mtp_head.parsed {
            assert_eq!(
                got,
                Some(c.has_embedded_mtp_head.value.unwrap_or(false)),
                "hasEmbeddedMTPHead {}",
                c.file
            );
        } else {
            assert_eq!(got, None, "hasEmbeddedMTPHead {} unparseable", c.file);
        }

        let got = GGUFMetadata::is_decision_model(path);
        if c.is_decision_model.parsed {
            assert_eq!(
                got,
                Some(c.is_decision_model.value.unwrap_or(false)),
                "isDecisionModel {}",
                c.file
            );
        } else {
            assert_eq!(got, None, "isDecisionModel {} unparseable", c.file);
        }
    }
}

#[test]
fn gguf_metadata_missing_file_is_unparseable() {
    let missing = fixtures_dir().join("gguf/does-not-exist.gguf");
    let missing = missing.to_str().unwrap();
    assert_eq!(GGUFMetadata::context_length(missing), None);
    assert_eq!(GGUFMetadata::has_embedded_mtp_head(missing), None);
    assert_eq!(GGUFMetadata::is_decision_model(missing), None);
}

// MARK: - HFCache

#[derive(Deserialize)]
struct NameCase {
    url: String,
    name: Option<String>,
}

#[derive(Deserialize)]
struct PathCase {
    url: String,
    path: Option<String>,
}

#[derive(Deserialize)]
struct PathHelpersFixture {
    #[serde(rename = "repoDirName")]
    repo_dir_name: Vec<NameCase>,
    #[serde(rename = "repoRelativePath")]
    repo_relative_path: Vec<PathCase>,
}

#[derive(Deserialize)]
struct ScanEntry {
    id: String,
    #[serde(rename = "ctxWindow")]
    ctx_window: i64,
    #[serde(rename = "fileSize")]
    file_size: i64,
    #[serde(rename = "draftHead")]
    draft_head: Option<String>,
    #[serde(rename = "isDecisionModel")]
    is_decision_model: bool,
    #[serde(rename = "usesMTP")]
    uses_mtp: bool,
    #[serde(rename = "mmprojFile")]
    mmproj_file: Option<String>,
    #[serde(rename = "draftSidecarFile")]
    draft_sidecar_file: Option<String>,
    #[serde(rename = "additionalParts")]
    additional_parts: Vec<String>,
    #[serde(rename = "hfRepoDirName")]
    hf_repo_dir_name: String,
    #[serde(rename = "modelFile")]
    model_file: String,
}

#[derive(Deserialize)]
struct ScanFixture {
    #[serde(rename = "cacheDir")]
    cache_dir: String,
    entries: Vec<ScanEntry>,
}

#[derive(Deserialize)]
struct PartialBytesCase {
    #[serde(rename = "modelId")]
    model_id: String,
    bytes: i64,
}

#[derive(Deserialize)]
struct ScanPlaceholdersFixture {
    #[serde(rename = "cacheDir")]
    cache_dir: String,
    paths: Vec<String>,
}

#[derive(Deserialize)]
struct SnapshotExistsCase {
    #[serde(rename = "repoDir")]
    repo_dir: String,
    filename: String,
    value: bool,
}

#[derive(Deserialize)]
struct WriteBlobFixture {
    #[serde(rename = "repoDir")]
    repo_dir: String,
    commit: String,
    #[serde(rename = "blobHash")]
    blob_hash: String,
    filename: String,
    #[serde(rename = "expectedSymlinkTarget")]
    expected_symlink_target: String,
    #[serde(rename = "expectedRefsMain")]
    expected_refs_main: String,
}

#[derive(Deserialize)]
struct DeleteExpected {
    #[serde(rename = "modelFileExists")]
    model_file_exists: bool,
    #[serde(rename = "mmprojFileExists")]
    mmproj_file_exists: bool,
    #[serde(rename = "blobsDirExists")]
    blobs_dir_exists: bool,
    #[serde(rename = "snapshotsDirExists")]
    snapshots_dir_exists: bool,
    #[serde(rename = "repoDirExists")]
    repo_dir_exists: bool,
    #[serde(rename = "refsMainExists")]
    refs_main_exists: bool,
}

#[derive(Deserialize)]
struct DeleteFixture {
    #[serde(rename = "repoDir")]
    repo_dir: String,
    #[serde(rename = "modelFile")]
    model_file: String,
    #[serde(rename = "additionalParts")]
    additional_parts: Vec<String>,
    #[serde(rename = "mmprojFile")]
    mmproj_file: Option<String>,
    expected: DeleteExpected,
}

#[derive(Deserialize)]
struct HFCacheFixture {
    #[serde(rename = "pathHelpers")]
    path_helpers: PathHelpersFixture,
    scan: ScanFixture,
    #[serde(rename = "partialBytes")]
    partial_bytes: Vec<PartialBytesCase>,
    #[serde(rename = "scanPlaceholders")]
    scan_placeholders: ScanPlaceholdersFixture,
    #[serde(rename = "snapshotFileExists")]
    snapshot_file_exists: Vec<SnapshotExistsCase>,
    #[serde(rename = "writeBlobAndLink")]
    write_blob_and_link: WriteBlobFixture,
    #[serde(rename = "deleteModelFiles")]
    delete_model_files: DeleteFixture,
}

fn hf_fixture() -> HFCacheFixture {
    load("hf_cache.json")
}

fn abs_cache() -> PathBuf {
    fixtures_dir().join("hf_cache").join("cache")
}

#[test]
fn hf_cache_path_helpers() {
    let f = hf_fixture();
    for c in &f.path_helpers.repo_dir_name {
        assert_eq!(HFCache::repo_dir_name_from_url(&c.url), c.name, "{}", c.url);
    }
    for c in &f.path_helpers.repo_relative_path {
        assert_eq!(
            HFCache::repo_relative_path_from_url(&c.url),
            c.path,
            "{}",
            c.url
        );
    }
}

fn sorted_entries(mut v: Vec<SideloadedEntry>) -> Vec<SideloadedEntry> {
    v.sort_by(|a, b| a.id.cmp(&b.id));
    v
}

#[test]
fn hf_cache_scan_for_sideloaded() {
    let f = hf_fixture();
    let cache = fixtures_dir().join(&f.scan.cache_dir);
    let cache_str = cache.to_str().unwrap();

    let mut got = sorted_entries(HFCache::scan_for_sideloaded(cache_str));
    let mut expected = f.scan.entries;
    expected.sort_by(|a, b| a.id.cmp(&b.id));

    assert_eq!(got.len(), expected.len(), "entry count");
    for (e, c) in got.iter_mut().zip(expected.iter()) {
        assert_eq!(e.id, c.id);
        assert_eq!(e.ctx_window, c.ctx_window, "{} ctx", c.id);
        assert_eq!(e.file_size, c.file_size, "{} size", c.id);
        assert_eq!(
            e.scanned_draft_head.map(|d| d.as_str().to_string()),
            c.draft_head,
            "{} draftHead",
            c.id
        );
        assert_eq!(
            e.is_decision_model, c.is_decision_model,
            "{} decision",
            c.id
        );
        assert_eq!(e.paths.uses_mtp, c.uses_mtp, "{} usesMTP", c.id);
        assert_eq!(e.paths.hf_repo_dir_name, c.hf_repo_dir_name);
        assert_eq!(
            e.paths.model_file,
            cache.join(&c.model_file).to_string_lossy(),
            "{} modelFile",
            c.id
        );
        assert_eq!(
            e.paths.mmproj_file,
            c.mmproj_file
                .as_ref()
                .map(|p| cache.join(p).to_string_lossy().into_owned()),
            "{} mmprojFile",
            c.id
        );
        assert_eq!(
            e.paths.draft_sidecar_file,
            c.draft_sidecar_file
                .as_ref()
                .map(|p| cache.join(p).to_string_lossy().into_owned()),
            "{} draftSidecarFile",
            c.id
        );
        let expected_parts: Vec<String> = c
            .additional_parts
            .iter()
            .map(|p| cache.join(p).to_string_lossy().into_owned())
            .collect();
        assert_eq!(e.paths.additional_parts, expected_parts, "{} parts", c.id);
    }
}

#[test]
fn hf_cache_partial_bytes() {
    let f = hf_fixture();
    let cache = fixtures_dir().join("hf_cache").join("cache");
    let cache_str = cache.to_str().unwrap();
    for c in &f.partial_bytes {
        assert_eq!(
            HFCache::partial_bytes(cache_str, &c.model_id),
            c.bytes,
            "partialBytes({})",
            c.model_id
        );
    }
}

#[test]
fn hf_cache_scan_placeholders() {
    let f = hf_fixture();
    let cache = fixtures_dir().join(&f.scan_placeholders.cache_dir);
    let mut got: Vec<String> = HFCache::scan_placeholders(cache.to_str().unwrap())
        .iter()
        .map(|p| {
            p.strip_prefix(&cache)
                .unwrap_or(p)
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    got.sort();
    let mut expected = f.scan_placeholders.paths;
    expected.sort();
    assert_eq!(got, expected);
}

#[test]
fn hf_cache_snapshot_file_exists() {
    let f = hf_fixture();
    let cache = fixtures_dir().join("hf_cache").join("cache");
    let cache_str = cache.to_str().unwrap();
    for c in &f.snapshot_file_exists {
        assert_eq!(
            HFCache::snapshot_file_exists(cache_str, &c.repo_dir, &c.filename),
            c.value,
            "{}/{}",
            c.repo_dir,
            c.filename
        );
    }
}

fn unique_temp_dir(tag: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("llama-phase2-{tag}-{}-{nanos}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn copy_tree(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).unwrap();
    for entry in fs::read_dir(src).unwrap().flatten() {
        let ft = entry.file_type().unwrap();
        let to = dst.join(entry.file_name());
        if ft.is_symlink() {
            let target = fs::read_link(entry.path()).unwrap();
            std::os::unix::fs::symlink(target, &to).unwrap();
        } else if ft.is_dir() {
            copy_tree(&entry.path(), &to);
        } else {
            fs::copy(entry.path(), &to).unwrap();
        }
    }
}

#[test]
fn hf_cache_write_blob_and_link() {
    let f = hf_fixture();
    let w = &f.write_blob_and_link;
    let tmp = unique_temp_dir("wbl");
    let cache = tmp.join("cache");
    fs::create_dir_all(&cache).unwrap();

    let temp_file = tmp.join("upload.tmp");
    fs::write(&temp_file, b"hello-blob").unwrap();

    HFCache::write_blob_and_link(
        cache.to_str().unwrap(),
        &w.repo_dir,
        &w.commit,
        &w.blob_hash,
        &w.filename,
        temp_file.to_str().unwrap(),
    )
    .unwrap();

    let blob = cache.join(&w.repo_dir).join("blobs").join(&w.blob_hash);
    assert!(blob.exists(), "blob should exist");
    assert_eq!(fs::read(&blob).unwrap(), b"hello-blob");
    assert!(!temp_file.exists(), "temp file should be moved");

    let symlink = cache
        .join(&w.repo_dir)
        .join("snapshots")
        .join(&w.commit)
        .join(&w.filename);
    assert_eq!(
        fs::read_link(&symlink).unwrap().to_string_lossy(),
        w.expected_symlink_target
    );

    let refs_main = cache.join(&w.repo_dir).join("refs").join("main");
    assert_eq!(
        fs::read_to_string(&refs_main).unwrap(),
        w.expected_refs_main
    );

    // Idempotent: a second write with the same blob leaves the blob intact and
    // just drops the temp file.
    let temp2 = tmp.join("upload2.tmp");
    fs::write(&temp2, b"hello-blob").unwrap();
    HFCache::write_blob_and_link(
        cache.to_str().unwrap(),
        &w.repo_dir,
        &w.commit,
        &w.blob_hash,
        &w.filename,
        temp2.to_str().unwrap(),
    )
    .unwrap();
    assert!(!temp2.exists());
    assert_eq!(fs::read(&blob).unwrap(), b"hello-blob");

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn hf_cache_delete_model_files() {
    let f = hf_fixture();
    let d = &f.delete_model_files;
    let tmp = unique_temp_dir("delete");
    let cache = tmp.join("cache");
    copy_tree(&abs_cache(), &cache);

    let paths = ResolvedPaths {
        model_file: cache.join(&d.model_file).to_string_lossy().into_owned(),
        additional_parts: d
            .additional_parts
            .iter()
            .map(|p| cache.join(p).to_string_lossy().into_owned())
            .collect(),
        mmproj_file: d
            .mmproj_file
            .as_ref()
            .map(|p| cache.join(p).to_string_lossy().into_owned()),
        uses_mtp: false,
        draft_sidecar_file: None,
        hf_repo_dir_name: d.repo_dir.clone(),
    };

    HFCache::delete_model_files(cache.to_str().unwrap(), &d.repo_dir, &paths).unwrap();

    let repo = cache.join(&d.repo_dir);
    assert_eq!(
        cache.join(&d.model_file).exists(),
        d.expected.model_file_exists
    );
    assert_eq!(
        d.mmproj_file
            .as_ref()
            .map(|p| cache.join(p).exists())
            .unwrap_or(false),
        d.expected.mmproj_file_exists
    );
    assert_eq!(
        repo.join("blobs").exists(),
        d.expected.blobs_dir_exists,
        "blobs dir"
    );
    assert_eq!(
        repo.join("snapshots").exists(),
        d.expected.snapshots_dir_exists,
        "snapshots dir"
    );
    assert_eq!(repo.exists(), d.expected.repo_dir_exists, "repo dir");
    assert_eq!(
        repo.join("refs").join("main").exists(),
        d.expected.refs_main_exists,
        "refs/main"
    );

    let _ = fs::remove_dir_all(&tmp);
}

#[test]
fn hf_cache_clean_installed_partials() {
    let tmp = unique_temp_dir("clean");
    let cache = tmp.join("cache");
    copy_tree(&abs_cache(), &cache);
    let cache_str = cache.to_str().unwrap();

    let mut ids: HashSet<String> = HashSet::new();
    ids.insert("unsloth/Tiny-GGUF:Q4_K_M".to_string());
    HFCache::clean_installed_partials(cache_str, &ids);

    assert!(
        !HFCache::partial_dir(cache_str, "unsloth/Tiny-GGUF:Q4_K_M").exists(),
        "installed partials removed"
    );
    // The empty org dir under the partial root is cleaned up too.
    assert!(!cache.join(".llama-partial").join("unsloth").exists());
    // The unrelated org's partials survive.
    assert!(HFCache::placeholder_url(cache_str, "org/Other-GGUF:Q8_0").exists());

    let _ = fs::remove_dir_all(&tmp);
}
