//! Exercises the C ABI directly: JSON round-trip, config push, the event
//! channel, and panic containment.

use llama_core_ffi::{
    llama_core_call, llama_core_free, llama_core_init, llama_core_partial_append,
    llama_core_set_config, llama_core_set_event_cb,
};
use std::ffi::{c_char, CStr, CString};
use std::sync::{Mutex, OnceLock};

/// Returns the response body, or `None` if the call returned NULL (e.g. panic).
fn call(method: &str, args: &str) -> Option<String> {
    let m = CString::new(method).unwrap();
    let a = CString::new(args).unwrap();
    let p = unsafe { llama_core_call(m.as_ptr(), a.as_ptr()) };
    if p.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { llama_core_free(p) };
    Some(s)
}

#[test]
fn dispatch_returns_json() {
    assert_eq!(call("ping", "{}").unwrap(), r#"{"pong":true}"#);
    assert_eq!(
        call("version.parse", r#"{"output":"b9370-aa50b2c2a"}"#).unwrap(),
        r#"{"build":9370}"#
    );
    assert_eq!(
        call("version.parse", r#"{"output":"garbage"}"#).unwrap(),
        r#"{"build":null}"#
    );
    assert_eq!(
        call("repo.parse", r#"{"repoDir":"models--org--repo--extra"}"#).unwrap(),
        r#"{"org":"org","repo":"repo--extra"}"#
    );
    assert_eq!(call("repo.parse", r#"{"repoDir":"nope"}"#).unwrap(), "null");
    assert_eq!(
        call(
            "repo.parseQuant",
            r#"{"filename":"Llama-3.2-1B-Q4_K_M.gguf"}"#
        )
        .unwrap(),
        r#""Q4_K_M""#
    );
    assert!(call("no.such.method", "{}")
        .unwrap()
        .contains("unknown method"));
}

#[test]
fn config_push_and_snapshot() {
    assert_eq!(unsafe { llama_core_init(std::ptr::null()) }, 0);

    // Compare parsed values so the test doesn't depend on JSON key order.
    let snapshot = |json: &str| serde_json::from_str::<serde_json::Value>(json).unwrap();
    let config_of = |v: &serde_json::Value| v["config"].clone();

    // With no config pushed, the snapshot is the canonical default.
    assert_eq!(
        config_of(&snapshot(&call("core.snapshot", "{}").unwrap())),
        snapshot(
            r#"{"schemaVersion":1,"hfCacheDirectory":"","hfToken":null,"serverPort":0,"sleepIdleSeconds":null,"agentMode":false,"extraServerArgs":[],"customWebUIDirectory":null,"gpuWorkingSetMb":null,"systemMemoryMb":null,"networkAvailable":true}"#
        )
    );

    let cfg = CString::new(
        r#"{"hfCacheDirectory":"/tmp/hf","serverPort":8080,"agentMode":true,"networkAvailable":false}"#,
    )
    .unwrap();
    assert_eq!(unsafe { llama_core_set_config(cfg.as_ptr()) }, 0);
    assert_eq!(unsafe { llama_core_init(cfg.as_ptr()) }, 0);

    // The snapshot reflects the pushed config, canonicalized (unknown keys
    // dropped, missing keys defaulted); the envelope carries the schema version.
    let snap = snapshot(&call("core.snapshot", "{}").unwrap());
    assert_eq!(snap["schemaVersion"], serde_json::json!(1));
    assert_eq!(
        config_of(&snap),
        snapshot(
            r#"{"schemaVersion":1,"hfCacheDirectory":"/tmp/hf","hfToken":null,"serverPort":8080,"sleepIdleSeconds":null,"agentMode":true,"extraServerArgs":[],"customWebUIDirectory":null,"gpuWorkingSetMb":null,"systemMemoryMb":null,"networkAvailable":false}"#
        )
    );

    // Unknown keys are ignored without error.
    let cfg = CString::new(r#"{"threads":4}"#).unwrap();
    assert_eq!(unsafe { llama_core_set_config(cfg.as_ptr()) }, 0);
    assert!(call("core.snapshot", "{}")
        .unwrap()
        .contains(r#""serverPort":0"#));
}

static EVENTS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

extern "C" fn on_event(p: *const c_char) {
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    EVENTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(s);
}

#[test]
fn event_channel_delivers() {
    EVENTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .clear();
    unsafe { llama_core_set_event_cb(Some(on_event)) };
    assert_eq!(
        call("emit.test", r#"{"payload":{"n":1}}"#).unwrap(),
        r#"{"ok":true}"#
    );
    // Semantic events carry the kind + payload; `known` reports whether the core
    // recognizes the kind.
    assert_eq!(
        call(
            "event.emit",
            r#"{"kind":"serverState","payload":{"state":"running"}}"#
        )
        .unwrap(),
        r#"{"known":true,"ok":true}"#
    );
    assert_eq!(
        call("event.emit", r#"{"kind":"mystery"}"#).unwrap(),
        r#"{"known":false,"ok":true}"#
    );
    unsafe { llama_core_set_event_cb(None) };

    let events = EVENTS.get().unwrap().lock().unwrap();
    assert_eq!(events.len(), 3);
    assert!(events[0].contains(r#""event":"test""#));
    assert!(events[0].contains(r#""n":1"#));
    assert!(events[1].contains(r#""event":"serverState""#));
    assert!(events[1].contains(r#""state":"running""#));
    assert!(events[2].contains(r#""event":"mystery""#));
}

#[test]
fn panic_is_contained() {
    assert_eq!(call("panic.test", "{}"), None);
    // The core is still usable after a contained panic.
    assert_eq!(call("ping", "{}").unwrap(), r#"{"pong":true}"#);
}

#[test]
fn fase1_quant_and_sidecar() {
    assert_eq!(
        call(
            "quant.tagForPath",
            r#"{"path":"unsloth/x-UD-Q4_K_XL.gguf"}"#
        )
        .unwrap(),
        r#"{"tag":"Q4_K_XL"}"#
    );
    assert_eq!(
        call("quant.parseLabel", r#"{"path":"x-Q4_0_4_4.gguf"}"#).unwrap(),
        r#"{"label":"Q4_0"}"#
    );
    assert_eq!(
        call("quant.parseLabel", r#"{"path":"noquant.gguf"}"#).unwrap(),
        r#"{"label":null}"#
    );
    assert_eq!(
        call("quant.quantBitsForPath", r#"{"path":"x-Q4_K_M.gguf"}"#).unwrap(),
        r#"{"bits":4}"#
    );
    assert_eq!(
        call("quant.canonicalTag", r#"{"label":"UD-Q4_K_XL"}"#).unwrap(),
        r#"{"tag":"Q4_K_XL"}"#
    );
    assert_eq!(
        call(
            "sidecar.draftHead",
            r#"{"names":["mtp-Q8_0.gguf","dflash-Q8_0.gguf"],"mainPath":"m-Q8_0.gguf","tag":"Q8_0"}"#
        )
        .unwrap(),
        r#"{"path":"dflash-Q8_0.gguf"}"#
    );
    assert_eq!(
        call("draftHead.fromSidecarPath", r#"{"path":"eagle3-x.gguf"}"#).unwrap(),
        r#"{"kind":null}"#
    );
}

#[test]
fn fase1_model_and_compat() {
    assert_eq!(
        call(
            "modelId.parse",
            r#"{"id":"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"}"#
        )
        .unwrap(),
        r#"{"displayOrg":null,"name":"gemma-3","org":"ggml-org","params":"4B","quant":"Q4_K_M","tags":["it"]}"#
    );
    assert_eq!(
        call(
            "modelId.plainName",
            r#"{"id":"unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M:Q4_K_M","showTags":false}"#
        )
        .unwrap(),
        r#"{"name":"unsloth/Qwen3-Coder 30B-A3B Q4_K_M"}"#
    );
    assert_eq!(
        call("model.makeId", r#"{"org":"a","repo":"b","tag":"Q4_K_M"}"#).unwrap(),
        r#"{"id":"a/b:Q4_K_M"}"#
    );
    assert_eq!(
        call(
            "compat.deviceBudget",
            r#"{"workingSetMb":24576,"physicalMb":32768,"useFullWorkingSet":false}"#
        )
        .unwrap(),
        r#"{"mb":23552.0}"#
    );
    assert!(call("contextTier.list", "{}").unwrap().starts_with(
        r#"[{"label":"4k ctx","rawValue":4096,"shortLabel":"4k","suffix":":4k ctx"}"#
    ));
}

#[test]
fn fase1_ini_and_aliases() {
    assert_eq!(
        call("ini.parse", r#"{"text":"[a]\nk = v\n\n[b]\nj = w\n"}"#).unwrap(),
        r#"{"sections":[{"name":"a","pairs":[{"key":"k","value":"v"}]},{"name":"b","pairs":[{"key":"j","value":"w"}]}]}"#
    );
    assert_eq!(
        call(
            "aliases.validationError",
            r#"{"name":"code","aliases":[{"name":"code","modelId":"m1"}],"replacing":null,"overrideNames":[]}"#
        )
        .unwrap(),
        r#"{"error":"There's already an alias named code."}"#
    );
    assert_eq!(
        call(
            "aliases.setAlias",
            r#"{"section":{"name":"M","pairs":[]},"names":["a","b"]}"#
        )
        .unwrap(),
        r#"{"section":{"name":"M","pairs":[{"key":"alias","value":"a,b"}]}}"#
    );
}

fn fixture_path(rel: &str) -> String {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Fixtures")
        .join(rel);
    p.to_str().unwrap().to_string()
}

#[test]
fn fase2_gguf_metadata() {
    let full = serde_json::json!({ "path": fixture_path("gguf/full.gguf") }).to_string();
    assert_eq!(
        call("gguf.contextLength", &full).unwrap(),
        r#"{"length":40960,"parsed":true}"#
    );
    assert_eq!(
        call("gguf.hasEmbeddedMTPHead", &full).unwrap(),
        r#"{"value":true}"#
    );
    assert_eq!(
        call("gguf.isDecisionModel", &full).unwrap(),
        r#"{"value":false}"#
    );

    // Header parses but the key is absent -> present object with null length.
    let nocontext = serde_json::json!({ "path": fixture_path("gguf/nocontext.gguf") }).to_string();
    assert_eq!(
        call("gguf.contextLength", &nocontext).unwrap(),
        r#"{"length":null,"parsed":true}"#
    );

    // Unparseable -> a whole `null` response.
    let junk = serde_json::json!({ "path": fixture_path("gguf/notgguf.gguf") }).to_string();
    assert_eq!(call("gguf.contextLength", &junk).unwrap(), "null");
    assert_eq!(call("gguf.hasEmbeddedMTPHead", &junk).unwrap(), "null");
}

#[test]
fn fase2_hf_cache() {
    let cache = fixture_path("hf_cache/cache");

    let url = serde_json::json!({
        "url": "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/file.gguf"
    })
    .to_string();
    assert_eq!(
        call("hfCache.repoDirName", &url).unwrap(),
        r#"{"name":"models--unsloth--Qwen3.5-2B-GGUF"}"#
    );
    assert_eq!(
        call("hfCache.repoRelativePath", &url).unwrap(),
        r#"{"path":"file.gguf"}"#
    );

    let scan = serde_json::json!({ "cacheDir": cache }).to_string();
    let json = call("hfCache.scanForSideloaded", &scan).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let entries = v.get("entries").unwrap().as_array().unwrap();
    let mut ids: Vec<&str> = entries
        .iter()
        .map(|e| e.get("id").unwrap().as_str().unwrap())
        .collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![
            "org/Decision-GGUF:Q4_K_M",
            "org/Embedded-GGUF:Q4_K_M",
            "unsloth/MTP-GGUF:Q8_0",
            "unsloth/Sharded-GGUF:Q4_K_M",
            "unsloth/Tiny-GGUF:Q4_K_M",
        ]
    );

    let pb =
        serde_json::json!({ "cacheDir": cache, "modelId": "unsloth/Tiny-GGUF:Q4_K_M" }).to_string();
    assert_eq!(
        call("hfCache.partialBytes", &pb).unwrap(),
        r#"{"bytes":20}"#
    );

    let sfe = serde_json::json!({
        "cacheDir": cache,
        "repoDir": "models--unsloth--Tiny-GGUF",
        "filename": "tiny-Q4_K_M.gguf"
    })
    .to_string();
    assert_eq!(
        call("hfCache.snapshotFileExists", &sfe).unwrap(),
        r#"{"value":true}"#
    );
}

#[test]
fn fase7_models_scan() {
    // `models.scan` cleans installed partial dirs, so scan a copy of the
    // read-only fixture tree.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = std::env::temp_dir().join(format!("llama-ffi-scan-{}-{nanos}", std::process::id()));
    let cache = tmp.join("cache");
    copy_tree(
        std::path::Path::new(&fixture_path("hf_cache/cache")),
        &cache,
    );
    let mem = tmp.join("mem");
    std::fs::create_dir_all(&mem).unwrap();

    // Seed one model's profile through the FFI, then scan.
    let seed = serde_json::json!({
        "cacheDir": mem.to_str().unwrap(),
        "modelId": "unsloth/Tiny-GGUF:Q4_K_M",
        "ctxBytesPer1kTokens": 4096,
        "residentBytes": 2048,
    })
    .to_string();
    assert!(call("engine.memCacheSet", &seed)
        .unwrap()
        .starts_with(r#"{"ok"#));

    let args = serde_json::json!({
        "cacheDir": cache.to_str().unwrap(),
        "memCacheDir": mem.to_str().unwrap(),
    })
    .to_string();
    let v: serde_json::Value = serde_json::from_str(&call("models.scan", &args).unwrap()).unwrap();

    let models = v["models"].as_array().unwrap();
    let mut ids: Vec<&str> = models.iter().map(|m| m["id"].as_str().unwrap()).collect();
    ids.sort();
    assert_eq!(
        ids,
        vec![
            "org/Decision-GGUF:Q4_K_M",
            "org/Embedded-GGUF:Q4_K_M",
            "unsloth/MTP-GGUF:Q8_0",
            "unsloth/Sharded-GGUF:Q4_K_M",
            "unsloth/Tiny-GGUF:Q4_K_M",
        ]
    );

    let seeded = models
        .iter()
        .find(|m| m["id"] == "unsloth/Tiny-GGUF:Q4_K_M")
        .unwrap();
    assert_eq!(seeded["ctxBytesPer1kTokens"], serde_json::json!(4096));
    assert_eq!(seeded["residentBytes"], serde_json::json!(2048));

    let needs: Vec<&str> = v["needsProfile"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p[0].as_str().unwrap())
        .collect();
    assert_eq!(
        needs.len(),
        4,
        "every model except the seeded one wants a probe"
    );
    assert!(!needs.contains(&"unsloth/Tiny-GGUF:Q4_K_M"));

    let _ = std::fs::remove_dir_all(&tmp);
}

/// Recursively copies `src` into `dst`, mirroring symlinks as symlinks.
fn copy_tree(src: &std::path::Path, dst: &std::path::Path) {
    std::fs::create_dir_all(dst).unwrap();
    for entry in std::fs::read_dir(src).unwrap().flatten() {
        let from = entry.path();
        let to = dst.join(entry.file_name());
        let ty = entry.file_type().unwrap();
        if ty.is_symlink() {
            let target = std::fs::read_link(&from).unwrap();
            std::os::unix::fs::symlink(target, &to).unwrap();
        } else if ty.is_dir() {
            copy_tree(&from, &to);
        } else {
            std::fs::copy(&from, &to).unwrap();
        }
    }
}

#[test]
fn fase2_hf_cache_write_blob_and_link() {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let tmp = std::env::temp_dir().join(format!("llama-ffi-wbl-{}-{nanos}", std::process::id()));
    let cache = tmp.join("cache");
    std::fs::create_dir_all(&cache).unwrap();
    let temp_file = tmp.join("upload.tmp");
    std::fs::write(&temp_file, b"blob-content").unwrap();

    let args = serde_json::json!({
        "cacheDir": cache.to_str().unwrap(),
        "repoDir": "models--x--y",
        "commit": "c0ffee",
        "blobHash": "abcdef",
        "filename": "sub/model-Q4_K_M.gguf",
        "tempFile": temp_file.to_str().unwrap(),
    })
    .to_string();
    assert_eq!(
        call("hfCache.writeBlobAndLink", &args).unwrap(),
        r#"{"ok":true}"#
    );

    let symlink = cache.join("models--x--y/snapshots/c0ffee/sub/model-Q4_K_M.gguf");
    assert_eq!(
        std::fs::read_link(&symlink).unwrap().to_string_lossy(),
        "../../../blobs/abcdef"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn fase3_network() {
    let resolve = serde_json::json!({
        "repo": "unsloth/Qwen3-4B-GGUF",
        "requestedQuant": null,
        "budgetMb": 100000,
        "siblings": [
            { "rfilename": "Qwen3-4B-Q8_0.gguf", "size": 4000000000i64 },
            { "rfilename": "Qwen3-4B-Q4_K_M.gguf", "size": 2500000000i64 }
        ]
    })
    .to_string();
    let out: serde_json::Value =
        serde_json::from_str(&call("net.resolveFromSiblings", &resolve).unwrap()).unwrap();
    assert_eq!(out["modelId"], "unsloth/Qwen3-4B-GGUF:Q4_K_M");
    assert_eq!(
        out["mainUrl"],
        "https://huggingface.co/unsloth/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf"
    );
    assert_eq!(out["approximateBytes"], 2500000000i64);

    let missing = serde_json::json!({
        "repo": "org/Missing-Shard-GGUF",
        "budgetMb": 100000,
        "siblings": [
            { "rfilename": "model-00001-of-00002.gguf", "size": 1 }
        ]
    })
    .to_string();
    let out: serde_json::Value =
        serde_json::from_str(&call("net.resolveFromSiblings", &missing).unwrap()).unwrap();
    assert_eq!(out["error"], "shardMissing");

    assert_eq!(
        call(
            "net.resolveUrl",
            r#"{"repo":"org/Space-GGUF","path":"my model-Q4_K_M.gguf"}"#
        )
        .unwrap(),
        r#"{"url":"https://huggingface.co/org/Space-GGUF/resolve/main/my%20model-Q4_K_M.gguf"}"#
    );

    assert_eq!(
        call("catalog.parseBytes", r#"{"label":"5.0 GB"}"#).unwrap(),
        r#"{"bytes":5000000000}"#
    );

    let catalog = serde_json::json!({
        "families": [{
            "brand": "Gemma",
            "featured": true,
            "sizes": [{
                "vision": false,
                "builds": [
                    { "quant": "Q4_K_M", "size": "3.0 GB", "repo": "g/x" },
                    { "quant": "Q8_0", "size": "5.0 GB", "repo": "g/x" }
                ]
            }]
        }],
        "budgetMb": 100000,
        "systemMemoryMb": 32768
    })
    .to_string();
    let out: serde_json::Value =
        serde_json::from_str(&call("catalog.featuredSuggestions", &catalog).unwrap()).unwrap();
    assert_eq!(out["suggestions"][0]["quant"], "Q8_0");
    assert_eq!(out["suggestions"][0]["brand"], "Gemma");

    assert_eq!(
        call(
            "hfRequest.userAgent",
            r#"{"shortVersion":"1.2.3","osMajor":15,"osMinor":2}"#
        )
        .unwrap(),
        r#"{"ua":"llama-app/1.2.3 (macOS 15.2; arm64)"}"#
    );

    let headers = serde_json::json!({
        "url": "https://huggingface.co/x",
        "token": "hf_abc",
        "userAgent": "llama-app/1.2.3 (macOS 15.2; arm64)"
    })
    .to_string();
    assert_eq!(
        call("hfRequest.make", &headers).unwrap(),
        r#"{"headers":[["User-Agent","llama-app/1.2.3 (macOS 15.2; arm64)"],["Authorization","Bearer hf_abc"]]}"#
    );

    assert_eq!(
        call(
            "hfRequest.isHuggingfaceHost",
            r#"{"url":"https://cdn-lfs.huggingface.co/x"}"#
        )
        .unwrap(),
        r#"{"value":true}"#
    );
    assert_eq!(
        call(
            "hfRequest.isHuggingfaceHost",
            r#"{"url":"https://cas-bridge.xethub.hf.co/x"}"#
        )
        .unwrap(),
        r#"{"value":false}"#
    );

    let bh = serde_json::json!({
        "status": 200,
        "headers": [["X-Linked-Etag", "\"abc123\""]]
    })
    .to_string();
    assert_eq!(call("net.blobHash", &bh).unwrap(), r#"{"hash":"abc123"}"#);

    let fm = serde_json::json!({
        "status": 302,
        "headers": [["X-Repo-Commit", "deadbeef"]]
    })
    .to_string();
    assert_eq!(
        call("net.fileMetadata", &fm).unwrap(),
        r#"{"metadata":{"blobHash":null,"commitHash":"deadbeef"}}"#
    );

    assert_eq!(
        call(
            "net.sameHostRedirect",
            r#"{"from":"huggingface.co","to":"cdn-lfs.huggingface.co"}"#
        )
        .unwrap(),
        r#"{"value":false}"#
    );
}

#[test]
fn fase4_engine() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn fixtures_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("Fixtures")
    }

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let root =
        std::env::temp_dir().join(format!("llama-core-ffi-phase4-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let cleanup = || {
        let _ = std::fs::remove_dir_all(&root);
    };

    fn make_exec(p: &std::path::Path) {
        let mut perms = std::fs::metadata(p).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(p, perms).unwrap();
    }

    fn script(name: &str) -> PathBuf {
        fixtures_dir().join("engine").join(name)
    }

    let home = root.join("home").to_string_lossy().into_owned();

    // managedPath (pure string mapping).
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "engine.managedPath",
            &serde_json::json!({ "home": "/Users/fixture" }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["path"], "/Users/fixture/.llama-app/llama");

    // metalConfig: matched and unsupported.
    assert_eq!(
        call("engine.metalConfig", r#"{"brand":"Apple M3 Pro"}"#).unwrap(),
        r#"{"config":"m3"}"#
    );
    let out: serde_json::Value =
        serde_json::from_str(&call("engine.metalConfig", r#"{"brand":"Apple M6 Pro"}"#).unwrap())
            .unwrap();
    assert_eq!(out["error"], "unsupportedHardware");

    // Artifact URLs.
    assert_eq!(
        call(
            "engine.artifactUrl",
            r#"{"version":"b11429","config":"m3"}"#
        )
        .unwrap(),
        r#"{"url":"https://huggingface.co/buckets/ggml-org/install.sh/resolve/b11429/aarch64/macos/metal/m3/llama-app.zst"}"#
    );
    assert_eq!(
        call("engine.unzstdUrl", r#"{"version":"b11429"}"#).unwrap(),
        r#"{"url":"https://huggingface.co/buckets/ggml-org/install.sh/resolve/b11429/aarch64/macos/unzstd"}"#
    );

    // binariesResolve: managed wins; else unmanaged; brew classified by Cellar.
    // No managed file, explicitly empty search list → none.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "engine.binariesResolve",
            &serde_json::json!({
                "home": home,
                "unmanagedDirs": [],
                "ignoreUnmanaged": false,
            })
            .to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(out.is_null());

    std::fs::create_dir_all(root.join("home/.llama-app")).unwrap();
    std::fs::write(root.join("home/.llama-app/llama"), "managed").unwrap();
    make_exec(&root.join("home/.llama-app/llama"));
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "engine.binariesResolve",
            &serde_json::json!({
                "home": home,
                "unmanagedDirs": [],
                "ignoreUnmanaged": false,
            })
            .to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["origin"], "managed");
    assert!(out["path"].as_str().unwrap().ends_with(".llama-app/llama"));

    let plain_dir = root.join("usr/local/bin");
    std::fs::create_dir_all(&plain_dir).unwrap();
    std::fs::write(plain_dir.join("llama"), "plain").unwrap();
    make_exec(&plain_dir.join("llama"));
    std::fs::remove_dir_all(root.join("home/.llama-app")).unwrap();
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "engine.binariesResolve",
            &serde_json::json!({
                "home": home,
                "unmanagedDirs": [plain_dir.to_string_lossy()],
                "ignoreUnmanaged": false,
            })
            .to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["origin"], "external");

    let brew_bin = root.join("opt/homebrew/bin");
    std::fs::create_dir_all(brew_bin.join("../Cellar/llama.cpp/1.0.0/bin")).unwrap();
    std::fs::write(brew_bin.join("../Cellar/llama.cpp/1.0.0/bin/llama"), "brew").unwrap();
    make_exec(&brew_bin.join("../Cellar/llama.cpp/1.0.0/bin/llama"));
    symlink(
        "../Cellar/llama.cpp/1.0.0/bin/llama",
        brew_bin.join("llama"),
    )
    .unwrap();
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "engine.binariesResolve",
            &serde_json::json!({
                "home": home,
                "unmanagedDirs": [brew_bin.to_string_lossy()],
                "ignoreUnmanaged": false,
            })
            .to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["origin"], "brew");

    // readVersion runs the binary. Copy script to temp, chmod, and probe.
    let bin = root.join("fake-llama");
    std::fs::copy(script("fake-llama-ok"), &bin).unwrap();
    make_exec(&bin);
    assert_eq!(
        call(
            "engine.readVersion",
            &serde_json::json!({ "path": bin.to_string_lossy() }).to_string()
        )
        .unwrap(),
        r#"{"build":11429}"#
    );

    // decompress: success + classified failure.
    let unzstd = root.join("unzstd");
    let unzstd_bad = root.join("unzstd-bad");
    std::fs::copy(script("fake-unzstd"), &unzstd).unwrap();
    std::fs::copy(script("fake-unzstd-fail"), &unzstd_bad).unwrap();
    make_exec(&unzstd);
    make_exec(&unzstd_bad);
    let zst = root.join("llama-app.zst");
    std::fs::write(&zst, "payload").unwrap();
    let dest = root.join("out/llama");
    let out = call(
        "engine.decompress",
        &serde_json::json!({
            "unzstd": unzstd.to_string_lossy(),
            "zst": zst.to_string_lossy(),
            "dest": dest.to_string_lossy(),
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(out, r#"{"ok":true}"#);
    assert_eq!(std::fs::read(&dest).unwrap(), b"payload");
    let dest_bad = root.join("out2/llama");
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "engine.decompress",
            &serde_json::json!({
                "unzstd": unzstd_bad.to_string_lossy(),
                "zst": zst.to_string_lossy(),
                "dest": dest_bad.to_string_lossy(),
            })
            .to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["error"], "decompressFailed");
    assert!(!dest_bad.exists());

    // ensurePathSymlink: fresh link + leave a real file alone.
    let live = root.join("live/llama");
    std::fs::create_dir_all(live.parent().unwrap()).unwrap();
    std::fs::write(&live, "x").unwrap();
    assert_eq!(
        call(
            "engine.ensurePathSymlink",
            &serde_json::json!({ "home": home, "llamaPath": live.to_string_lossy() }).to_string()
        )
        .unwrap(),
        r#"{"outcome":"linked"}"#
    );
    let link_path = root.join("home/.local/bin/llama");
    std::fs::remove_file(&link_path).unwrap();
    std::fs::write(&link_path, "user-file").unwrap();
    assert_eq!(
        call(
            "engine.ensurePathSymlink",
            &serde_json::json!({ "home": home, "llamaPath": live.to_string_lossy() }).to_string()
        )
        .unwrap(),
        r#"{"outcome":"leftExisting"}"#
    );
    std::fs::remove_file(&link_path).unwrap();

    // promoteStaged: promoted when the staged binary matches target.
    let staged = root.join("staged/llama");
    std::fs::create_dir_all(staged.parent().unwrap()).unwrap();
    std::fs::copy(script("fake-llama-ok"), &staged).unwrap();
    make_exec(&staged);
    let live2 = root.join("live2/llama");
    assert_eq!(
        call(
            "engine.promoteStaged",
            &serde_json::json!({
                "staged": staged.to_string_lossy(),
                "live": live2.to_string_lossy(),
                "targetBuild": 11429,
                "home": home,
            })
            .to_string()
        )
        .unwrap(),
        r#"{"outcome":"promoted"}"#
    );
    assert_eq!(
        std::fs::metadata(&live2).unwrap().permissions().mode() & 0o111,
        0o111
    );

    // fit / parseFitp / memCache round-trips.
    assert_eq!(
        call(
            "engine.fit",
            r#"{"loMib":2000,"hiMib":5000,"ctxLo":4096,"ctxHi":131072}"#
        )
        .unwrap(),
        r#"{"profile":{"ctxBytesPer1kTokens":24774193,"residentBytes":1995676903,"schemaVersion":2}}"#
    );
    assert_eq!(
        call(
            "engine.fit",
            r#"{"loMib":3000,"hiMib":2000,"ctxLo":4096,"ctxHi":131072}"#
        )
        .unwrap(),
        "null"
    );
    assert_eq!(
        call("engine.parseFitp", r#"{"output":"MTL0 217 145 0"}"#).unwrap(),
        r#"{"sum":362}"#
    );
    assert_eq!(
        call(
            "engine.memCacheFileName",
            r#"{"modelId":"unsloth/Tiny-GGUF:Q4_K_M"}"#
        )
        .unwrap(),
        r#"{"hash":"26fce773f312d183395cda0d5faa7fd8"}"#
    );
    let cache_dir = root.join("Caches").to_string_lossy().into_owned();
    let get = call(
        "engine.memCacheGet",
        &serde_json::json!({ "cacheDir": cache_dir, "modelId": "org/Model:Q4_0" }).to_string(),
    )
    .unwrap();
    assert!(get.contains(r#""profile":null"#));
    let set = call(
        "engine.memCacheSet",
        &serde_json::json!({
            "cacheDir": cache_dir,
            "modelId": "org/Model:Q4_0",
            "ctxBytesPer1kTokens": 62000,
            "residentBytes": 505413632,
        })
        .to_string(),
    )
    .unwrap();
    assert_eq!(set, r#"{"ok":true}"#);
    let get = call(
        "engine.memCacheGet",
        &serde_json::json!({ "cacheDir": cache_dir, "modelId": "org/Model:Q4_0" }).to_string(),
    )
    .unwrap();
    assert!(get.contains(r#""ctxBytesPer1kTokens":62000"#));
    assert!(get.contains(r#""residentBytes":505413632"#));

    cleanup();
}

#[test]
fn fase5_download() {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let root =
        std::env::temp_dir().join(format!("llama-core-ffi-phase5-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();

    // -- Pure response leaves. --
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "download.extractFullSize",
            r#"{"status":206,"headers":[["Content-Range","bytes 302-90210/100000"]]}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["size"], 100_000);

    assert_eq!(
        call("download.userMessage", r#"{"status":401}"#).unwrap(),
        r#"{"message":"Hugging Face requires authentication for this download. Set a Hugging Face token in Settings and try again."}"#
    );
    assert_eq!(
        call("download.userMessage", r#"{"status":200}"#).unwrap(),
        r#"{"message":"Download failed with HTTP 200."}"#
    );

    assert_eq!(
        call("download.isTransient", r#"{"code":-1001}"#).unwrap(),
        r#"{"transient":true}"#
    );
    assert_eq!(
        call("download.isTransient", r#"{"code":-999}"#).unwrap(),
        r#"{"transient":false}"#
    );

    assert_eq!(
        call("download.retryDelay", r#"{"attempts":2}"#).unwrap(),
        r#"{"delay":8.0}"#
    );

    assert_eq!(
        call(
            "download.transientDecision",
            r#"{"isNetworkAvailable":true,"attempts":0,"maxAttempts":3}"#
        )
        .unwrap(),
        r#"{"outcome":"retry"}"#
    );
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "download.transientDecision",
            r#"{"isNetworkAvailable":false,"attempts":1,"maxAttempts":3}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["outcome"], "park");
    assert_eq!(out["resumeOnReconnect"], true);

    // Progress floors at 1 and never shrinks.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "download.refreshProgress",
            r#"{"totalUnitCount":0,"completedFilesBytes":0,"activeBytes":0,"expectedActiveBytes":0}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["totalUnitCount"], 1);

    let out: serde_json::Value = serde_json::from_str(
        &call(
            "download.rate",
            r#"{"samples":[[0.0,0],[2.0,3000]],"window":5.0}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["rate"], 1500);
    let out: serde_json::Value =
        serde_json::from_str(&call("download.rate", r#"{"samples":[],"window":5.0}"#).unwrap())
            .unwrap();
    assert!(out["rate"].is_null());

    let out: serde_json::Value = serde_json::from_str(
        &call(
            "download.errorMessage",
            r#"{"kind":"notCompatible","reason":"requires 32 GB"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["description"], "Model Not Compatible");
    assert_eq!(
        out["recovery"],
        "This model requires 32 GB. Choose a smaller model or upgrade your system memory."
    );

    assert_eq!(
        call(
            "download.verifyHash",
            r#"{"computed":"aabb","expected":"aabb"}"#
        )
        .unwrap(),
        r#"{"matches":true}"#
    );
    assert_eq!(
        call("download.isTooSmall", r#"{"size":0}"#).unwrap(),
        r#"{"tooSmall":true}"#
    );
    assert_eq!(
        call("download.isTooSmall", r#"{"size":1000001}"#).unwrap(),
        r#"{"tooSmall":false}"#
    );

    // -- Partial writer lifecycle through the registry. --
    assert_eq!(
        call("partial.sha256", r#"{"text":"abc"}"#).unwrap(),
        r#"{"hex":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}"#
    );

    let partial = root
        .join("model.gguf.partial")
        .to_string_lossy()
        .into_owned();

    // Feed-hash a missing path is a classified openFailed, not a panic.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.feedHash",
            r#"{"path":"/nonexistent/llama-core/x"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["error"], "openFailed");

    // Open fresh: empty digest, zero bytes.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.open",
            &serde_json::json!({ "path": partial }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    let handle = out["handle"].as_u64().expect("handle");
    assert_eq!(out["bytesWritten"], 0);
    assert_eq!(
        out["digest"],
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );

    // Append streamed chunks: running hash == sha256 of concatenation.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.append",
            &serde_json::json!({ "handle": handle, "text": "abc" }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["ok"], true);
    assert_eq!(out["bytesWritten"], 3);
    assert_eq!(
        out["digest"],
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );

    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.append",
            &serde_json::json!({ "handle": handle, "text": "def" }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["bytesWritten"], 6);
    assert_eq!(
        out["digest"],
        "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
    );

    // The on-disk file re-hashes to the same digest (resume path).
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.feedHash",
            &serde_json::json!({ "path": partial }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        out["digest"],
        "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1fc6c5c6dcd93c4721"
    );

    // 200 restart: truncate + reset hash + counter.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.restart",
            &serde_json::json!({ "handle": handle }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["ok"], true);
    assert_eq!(out["bytesWritten"], 0);
    assert_eq!(
        out["digest"],
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );

    // Unknown handle: classified notFound.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.append",
            &serde_json::json!({ "handle": 424242, "text": "x" }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["error"], "notFound");

    // Close drops the registry entry.
    assert_eq!(
        call(
            "partial.close",
            &serde_json::json!({ "handle": handle }).to_string()
        )
        .unwrap(),
        r#"{"ok":true}"#
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fase5_partial_append_takes_raw_binary() {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let root =
        std::env::temp_dir().join(format!("llama-core-ffi-pwraw-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join("binary.gguf.partial");

    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.open",
            &serde_json::json!({ "path": path.to_string_lossy() }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    let handle = out["handle"].as_u64().expect("handle");

    // GGUF is arbitrary binary: NULs, bytes that are not valid UTF-8 (0xff,
    // 0xfe), and a lone 0xc3 that would begin a two-byte sequence. None of it
    // can ride the JSON `char *` bridge, which is why the raw entry point exists.
    let bytes: [u8; 21] = [
        0x47, 0x47, 0x55, 0x46, 0x00, 0x01, 0x02, 0x7f, 0x80, 0xff, 0xfe, 0xc3, 0x28, 0x0a, 0x00,
        0x54, 0x49, 0x54, 0x4c, 0x45, 0x00,
    ];
    let rc = unsafe { llama_core_partial_append(handle, bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, 0, "raw append must succeed");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        bytes,
        "bytes land byte-for-byte"
    );

    // The running digest equals the one-pass digest over the exact bytes —
    // so nothing was dropped, reinterpreted as UTF-8, or truncated.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "partial.digest",
            &serde_json::json!({ "handle": handle }).to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        out["digest"],
        "eca56b38b9227c6089910f3d2c00bb395c9c5f00e3d80b8531d07dc05769b238"
    );

    // A null pointer with a non-zero length is rejected, never dereferenced.
    let rc = unsafe { llama_core_partial_append(handle, std::ptr::null(), 8) };
    assert_eq!(rc, -1);

    // An unknown handle is classified, not a panic.
    let rc = unsafe { llama_core_partial_append(424242, bytes.as_ptr(), 0) };
    assert_eq!(rc, -2);

    assert_eq!(
        call(
            "partial.close",
            &serde_json::json!({ "handle": handle }).to_string()
        )
        .unwrap(),
        r#"{"ok":true}"#
    );

    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn fase6_server() {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let root =
        std::env::temp_dir().join(format!("llama-core-ffi-phase6-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&root).unwrap();
    let tmp = root.to_string_lossy().into_owned();
    std::fs::write(root.join("index.html"), "<html></html>").unwrap();

    // -- Shell quoting and rendered invocations. --
    assert_eq!(
        call("server.quote", r#"{"text":"a' b"}"#).unwrap(),
        r#"{"quoted":"'a'\\'' b'"}"#
    );
    assert_eq!(
        call("server.quote", r#"{"text":"/usr/local/bin/llama"}"#).unwrap(),
        r#"{"quoted":"/usr/local/bin/llama"}"#
    );
    let command: serde_json::Value = serde_json::from_str(
        &call(
            "server.displayCommand",
            &serde_json::json!({
                "executablePath": format!("{tmp}/bin/llama"),
                "arguments": ["serve", "--port", "8080"],
                "extraArgCount": 0,
                "env": [["GGML_METAL_NO_RESIDENCY", "1"]],
            })
            .to_string(),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(
        command["command"],
        format!("export GGML_METAL_NO_RESIDENCY=1\n\n{tmp}/bin/llama serve \\\n  --port 8080")
    );

    // -- Host argument gating on the multi-host engine version. --
    assert_eq!(
        call(
            "server.hostArgument",
            r#"{"bindAddress":"100.64.0.1","versionBuild":11200}"#
        )
        .unwrap(),
        r#"{"host":"127.0.0.1,100.64.0.1"}"#
    );
    assert_eq!(
        call(
            "server.hostArgument",
            r#"{"bindAddress":"100.64.0.1","versionBuild":11103}"#
        )
        .unwrap(),
        r#"{"host":"100.64.0.1"}"#
    );

    // -- Tailscale + interface classification. --
    assert_eq!(
        call("server.isTailscaleAddress", r#"{"address":"100.64.0.1"}"#).unwrap(),
        r#"{"isTailscale":true}"#
    );
    assert_eq!(
        call("server.isTailscaleAddress", r#"{"address":"192.168.1.1"}"#).unwrap(),
        r#"{"isTailscale":false}"#
    );
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "server.pickIpAddress",
            r#"{"addresses":[["en0","192.168.1.5"],["en1","192.168.2.9"]],"primary":"en1"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["address"], "192.168.2.9");
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "server.tailscaleAddress",
            r#"{"addresses":[["en0","192.168.1.5"],["utun5","100.64.0.3"],["utun4","100.64.0.9"]]}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["address"], "100.64.0.3");

    // -- Bind/host resolution + fit target math. --
    assert_eq!(
        call(
            "server.effectiveBindAddress",
            r#"{"configured":"0.0.0.0","localAddrs":["192.168.1.5"]}"#
        )
        .unwrap(),
        r#"{"address":"0.0.0.0"}"#
    );
    assert_eq!(
        call(
            "server.effectiveBindAddress",
            r#"{"configured":"10.9.9.9","localAddrs":["192.168.1.5"]}"#
        )
        .unwrap(),
        r#"{"address":null}"#
    );
    assert_eq!(
        call(
            "server.resolvedHost",
            r#"{"effectiveBind":"0.0.0.0","addresses":[["en0","192.168.1.5"]],"primary":"en0"}"#
        )
        .unwrap(),
        r#"{"host":"192.168.1.5"}"#
    );
    assert_eq!(
        call(
            "server.resolvedHost",
            r#"{"effectiveBind":"100.64.0.1","addresses":[],"primary":null}"#
        )
        .unwrap(),
        r#"{"host":"100.64.0.1"}"#
    );
    assert_eq!(
        call("server.localHost", r#"{"effectiveBind":"0.0.0.0"}"#).unwrap(),
        r#"{"host":"localhost"}"#
    );
    assert_eq!(
        call(
            "server.fitTargetMb",
            r#"{"gpuWorkingSetMb":6144,"memoryMb":8192,"useFullWorkingSet":false}"#
        )
        .unwrap(),
        r#"{"mb":2048}"#
    );

    // -- Preset rejection + model statuses. --
    assert_eq!(
        call(
            "server.presetRejectionOption",
            r#"{"message":"option 'temperture' not recognized in preset 'models.user.ini'\n"}"#,
        )
        .unwrap(),
        r#"{"option":"temperture"}"#
    );
    assert_eq!(
        json_call("server.presetRejectionOption", r#"{"message":"quiet"}"#),
        r#"{"option":null}"#
    );

    let out: serde_json::Value = serde_json::from_str(
        &call(
            "server.parseModelStatuses",
            r#"{"body":"{\"data\":[{\"id\":\"org/Model:Q4_0\",\"status\":{\"value\":\"loaded\"}}]}"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["statuses"][0][0], "org/Model:Q4_0");
    assert_eq!(out["statuses"][0][1], "loaded");
    let out: serde_json::Value =
        serde_json::from_str(&call("server.parseModelStatuses", r#"{"body":"garbage"}"#).unwrap())
            .unwrap();
    assert_eq!(out["error"], "malformed");

    // -- Reclaim port leaves. --
    assert_eq!(
        call("server.extractPids", r#"{"lsofOutput":"123\n456\n"}"#).unwrap(),
        r#"{"pids":[123,456]}"#
    );
    assert_eq!(
        call("server.isLlamaProcess", r#"{"comm":"/usr/sbin/lsof"}"#).unwrap(),
        r#"{"isLlama":false}"#
    );
    assert_eq!(
        call(
            "server.lastPathComponent",
            r#"{"path":"/opt/homebrew/bin/llama"}"#
        )
        .unwrap(),
        r#"{"name":"llama"}"#
    );

    // -- buildLaunchSpec: full tree + effective webui (real index.html). --
    let build = |json: serde_json::Value| -> serde_json::Value {
        serde_json::from_str(&call("server.buildLaunchSpec", &json.to_string()).unwrap()).unwrap()
    };
    let spec = build(serde_json::json!({
        "llamaPath": format!("{tmp}/bin/llama"),
        "presetsPath": format!("{tmp}/models.ini"),
        "logFilePath": format!("{tmp}/llama-server.log"),
        "hfCacheDirectory": format!("{tmp}/hf-cache"),
        "emptyCacheDirectory": format!("{tmp}/empty-cache"),
        "port": 9931,
        "fitTargetMb": 1024,
        "configuredBindAddress": "100.64.0.1",
        "addresses": [["utun4", "100.64.0.1"], ["en0", "192.168.1.5"]],
        "primaryInterface": "en0",
        "currentVersionBuild": 11200,
        "configuredWebUiDirectory": tmp,
        "sleepIdleSeconds": null,
        "agentMode": true,
        "apiKey": "ffi-key",
        "extraArgs": ["--ctx-size", "8192"],
    }));
    assert_eq!(spec["spec"]["executablePath"], format!("{tmp}/bin/llama"));
    let args: Vec<String> = spec["spec"]["arguments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(args[13], "--host");
    assert_eq!(args[14], "127.0.0.1,100.64.0.1");
    assert_eq!(args.last().unwrap(), "8192");
    assert_eq!(spec["spec"]["extraArgCount"], 2);
    assert_eq!(spec["spec"]["env"][1][0], "HF_HUB_CACHE");
    // Exposed bind => the key is delivered via the environment, not a flag.
    assert_eq!(spec["spec"]["env"][3][0], "LLAMA_API_KEY");
    assert_eq!(spec["spec"]["env"][3][1], "ffi-key");

    // stale bind falls back to loopback => explicit --host 127.0.0.1.
    let spec = build(serde_json::json!({
        "llamaPath": format!("{tmp}/bin/llama"),
        "presetsPath": format!("{tmp}/models.ini"),
        "logFilePath": format!("{tmp}/llama-server.log"),
        "hfCacheDirectory": format!("{tmp}/hf-cache"),
        "emptyCacheDirectory": format!("{tmp}/empty-cache"),
        "port": 9931,
        "fitTargetMb": 1024,
        "configuredBindAddress": "100.64.0.1",
        "addresses": [["en0", "192.168.1.5"]],
        "primaryInterface": "en0",
        "currentVersionBuild": 11200,
        "configuredWebUiDirectory": tmp,
        "sleepIdleSeconds": null,
        "agentMode": false,
        "extraArgs": [],
    }));
    let args: Vec<String> = spec["spec"]["arguments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let host_idx = args.iter().position(|a| a == "--host").unwrap();
    assert_eq!(args[host_idx + 1], "127.0.0.1");
    assert!(args.contains(&"--spec-default".to_string()));

    // no configured bind at all => still an explicit loopback --host.
    let spec = build(serde_json::json!({
        "llamaPath": format!("{tmp}/bin/llama"),
        "presetsPath": format!("{tmp}/models.ini"),
        "logFilePath": format!("{tmp}/llama-server.log"),
        "hfCacheDirectory": format!("{tmp}/hf-cache"),
        "emptyCacheDirectory": format!("{tmp}/empty-cache"),
        "port": 9931,
        "fitTargetMb": 1024,
        "addresses": [["en0", "192.168.1.5"]],
        "primaryInterface": "en0",
        "agentMode": false,
        "extraArgs": [],
    }));
    let args: Vec<String> = spec["spec"]["arguments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let host_idx = args.iter().position(|a| a == "--host").unwrap();
    assert_eq!(args[host_idx + 1], "127.0.0.1");

    // no binary => classified, not a panic.
    let out: serde_json::Value = serde_json::from_str(
        &call(
            "server.buildLaunchSpec",
            r#"{"llamaPath":null,"presetsPath":"/tmp/m","logFilePath":"/tmp/l","hfCacheDirectory":"/tmp/h","port":1,"fitTargetMb":0,"addresses":[],"extraArgs":[]}"#,
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(out["error"], "noBinary");

    let _ = std::fs::remove_dir_all(&root);
}

/// Small alias to keep the JSON-string friction out of the test body.
fn json_call(method: &str, args: &str) -> String {
    call(method, args).unwrap()
}
