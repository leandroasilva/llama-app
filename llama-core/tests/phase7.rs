//! Fase 7 equivalence harness (bridges: `CoreConfig` + core events). Shares
//! `Fixtures/core_config.json` and `Fixtures/core_events.json` with the Swift
//! `LlamaTests` target so both sides are pinned to the same shape.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

use llama_core::config::core_config::CoreConfig;
use llama_core::engine::mem_profile::{cache_set, MemProfile};
use llama_core::events::{is_known_event, notification_for_event, KNOWN_EVENT_KINDS};
use llama_core::models::scan::scan;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Fixtures")
}

fn load(name: &str) -> Value {
    let path = fixtures_dir().join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn temp_dir(tag: &str) -> PathBuf {
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("llama-phase7-{tag}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Recursively copies `src` into `dst`, mirroring symlinks as symlinks.
fn copy_tree(src: &Path, dst: &Path) {
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
fn core_config_normalizes_to_fixture() {
    let f = load("core_config.json");
    let cases = f["cases"].as_array().expect("cases");
    assert!(!cases.is_empty());
    for c in cases {
        let name = c["name"].as_str().unwrap_or("?");
        let got = CoreConfig::from_value(&c["input"]).to_value();
        assert_eq!(&got, &c["normalized"], "core_config case {name}");
    }
}

#[test]
fn core_config_snapshot_shape_is_stable() {
    let f = load("core_config.json");
    let shape: BTreeSet<String> = f["snapshotShape"]
        .as_array()
        .expect("snapshotShape")
        .iter()
        .map(|k| k.as_str().unwrap().to_string())
        .collect();

    let got: BTreeSet<String> = CoreConfig::default()
        .to_value()
        .as_object()
        .unwrap()
        .keys()
        .cloned()
        .collect();

    assert_eq!(got, shape, "snapshot keys drifted from the fixture");
}

#[test]
fn core_event_notification_mapping_matches_fixture() {
    let f = load("core_events.json");
    let events = f["events"].as_array().expect("events");

    let mut fixture_kinds = BTreeSet::new();
    for e in events {
        let kind = e["kind"].as_str().expect("kind");
        let notification = e["notification"].as_str().expect("notification");
        assert_eq!(
            notification_for_event(kind),
            Some(notification),
            "mapping for {kind}"
        );
        assert!(is_known_event(kind));
        fixture_kinds.insert(kind.to_string());
    }

    // The fixture must enumerate exactly the kinds the core can emit.
    let known: BTreeSet<String> = KNOWN_EVENT_KINDS.iter().map(|k| k.to_string()).collect();
    assert_eq!(
        fixture_kinds, known,
        "event kind set drifted from the fixture"
    );
}

#[test]
fn models_scan_matches_fixture() {
    let f = load("model_scan.json");

    let cache = temp_dir("cache");
    copy_tree(
        &fixtures_dir().join(f["cacheDir"].as_str().unwrap()),
        &cache,
    );

    let mem_dir = temp_dir("mem");
    let seeded = &f["seededProfile"];
    let seeded_id = seeded["modelId"].as_str().unwrap();
    let seeded_ctx = seeded["ctxBytesPer1kTokens"].as_i64().unwrap();
    let seeded_resident = seeded["residentBytes"].as_i64().unwrap();
    cache_set(
        mem_dir.to_str().unwrap(),
        seeded_id,
        &MemProfile::new(seeded_ctx, seeded_resident),
    )
    .unwrap();

    let out = scan(cache.to_str().unwrap(), mem_dir.to_str().unwrap());

    let got: BTreeSet<String> = out.models.iter().map(|m| m.entry.id.clone()).collect();
    let expected: BTreeSet<String> = f["expectedModelIds"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    assert_eq!(got, expected, "scanned model set drifted from the fixture");

    for m in &out.models {
        let needs = out.needs_profile.iter().any(|(id, _)| id == &m.entry.id);
        if m.entry.id == seeded_id {
            assert_eq!(m.ctx_bytes_per_1k_tokens, seeded_ctx);
            assert_eq!(m.resident_bytes, seeded_resident);
            assert!(!needs, "seeded model must not need a profile");
        } else {
            assert_eq!(m.ctx_bytes_per_1k_tokens, 0);
            assert_eq!(m.resident_bytes, 0);
            assert!(needs, "unseeded model must need a profile");
        }
    }

    let _ = std::fs::remove_dir_all(&cache);
    let _ = std::fs::remove_dir_all(&mem_dir);
}
