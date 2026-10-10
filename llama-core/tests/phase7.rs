//! Fase 7 equivalence harness (bridges: `CoreConfig` + core events). Shares
//! `Fixtures/core_config.json` and `Fixtures/core_events.json` with the Swift
//! `LlamaTests` target so both sides are pinned to the same shape.

use std::collections::BTreeSet;
use std::path::PathBuf;

use serde_json::Value;

use llama_core::config::core_config::CoreConfig;
use llama_core::events::{is_known_event, notification_for_event, KNOWN_EVENT_KINDS};

fn load(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Fixtures")
        .join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
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
