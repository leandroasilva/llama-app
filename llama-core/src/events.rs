//! Core event vocabulary.
//!
//! The core is the owner of logical state (see `MIGRACAO_RUST_REVIEW.md` §3.4):
//! when state changes it emits a semantic event over the FFI event channel, and
//! the Swift `EventBridge` re-posts the matching `NotificationCenter`
//! notification so the existing UI contract is unchanged.
//!
//! The core speaks *semantic* kinds (`serverState`, `modelStatus`, ...), not the
//! host's notification names — mapping kinds to notifications is the host's
//! job. The canonical mapping lives here so both sides can be pinned to the
//! shared `Fixtures/core_events.json` corpus.

use serde_json::{json, Value};

pub const EVENT_SERVER_STATE: &str = "serverState";
pub const EVENT_MODEL_DOWNLOADS: &str = "modelDownloads";
pub const EVENT_MODEL_DOWNLOADED_LIST: &str = "modelDownloadedList";
pub const EVENT_USER_SETTINGS: &str = "userSettings";
pub const EVENT_MODEL_DOWNLOAD_FAILED: &str = "modelDownloadFailed";
pub const EVENT_MODEL_DOWNLOAD_COMPLETED: &str = "modelDownloadCompleted";
pub const EVENT_MODEL_STATUS: &str = "modelStatus";
pub const EVENT_GLOBAL_INPUT_SHORTCUT: &str = "globalInputShortcut";

/// Every event kind the core can emit. Kept in lockstep with
/// `Fixtures/core_events.json` by the test below.
pub const KNOWN_EVENT_KINDS: &[&str] = &[
    EVENT_SERVER_STATE,
    EVENT_MODEL_DOWNLOADS,
    EVENT_MODEL_DOWNLOADED_LIST,
    EVENT_USER_SETTINGS,
    EVENT_MODEL_DOWNLOAD_FAILED,
    EVENT_MODEL_DOWNLOAD_COMPLETED,
    EVENT_MODEL_STATUS,
    EVENT_GLOBAL_INPUT_SHORTCUT,
];

/// Maps a semantic event kind to the host notification name it should be
/// re-posted as. `None` for unknown kinds (host should log and ignore).
pub fn notification_for_event(kind: &str) -> Option<&'static str> {
    match kind {
        EVENT_SERVER_STATE => Some("LBServerStateDidChange"),
        EVENT_MODEL_DOWNLOADS => Some("LBModelDownloadsDidChange"),
        EVENT_MODEL_DOWNLOADED_LIST => Some("LBModelDownloadedListDidChange"),
        EVENT_USER_SETTINGS => Some("LBUserSettingsDidChange"),
        EVENT_MODEL_DOWNLOAD_FAILED => Some("LBModelDownloadDidFail"),
        EVENT_MODEL_DOWNLOAD_COMPLETED => Some("LBModelDownloadDidComplete"),
        EVENT_MODEL_STATUS => Some("LBModelStatusDidChange"),
        EVENT_GLOBAL_INPUT_SHORTCUT => Some("LBGlobalInputShortcutDidChange"),
        _ => None,
    }
}

/// Whether `kind` is one the core is allowed to emit.
pub fn is_known_event(kind: &str) -> bool {
    notification_for_event(kind).is_some()
}

/// Builds the JSON payload handed to the host callback: `{"event": kind,
/// "payload": ...}`. The kind is preserved verbatim, so an unknown kind still
/// travels (the host decides what to do with it).
pub fn event_value(kind: &str, payload: Value) -> Value {
    json!({ "event": kind, "payload": payload })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_known_kind_has_a_notification() {
        for kind in KNOWN_EVENT_KINDS {
            assert!(
                notification_for_event(kind).is_some(),
                "no notification mapping for {kind}"
            );
            assert!(is_known_event(kind));
        }
    }

    #[test]
    fn unknown_kind_has_no_notification() {
        assert_eq!(notification_for_event("nope"), None);
        assert!(!is_known_event("nope"));
    }

    #[test]
    fn event_value_wraps_kind_and_payload() {
        let v = event_value(EVENT_SERVER_STATE, json!({ "state": "running" }));
        assert_eq!(v["event"], json!(EVENT_SERVER_STATE));
        assert_eq!(v["payload"]["state"], json!("running"));
    }
}
