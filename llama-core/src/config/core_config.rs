//! `CoreConfig` — the immutable environment snapshot the Swift host pushes into
//! the core (`llama_core_init` / `llama_core_set_config`).
//!
//! Rust never reads `UserDefaults`; everything the core needs about the host
//! environment (cache location, port, network state, measured memory) travels
//! in this JSON object. `from_value` is tolerant — missing keys fall back to
//! defaults so an older host can talk to a newer core — while `to_value` always
//! emits the full, canonical shape (every key present) so snapshots are stable.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Schema version the core speaks. Bumped when the shape changes incompatibly.
pub const CORE_CONFIG_SCHEMA_VERSION: u32 = 1;

/// Host environment snapshot. Serialized `camelCase` to match the Swift side.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct CoreConfig {
    pub schema_version: u32,
    /// Hugging Face cache directory (`~/.cache/huggingface/hub` by default).
    pub hf_cache_directory: String,
    /// Optional Hugging Face access token for gated downloads.
    pub hf_token: Option<String>,
    /// Preferred llama-server port.
    pub server_port: u16,
    /// Idle-unload timeout in seconds; `None` means "never unload".
    pub sleep_idle_seconds: Option<u64>,
    /// Whether agent mode is enabled.
    pub agent_mode: bool,
    /// Extra user-supplied llama-server arguments.
    pub extra_server_args: Vec<String>,
    /// Custom web UI directory, if the user set one.
    #[serde(rename = "customWebUIDirectory")]
    pub custom_web_ui_directory: Option<String>,
    /// Metal recommended working-set size, measured by the host.
    pub gpu_working_set_mb: Option<u64>,
    /// Physical RAM, measured by the host.
    pub system_memory_mb: Option<u64>,
    /// Latest `NWPathMonitor` status, injected by the host. Optimistically
    /// `true` until the first update so a download started at launch isn't
    /// treated as offline.
    pub network_available: bool,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            schema_version: CORE_CONFIG_SCHEMA_VERSION,
            hf_cache_directory: String::new(),
            hf_token: None,
            server_port: 0,
            sleep_idle_seconds: None,
            agent_mode: false,
            extra_server_args: Vec::new(),
            custom_web_ui_directory: None,
            gpu_working_set_mb: None,
            system_memory_mb: None,
            // Optimistic: matches the Swift `isNetworkAvailable` seed.
            network_available: true,
        }
    }
}

impl CoreConfig {
    /// Decodes a pushed snapshot, tolerating missing/extra keys and never
    /// failing: a malformed payload yields defaults. The schema version is
    /// always stamped to the version the core speaks.
    pub fn from_value(v: &Value) -> Self {
        let mut cfg: Self = serde_json::from_value(v.clone()).unwrap_or_default();
        cfg.schema_version = CORE_CONFIG_SCHEMA_VERSION;
        cfg
    }

    /// Canonical JSON: every field present, schema version stamped.
    pub fn to_value(&self) -> Value {
        let mut v = serde_json::to_value(self).unwrap_or(Value::Null);
        if let Some(obj) = v.as_object_mut() {
            obj.insert(
                "schemaVersion".into(),
                Value::from(CORE_CONFIG_SCHEMA_VERSION),
            );
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn defaults_are_optimistic_online() {
        let c = CoreConfig::default();
        assert_eq!(c.schema_version, CORE_CONFIG_SCHEMA_VERSION);
        assert!(c.network_available);
        assert!(c.sleep_idle_seconds.is_none());
    }

    #[test]
    fn empty_input_yields_defaults() {
        assert_eq!(CoreConfig::from_value(&Value::Null), CoreConfig::default());
        assert_eq!(CoreConfig::from_value(&json!({})), CoreConfig::default());
    }

    #[test]
    fn malformed_input_yields_defaults_not_error() {
        // Wrong types: tolerance means fall back to defaults, never panic.
        let c = CoreConfig::from_value(&json!({ "serverPort": "not-a-number" }));
        assert_eq!(c, CoreConfig::default());
    }

    #[test]
    fn round_trips_known_fields() {
        let c = CoreConfig::from_value(&json!({
            "hfCacheDirectory": "/tmp/hf",
            "hfToken": "hf_abc",
            "serverPort": 8080,
            "sleepIdleSeconds": 900,
            "agentMode": true,
            "extraServerArgs": ["--verbose"],
            "customWebUIDirectory": "/tmp/webui",
            "gpuWorkingSetMb": 12288,
            "systemMemoryMb": 16384,
            "networkAvailable": false,
        }));
        assert_eq!(c.hf_cache_directory, "/tmp/hf");
        assert_eq!(c.hf_token.as_deref(), Some("hf_abc"));
        assert_eq!(c.server_port, 8080);
        assert_eq!(c.sleep_idle_seconds, Some(900));
        assert!(c.agent_mode);
        assert_eq!(c.extra_server_args, vec!["--verbose".to_string()]);
        assert_eq!(c.custom_web_ui_directory.as_deref(), Some("/tmp/webui"));
        assert_eq!(c.gpu_working_set_mb, Some(12288));
        assert_eq!(c.system_memory_mb, Some(16384));
        assert!(!c.network_available);
    }

    #[test]
    fn to_value_is_canonical_and_versioned() {
        let v = CoreConfig::from_value(&json!({ "serverPort": 9000 })).to_value();
        let obj = v.as_object().unwrap();
        // Every key present even though only one was supplied.
        for key in [
            "schemaVersion",
            "hfCacheDirectory",
            "hfToken",
            "serverPort",
            "sleepIdleSeconds",
            "agentMode",
            "extraServerArgs",
            "customWebUIDirectory",
            "gpuWorkingSetMb",
            "systemMemoryMb",
            "networkAvailable",
        ] {
            assert!(obj.contains_key(key), "missing {key} in {v}");
        }
        assert_eq!(obj["schemaVersion"], json!(CORE_CONFIG_SCHEMA_VERSION));
        assert_eq!(obj["serverPort"], json!(9000));
        assert_eq!(obj["networkAvailable"], json!(true));
    }
}
