use std::collections::BTreeMap;
use std::path::PathBuf;

use llama_core::config::preset::{ContextLengthPolicy, KvCacheType, PerformanceConfig};
use serde::{Deserialize, Serialize};

use crate::paths;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PerfSettings {
    pub flash_attention: bool,
    pub kv_cache_type_k: String,
    pub kv_cache_type_v: String,
    pub parallel: i64,
    pub batch_size: i64,
    pub ubatch_size: i64,
    pub no_kv_offload: bool,
}

impl Default for PerfSettings {
    fn default() -> Self {
        Self {
            flash_attention: true,
            kv_cache_type_k: "q8_0".into(),
            kv_cache_type_v: "q8_0".into(),
            parallel: 1,
            batch_size: 512,
            ubatch_size: 256,
            no_kv_offload: false,
        }
    }
}

impl PerfSettings {
    pub fn from_core(c: PerformanceConfig) -> Self {
        Self {
            flash_attention: c.flash_attention,
            kv_cache_type_k: c.kv_cache_type_k.as_str().into(),
            kv_cache_type_v: c.kv_cache_type_v.as_str().into(),
            parallel: c.parallel,
            batch_size: c.batch_size,
            ubatch_size: c.ubatch_size,
            no_kv_offload: c.no_kv_offload,
        }
    }

    pub fn normalized(self) -> Self {
        Self::from_core(self.to_core())
    }

    pub fn to_core(&self) -> PerformanceConfig {
        PerformanceConfig {
            flash_attention: self.flash_attention,
            kv_cache_type_k: KvCacheType::parse(&self.kv_cache_type_k).unwrap_or(KvCacheType::Q8_0),
            kv_cache_type_v: KvCacheType::parse(&self.kv_cache_type_v).unwrap_or(KvCacheType::Q8_0),
            parallel: self.parallel,
            batch_size: self.batch_size,
            ubatch_size: self.ubatch_size,
            no_kv_offload: self.no_kv_offload,
        }
        .normalized()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub server_port: i64,
    pub bind_all: bool,
    pub hf_cache: String,
    pub hf_token: String,
    pub backend: String,
    pub context_policy: String,
    pub performance: PerfSettings,
    pub selected_tiers: BTreeMap<String, i64>,
    pub performance_overrides: BTreeMap<String, PerfSettings>,
    pub extra_args: String,
    pub agent_mode: bool,
    pub sleep_idle_seconds: Option<i64>,
    pub server_api_key: String,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            server_port: 9931,
            bind_all: false,
            hf_cache: paths::default_hf_cache().to_string_lossy().into_owned(),
            hf_token: String::new(),
            backend: "cpu".into(),
            context_policy: "largestFitting".into(),
            performance: PerfSettings::default(),
            selected_tiers: BTreeMap::new(),
            performance_overrides: BTreeMap::new(),
            extra_args: String::new(),
            agent_mode: false,
            sleep_idle_seconds: None,
            server_api_key: String::new(),
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let path = paths::settings_file();
        let Ok(data) = std::fs::read(&path) else {
            return Self::default();
        };
        serde_json::from_slice(&data).unwrap_or_else(|_| Self::default())
    }

    pub fn save(&self) -> Result<(), String> {
        let path = paths::settings_file();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let data = serde_json::to_vec_pretty(self).map_err(|e| e.to_string())?;
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &data).map_err(|e| e.to_string())?;
        set_private(&tmp)?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
        set_private(&path)?;
        Ok(())
    }

    pub fn hf_cache_path(&self) -> PathBuf {
        PathBuf::from(if self.hf_cache.trim().is_empty() {
            paths::default_hf_cache().to_string_lossy().into_owned()
        } else {
            self.hf_cache.clone()
        })
    }

    pub fn context_policy(&self) -> ContextLengthPolicy {
        ContextLengthPolicy::parse(&self.context_policy)
            .unwrap_or(ContextLengthPolicy::LargestFitting)
    }

    pub fn extra_args(&self) -> Vec<String> {
        const RESERVED: [&str; 8] = [
            "--host",
            "--port",
            "--bind",
            "-H",
            "-p",
            "--models-preset",
            "--log-file",
            "--api-key",
        ];
        let mut out = Vec::new();
        let mut tokens = self.extra_args.split_whitespace();
        while let Some(token) = tokens.next() {
            if RESERVED.contains(&token) {
                tokens.next();
                continue;
            }
            if RESERVED
                .iter()
                .any(|flag| token.starts_with(&format!("{flag}=")))
            {
                continue;
            }
            out.push(token.to_string());
        }
        out
    }

    pub fn token(&self) -> Option<&str> {
        let t = self.hf_token.trim();
        (!t.is_empty()).then_some(t)
    }

    pub fn ensure_api_key(&mut self) -> String {
        if self.server_api_key.is_empty() {
            let key = random_hex_32();
            self.server_api_key = key.clone();
            let _ = self.save();
            return key;
        }
        self.server_api_key.clone()
    }
}

#[cfg(unix)]
fn set_private(path: &std::path::Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))
        .map_err(|e| e.to_string())
}

#[cfg(windows)]
fn set_private(_path: &std::path::Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn random_hex_32() -> String {
    let mut bytes = [0u8; 32];
    if let Ok(mut f) = std::fs::File::open("/dev/urandom") {
        use std::io::Read;
        if f.read_exact(&mut bytes).is_ok() {
            return llama_core::engine::partial_writer::hex(&bytes);
        }
    }
    fallback_hex_32()
}

#[cfg(windows)]
fn random_hex_32() -> String {
    use std::ffi::c_void;

    #[link(name = "bcrypt")]
    unsafe extern "system" {
        fn BCryptGenRandom(algorithm: *mut c_void, buffer: *mut u8, length: u32, flags: u32)
            -> i32;
    }

    const BCRYPT_USE_SYSTEM_PREFERRED_RNG: u32 = 0x0000_0002;
    let mut bytes = [0u8; 32];
    let status = unsafe {
        BCryptGenRandom(
            std::ptr::null_mut(),
            bytes.as_mut_ptr(),
            bytes.len() as u32,
            BCRYPT_USE_SYSTEM_PREFERRED_RNG,
        )
    };
    if status == 0 {
        return llama_core::engine::partial_writer::hex(&bytes);
    }
    fallback_hex_32()
}

fn fallback_hex_32() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let input = format!(
        "{}:{}:{}",
        std::process::id(),
        nanos,
        COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    llama_core::engine::partial_writer::sha256_hex(input.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extra_args_drop_reserved_flags_and_values() {
        let settings = Settings {
            extra_args: "--ctx-size 8192 --port 1 -p 2 --host=0.0.0.0 --api-key secret --verbose"
                .into(),
            ..Settings::default()
        };
        assert_eq!(
            settings.extra_args(),
            vec!["--ctx-size", "8192", "--verbose"]
        );
    }

    #[test]
    fn performance_normalizes_to_supported_values() {
        let perf = PerfSettings {
            parallel: 99,
            batch_size: 333,
            ubatch_size: 4096,
            ..PerfSettings::default()
        }
        .normalized();
        assert_eq!(perf.parallel, 8);
        assert_eq!(perf.batch_size, 512);
        assert_eq!(perf.ubatch_size, 256);
    }
}
