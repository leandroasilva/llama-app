use std::path::{Path, PathBuf};

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

fn xdg(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(fallback))
}

pub fn data_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("llama")
}

pub fn config_dir() -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join("llama")
}

pub fn state_dir() -> PathBuf {
    xdg("XDG_STATE_HOME", ".local/state").join("llama")
}

pub fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", ".cache").join("llama")
}

pub fn home_dir() -> PathBuf {
    home()
}

pub fn default_hf_cache() -> PathBuf {
    home().join(".cache/huggingface/hub")
}

pub fn models_ini() -> PathBuf {
    data_dir().join("models.ini")
}

pub fn user_ini() -> PathBuf {
    config_dir().join("models.user.ini")
}

pub fn settings_file() -> PathBuf {
    config_dir().join("gtk-settings.json")
}

pub fn log_file() -> PathBuf {
    state_dir().join("llama-server.log")
}

pub fn empty_cache() -> PathBuf {
    data_dir().join("empty-cache")
}

pub fn mem_profile_dir() -> PathBuf {
    cache_dir().join("mem-profile")
}

pub fn engine_download_dir() -> PathBuf {
    cache_dir().join("engine")
}

pub fn ensure_dirs() {
    for dir in [
        data_dir(),
        config_dir(),
        state_dir(),
        cache_dir(),
        empty_cache(),
        mem_profile_dir(),
        engine_download_dir(),
    ] {
        let _ = std::fs::create_dir_all(dir);
    }
}

pub fn display_path(path: &Path) -> String {
    let home = home();
    match path.strip_prefix(&home) {
        Ok(rest) => format!("~/{}", rest.to_string_lossy()),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

pub fn physical_memory_mb() -> i64 {
    let Ok(text) = std::fs::read_to_string("/proc/meminfo") else {
        return 8192;
    };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            let kb: i64 = rest
                .split_whitespace()
                .next()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if kb > 0 {
                return kb / 1024;
            }
        }
    }
    8192
}
