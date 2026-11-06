use std::path::{Path, PathBuf};

#[cfg(not(windows))]
fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

#[cfg(windows)]
fn home() -> PathBuf {
    if let Some(profile) = std::env::var_os("USERPROFILE").filter(|v| !v.is_empty()) {
        return PathBuf::from(profile);
    }
    let drive = std::env::var_os("HOMEDRIVE").unwrap_or_default();
    let path = std::env::var_os("HOMEPATH").unwrap_or_default();
    if !drive.is_empty() || !path.is_empty() {
        return PathBuf::from(format!(
            "{}{}",
            drive.to_string_lossy(),
            path.to_string_lossy()
        ));
    }
    PathBuf::from("C:\\")
}

#[cfg(not(windows))]
fn xdg(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(fallback))
}

#[cfg(windows)]
fn local_app_data() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("AppData/Local"))
}

#[cfg(windows)]
fn roaming_app_data() -> PathBuf {
    std::env::var_os("APPDATA")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join("AppData/Roaming"))
}

#[cfg(not(windows))]
pub fn data_dir() -> PathBuf {
    xdg("XDG_DATA_HOME", ".local/share").join("llama")
}

#[cfg(windows)]
pub fn data_dir() -> PathBuf {
    local_app_data().join("llama")
}

#[cfg(not(windows))]
pub fn config_dir() -> PathBuf {
    xdg("XDG_CONFIG_HOME", ".config").join("llama")
}

#[cfg(windows)]
pub fn config_dir() -> PathBuf {
    roaming_app_data().join("llama")
}

#[cfg(not(windows))]
pub fn state_dir() -> PathBuf {
    xdg("XDG_STATE_HOME", ".local/state").join("llama")
}

#[cfg(windows)]
pub fn state_dir() -> PathBuf {
    local_app_data().join("llama").join("logs")
}

#[cfg(not(windows))]
pub fn cache_dir() -> PathBuf {
    xdg("XDG_CACHE_HOME", ".cache").join("llama")
}

#[cfg(windows)]
pub fn cache_dir() -> PathBuf {
    local_app_data().join("llama").join("cache")
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

/// Point GTK/GLib at the runtime files shipped beside the executable. On other
/// platforms the system's XDG/GTK prefix is already correct.
#[cfg(windows)]
pub fn prepare_runtime() {
    let Ok(exe) = std::env::current_exe() else {
        return;
    };
    let Some(dir) = exe.parent() else {
        return;
    };
    let _ = std::env::set_current_dir(dir);

    let share = dir.join("share");
    let schemas = share.join("glib-2.0").join("schemas");
    if schemas.is_dir() {
        std::env::set_var("GSETTINGS_SCHEMA_DIR", &schemas);
    }

    let pixbuf = dir.join("lib").join("gdk-pixbuf-2.0").join("2.10.0");
    if pixbuf.join("loaders.cache").is_file() {
        std::env::set_var("GDK_PIXBUF_MODULE_FILE", pixbuf.join("loaders.cache"));
    }
    if pixbuf.join("loaders").is_dir() {
        std::env::set_var("GDK_PIXBUF_MODULEDIR", pixbuf.join("loaders"));
    }

    let gio = dir.join("lib").join("gio").join("modules");
    if gio.is_dir() {
        std::env::set_var("GIO_EXTRA_MODULES", gio);
    }

    if share.is_dir() {
        let mut data_dirs = vec![share];
        if let Some(existing) = std::env::var_os("XDG_DATA_DIRS") {
            data_dirs.extend(std::env::split_paths(&existing));
        }
        if let Ok(joined) = std::env::join_paths(data_dirs) {
            std::env::set_var("XDG_DATA_DIRS", joined);
        }
        std::env::set_var("GTK_DATA_PREFIX", dir);
    }
    std::env::set_var("GTK_EXE_PREFIX", dir);
}

#[cfg(not(windows))]
pub fn prepare_runtime() {}

pub fn display_path(path: &Path) -> String {
    let home = home();
    match path.strip_prefix(&home) {
        Ok(rest) => format!("~{}{}", std::path::MAIN_SEPARATOR, rest.to_string_lossy()),
        Err(_) => path.to_string_lossy().into_owned(),
    }
}

#[cfg(not(windows))]
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

#[cfg(windows)]
pub fn physical_memory_mb() -> i64 {
    #[repr(C)]
    struct MemoryStatusEx {
        dw_length: u32,
        dw_memory_load: u32,
        ull_total_phys: u64,
        ull_avail_phys: u64,
        ull_total_page_file: u64,
        ull_avail_page_file: u64,
        ull_total_virtual: u64,
        ull_avail_virtual: u64,
        ull_avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        dw_length: std::mem::size_of::<MemoryStatusEx>() as u32,
        dw_memory_load: 0,
        ull_total_phys: 0,
        ull_avail_phys: 0,
        ull_total_page_file: 0,
        ull_avail_page_file: 0,
        ull_total_virtual: 0,
        ull_avail_virtual: 0,
        ull_avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok != 0 && status.ull_total_phys > 0 {
        (status.ull_total_phys / 1_048_576) as i64
    } else {
        8192
    }
}
