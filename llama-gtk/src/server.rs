use std::io::Write;
use std::process::{Child, Command, Stdio};

use llama_core::engine::binaries;
use llama_core::engine::server::{build_launch_spec, LaunchSpec, LaunchSpecInput};

use crate::engine;
use crate::paths;
use crate::settings::Settings;

pub struct ServerHandle {
    child: Child,
    pub port: i64,
}

impl ServerHandle {
    pub fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }

    pub fn try_wait(&mut self) -> Option<std::process::ExitStatus> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status),
            _ => None,
        }
    }
}

pub fn start(settings: &mut Settings, engine_path: &str) -> Result<ServerHandle, String> {
    paths::ensure_dirs();
    let version = binaries::read_version(engine_path);
    let api_key = if settings.bind_all {
        Some(settings.ensure_api_key())
    } else {
        None
    };
    let address_by_interface: [(String, String); 0] = [];
    let configured_bind = settings.bind_all.then_some("0.0.0.0");
    let presets = paths::models_ini();
    let log = paths::log_file();
    let empty = paths::empty_cache();
    let hf = settings.hf_cache_path();
    let extra_args = settings.extra_args();
    let fit = (paths::physical_memory_mb() - 4096).max(0);

    let input = LaunchSpecInput {
        llama_path: Some(engine_path),
        presets_path: &presets.to_string_lossy(),
        log_file_path: &log.to_string_lossy(),
        hf_cache_directory: &hf.to_string_lossy(),
        empty_cache_directory: &empty.to_string_lossy(),
        port: settings.server_port,
        fit_target_mb: fit,
        configured_bind_address: configured_bind,
        address_by_interface: &address_by_interface,
        primary_interface: None,
        current_version: version,
        configured_webui_dir: None,
        sleep_idle_seconds: settings.sleep_idle_seconds,
        agent_mode: settings.agent_mode,
        api_key: api_key.as_deref(),
        extra_args: &extra_args,
    };
    let spec: LaunchSpec =
        build_launch_spec(&input).ok_or_else(|| "No llama binary found.".to_string())?;

    if let Some(parent) = log.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::create_dir_all(&empty).map_err(|e| e.to_string())?;

    let mut command = Command::new(&spec.executable_path);
    command
        .args(&spec.arguments)
        .current_dir(paths::data_dir())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    if let Some(dir) = engine::library_dir(&spec.executable_path) {
        let existing = std::env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
        let mut value = dir.to_string_lossy().into_owned();
        if !existing.is_empty() {
            value.push(':');
            value.push_str(&existing.to_string_lossy());
        }
        command.env("LD_LIBRARY_PATH", value);
    }
    for key in std::env::vars()
        .map(|(k, _)| k)
        .filter(|k| k.starts_with("LLAMA_ARG_"))
    {
        command.env_remove(key);
    }
    for (key, value) in &spec.env {
        command.env(key, value);
    }

    let child = command
        .spawn()
        .map_err(|e| format!("Couldn't start llama-server: {e}"))?;

    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log)
        .map_err(|e| e.to_string())?;
    let _ = writeln!(
        log_file,
        "\n[llama-gtk] started {} on port {}\n{}",
        spec.executable_path,
        settings.server_port,
        spec.display_command()
    );

    Ok(ServerHandle {
        child,
        port: settings.server_port,
    })
}

pub fn open_web_ui(port: i64) {
    let url = format!("http://localhost:{port}");
    let _ = Command::new("xdg-open").arg(&url).spawn();
}

pub fn port_in_use(port: i64) -> bool {
    use std::net::TcpListener;
    TcpListener::bind(("127.0.0.1", port as u16)).is_err()
}
