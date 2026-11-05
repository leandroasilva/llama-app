use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use flate2::read::GzDecoder;
use llama_core::engine::binaries;
use llama_core::engine::installer;
use llama_core::engine::platform::{self, LinuxBackend, Target, TargetArch, TargetOs};
use llama_core::engine::version::LlamaVersion;
use sha2::{Digest, Sha256};

use crate::paths;

pub const USER_AGENT: &str = "llama-app/0.1 (Linux; llama-gtk)";

#[derive(Debug, Clone)]
pub struct Engine {
    pub path: String,
    pub origin: binaries::Origin,
    pub version: Option<LlamaVersion>,
}

pub fn resolve_engine() -> Option<Engine> {
    let home = paths::home_dir();
    let dirs = vec![
        home.join(".local/bin").to_string_lossy().into_owned(),
        "/usr/local/bin".into(),
        "/usr/bin".into(),
    ];
    let (path, origin) = binaries::resolve(&home.to_string_lossy(), &dirs, false)?;
    let version = binaries::read_version(&path);
    Some(Engine {
        path,
        origin,
        version,
    })
}

pub fn target_for_backend(backend: &str) -> Result<Target, String> {
    let backend = if backend == "vulkan" {
        LinuxBackend::Vulkan
    } else {
        LinuxBackend::Cpu
    };
    let arch = TargetArch::current().ok_or_else(|| {
        format!(
            "No prebuilt llama.cpp archive for {}",
            std::env::consts::ARCH
        )
    })?;
    platform::target_for(TargetOs::Linux, arch, backend)
        .ok_or_else(|| "No prebuilt llama.cpp archive for this platform.".to_string())
}

/// Downloads, verifies, unpacks, and atomically links the pinned engine.
///
/// The progress callback receives `(downloaded_bytes, total_bytes, label)`.
pub fn install(
    version: &str,
    backend: &str,
    progress: &(dyn Fn(u64, u64, &str) + Send + Sync),
) -> Result<String, String> {
    let target = target_for_backend(backend)?;
    let artifact = platform::artifact_for(version, target);
    let download_dir = paths::engine_download_dir();
    std::fs::create_dir_all(&download_dir).map_err(|e| e.to_string())?;
    let archive = download_dir.join(&artifact.file_name);

    let expected = artifact.sha256.clone();
    download_archive(&artifact.url, &archive, expected.as_deref(), progress)?;

    let home = paths::home_dir();
    let install_dir = home.join(".llama-app");
    let releases = install_dir.join("releases");
    let staging = install_dir.join(format!(".extract-{}", unique_suffix()));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

    let result = (|| -> Result<String, String> {
        extract(&archive, &staging)?;
        let bundle = staging.join(&artifact.bundle_name);
        let binary = bundle.join("llama");
        if !binaries::is_executable_file(&binary.to_string_lossy()) {
            return Err("The llama.cpp release archive is incomplete.".into());
        }
        let got = binaries::read_version(&binary.to_string_lossy());
        if got.map(|v| v.tag()) != Some(version.to_string()) {
            return Err(format!(
                "The release archive did not contain llama {version}."
            ));
        }

        std::fs::create_dir_all(&releases).map_err(|e| e.to_string())?;
        let destination = releases.join(format!("llama-{version}"));
        if destination.exists() {
            std::fs::remove_dir_all(&destination).map_err(|e| e.to_string())?;
        }
        std::fs::rename(&bundle, &destination)
            .map_err(|e| format!("Couldn't move the release into place: {e}"))?;

        let live = install_dir.join("llama");
        let release_binary = destination.join("llama");
        point_link(&live, &release_binary)?;
        if let installer::PathSymlinkOutcome::Failed(e) =
            installer::ensure_path_symlink(&home.to_string_lossy(), &live.to_string_lossy())
        {
            eprintln!("Couldn't create ~/.local/bin/llama: {e}");
        }
        Ok(release_binary.to_string_lossy().into_owned())
    })();

    let _ = std::fs::remove_dir_all(&staging);
    let _ = std::fs::remove_file(&archive);
    result
}

fn download_archive(
    url: &str,
    dest: &Path,
    expected_sha256: Option<&str>,
    progress: &(dyn Fn(u64, u64, &str) + Send + Sync),
) -> Result<(), String> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60 * 60))
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client.get(url).header("User-Agent", USER_AGENT);
    if let Some(token) = std::env::var_os("HF_TOKEN") {
        if llama_core::net::hf_request::is_huggingface_host(url) {
            request = request.header(
                "Authorization",
                format!("Bearer {}", token.to_string_lossy()),
            );
        }
    }
    let mut response = request.send().map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("{url} (HTTP {})", response.status()));
    }
    let total = response.content_length().unwrap_or(0);
    let tmp = dest.with_extension("download");
    let mut file = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 1 << 20];
    progress(0, total, "Downloading llama.cpp");
    loop {
        let n = response.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        hasher.update(&buf[..n]);
        downloaded += n as u64;
        progress(downloaded, total, "Downloading llama.cpp");
    }
    file.sync_all().map_err(|e| e.to_string())?;
    drop(file);

    if let Some(expected) = expected_sha256.filter(|s| !s.is_empty()) {
        let computed = format!("{:x}", hasher.finalize());
        if !computed.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(&tmp);
            return Err("llama.cpp archive failed its SHA256 check.".into());
        }
    }
    std::fs::rename(&tmp, dest).map_err(|e| e.to_string())
}

fn extract(archive: &Path, staging: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive).map_err(|e| e.to_string())?;
    let gz = GzDecoder::new(file);
    let mut tar = tar::Archive::new(gz);
    tar.unpack(staging)
        .map_err(|e| format!("Couldn't unpack the llama.cpp archive: {e}"))
}

fn point_link(link: &Path, binary: &Path) -> Result<(), String> {
    let parent = link.parent().ok_or("invalid install path")?;
    std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    let tmp = parent.join(format!(".llama-link-{}", unique_suffix()));
    let _ = std::fs::remove_file(&tmp);
    std::os::unix::fs::symlink(binary, &tmp).map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, link).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        e.to_string()
    })
}

pub fn library_dir(engine_path: &str) -> Option<PathBuf> {
    let canonical = std::fs::canonicalize(engine_path).ok()?;
    canonical.parent().map(Path::to_path_buf)
}

fn unique_suffix() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{}-{n}", std::process::id())
}
