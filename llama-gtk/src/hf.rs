use std::io::Read;
use std::path::Path;
use std::time::Duration;

use llama_core::config::preset::{ContextLengthPolicy, ModelPreset, PerformanceConfig};
use llama_core::engine::download::extract_full_size;
use llama_core::engine::mem_profile::{self, MemProfile};
use llama_core::engine::partial_writer::PartialWriter;
use llama_core::hf::HFCache;
use llama_core::models::compat::ModelMem;
use llama_core::models::scan::{self, ScannedModel};
use llama_core::models::ResolvedPaths;
use llama_core::net::client::{HttpClient, ReqwestClient};
use llama_core::net::hf_api::{fetch_file_metadata, fetch_siblings};
use llama_core::net::hf_repo_resolver::{resolve_from_siblings, Resolved};
use llama_core::net::hf_request;
use llama_core::pathutil::last_path_component;

use crate::paths;
use crate::settings::Settings;

pub const USER_AGENT: &str = "llama-app/0.1 (llama-gtk)";

#[derive(Debug, Clone)]
pub struct UiModel {
    pub id: String,
    pub display_name: String,
    pub model_path: String,
    pub paths: ResolvedPaths,
    pub ctx_window: i64,
    pub file_size: i64,
    pub supported_tiers: Vec<i64>,
    pub effective_tier: Option<i64>,
    pub selected_tier: Option<i64>,
    pub is_decision_model: bool,
}

pub struct ScanResult {
    pub models: Vec<UiModel>,
    pub profiled: usize,
}

pub fn normalize_repo(input: &str) -> Result<String, String> {
    let trimmed = input.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err("Enter a Hugging Face repo, for example `unsloth/Qwen3.5-2B-GGUF`.".into());
    }
    let without_host = trimmed
        .strip_prefix("https://huggingface.co/")
        .or_else(|| trimmed.strip_prefix("http://huggingface.co/"))
        .unwrap_or(trimmed)
        .trim_start_matches('/');
    let mut parts = without_host.split('/');
    let org = parts.next().unwrap_or("");
    let repo = parts.next().unwrap_or("");
    if org.is_empty() || repo.is_empty() || parts.next().is_some() {
        return Err("Use the form `org/repo`, not a file URL.".into());
    }
    Ok(format!("{org}/{repo}"))
}

pub fn resolve(
    repo: &str,
    quant: Option<&str>,
    budget_mb: f64,
    token: Option<&str>,
) -> Result<Resolved, String> {
    let client = ReqwestClient::new(USER_AGENT.to_string()).map_err(|e| e.to_string())?;
    let siblings = fetch_siblings(&client, repo, token).map_err(|e| e.message())?;
    resolve_from_siblings(repo, quant, &siblings, budget_mb).map_err(|e| e.message())
}

pub fn download_resolved(
    resolved: &Resolved,
    settings: &Settings,
    progress: &(dyn Fn(u64, u64, &str) + Send + Sync),
) -> Result<String, String> {
    let cache = settings.hf_cache_path();
    let cache = cache.to_string_lossy().into_owned();
    std::fs::create_dir_all(&cache).map_err(|e| e.to_string())?;

    let mut urls = vec![resolved.main_url.clone()];
    urls.extend(resolved.additional_parts.iter().cloned());
    if let Some(m) = &resolved.mmproj_url {
        urls.push(m.clone());
    }
    if let Some(d) = &resolved.draft_url {
        urls.push(d.clone());
    }

    let token = settings.token();
    let head_client = ReqwestClient::new(USER_AGENT.to_string()).map_err(|e| e.to_string())?;
    let metadata = fetch_file_metadata(&head_client, &urls, token);
    let commit = metadata
        .values()
        .find_map(|m| m.commit_hash.clone())
        .unwrap_or_else(|| "main".to_string());

    let repo_dir = repo_dir_for(&resolved.model_id);

    let mut files = Vec::with_capacity(urls.len());
    for url in &urls {
        let size = head_client
            .head(url, token)
            .ok()
            .map(|r| extract_full_size(r.status as i64, &r.headers).max(0) as u64)
            .unwrap_or(0);
        files.push((url.clone(), size));
    }
    let total: u64 = files.iter().map(|(_, size)| *size).sum();
    let mut base: u64 = 0;
    for (url, size) in &files {
        let filename = HFCache::repo_relative_path_from_url(url)
            .or_else(|| Some(last_path_component(url).to_string()))
            .unwrap_or_else(|| "model.gguf".to_string());
        let already = HFCache::snapshot_file_exists(&cache, &repo_dir, &filename);
        if !already {
            let meta = metadata.get(url);
            let expected_hash = meta.and_then(|m| m.blob_hash.clone());
            progress(
                base,
                total,
                &format!("Downloading {}", last_path_component(&filename)),
            );
            let job = DownloadJob {
                url,
                cache: &cache,
                model_id: &resolved.model_id,
                filename: &filename,
                expected_hash: expected_hash.as_deref(),
                commit: &commit,
                token,
                base,
                total,
            };
            download_one(&job, progress)?;
        }
        base += size;
    }
    HFCache::remove_partials(&cache, &resolved.model_id);

    let scan = HFCache::scan_for_sideloaded(&cache);
    let model = scan
        .into_iter()
        .find(|m| m.id == resolved.model_id)
        .ok_or_else(|| {
            "Download finished, but the model was not found in the cache.".to_string()
        })?;
    Ok(model.paths.model_file)
}

struct DownloadJob<'a> {
    url: &'a str,
    cache: &'a str,
    model_id: &'a str,
    filename: &'a str,
    expected_hash: Option<&'a str>,
    commit: &'a str,
    token: Option<&'a str>,
    base: u64,
    total: u64,
}

fn download_one(
    job: &DownloadJob<'_>,
    progress: &(dyn Fn(u64, u64, &str) + Send + Sync),
) -> Result<(), String> {
    let DownloadJob {
        url,
        cache,
        model_id,
        filename,
        expected_hash,
        commit,
        token,
        base,
        total,
    } = *job;
    let partial_path = HFCache::partial_path(cache, model_id, filename);
    let partial_path = partial_path.to_string_lossy().into_owned();
    let mut writer = PartialWriter::open(&partial_path, expected_hash.map(str::to_string))
        .map_err(|e| e.message())?;
    writer.set_total_expected(total as i64);

    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(60 * 60 * 24))
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| e.to_string())?;
    let mut request = client.get(url);
    for (k, v) in hf_request::make(url, token, USER_AGENT) {
        request = request.header(k, v);
    }
    if writer.bytes_written() > 0 {
        request = request.header("Range", format!("bytes={}-", writer.bytes_written()));
    }
    let mut response = request.send().map_err(|e| e.to_string())?;
    if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
        let _ = std::fs::remove_file(&partial_path);
        return Err("The partial download was inconsistent and has been reset. Try again.".into());
    }
    if !response.status().is_success() {
        return Err(format!(
            "Download failed with HTTP {}: {}",
            response.status(),
            llama_core::engine::download::user_message(response.status().as_u16() as i64)
        ));
    }
    if response.status() == reqwest::StatusCode::OK && writer.bytes_written() > 0 {
        writer.restart().map_err(|e| e.message())?;
    }

    let mut downloaded = writer.bytes_written() as u64;
    progress(
        base + downloaded,
        total,
        &format!("Downloading {}", last_path_component(filename)),
    );
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = response.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        writer.append(&buf[..n]).map_err(|e| e.message())?;
        downloaded += n as u64;
        progress(
            base + downloaded,
            total,
            &format!("Downloading {}", last_path_component(filename)),
        );
    }

    if let Some(expected) = expected_hash {
        let computed = writer.digest_hex();
        if !computed.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(&partial_path);
            return Err(format!(
                "{} failed its integrity check (SHA256 mismatch).",
                last_path_component(filename)
            ));
        }
    }
    let blob_hash = expected_hash
        .map(str::to_string)
        .unwrap_or_else(|| writer.digest_hex());
    drop(writer);
    HFCache::write_blob_and_link(
        cache,
        &repo_dir_for(model_id),
        commit,
        &blob_hash,
        filename,
        &partial_path,
    )
    .map_err(|e| {
        format!(
            "Couldn't write {} into the cache: {e}",
            last_path_component(filename)
        )
    })
}

fn repo_dir_for(model_id: &str) -> String {
    let repo = model_id.split(':').next().unwrap_or(model_id);
    format!("models--{}", repo.replace('/', "--"))
}

pub fn scan_models(settings: &Settings, engine_path: Option<&str>, profile: bool) -> ScanResult {
    let cache = settings.hf_cache_path();
    let cache = cache.to_string_lossy().into_owned();
    let mem_dir = paths::mem_profile_dir();
    let mem_dir = mem_dir.to_string_lossy().into_owned();
    let mut scanned = scan::scan(&cache, &mem_dir);
    let mut profiled = 0;

    if profile {
        if let Some(engine) = engine_path {
            for (id, path) in &scanned.needs_profile {
                if let Some(profile) = probe_profile(engine, path) {
                    if mem_profile::cache_set(&mem_dir, id, &profile).is_ok() {
                        profiled += 1;
                    }
                }
            }
            if profiled > 0 {
                scanned = scan::scan(&cache, &mem_dir);
            }
        }
    }

    let physical = paths::physical_memory_mb() as f64;
    let budget = (physical - 4096.0).max(0.0);
    let policy = settings.context_policy();
    let global = settings.performance.to_core();

    let mut models: Vec<UiModel> = scanned
        .models
        .into_iter()
        .map(|m| ui_model(m, settings, budget, policy, &global))
        .collect();
    models.sort_by_key(|a| a.display_name.to_lowercase());
    ScanResult { models, profiled }
}

fn ui_model(
    m: ScannedModel,
    settings: &Settings,
    budget: f64,
    policy: ContextLengthPolicy,
    _global: &PerformanceConfig,
) -> UiModel {
    let mem = ModelMem {
        ctx_window: m.entry.ctx_window,
        file_size: m.entry.file_size,
        ctx_bytes_per_1k_tokens: m.ctx_bytes_per_1k_tokens,
        resident_bytes: m.resident_bytes,
    };
    let supported = mem.supported_context_tiers(budget);
    let selected = settings
        .selected_tiers
        .get(&m.entry.id)
        .copied()
        .and_then(llama_core::models::ContextTier::from_raw);
    let effective = mem.effective_ctx_tier(
        budget,
        selected,
        policy == ContextLengthPolicy::LargestFitting,
    );
    UiModel {
        id: m.entry.id.clone(),
        display_name: display_name(&m.entry.id),
        model_path: m.entry.paths.model_file.clone(),
        paths: m.entry.paths.clone(),
        ctx_window: m.entry.ctx_window,
        file_size: m.entry.file_size,
        supported_tiers: supported.into_iter().map(|t| t.raw_value()).collect(),
        effective_tier: effective.map(|t| t.raw_value()),
        selected_tier: selected.map(|t| t.raw_value()),
        is_decision_model: m.entry.is_decision_model,
    }
}

fn display_name(id: &str) -> String {
    let repo = id.split(':').next().unwrap_or(id);
    repo.rsplit('/').next().unwrap_or(repo).to_string()
}

pub fn write_models_ini(settings: &Settings, models: &[UiModel]) -> Result<bool, String> {
    let user_ini = paths::user_ini();
    if !user_ini.exists() {
        if let Some(parent) = user_ini.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        std::fs::write(&user_ini, llama_core::config::preset::user_ini_template())
            .map_err(|e| e.to_string())?;
    }
    let override_text = std::fs::read_to_string(&user_ini).unwrap_or_default();
    let overrides = llama_core::config::ini::parse(&override_text);
    let global = settings.performance.to_core();
    let presets: Vec<ModelPreset> = models
        .iter()
        .filter_map(|m| {
            let tier = m.effective_tier?;
            Some(ModelPreset {
                id: m.id.clone(),
                model_path: m.paths.model_file.clone(),
                ctx_size: tier,
                mmproj_path: m.paths.mmproj_file.clone(),
                draft_sidecar_kind: m.paths.draft_sidecar_kind(),
                draft_sidecar_path: m.paths.draft_sidecar_file.clone(),
                uses_mtp: m.paths.uses_mtp,
                is_decision_model: m.is_decision_model,
                performance_override: settings
                    .performance_overrides
                    .get(&m.id)
                    .map(|p| p.to_core()),
                aliases: vec![],
            })
        })
        .collect();
    let content = llama_core::config::preset::render_models_ini(
        &paths::display_path(&user_ini),
        &global,
        &presets,
        &overrides,
        false,
    );
    let path = paths::models_ini();
    let existing = std::fs::read_to_string(&path).unwrap_or_default();
    if existing == content {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&path, content).map_err(|e| e.to_string())?;
    Ok(true)
}

fn probe_profile(engine: &str, model_path: &str) -> Option<MemProfile> {
    let lo = probe_mib(engine, model_path, 4096)?;
    let hi = probe_mib(engine, model_path, 131072)?;
    mem_profile::fit(lo, hi, 4096, 131072)
}

fn probe_mib(engine: &str, model_path: &str, ctx: i64) -> Option<i64> {
    let out = std::process::Command::new(engine)
        .args([
            "fit-params",
            "-m",
            model_path,
            "-c",
            &ctx.to_string(),
            "-fitp",
            "on",
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    mem_profile::parse_fitp_output(&String::from_utf8_lossy(&out.stdout))
}

pub fn delete_model(cache_dir: &Path, model: &UiModel) -> Result<(), String> {
    let cache = cache_dir.to_string_lossy().into_owned();
    HFCache::delete_model_files(&cache, &model.paths.hf_repo_dir_name, &model.paths)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_repo_accepts_urls_and_rejects_file_urls() {
        assert_eq!(
            normalize_repo("https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/").unwrap(),
            "unsloth/Qwen3.5-2B-GGUF"
        );
        assert!(normalize_repo(
            "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/x.gguf"
        )
        .is_err());
        assert_eq!(normalize_repo("org/repo").unwrap(), "org/repo");
    }
}
