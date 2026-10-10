//! Core-owned model catalog scan (the computation behind `models.scan`).
//!
//! The Swift `ModelManager.scan` used to orchestrate the installed-model pass:
//! discover runnable models in the HF cache, fold each model's cached memory
//! profile, then garbage-collect partial download dirs whose target is now
//! installed. This module moves that orchestration into the core, so the cache
//! layout is understood in exactly one place. The Swift host still rehydrates
//! *paused* placeholder models (that path needs Swift's `Codable` `Model`), but
//! the discovered/installed catalog is produced here.

use std::collections::HashSet;

use crate::engine::mem_profile;
use crate::hf::{HFCache, SideloadedEntry};

/// A discovered installed model with its cached memory profile folded in.
#[derive(Debug, Clone, PartialEq)]
pub struct ScannedModel {
    pub entry: SideloadedEntry,
    /// Cached `MemProfile::ctx_bytes_per_1k_tokens`, or 0 when none is cached.
    pub ctx_bytes_per_1k_tokens: i64,
    /// Cached `MemProfile::resident_bytes`, or 0 when none is cached.
    pub resident_bytes: i64,
}

/// The result of [`scan`]: the installed catalog plus the models still needing
/// a memory-profile probe (id + absolute model-file path).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CatalogScan {
    pub models: Vec<ScannedModel>,
    pub needs_profile: Vec<(String, String)>,
}

/// Scans `cache_dir` for installed models, folds each one's cached memory profile
/// from `mem_cache_dir`, and cleans up partial download dirs whose target is now
/// installed. Mirrors the former Swift `ModelManager.scan` discovery pass.
pub fn scan(cache_dir: &str, mem_cache_dir: &str) -> CatalogScan {
    let discovered = HFCache::scan_for_sideloaded(cache_dir);

    let mut models: Vec<ScannedModel> = Vec::with_capacity(discovered.len());
    let mut needs_profile: Vec<(String, String)> = Vec::new();

    for entry in discovered {
        let cached = mem_profile::cache_get(mem_cache_dir, &entry.id);
        let (ctx_bytes_per_1k_tokens, resident_bytes) = match &cached {
            Some(p) => (p.ctx_bytes_per_1k_tokens, p.resident_bytes),
            None => (0, 0),
        };
        if cached.is_none() {
            needs_profile.push((entry.id.clone(), entry.paths.model_file.clone()));
        }
        models.push(ScannedModel {
            entry,
            ctx_bytes_per_1k_tokens,
            resident_bytes,
        });
    }

    let installed: HashSet<String> = models.iter().map(|m| m.entry.id.clone()).collect();
    HFCache::clean_installed_partials(cache_dir, &installed);

    CatalogScan {
        models,
        needs_profile,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixtures_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../Fixtures")
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let n = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("llama-scan-{tag}-{}-{n}", std::process::id()));
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
    fn scan_folds_cached_profile_and_reports_the_rest() {
        // `scan` cleans installed partial dirs, so run against a copy of the
        // read-only fixture tree.
        let cache = temp_dir("cache");
        copy_tree(&fixtures_dir().join("hf_cache").join("cache"), &cache);
        let cache = cache.to_str().unwrap();

        let mem_dir = temp_dir("mem");
        let seeded_id = "unsloth/Tiny-GGUF:Q4_K_M";
        mem_profile::cache_set(
            mem_dir.to_str().unwrap(),
            seeded_id,
            &mem_profile::MemProfile::new(4096, 2048),
        )
        .unwrap();

        let out = scan(cache, mem_dir.to_str().unwrap());

        assert!(!out.models.is_empty(), "some models discovered");

        let seeded = out
            .models
            .iter()
            .find(|m| m.entry.id == seeded_id)
            .expect("seeded model discovered");
        assert_eq!(seeded.ctx_bytes_per_1k_tokens, 4096);
        assert_eq!(seeded.resident_bytes, 2048);

        for m in &out.models {
            if m.entry.id == seeded_id {
                assert_eq!(m.ctx_bytes_per_1k_tokens, 4096);
                assert_eq!(m.resident_bytes, 2048);
                assert!(!out.needs_profile.iter().any(|(id, _)| id == seeded_id));
            } else {
                assert_eq!(m.ctx_bytes_per_1k_tokens, 0);
                assert_eq!(m.resident_bytes, 0);
                assert!(out.needs_profile.iter().any(|(id, _)| id == &m.entry.id));
            }
        }

        let _ = std::fs::remove_dir_all(cache);
        let _ = std::fs::remove_dir_all(&mem_dir);
    }
}
