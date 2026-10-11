//! Port of `Llama/HF/HFCache.swift` (the filesystem half; the HEAD-request
//! metadata fetch stays in Swift with the network layer).
//!
//! Utility for working with the standard HuggingFace Hub cache layout:
//! ```text
//! <cacheDir>/
//! └── models--{org}--{repo}/
//!     ├── blobs/{sha256}            # actual file, named by content hash
//!     ├── refs/main                 # text file containing commit hash
//!     └── snapshots/{commit}/
//!         └── filename.gguf -> ../../blobs/{sha256}
//! ```

use crate::hf::{GGUFMetadata, GGUFQuant, HFRepoParser, SidecarPicker};
use crate::models::{model, DraftHead, ResolvedPaths};
use crate::pathutil::last_path_component;
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

/// Name of the hidden dir under the HF cache that holds in-progress `.partial`
/// files.
pub const PARTIAL_ROOT_DIR_NAME: &str = ".llama-partial";

/// File name of the per-download placeholder dropped next to a model's
/// `.partial` files.
pub const PLACEHOLDER_FILE_NAME: &str = ".model.json";

/// A model discovered on disk by [`HFCache::scan_for_sideloaded`]. This mirrors
/// the subset of Swift's `Model` those scan paths actually compute; the Swift
/// host reconstructs its own `Model` from these fields.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideloadedEntry {
    pub id: String,
    pub ctx_window: i64,
    pub file_size: i64,
    pub scanned_draft_head: Option<DraftHead>,
    pub is_decision_model: bool,
    pub paths: ResolvedPaths,
}

/// Namespace for the HF cache helpers (mirrors the Swift `enum HFCache`).
pub struct HFCache;

// MARK: - Path Helpers (pure, no I/O)

/// The path components of a URL, mirroring Foundation's `url.pathComponents`
/// for the shapes we care about: the leading empty component (from the root
/// slash) is kept, so `components[3] == "resolve"` and `components[1..3]` are
/// org/repo, exactly as the Swift code indexes them.
fn url_path_components(url: &str) -> Vec<String> {
    let after_scheme = match url.find("://") {
        Some(i) => &url[i + 3..],
        None => url,
    };
    let path = match after_scheme.find('/') {
        Some(j) => &after_scheme[j..],
        None => "",
    };
    let path = path.split(['?', '#']).next().unwrap_or(path);
    path.split('/').map(str::to_string).collect()
}

impl HFCache {
    /// Parses a HF download URL and returns the repo directory name.
    /// e.g. `https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/file.gguf`
    /// → `"models--unsloth--Qwen3.5-2B-GGUF"`.
    pub fn repo_dir_name_from_url(url: &str) -> Option<String> {
        let c = url_path_components(url);
        if c.len() < 4 || c[3] != "resolve" {
            return None;
        }
        Some(format!("models--{}--{}", c[1], c[2]))
    }

    /// The repo-relative path a HF download URL points at (everything past
    /// `resolve/{branch}/`), preserving subdir structure.
    pub fn repo_relative_path_from_url(url: &str) -> Option<String> {
        let c = url_path_components(url);
        if c.len() < 6 || c[3] != "resolve" {
            return None;
        }
        Some(c[5..].join("/"))
    }

    /// Directory that holds in-progress `.partial` files for a model. Lives under
    /// the HF cache so promoting `.partial` → `blobs/<hash>` stays on the same
    /// filesystem (`moveItem` is atomic).
    pub fn partial_dir(cache_dir: &str, model_id: &str) -> PathBuf {
        Path::new(cache_dir)
            .join(PARTIAL_ROOT_DIR_NAME)
            .join(model_id)
    }

    /// Path to an in-progress `.partial` file for a single remote file.
    pub fn partial_path(cache_dir: &str, model_id: &str, filename: &str) -> PathBuf {
        Self::partial_dir(cache_dir, model_id).join(format!("{filename}.partial"))
    }

    /// Path to a download's placeholder file (the serialized `Model`).
    pub fn placeholder_url(cache_dir: &str, model_id: &str) -> PathBuf {
        Self::partial_dir(cache_dir, model_id).join(PLACEHOLDER_FILE_NAME)
    }

    // MARK: - Partials / placeholders

    /// Removes the model's partial directory (and any files under it). Silently
    /// ignores missing dirs -- used by cancel/delete/cleanup paths.
    pub fn remove_partials(cache_dir: &str, model_id: &str) {
        let dir = Self::partial_dir(cache_dir, model_id);
        let _ = fs::remove_dir_all(&dir);

        // A model id has a slash (`org/repo:quant`), so its dir sits inside an
        // org dir that would otherwise be left behind empty. Remove that one
        // level, but only if it's empty -- POSIX `rmdir` refuses a non-empty dir.
        let root = Path::new(cache_dir).join(PARTIAL_ROOT_DIR_NAME);
        if let Some(org_dir) = dir.parent() {
            if org_dir != root {
                let _ = fs::remove_dir(org_dir);
            }
        }
    }

    /// Sum of `.partial` file sizes in a single model's staging dir. Returns 0 if
    /// the dir doesn't exist or contains no `.partial` files. Walks recursively,
    /// skipping hidden files.
    pub fn partial_bytes(cache_dir: &str, model_id: &str) -> i64 {
        let dir = Self::partial_dir(cache_dir, model_id);
        let mut total: i64 = 0;
        walk_files(&dir, true, &mut |path| {
            if path.extension().and_then(|e| e.to_str()) == Some("partial") {
                if let Ok(meta) = fs::symlink_metadata(path) {
                    if meta.is_file() {
                        total += meta.len() as i64;
                    }
                }
            }
        });
        total
    }

    /// Finds every download placeholder under the partial root -- one per
    /// interrupted/paused download from a previous session. Returned order is
    /// filesystem-dependent; callers that need determinism should sort.
    pub fn scan_placeholders(cache_dir: &str) -> Vec<PathBuf> {
        let root = Path::new(cache_dir).join(PARTIAL_ROOT_DIR_NAME);
        let mut result = Vec::new();
        walk_files(&root, false, &mut |path| {
            if path.file_name().and_then(|n| n.to_str()) == Some(PLACEHOLDER_FILE_NAME) {
                result.push(path.to_path_buf());
            }
        });
        result
    }

    /// Cleans up `<partialRootDirName>/<id>` subdirs for ids that are already
    /// installed (e.g. a deeplink download that landed but left its partial dir
    /// behind).
    pub fn clean_installed_partials(cache_dir: &str, installed_ids: &HashSet<String>) {
        for model_id in installed_ids {
            Self::remove_partials(cache_dir, model_id);
        }
        clean_empty_dirs(&Path::new(cache_dir).join(PARTIAL_ROOT_DIR_NAME));
    }

    /// Returns true if `filename` (repo-relative) exists in any snapshot commit
    /// of the repo -- i.e. the file is already installed in the cache.
    pub fn snapshot_file_exists(cache_dir: &str, repo_dir: &str, filename: &str) -> bool {
        let snapshots_dir = Path::new(cache_dir).join(repo_dir).join("snapshots");
        let Ok(commits) = fs::read_dir(&snapshots_dir) else {
            return false;
        };
        for commit in commits.flatten() {
            if commit.path().join(filename).exists() {
                return true;
            }
        }
        false
    }

    // MARK: - File Operations

    /// Writes a downloaded file into the HF cache layout.
    ///
    /// 1. Creates directory structure (`blobs/`, `snapshots/{commit}/`, `refs/`)
    /// 2. Moves temp file → `blobs/{sha256}`
    /// 3. Creates symlink `snapshots/{commit}/{filename}` → `../../blobs/{sha256}`
    /// 4. Writes `refs/main` with the commit hash
    pub fn write_blob_and_link(
        cache_dir: &str,
        repo_dir: &str,
        commit: &str,
        blob_hash: &str,
        filename: &str,
        temp_file: &str,
    ) -> io::Result<()> {
        let repo_base = Path::new(cache_dir).join(repo_dir);
        let blobs_dir = repo_base.join("blobs");
        let snapshot_dir = repo_base.join("snapshots").join(commit);
        let refs_dir = repo_base.join("refs");

        fs::create_dir_all(&blobs_dir)?;
        fs::create_dir_all(&snapshot_dir)?;
        fs::create_dir_all(&refs_dir)?;

        // Move temp file to blob (atomic within same filesystem). If the blob
        // already exists (identical content from a concurrent download), just
        // clean up temp.
        let blob_dest = blobs_dir.join(blob_hash);
        if blob_dest.exists() {
            let _ = fs::remove_file(temp_file);
        } else {
            fs::rename(temp_file, &blob_dest)?;
        }

        // Create symlink: snapshots/{commit}/{filename} → ../../blobs/{sha256}
        // `filename` is repo-relative and may contain a subdir; the relative
        // target needs one extra `..` per extra path component.
        let symlink_path = snapshot_dir.join(filename);
        if let Some(parent) = symlink_path.parent() {
            fs::create_dir_all(parent)?;
        }
        if symlink_path.exists() || fs::read_link(&symlink_path).is_ok() {
            let _ = fs::remove_file(&symlink_path);
        }
        let depth = filename.split('/').count().saturating_sub(1);
        let relative_target = format!("{}blobs/{}", "../".repeat(2 + depth), blob_hash);
        std::os::unix::fs::symlink(relative_target, &symlink_path)?;

        // Write refs/main with the commit hash.
        fs::write(refs_dir.join("main"), commit)?;
        Ok(())
    }

    // MARK: - Deletion

    /// Deletes a model's files from the HF cache. Removes blobs (resolved from
    /// symlinks), snapshot symlinks, and cleans up empty dirs.
    pub fn delete_model_files(
        cache_dir: &str,
        repo_dir: &str,
        paths: &ResolvedPaths,
    ) -> io::Result<()> {
        let mut blobs_to_delete: HashSet<String> = HashSet::new();
        let mut symlinks_to_delete: Vec<String> = Vec::new();

        for path in paths.all_paths() {
            if let Ok(dest) = fs::read_link(&path) {
                // dest is relative like "../../blobs/{hash}"; resolve lexically.
                let symlink_dir = Path::new(&path).parent().unwrap_or(Path::new(""));
                let blob_absolute = lexical_normalize(&symlink_dir.join(dest));
                blobs_to_delete.insert(blob_absolute.to_string_lossy().into_owned());
                symlinks_to_delete.push(path);
            } else if Path::new(&path).exists() {
                // Direct file (not a symlink) -- just delete it.
                fs::remove_file(&path)?;
            }
        }

        for symlink in &symlinks_to_delete {
            let _ = fs::remove_file(symlink);
        }
        for blob in &blobs_to_delete {
            if Path::new(blob).exists() {
                let _ = fs::remove_file(blob);
            }
        }

        // Delete the refs/ directory (contains commit hash reference) so the
        // parent repo dir can be cleaned up when empty.
        let refs_dir = Path::new(cache_dir).join(repo_dir).join("refs");
        if refs_dir.exists() {
            let _ = fs::remove_dir_all(&refs_dir);
        }

        clean_empty_dirs(&Path::new(cache_dir).join(repo_dir));
        Ok(())
    }

    // MARK: - Discovery

    /// Scans the HF cache for GGUF files and builds a [`SideloadedEntry`] per
    /// discovered file (or shard group). Returned order is filesystem-dependent;
    /// callers that need determinism should sort by id.
    pub fn scan_for_sideloaded(cache_dir: &str) -> Vec<SideloadedEntry> {
        let mut results: Vec<SideloadedEntry> = Vec::new();

        let Ok(repo_dirs) = fs::read_dir(cache_dir) else {
            return results;
        };

        for repo_entry in repo_dirs.flatten() {
            let repo_dir = match repo_entry.file_name().into_string() {
                Ok(s) => s,
                Err(_) => continue,
            };
            if !repo_dir.starts_with("models--") {
                continue;
            }

            let snapshots_dir = Path::new(cache_dir).join(&repo_dir).join("snapshots");
            let Ok(commits) = fs::read_dir(&snapshots_dir) else {
                continue;
            };

            let mut seen_ids: HashSet<String> = HashSet::new();

            for commit in commits.flatten() {
                let snapshot_dir = commit.path();
                // `all_files` keeps the sidecars the runnable list drops -- the
                // sidecar pickers rank across the whole snapshot.
                let all_files = all_gguf_files(&snapshot_dir);
                let gguf_files: Vec<String> = all_files
                    .iter()
                    .filter(|f| is_runnable_model(f))
                    .cloned()
                    .collect();
                if gguf_files.is_empty() {
                    continue;
                }

                // Group split shards: "model-00001-of-00003.gguf" etc.
                let mut shard_groups: HashMap<String, Vec<String>> = HashMap::new();
                let mut standalone_files: Vec<String> = Vec::new();
                for filename in gguf_files {
                    if HFRepoParser::is_split_shard(&filename) {
                        if let Some(base) = HFRepoParser::split_shard_base_name(&filename) {
                            shard_groups.entry(base).or_default().push(filename);
                        }
                    } else {
                        standalone_files.push(filename);
                    }
                }

                for filename in standalone_files {
                    if let Some(result) = build_sideloaded_entry(
                        &repo_dir,
                        &filename,
                        None,
                        &snapshot_dir,
                        &all_files,
                    ) {
                        if seen_ids.insert(result.id.clone()) {
                            results.push(result);
                        }
                    }
                }

                for shard_filenames in shard_groups.values() {
                    let mut sorted = shard_filenames.clone();
                    sorted.sort();
                    // Only include complete groups: first shard present AND every
                    // shard of the declared `-of-NNNNN` count on disk.
                    let Some(first_shard) = sorted.first().cloned() else {
                        continue;
                    };
                    if !HFRepoParser::is_first_shard(&first_shard)
                        || Some(sorted.len() as i64) != HFRepoParser::shard_total(&first_shard)
                    {
                        continue;
                    }
                    if let Some(result) = build_sideloaded_entry(
                        &repo_dir,
                        &first_shard,
                        Some(&sorted),
                        &snapshot_dir,
                        &all_files,
                    ) {
                        if seen_ids.insert(result.id.clone()) {
                            results.push(result);
                        }
                    }
                }
            }
        }

        results
    }
}

// MARK: - Helpers

/// Recursively walks `dir`, invoking `f` for every file. When `skip_hidden` is
/// true, entries (and their subtrees) whose name starts with `.` are skipped,
/// mirroring `FileManager.enumerator`'s `.skipsHiddenFiles`.
fn walk_files(dir: &Path, skip_hidden: bool, f: &mut dyn FnMut(&Path)) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        if skip_hidden && name.to_string_lossy().starts_with('.') {
            continue;
        }
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => walk_files(&path, skip_hidden, f),
            Ok(ft) if ft.is_file() || ft.is_symlink() => f(&path),
            _ => {}
        }
    }
}

/// Resolves `.` and `..` components lexically, without touching the filesystem
/// (Foundation's `URL.standardized`).
fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in path.components() {
        match comp {
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Recursively removes empty directories under `url`, and `url` itself if it
/// becomes empty.
fn clean_empty_dirs(url: &Path) {
    let Ok(entries) = fs::read_dir(url) else {
        return;
    };
    for entry in entries.flatten() {
        let item_url = entry.path();
        if entry.path().is_dir() {
            clean_empty_dirs(&item_url);
        }
    }
    if let Ok(mut remaining) = fs::read_dir(url) {
        if remaining.next().is_none() {
            let _ = fs::remove_dir(url);
        }
    }
}

/// Collects every GGUF in a snapshot dir and one level of subdirs, sidecars
/// included. Returned paths are relative to `snapshot_dir`.
fn all_gguf_files(snapshot_dir: &Path) -> Vec<String> {
    let Ok(top_files) = fs::read_dir(snapshot_dir) else {
        return Vec::new();
    };

    let mut all_files: Vec<String> = Vec::new();
    for entry in top_files.flatten() {
        let item = match entry.file_name().into_string() {
            Ok(s) => s,
            Err(_) => continue,
        };
        let item_path = entry.path();
        if item_path.is_dir() {
            if let Ok(sub_files) = fs::read_dir(&item_path) {
                for sub in sub_files.flatten() {
                    if let Ok(sub_name) = sub.file_name().into_string() {
                        all_files.push(format!("{item}/{sub_name}"));
                    }
                }
            }
        } else {
            all_files.push(item);
        }
    }

    all_files.retain(|f| f.to_lowercase().ends_with(".gguf"));
    all_files
}

/// True for a GGUF that can stand on its own as a model entry.
fn is_runnable_model(relative_path: &str) -> bool {
    !SidecarPicker::is_mmproj(relative_path) && !SidecarPicker::is_draft_head(relative_path)
}

fn file_has_mtp_head(file_base_name: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"(?i)(^|[-_.])mtp([-_.]|$)").unwrap());
    re.is_match(file_base_name)
}

/// Resolves symlinks (like `URL.resolvingSymlinksInPath`), falling back to the
/// original path when it can't be resolved, then reads the file size via
/// `lstat` (matching `FileManager.fileSize(atPath:)`, which does not follow
/// symlinks -- hence the explicit resolution first).
fn resolved_file_size(path: &str) -> i64 {
    let resolved = fs::canonicalize(path).unwrap_or_else(|_| PathBuf::from(path));
    fs::symlink_metadata(&resolved)
        .map(|m| m.len() as i64)
        .unwrap_or(0)
}

fn build_sideloaded_entry(
    repo_dir: &str,
    filename: &str,
    shard_files: Option<&[String]>,
    snapshot_dir: &Path,
    siblings: &[String],
) -> Option<SideloadedEntry> {
    let (org, repo) = HFRepoParser::parse(repo_dir)?;

    let quant = GGUFQuant::tag_for_path(filename, None);

    let mmproj_file = SidecarPicker::mmproj(siblings, filename)
        .map(|f| snapshot_dir.join(&f).to_string_lossy().into_owned());

    let mut file_paths: Vec<String> = match shard_files {
        Some(shards) => shards
            .iter()
            .map(|f| snapshot_dir.join(f).to_string_lossy().into_owned())
            .collect(),
        None => vec![snapshot_dir.join(filename).to_string_lossy().into_owned()],
    };
    if let Some(mmproj) = &mmproj_file {
        file_paths.push(mmproj.clone());
    }

    let total_file_size: i64 = file_paths.iter().map(|p| resolved_file_size(p)).sum();

    let model_id = model::make_id(&org, &repo, &quant);

    let main_file_path = snapshot_dir.join(filename);
    let main_file_path = main_file_path.to_string_lossy().into_owned();

    let ctx_window: i64 = match GGUFMetadata::context_length(&main_file_path) {
        Some(Some(length)) => length,
        Some(None) => return None, // header parsed, key absent -- not an LLM
        None => 131_072,           // header unparseable -- fall back
    };

    let draft_sidecar = SidecarPicker::draft_head(siblings, filename, Some(&quant))
        .map(|f| snapshot_dir.join(&f).to_string_lossy().into_owned());

    let has_embedded_head = GGUFMetadata::has_embedded_mtp_head(&main_file_path)
        .unwrap_or_else(|| file_has_mtp_head(last_path_component(filename)));

    let is_decision_model = GGUFMetadata::is_decision_model(&main_file_path).unwrap_or(false);

    let scanned_draft_head = draft_sidecar
        .as_deref()
        .and_then(DraftHead::from_sidecar_path)
        .or(if has_embedded_head {
            Some(DraftHead::Mtp)
        } else {
            None
        });

    let additional_parts: Vec<String> = match shard_files {
        Some(shards) if shards.len() > 1 => shards
            .iter()
            .skip(1)
            .map(|f| snapshot_dir.join(f).to_string_lossy().into_owned())
            .collect(),
        _ => Vec::new(),
    };

    let paths = ResolvedPaths {
        model_file: main_file_path,
        additional_parts,
        mmproj_file,
        uses_mtp: draft_sidecar.is_none() && has_embedded_head,
        draft_sidecar_file: draft_sidecar,
        hf_repo_dir_name: repo_dir.to_string(),
    };

    Some(SideloadedEntry {
        id: model_id,
        ctx_window,
        file_size: total_file_size,
        scanned_draft_head,
        is_decision_model,
        paths,
    })
}
