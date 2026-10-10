//! Port of the selection logic in `Llama/Deeplinks/HFRepoResolver.swift`.
//!
//! The network lookup (`fetchSiblings`) lives in the transport layer; this
//! module takes the already-fetched sibling list and applies the same
//! quant/shard/mmproj/draft selection the Swift resolver does. The device
//! memory budget (`Model.deviceMemoryBudgetMb`, which reads `SystemMemory`) is
//! injected as `budget_mb`.

use crate::hf::{GGUFQuant, HFRepoParser, SidecarPicker};
use crate::models::{compat, model};

use percent_encoding::{utf8_percent_encode, AsciiSet, CONTROLS};
use serde::Deserialize;

/// Minimal sibling representation — `rfilename` is a repo-relative path
/// (e.g. `Q4_K_M/model.gguf`), `size` is bytes when the HF API supplies it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Sibling {
    pub rfilename: String,
    pub size: Option<i64>,
}

impl Sibling {
    pub fn new(rfilename: impl Into<String>, size: Option<i64>) -> Self {
        Self {
            rfilename: rfilename.into(),
            size,
        }
    }
}

/// A resolved deeplink target. Mirrors `HFRepoResolver.Resolved`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Resolved {
    pub model_id: String,
    pub main_url: String,
    pub additional_parts: Vec<String>,
    pub mmproj_url: Option<String>,
    pub draft_url: Option<String>,
    pub approximate_bytes: i64,
}

/// `HFRepoResolver.ResolveError` with the same user-facing surface text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    InvalidQuant(String),
    RepoNotFound(String),
    Gated(String),
    NetworkUnavailable(String),
    NoGgufFiles(String),
    QuantNotFound { repo: String, quant: String },
    ShardMissing { repo: String, shard: String },
    NoCompatibleFile(String),
}

impl ResolveError {
    pub fn kind(&self) -> &'static str {
        match self {
            Self::InvalidQuant(_) => "invalidQuant",
            Self::RepoNotFound(_) => "repoNotFound",
            Self::Gated(_) => "gated",
            Self::NetworkUnavailable(_) => "networkUnavailable",
            Self::NoGgufFiles(_) => "noGgufFiles",
            Self::QuantNotFound { .. } => "quantNotFound",
            Self::ShardMissing { .. } => "shardMissing",
            Self::NoCompatibleFile(_) => "noCompatibleFile",
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::InvalidQuant(q) => format!("“{q}” isn’t a known GGUF quantization label."),
            Self::RepoNotFound(repo) => format!("Hugging Face returned 404 for {repo}."),
            Self::Gated(repo) => format!("Hugging Face returned 401/403 for {repo} (gated)."),
            Self::NetworkUnavailable(repo) => {
                format!("Couldn’t reach Hugging Face for {repo}.")
            }
            Self::NoGgufFiles(repo) => format!("{repo} ships no runnable GGUF files."),
            Self::QuantNotFound { repo, quant } => {
                format!("{repo} has no {quant} build.")
            }
            Self::ShardMissing { repo, shard } => {
                format!("{repo} is missing the shard {shard}.")
            }
            Self::NoCompatibleFile(repo) => {
                format!("No build of {repo} fits this Mac’s memory budget.")
            }
        }
    }
}

/// The outcome of the main-file pick.
struct Pick {
    rfilename: String,
    tag: String,
}

/// Resolves a repo's siblings to a single main build plus its sidecars, with
/// the device memory budget injected. Mirrors `HFRepoResolver.resolve` minus the
/// HTTP fetch.
pub fn resolve_from_siblings(
    repo: &str,
    requested_quant: Option<&str>,
    siblings: &[Sibling],
    budget_mb: f64,
) -> Result<Resolved, ResolveError> {
    if let Some(quant) = requested_quant {
        if GGUFQuant::parse_label(quant).is_none() {
            return Err(ResolveError::InvalidQuant(quant.to_string()));
        }
    }

    let pick = select_main(repo, requested_quant, siblings, budget_mb)?;

    let shards = expand_shards(&pick.rfilename, siblings, repo)?;
    let mmproj = pick_mmproj(&pick.rfilename, siblings);
    let draft = pick_draft_head(&pick.rfilename, siblings, &pick.tag);

    let mut all_picked: Vec<String> = shards.clone();
    if let Some(m) = &mmproj {
        all_picked.push(m.rfilename.clone());
    }
    if let Some(d) = &draft {
        all_picked.push(d.rfilename.clone());
    }
    let approx_bytes: i64 = all_picked
        .iter()
        .map(|path| {
            siblings
                .iter()
                .find(|s| &s.rfilename == path)
                .and_then(|s| s.size)
                .unwrap_or(0)
        })
        .sum();

    let model_id = model::make_id_org_slash_repo(repo, &pick.tag);

    let main_url = resolve_url(repo, &pick.rfilename);
    let additional_parts = shards
        .iter()
        .skip(1)
        .map(|p| resolve_url(repo, p))
        .collect();
    let mmproj_url = mmproj.as_ref().map(|s| resolve_url(repo, &s.rfilename));
    let draft_url = draft.as_ref().map(|s| resolve_url(repo, &s.rfilename));

    Ok(Resolved {
        model_id,
        main_url,
        additional_parts,
        mmproj_url,
        draft_url,
        approximate_bytes: approx_bytes,
    })
}

fn is_gguf_candidate(path: &str) -> bool {
    path.to_lowercase().ends_with(".gguf")
}

fn last_component(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn select_main(
    repo: &str,
    requested_quant: Option<&str>,
    siblings: &[Sibling],
    budget_mb: f64,
) -> Result<Pick, ResolveError> {
    let all_ggufs: Vec<&Sibling> = siblings
        .iter()
        .filter(|sib| {
            if !is_gguf_candidate(&sib.rfilename)
                || SidecarPicker::is_draft_head(&sib.rfilename)
                || SidecarPicker::is_mmproj(&sib.rfilename)
            {
                return false;
            }
            let name = last_component(&sib.rfilename);
            if name.to_lowercase().contains("imatrix") {
                return false;
            }
            !HFRepoParser::is_split_shard(name) || HFRepoParser::is_first_shard(name)
        })
        .collect();
    if all_ggufs.is_empty() {
        return Err(ResolveError::NoGgufFiles(repo.to_string()));
    }

    // 1. Explicit quant.
    if let Some(requested) = requested_quant {
        let label = |sib: &Sibling| -> Option<String> {
            GGUFQuant::parse_label(&sib.rfilename)
                .or_else(|| HFRepoParser::parse_quant(last_component(&sib.rfilename)))
        };

        let mut matches: Vec<&Sibling> = all_ggufs
            .iter()
            .copied()
            .filter(|sib| match label(sib) {
                Some(label) => GGUFQuant::matches(&label, requested),
                None => false,
            })
            .collect();

        if matches.is_empty() {
            let requested_tag = GGUFQuant::canonical_tag(requested);
            matches = all_ggufs
                .iter()
                .copied()
                .filter(|sib| match label(sib) {
                    Some(label) => {
                        GGUFQuant::matches(&GGUFQuant::canonical_tag(&label), &requested_tag)
                    }
                    None => false,
                })
                .collect();
        }

        let picked =
            largest(&matches, siblings, repo).ok_or_else(|| ResolveError::QuantNotFound {
                repo: repo.to_string(),
                quant: requested.to_string(),
            })?;
        return Ok(Pick {
            rfilename: picked.rfilename.clone(),
            tag: GGUFQuant::tag_for_path(&picked.rfilename, Some(requested)),
        });
    }

    let compatible: Vec<&Sibling> = all_ggufs
        .iter()
        .copied()
        .filter(|sib| {
            compat::estimated_weight_fits(estimated_model_bytes(sib, siblings, repo), budget_mb)
        })
        .collect();

    // Tag preference: try each in order, first matching candidate wins. The
    // boundary `[.-]` mirrors llama.cpp's regex.
    let preferred = ["Q4_K_M", "Q4_0"].iter().find_map(|tag| {
        let re = regex::Regex::new(&format!("(?i){}[.-]", regex::escape(tag))).ok()?;
        compatible
            .iter()
            .find(|sib| re.is_match(&sib.rfilename))
            .copied()
    });

    let best = match preferred {
        Some(sib) => sib,
        None => largest(&compatible, siblings, repo)
            .ok_or_else(|| ResolveError::NoCompatibleFile(repo.to_string()))?,
    };
    Ok(Pick {
        rfilename: best.rfilename.clone(),
        tag: GGUFQuant::tag_for_path(&best.rfilename, None),
    })
}

/// Estimated ordering key: `(bytes, rfilename)` ascending. Used to pick the
/// largest build, with rfilename as a deterministic tie-break.
fn order_key(sib: &Sibling, siblings: &[Sibling], repo: &str) -> (i64, String) {
    (
        estimated_model_bytes(sib, siblings, repo).unwrap_or(0),
        sib.rfilename.clone(),
    )
}

fn largest<'a>(sibs: &[&'a Sibling], siblings: &[Sibling], repo: &str) -> Option<&'a Sibling> {
    let mut best: Option<&Sibling> = None;
    for sib in sibs {
        match best {
            None => best = Some(sib),
            Some(b) => {
                if order_key(sib, siblings, repo) > order_key(b, siblings, repo) {
                    best = Some(sib);
                }
            }
        }
    }
    best
}

/// Estimated bytes for a runnable quant: shard sum when `main` is the first
/// shard of a complete set and every size is known, else `main.size`.
fn estimated_model_bytes(main: &Sibling, siblings: &[Sibling], repo: &str) -> Option<i64> {
    let shard_paths = match expand_shards(&main.rfilename, siblings, repo) {
        Ok(paths) => paths,
        Err(_) => return main.size,
    };
    if shard_paths.len() <= 1 {
        return main.size;
    }
    let mut total: i64 = 0;
    for path in &shard_paths {
        match siblings
            .iter()
            .find(|s| &s.rfilename == path)
            .and_then(|s| s.size)
        {
            Some(size) => total += size,
            None => return main.size,
        }
    }
    Some(total)
}

/// Ordered shard filenames (main included) if `main` is a split shard, else
/// `[main]`. Errors if any shard in `1..=total` is missing.
fn expand_shards(
    main: &str,
    siblings: &[Sibling],
    repo: &str,
) -> Result<Vec<String>, ResolveError> {
    let name = last_component(main);
    let (Some(total), Some(stem)) = (
        HFRepoParser::shard_total(name),
        HFRepoParser::split_shard_base_name(name),
    ) else {
        return Ok(vec![main.to_string()]);
    };

    let prefix = match main.rfind('/') {
        Some(i) => &main[..=i],
        None => "",
    };

    let mut shards = Vec::with_capacity(total as usize);
    for i in 1..=total {
        let shard = format!("{stem}-{i:05}-of-{total:05}.gguf");
        let full = format!("{prefix}{shard}");
        if !siblings.iter().any(|s| s.rfilename == full) {
            return Err(ResolveError::ShardMissing {
                repo: repo.to_string(),
                shard,
            });
        }
        shards.push(full);
    }
    Ok(shards)
}

fn pick_mmproj(main: &str, siblings: &[Sibling]) -> Option<Sibling> {
    let names: Vec<String> = siblings.iter().map(|s| s.rfilename.clone()).collect();
    let picked = SidecarPicker::mmproj(&names, main)?;
    siblings.iter().find(|s| s.rfilename == picked).cloned()
}

fn pick_draft_head(main: &str, siblings: &[Sibling], main_quant: &str) -> Option<Sibling> {
    let names: Vec<String> = siblings.iter().map(|s| s.rfilename.clone()).collect();
    let picked = SidecarPicker::draft_head(&names, main, Some(main_quant))?;
    siblings.iter().find(|s| s.rfilename == picked).cloned()
}

/// Path segment set that mirrors Foundation's `.urlPathAllowed` for the shapes
/// that appear in model repos.
const PATH_SEGMENT_SET: &AsciiSet = &CONTROLS
    .add(b' ')
    .add(b'"')
    .add(b'#')
    .add(b'<')
    .add(b'>')
    .add(b'?')
    .add(b'`')
    .add(b'{')
    .add(b'}')
    .add(b'|')
    .add(b'\\')
    .add(b'^')
    .add(b'[')
    .add(b']');

/// `https://huggingface.co/{repo}/resolve/main/{path}`, percent-encoding each
/// path segment separately so `/` stays a delimiter.
pub fn resolve_url(repo: &str, path: &str) -> String {
    let encoded = path
        .split('/')
        .map(|seg| utf8_percent_encode(seg, PATH_SEGMENT_SET).to_string())
        .collect::<Vec<_>>()
        .join("/");
    format!("https://huggingface.co/{repo}/resolve/main/{encoded}")
}
