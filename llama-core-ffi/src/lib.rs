//! C ABI for `llama-core`.
//!
//! Contract (see `MIGRACAO_RUST_REVIEW.md`):
//! - Rust allocates every returned string; Swift releases it with
//!   [`llama_core_free`].
//! - Inputs are UTF-8, NUL-terminated, and only need to live for the call.
//! - Every entry point catches Rust panics so a panic never crosses the FFI
//!   boundary (which would be undefined behavior).
//! - Complex payloads travel as JSON; this keeps the ABI small and stable.

use llama_core::config::{aliases, ini};
use llama_core::engine::{
    binaries, download, installer, mem_profile, partial_writer, server, LlamaVersion,
};
use llama_core::hf::{
    GGUFMetadata, GGUFQuant, HFCache, HFRepoParser, SidecarPicker, SideloadedEntry,
};
use llama_core::models::{model, ContextTier, DraftHead, ModelIdParser, ModelMem, ResolvedPaths};
use llama_core::net::catalog;
use llama_core::net::hf_repo_resolver::{
    resolve_from_siblings, resolve_url, ResolveError, Resolved, Sibling,
};
use llama_core::net::hf_request;
use llama_core::net::response::{
    blob_hash, commit_hash, file_metadata, same_host_redirect, HttpResponse,
};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// A host callback that receives events (progress, completion, ...) marshaled
/// as a JSON string. The pointer is only valid for the duration of the call.
/// The `Option` makes the function pointer nullable so the host can pass NULL
/// to clear it.
type EventCb = Option<extern "C" fn(*const c_char)>;

struct Core {
    config: Value,
    event_cb: EventCb,
}

fn core() -> &'static Mutex<Core> {
    static CORE: OnceLock<Mutex<Core>> = OnceLock::new();
    CORE.get_or_init(|| {
        Mutex::new(Core {
            config: Value::Null,
            event_cb: None,
        })
    })
}

/// Open `PartialWriter`s keyed by an opaque handle. The JSON dispatch is
/// stateless, so a stream surviving across `partial.*` calls lives here — the
/// Rust analogue of Swift's `WriterTable` (minus the per-task URLSession map).
fn partial_writers() -> &'static Mutex<HashMap<u64, partial_writer::PartialWriter>> {
    static WRITERS: OnceLock<Mutex<HashMap<u64, partial_writer::PartialWriter>>> = OnceLock::new();
    WRITERS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_partial_handle() -> u64 {
    static SEQ: AtomicU64 = AtomicU64::new(1);
    SEQ.fetch_add(1, Ordering::SeqCst)
}

unsafe fn cstr_to_str<'a>(p: *const c_char) -> &'a str {
    if p.is_null() {
        return "";
    }
    CStr::from_ptr(p).to_str().unwrap_or("")
}

fn json_to_c(v: &Value) -> *mut c_char {
    match serde_json::to_string(v) {
        Ok(s) => match CString::new(s) {
            Ok(c) => c.into_raw(),
            Err(_) => std::ptr::null_mut(),
        },
        Err(_) => std::ptr::null_mut(),
    }
}

/// Emits an event to the registered callback, if any. The lock is released
/// before invoking the callback so a re-entrant call from the host can't
/// deadlock.
fn emit(value: &Value) {
    let cb = core().lock().ok().and_then(|c| c.event_cb);
    let Some(cb) = cb else { return };
    if let Ok(s) = serde_json::to_string(value) {
        if let Ok(cstr) = CString::new(s) {
            cb(cstr.as_ptr());
        }
    }
}

/// Initializes the core with a JSON `CoreConfig` snapshot pushed by the host.
/// Passing NULL or an empty string means "no config yet". Returns 0 on success.
///
/// # Safety
/// `config_json` must be NULL or point to a NUL-terminated UTF-8 string that is
/// valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn llama_core_init(config_json: *const c_char) -> c_int {
    catch_unwind(AssertUnwindSafe(|| {
        let s = unsafe { cstr_to_str(config_json) };
        let v = if s.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(s).unwrap_or(Value::Null)
        };
        if let Ok(mut c) = core().lock() {
            c.config = v;
            0
        } else {
            -1
        }
    }))
    .unwrap_or(-1)
}

/// Replaces the `CoreConfig` snapshot (called on every settings change in the
/// host). Returns 0 on success.
///
/// # Safety
/// Same requirements as [`llama_core_init`].
#[no_mangle]
pub unsafe extern "C" fn llama_core_set_config(config_json: *const c_char) -> c_int {
    // SAFETY: same contract as `llama_core_init`.
    unsafe { llama_core_init(config_json) }
}

/// Registers (or clears, with NULL) the event callback.
///
/// # Safety
/// `cb` must be NULL or a valid C function pointer that stays callable for as
/// long as the core may emit events. The core holds it until replaced.
#[no_mangle]
pub unsafe extern "C" fn llama_core_set_event_cb(cb: Option<extern "C" fn(*const c_char)>) {
    let _ = catch_unwind(AssertUnwindSafe(|| {
        if let Ok(mut c) = core().lock() {
            c.event_cb = cb;
        }
    }));
}

/// Dispatch entry point. `method` selects the operation; `args_json` is a JSON
/// object. Returns a heap JSON string owned by the core, or NULL on panic.
///
/// The caller must free the result with [`llama_core_free`].
///
/// # Safety
/// Both pointers must be NULL or point to NUL-terminated UTF-8 strings valid
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn llama_core_call(
    method: *const c_char,
    args_json: *const c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| {
        let method = unsafe { cstr_to_str(method) };
        let args_s = unsafe { cstr_to_str(args_json) };
        let args: Value = if args_s.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(args_s).unwrap_or(Value::Null)
        };
        dispatch(method, &args)
    }));
    match result {
        Ok(v) => json_to_c(&v),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Frees a string previously returned by [`llama_core_call`].
///
/// # Safety
/// `s` must be NULL or a pointer previously returned by [`llama_core_call`] and
/// not yet freed; it must not be used after this call.
#[no_mangle]
pub unsafe extern "C" fn llama_core_free(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(CString::from_raw(s));
    }));
}

fn section_from_value(v: &Value) -> ini::Section {
    let name = v
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    let mut s = ini::Section::new(name);
    if let Some(pairs) = v.get("pairs").and_then(Value::as_array) {
        for p in pairs {
            let k = p.get("key").and_then(Value::as_str).unwrap_or("");
            let val = p.get("value").and_then(Value::as_str).unwrap_or("");
            s.set(k, val);
        }
    }
    s
}

fn section_to_value(s: &ini::Section) -> Value {
    let pairs: Vec<Value> = s
        .pairs
        .iter()
        .map(|(k, v)| json!({ "key": k, "value": v }))
        .collect();
    json!({ "name": s.name, "pairs": pairs })
}

fn sections_from_value(v: &Value) -> Vec<ini::Section> {
    v.as_array()
        .map(|a| a.iter().map(section_from_value).collect())
        .unwrap_or_default()
}

fn sections_to_value(sections: &[ini::Section]) -> Value {
    Value::Array(sections.iter().map(section_to_value).collect())
}

fn strings_from_value(v: &Value) -> Vec<String> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_str)
                .map(str::to_string)
                .collect()
        })
        .unwrap_or_default()
}

/// Decodes `[[k, v], ...]` into owned pairs (header/env/address lists).
fn pairs_from_value(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|pair| {
                    let p = pair.as_array()?;
                    Some((
                        p.first()?.as_str()?.to_string(),
                        p.get(1)?.as_str()?.to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Decodes `addresses` ([[interface, ip], ...]) in `args["addresses"]`.
fn interface_adresses(args: &Value) -> Vec<(String, String)> {
    pairs_from_value(args.get("addresses").unwrap_or(&Value::Null))
}

fn tiers_to_value(tiers: &[ContextTier]) -> Value {
    Value::Array(tiers.iter().map(|t| json!(t.raw_value())).collect())
}

fn parsed_to_value(p: &llama_core::models::Parsed) -> Value {
    json!({
        "org": p.org,
        "displayOrg": p.display_org(),
        "name": p.name,
        "params": p.params,
        "quant": p.quant,
        "tags": p.tags,
    })
}

fn resolved_paths_from_value(v: &Value) -> ResolvedPaths {
    let opt = |key: &str| v.get(key).and_then(Value::as_str).map(str::to_string);
    ResolvedPaths {
        model_file: v
            .get("modelFile")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
        additional_parts: strings_from_value(v.get("additionalParts").unwrap_or(&Value::Null)),
        mmproj_file: opt("mmprojFile"),
        uses_mtp: v.get("usesMTP").and_then(Value::as_bool).unwrap_or(false),
        draft_sidecar_file: opt("draftSidecarFile"),
        hf_repo_dir_name: v
            .get("hfRepoDirName")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string(),
    }
}

fn resolved_paths_to_value(rp: &ResolvedPaths) -> Value {
    json!({
        "modelFile": rp.model_file,
        "additionalParts": rp.additional_parts,
        "mmprojFile": rp.mmproj_file,
        "usesMTP": rp.uses_mtp,
        "draftSidecarFile": rp.draft_sidecar_file,
        "hfRepoDirName": rp.hf_repo_dir_name,
    })
}

fn sideloaded_entry_to_value(e: &SideloadedEntry) -> Value {
    json!({
        "id": e.id,
        "ctxWindow": e.ctx_window,
        "fileSize": e.file_size,
        "draftHead": e.scanned_draft_head.map(|d| d.as_str()),
        "isDecisionModel": e.is_decision_model,
        "paths": resolved_paths_to_value(&e.paths),
    })
}

fn siblings_from_value(v: &Value) -> Vec<Sibling> {
    v.as_array()
        .map(|a| {
            a.iter()
                .map(|s| {
                    Sibling::new(
                        s.get("rfilename")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                        s.get("size").and_then(Value::as_i64),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

fn families_from_value(v: &Value) -> Vec<catalog::Family> {
    v.as_array()
        .map(|a| {
            a.iter()
                .filter_map(|f| serde_json::from_value(f.clone()).ok())
                .collect()
        })
        .unwrap_or_default()
}

fn resolved_to_value(r: &Resolved) -> Value {
    json!({
        "modelId": r.model_id,
        "mainUrl": r.main_url,
        "additionalParts": r.additional_parts,
        "mmprojUrl": r.mmproj_url,
        "draftUrl": r.draft_url,
        "approximateBytes": r.approximate_bytes,
    })
}

fn resolve_error_to_value(e: &ResolveError) -> Value {
    let (repo, quant, shard) = match e {
        ResolveError::InvalidQuant(q) => (None, Some(q.as_str()), None),
        ResolveError::RepoNotFound(r)
        | ResolveError::Gated(r)
        | ResolveError::NetworkUnavailable(r)
        | ResolveError::NoGgufFiles(r)
        | ResolveError::NoCompatibleFile(r) => (Some(r.as_str()), None, None),
        ResolveError::QuantNotFound { repo, quant } => {
            (Some(repo.as_str()), Some(quant.as_str()), None)
        }
        ResolveError::ShardMissing { repo, shard } => {
            (Some(repo.as_str()), None, Some(shard.as_str()))
        }
    };
    json!({
        "error": e.kind(),
        "message": e.message(),
        "repo": repo,
        "quant": quant,
        "shard": shard,
    })
}

fn suggestions_to_value(sa: &[catalog::Suggestion]) -> Value {
    Value::Array(
        sa.iter()
            .map(|s| {
                json!({
                    "brand": s.brand,
                    "repo": s.repo,
                    "quant": s.quant,
                    "sizeLabel": s.size_label,
                    "hasVision": s.has_vision,
                })
            })
            .collect(),
    )
}

fn response_from_value(v: &Value) -> HttpResponse {
    let headers: Vec<(String, String)> = v
        .get("headers")
        .and_then(Value::as_array)
        .map(|a| {
            a.iter()
                .filter_map(|p| {
                    let pair = p.as_array()?;
                    let k = pair.first()?.as_str()?;
                    let val = pair.get(1)?.as_str()?;
                    Some((k.to_string(), val.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();
    HttpResponse {
        status: v.get("status").and_then(Value::as_u64).unwrap_or(200) as u16,
        headers,
        body: Vec::new(),
    }
}

fn dispatch(method: &str, args: &Value) -> Value {
    // Development-only escape hatch used by tests to prove that a panic is
    // caught at the boundary rather than unwinding into the host.
    #[cfg(debug_assertions)]
    if method == "panic.test" {
        panic!("intentional panic for tests");
    }

    let arg_str = |key: &str| args.get(key).and_then(Value::as_str).unwrap_or("");
    let arg_f = |key: &str| args.get(key).and_then(Value::as_f64).unwrap_or(0.0);
    let arg_i = |key: &str| args.get(key).and_then(Value::as_i64).unwrap_or(0);
    match method {
        "ping" => json!({ "pong": true }),

        "version.parse" => {
            json!({ "build": LlamaVersion::parse(arg_str("output")).map(|v| v.build) })
        }

        "repo.parse" => match HFRepoParser::parse(arg_str("repoDir")) {
            Some((org, repo)) => json!({ "org": org, "repo": repo }),
            None => Value::Null,
        },
        "repo.parseQuant" => json!(HFRepoParser::parse_quant(arg_str("filename"))),
        "repo.isSplitShard" => json!(HFRepoParser::is_split_shard(arg_str("filename"))),
        "repo.isFirstShard" => json!(HFRepoParser::is_first_shard(arg_str("filename"))),
        "repo.shardTotal" => json!(HFRepoParser::shard_total(arg_str("filename"))),
        "repo.splitShardBaseName" => {
            json!(HFRepoParser::split_shard_base_name(arg_str("filename")))
        }

        // ---- Fase 1: GGUF quant ----
        "quant.parseLabel" => {
            json!({ "label": GGUFQuant::parse_label(arg_str("path")) })
        }
        "quant.tagForPath" => {
            let requested = args.get("requested").and_then(Value::as_str);
            json!({ "tag": GGUFQuant::tag_for_path(arg_str("path"), requested) })
        }
        "quant.quantBitsForPath" => {
            json!({ "bits": GGUFQuant::quant_bits_for_path(arg_str("path")) })
        }
        "quant.quantBitsForTag" => {
            json!({ "bits": GGUFQuant::quant_bits_for_tag(arg_str("tag")) })
        }
        "quant.canonicalTag" => {
            json!({ "tag": GGUFQuant::canonical_tag(arg_str("label")) })
        }
        "quant.matches" => {
            json!({ "value": GGUFQuant::matches(arg_str("a"), arg_str("b")) })
        }

        // ---- Fase 2: GGUF metadata header ----
        "gguf.contextLength" => match GGUFMetadata::context_length(arg_str("path")) {
            None => Value::Null,
            Some(length) => json!({ "parsed": true, "length": length }),
        },
        "gguf.hasEmbeddedMTPHead" => match GGUFMetadata::has_embedded_mtp_head(arg_str("path")) {
            None => Value::Null,
            Some(value) => json!({ "value": value }),
        },
        "gguf.isDecisionModel" => match GGUFMetadata::is_decision_model(arg_str("path")) {
            None => Value::Null,
            Some(value) => json!({ "value": value }),
        },

        // ---- Fase 2: HFCache ----
        "hfCache.repoDirName" => {
            json!({ "name": HFCache::repo_dir_name_from_url(arg_str("url")) })
        }
        "hfCache.repoRelativePath" => {
            json!({ "path": HFCache::repo_relative_path_from_url(arg_str("url")) })
        }
        "hfCache.partialDir" => json!({
            "path": HFCache::partial_dir(arg_str("cacheDir"), arg_str("modelId"))
                .to_string_lossy()
        }),
        "hfCache.partialPath" => json!({
            "path": HFCache::partial_path(arg_str("cacheDir"), arg_str("modelId"), arg_str("filename"))
                .to_string_lossy()
        }),
        "hfCache.placeholderURL" => json!({
            "path": HFCache::placeholder_url(arg_str("cacheDir"), arg_str("modelId"))
                .to_string_lossy()
        }),
        "hfCache.removePartials" => {
            HFCache::remove_partials(arg_str("cacheDir"), arg_str("modelId"));
            json!({ "ok": true })
        }
        "hfCache.partialBytes" => json!({
            "bytes": HFCache::partial_bytes(arg_str("cacheDir"), arg_str("modelId"))
        }),
        "hfCache.scanPlaceholders" => {
            let paths: Vec<String> = HFCache::scan_placeholders(arg_str("cacheDir"))
                .iter()
                .map(|p| p.to_string_lossy().into_owned())
                .collect();
            json!({ "paths": paths })
        }
        "hfCache.cleanInstalledPartials" => {
            let ids: std::collections::HashSet<String> =
                strings_from_value(args.get("installedIds").unwrap_or(&Value::Null))
                    .into_iter()
                    .collect();
            HFCache::clean_installed_partials(arg_str("cacheDir"), &ids);
            json!({ "ok": true })
        }
        "hfCache.snapshotFileExists" => json!({
            "value": HFCache::snapshot_file_exists(
                arg_str("cacheDir"), arg_str("repoDir"), arg_str("filename")
            )
        }),
        "hfCache.writeBlobAndLink" => {
            match HFCache::write_blob_and_link(
                arg_str("cacheDir"),
                arg_str("repoDir"),
                arg_str("commit"),
                arg_str("blobHash"),
                arg_str("filename"),
                arg_str("tempFile"),
            ) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        "hfCache.deleteModelFiles" => {
            let paths = resolved_paths_from_value(args.get("paths").unwrap_or(&Value::Null));
            match HFCache::delete_model_files(arg_str("cacheDir"), arg_str("repoDir"), &paths) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        "hfCache.scanForSideloaded" => {
            let entries: Vec<Value> = HFCache::scan_for_sideloaded(arg_str("cacheDir"))
                .iter()
                .map(sideloaded_entry_to_value)
                .collect();
            json!({ "entries": entries })
        }

        // ---- Fase 1: DraftHead ----
        "draftHead.fromSidecarPath" => {
            json!({ "kind": DraftHead::from_sidecar_path(arg_str("path")).map(DraftHead::as_str) })
        }
        "draftHead.describe" => match DraftHead::from_str_name(arg_str("kind")) {
            Some(k) => json!({
                "prefix": k.sidecar_prefix(),
                "specType": k.spec_type(),
                "draftNMax": k.draft_n_max(),
                "label": k.label(),
            }),
            None => Value::Null,
        },
        "draftHead.preferenceOrder" => {
            json!(DraftHead::PREFERENCE_ORDER
                .iter()
                .map(|k| k.as_str())
                .collect::<Vec<_>>())
        }

        // ---- Fase 1: SidecarPicker ----
        "sidecar.isMmproj" => json!({ "value": SidecarPicker::is_mmproj(arg_str("path")) }),
        "sidecar.isDraftHead" => json!({ "value": SidecarPicker::is_draft_head(arg_str("path")) }),
        "sidecar.mmproj" => {
            let names = strings_from_value(args.get("names").unwrap_or(&Value::Null));
            json!({ "path": SidecarPicker::mmproj(&names, arg_str("mainPath")) })
        }
        "sidecar.draftHead" => {
            let names = strings_from_value(args.get("names").unwrap_or(&Value::Null));
            let tag = args.get("tag").and_then(Value::as_str);
            json!({ "path": SidecarPicker::draft_head(&names, arg_str("mainPath"), tag) })
        }

        // ---- Fase 1: ModelIdParser ----
        "modelId.parse" => parsed_to_value(&ModelIdParser::parse(arg_str("id"))),
        "modelId.plainName" => {
            let show_tags = args
                .get("showTags")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({ "name": ModelIdParser::parse(arg_str("id")).plain_name(show_tags) })
        }
        "modelId.displayKey" => {
            json!({ "key": ModelIdParser::parse(arg_str("id")).display_key() })
        }

        // ---- Fase 1: Model id/sort helpers ----
        "model.makeId" => json!({
            "id": model::make_id(arg_str("org"), arg_str("repo"), arg_str("tag"))
        }),
        "model.idBaseOf" => json!({ "base": model::id_base_of(arg_str("id")) }),
        "model.sortKey" => json!({ "key": model::sort_key(arg_str("id")) }),
        "model.displayOrder" => {
            let order = model::display_order(arg_str("lhsId"), arg_str("rhsId"));
            json!({ "order": order as i32 })
        }

        // ---- Fase 1: ContextTier ----
        "contextTier.list" => {
            let tiers: Vec<Value> = ContextTier::ALL
                .iter()
                .map(|t| {
                    json!({
                        "rawValue": t.raw_value(),
                        "shortLabel": t.short_label(),
                        "label": t.label(),
                        "suffix": t.suffix(),
                    })
                })
                .collect();
            Value::Array(tiers)
        }

        // ---- Fase 1: ResolvedPaths ----
        "resolvedPaths.allPaths" => {
            let rp = ResolvedPaths {
                model_file: arg_str("modelFile").to_string(),
                additional_parts: strings_from_value(
                    args.get("additionalParts").unwrap_or(&Value::Null),
                ),
                mmproj_file: args
                    .get("mmprojFile")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                uses_mtp: args
                    .get("usesMTP")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                draft_sidecar_file: args
                    .get("draftSidecarFile")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                hf_repo_dir_name: arg_str("hfRepoDirName").to_string(),
            };
            json!({ "paths": rp.all_paths() })
        }
        "resolvedPaths.draftSidecarKind" => {
            let rp = ResolvedPaths {
                model_file: String::new(),
                additional_parts: vec![],
                mmproj_file: None,
                uses_mtp: false,
                draft_sidecar_file: args
                    .get("draftSidecarFile")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                hf_repo_dir_name: String::new(),
            };
            json!({ "kind": rp.draft_sidecar_kind().map(DraftHead::as_str) })
        }

        // ---- Fase 1: Model+Compatibility ----
        "compat.deviceBudget" => {
            let use_full = args
                .get("useFullWorkingSet")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({ "mb": llama_core::models::compat::device_memory_budget_mb(
                arg_f("workingSetMb"), arg_f("physicalMb"), use_full
            ) })
        }
        "compat.fitTarget" => {
            json!({ "mb": llama_core::models::compat::fit_target_mb(arg_f("workingSetMb"), arg_f("budgetMb")) })
        }
        "compat.estimatedWeightFits" => {
            let bytes = args.get("bytes").and_then(Value::as_i64);
            json!({ "value": llama_core::models::compat::estimated_weight_fits(bytes, arg_f("budgetMb")) })
        }
        "compat.info" => {
            let mem = ModelMem {
                ctx_window: arg_i("ctxWindow"),
                file_size: arg_i("fileSize"),
                ctx_bytes_per_1k_tokens: arg_i("ctxBytesPer1kTokens"),
                resident_bytes: arg_i("residentBytes"),
            };
            let info = mem.compatibility_info(arg_f("ctxWindowTokens"), arg_f("budgetMb"));
            json!({
                "isCompatible": info.is_compatible,
                "incompatibilitySummary": info.incompatibility_summary,
            })
        }
        "compat.runtimeMemoryMb" => {
            let mem = ModelMem {
                ctx_window: arg_i("ctxWindow"),
                file_size: arg_i("fileSize"),
                ctx_bytes_per_1k_tokens: arg_i("ctxBytesPer1kTokens"),
                resident_bytes: arg_i("residentBytes"),
            };
            json!({ "mb": mem.runtime_memory_usage_mb(arg_f("ctxWindowTokens")) })
        }
        "compat.supportedTiers" => {
            let mem = ModelMem {
                ctx_window: arg_i("ctxWindow"),
                file_size: arg_i("fileSize"),
                ctx_bytes_per_1k_tokens: arg_i("ctxBytesPer1kTokens"),
                resident_bytes: arg_i("residentBytes"),
            };
            json!({ "tiers": tiers_to_value(&mem.supported_context_tiers(arg_f("budgetMb"))) })
        }
        "compat.displayTiers" => {
            let mem = ModelMem {
                ctx_window: arg_i("ctxWindow"),
                file_size: arg_i("fileSize"),
                ctx_bytes_per_1k_tokens: arg_i("ctxBytesPer1kTokens"),
                resident_bytes: arg_i("residentBytes"),
            };
            json!({ "tiers": tiers_to_value(&mem.display_context_tiers()) })
        }
        "compat.nativeMaxTier" => {
            let mem = ModelMem {
                ctx_window: arg_i("ctxWindow"),
                file_size: 0,
                ctx_bytes_per_1k_tokens: 0,
                resident_bytes: 0,
            };
            json!({ "tier": mem.native_max_tier().map(|t| t.raw_value()) })
        }
        "compat.effectiveTier" => {
            let mem = ModelMem {
                ctx_window: arg_i("ctxWindow"),
                file_size: arg_i("fileSize"),
                ctx_bytes_per_1k_tokens: arg_i("ctxBytesPer1kTokens"),
                resident_bytes: arg_i("residentBytes"),
            };
            let selected = args
                .get("selected")
                .and_then(Value::as_i64)
                .and_then(ContextTier::from_raw);
            json!({ "tier": mem.effective_ctx_tier(arg_f("budgetMb"), selected).map(|t| t.raw_value()) })
        }

        // ---- Fase 1: UserModelOverrides (pure INI) ----
        "ini.parse" => {
            let sections = ini::parse(arg_str("text"));
            json!({ "sections": sections_to_value(&sections) })
        }
        "ini.serialize" => {
            let sections = sections_from_value(args.get("sections").unwrap_or(&Value::Null));
            let text: String = sections.iter().map(ini::Section::serialized).collect();
            json!({ "text": text })
        }
        "ini.apply" => {
            let generated = sections_from_value(args.get("generated").unwrap_or(&Value::Null));
            let overrides = sections_from_value(args.get("overrides").unwrap_or(&Value::Null));
            let suspended = args
                .get("suspended")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({ "sections": sections_to_value(&ini::apply(&generated, &overrides, suspended)) })
        }

        // ---- Fase 1: ModelAliases (pure) ----
        "aliases.validationError" => {
            let aliases: Vec<aliases::Alias> = args
                .get("aliases")
                .and_then(Value::as_array)
                .map(|a| {
                    a.iter()
                        .map(|v| aliases::Alias {
                            name: v
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            model_id: v
                                .get("modelId")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();
            let replacing = args.get("replacing").and_then(|v| {
                v.as_object().map(|o| aliases::Alias {
                    name: o
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    model_id: o
                        .get("modelId")
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                })
            });
            let override_names =
                strings_from_value(args.get("overrideNames").unwrap_or(&Value::Null));
            json!({ "error": aliases::validation_error(
                arg_str("name"), &aliases, replacing.as_ref(), &override_names
            ) })
        }
        "aliases.removeConflicts" => {
            let sections = sections_from_value(args.get("sections").unwrap_or(&Value::Null));
            json!({ "sections": sections_to_value(&aliases::remove_conflicts(&sections)) })
        }
        "aliases.setAlias" => {
            let mut section = section_from_value(args.get("section").unwrap_or(&Value::Null));
            let names = strings_from_value(args.get("names").unwrap_or(&Value::Null));
            aliases::set_alias(&mut section, &names);
            json!({ "section": section_to_value(&section) })
        }

        // ---- Fase 3: network + remote selection ----
        "net.resolveFromSiblings" => {
            let requested = args.get("requestedQuant").and_then(Value::as_str);
            let siblings = siblings_from_value(args.get("siblings").unwrap_or(&Value::Null));
            match resolve_from_siblings(arg_str("repo"), requested, &siblings, arg_f("budgetMb")) {
                Ok(r) => resolved_to_value(&r),
                Err(e) => resolve_error_to_value(&e),
            }
        }
        "net.resolveUrl" => {
            json!({ "url": resolve_url(arg_str("repo"), arg_str("path")) })
        }
        "catalog.parseBytes" => {
            let label = args.get("label").and_then(Value::as_str);
            json!({ "bytes": catalog::parse_bytes(label) })
        }
        "catalog.featuredSuggestions" => {
            let families = families_from_value(args.get("families").unwrap_or(&Value::Null));
            let suggestions = catalog::featured_suggestions(
                &families,
                arg_f("budgetMb"),
                arg_i("systemMemoryMb") as u64,
            );
            json!({ "suggestions": suggestions_to_value(&suggestions) })
        }
        "hfRequest.userAgent" => json!({
            "ua": hf_request::user_agent(arg_str("shortVersion"), arg_i("osMajor"), arg_i("osMinor"))
        }),
        "hfRequest.make" => {
            let token = args.get("token").and_then(Value::as_str);
            let headers: Vec<Value> = hf_request::make(arg_str("url"), token, arg_str("userAgent"))
                .iter()
                .map(|(k, v)| json!([k, v]))
                .collect();
            json!({ "headers": headers })
        }
        "net.blobHash" => json!({ "hash": blob_hash(&response_from_value(args)) }),
        "net.commitHash" => json!({ "hash": commit_hash(&response_from_value(args)) }),
        "net.fileMetadata" => {
            let meta = file_metadata(&response_from_value(args))
                .map(|m| json!({ "blobHash": m.blob_hash, "commitHash": m.commit_hash }));
            json!({ "metadata": meta })
        }
        "net.sameHostRedirect" => json!({
            "value": same_host_redirect(
                args.get("from").and_then(Value::as_str),
                args.get("to").and_then(Value::as_str),
            )
        }),

        // ---- Fase 4: Engine (LlamaBinaries / LlamaInstaller leaves / MemProfile) ----
        "engine.managedPath" => json!({
            "path": binaries::managed_path(arg_str("home"))
        }),
        "engine.metalConfig" => match installer::metal_config(arg_str("brand")) {
            Ok(config) => json!({ "config": config }),
            Err(e) => json!({ "error": e.kind(), "message": e.message() }),
        },
        "engine.artifactUrl" => json!({
            "url": installer::artifact_url(arg_str("version"), arg_str("config"))
        }),
        "engine.unzstdUrl" => json!({
            "url": installer::unzstd_url(arg_str("version"))
        }),
        "engine.binariesResolve" => {
            let unmanaged_dirs: Vec<String> =
                match args.get("unmanagedDirs").and_then(Value::as_array) {
                    // An array is used verbatim (empty = "search nothing") so the
                    // host can pin a temp dir in tests; absent defaults to the
                    // Well-Known Homebrew dirs for real use.
                    Some(a) => a
                        .iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect(),
                    None => binaries::UNMANAGED_DIRS
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                };
            let ignore = args
                .get("ignoreUnmanaged")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            match binaries::resolve(arg_str("home"), &unmanaged_dirs, ignore) {
                Some((path, origin)) => json!({ "origin": origin.as_str(), "path": path }),
                None => Value::Null,
            }
        }
        "engine.readVersion" => json!({
            "build": binaries::read_version(arg_str("path")).map(|v| v.build)
        }),
        "engine.decompress" => {
            match installer::decompress(arg_str("unzstd"), arg_str("zst"), arg_str("dest")) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.kind(), "message": e.message() }),
            }
        }
        "engine.atomicInstall" => {
            match installer::atomic_install(arg_str("src"), arg_str("dest")) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.kind(), "message": e.message() }),
            }
        }
        "engine.ensurePathSymlink" => {
            match installer::ensure_path_symlink(arg_str("home"), arg_str("llamaPath")) {
                installer::PathSymlinkOutcome::Linked => json!({ "outcome": "linked" }),
                installer::PathSymlinkOutcome::LeftExisting => {
                    json!({ "outcome": "leftExisting" })
                }
                installer::PathSymlinkOutcome::Failed(message) => {
                    json!({ "outcome": "failed", "message": message })
                }
            }
        }
        "engine.promoteStaged" => {
            let target = LlamaVersion {
                build: arg_i("targetBuild"),
            };
            match installer::promote_staged(
                arg_str("staged"),
                arg_str("live"),
                target,
                arg_str("home"),
            ) {
                installer::PromoteOutcome::NothingStaged => json!({ "outcome": "nothingStaged" }),
                installer::PromoteOutcome::DiscardedStale => {
                    json!({ "outcome": "discardedStale" })
                }
                installer::PromoteOutcome::Failed(message) => {
                    json!({ "outcome": "failed", "message": message })
                }
                installer::PromoteOutcome::Promoted => json!({ "outcome": "promoted" }),
            }
        }
        "engine.fit" => {
            match mem_profile::fit(
                arg_i("loMib"),
                arg_i("hiMib"),
                arg_i("ctxLo"),
                arg_i("ctxHi"),
            ) {
                Some(p) => json!({
                    "profile": {
                        "ctxBytesPer1kTokens": p.ctx_bytes_per_1k_tokens,
                        "residentBytes": p.resident_bytes,
                        "schemaVersion": p.schema_version,
                    }
                }),
                None => Value::Null,
            }
        }
        "engine.parseFitp" => json!({ "sum": mem_profile::parse_fitp_output(arg_str("output")) }),
        "engine.memCacheFileName" => {
            json!({ "hash": mem_profile::cache_file_name(arg_str("modelId")) })
        }
        "engine.memCachePath" => {
            json!({ "path": mem_profile::cache_path(arg_str("cacheDir"), arg_str("modelId")) })
        }
        "engine.memCacheGet" => {
            match mem_profile::cache_get(arg_str("cacheDir"), arg_str("modelId")) {
                Some(p) => json!({
                    "profile": {
                        "ctxBytesPer1kTokens": p.ctx_bytes_per_1k_tokens,
                        "residentBytes": p.resident_bytes,
                        "schemaVersion": p.schema_version,
                    }
                }),
                None => json!({ "profile": Value::Null }),
            }
        }
        "engine.memCacheSet" => {
            let profile =
                mem_profile::MemProfile::new(arg_i("ctxBytesPer1kTokens"), arg_i("residentBytes"));
            match mem_profile::cache_set(arg_str("cacheDir"), arg_str("modelId"), &profile) {
                Ok(()) => json!({ "ok": true }),
                Err(message) => json!({ "error": message }),
            }
        }

        // ---- Fase 5: Download engine (ModelManager+Transfer leaves, ActiveDownload, DownloadError) ----
        "download.extractFullSize" => {
            let headers: Vec<(String, String)> = match args.get("headers").and_then(Value::as_array)
            {
                Some(a) => a
                    .iter()
                    .filter_map(|pair| {
                        let p = pair.as_array()?;
                        Some((
                            p.first()?.as_str()?.to_string(),
                            p.get(1)?.as_str()?.to_string(),
                        ))
                    })
                    .collect(),
                None => Vec::new(),
            };
            json!({ "size": download::extract_full_size(arg_i("status"), &headers) })
        }
        "download.userMessage" => json!({ "message": download::user_message(arg_i("status")) }),
        "download.isTransient" => {
            json!({ "transient": download::is_transient_error(arg_i("code")) })
        }
        "download.retryDelay" => {
            let base = args
                .get("base")
                .and_then(Value::as_f64)
                .unwrap_or(download::BASE_RETRY_DELAY);
            json!({ "delay": download::retry_delay_seconds(arg_i("attempts"), base) })
        }
        "download.transientDecision" => {
            let decision = download::transient_decision(
                args.get("isNetworkAvailable")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                arg_i("attempts"),
                args.get("maxAttempts")
                    .and_then(Value::as_i64)
                    .unwrap_or(download::MAX_RETRY_ATTEMPTS),
            );
            match decision {
                download::TransientDecision::Retry => json!({ "outcome": "retry" }),
                download::TransientDecision::Park {
                    resume_on_reconnect,
                } => {
                    json!({ "outcome": "park", "resumeOnReconnect": resume_on_reconnect })
                }
            }
        }
        "download.refreshProgress" => {
            let s = download::refresh_progress(
                arg_i("totalUnitCount"),
                arg_i("completedFilesBytes"),
                arg_i("activeBytes"),
                arg_i("expectedActiveBytes"),
            );
            json!({
                "totalUnitCount": s.total_unit_count,
                "completedUnitCount": s.completed_unit_count,
            })
        }
        "download.rate" => {
            let samples: Vec<(f64, i64)> = match args.get("samples").and_then(Value::as_array) {
                Some(a) => a
                    .iter()
                    .filter_map(|s| {
                        let s = s.as_array()?;
                        Some((s.first()?.as_f64()?, s.get(1)?.as_i64()?))
                    })
                    .collect(),
                None => Vec::new(),
            };
            let window = args.get("window").and_then(Value::as_f64).unwrap_or(5.0); // ActiveDownload.rateWindow
            json!({ "rate": download::rate_bytes_per_second(&samples, window) })
        }
        "download.errorMessage" => {
            let e = if arg_str("kind") == "notCompatible" {
                download::DownloadError::NotCompatible(arg_str("reason").to_string())
            } else {
                download::DownloadError::NotEnoughDiskSpace {
                    required: arg_str("required").to_string(),
                    available: arg_str("available").to_string(),
                }
            };
            json!({
                "description": e.description(),
                "recovery": e.recovery_suggestion(),
            })
        }
        "download.verifyHash" => json!({
            "matches": download::expected_hash_matches(arg_str("computed"), arg_str("expected"))
        }),
        "download.isTooSmall" => json!({ "tooSmall": download::is_too_small(arg_i("size")) }),

        // ---- Fase 5: partial file writer (streaming + resume re-hash) ----
        "partial.sha256" => {
            json!({ "hex": partial_writer::sha256_hex(arg_str("text").as_bytes()) })
        }
        "partial.feedHash" => match partial_writer::feed_sha256(arg_str("path")) {
            Ok(digest) => json!({ "digest": digest }),
            Err(e) => json!({ "error": e.kind(), "message": e.message() }),
        },
        "partial.open" => {
            let expected = args
                .get("expectedBlobHash")
                .and_then(Value::as_str)
                .map(str::to_string);
            match partial_writer::PartialWriter::open(arg_str("path"), expected) {
                Ok(w) => {
                    let bytes = w.bytes_written();
                    let digest = w.digest_hex();
                    let handle = next_partial_handle();
                    partial_writers()
                        .lock()
                        .expect("partial writer registry poisoned")
                        .insert(handle, w);
                    json!({
                        "handle": handle,
                        "bytesWritten": bytes,
                        "digest": digest,
                    })
                }
                Err(e) => json!({ "error": e.kind(), "message": e.message() }),
            }
        }
        "partial.append" => {
            let handle = arg_i("handle") as u64;
            let mut locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get_mut(&handle) {
                Some(w) => match w.append(arg_str("text").as_bytes()) {
                    Ok(()) => json!({
                        "ok": true,
                        "bytesWritten": w.bytes_written(),
                        "digest": w.digest_hex(),
                    }),
                    Err(e) => json!({ "error": e.kind(), "message": e.message() }),
                },
                None => json!({
                    "error": "notFound",
                    "message": format!("no partial writer for handle {handle}"),
                }),
            }
        }
        "partial.restart" => {
            let handle = arg_i("handle") as u64;
            let mut locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get_mut(&handle) {
                Some(w) => match w.restart() {
                    Ok(()) => json!({
                        "ok": true,
                        "bytesWritten": w.bytes_written(),
                        "digest": w.digest_hex(),
                    }),
                    Err(e) => json!({ "error": e.kind(), "message": e.message() }),
                },
                None => json!({
                    "error": "notFound",
                    "message": format!("no partial writer for handle {handle}"),
                }),
            }
        }
        "partial.setTotalExpected" => {
            let handle = arg_i("handle") as u64;
            let mut locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get_mut(&handle) {
                Some(w) => {
                    w.set_total_expected(arg_i("size"));
                    json!({ "ok": true, "totalExpected": w.total_expected() })
                }
                None => json!({
                    "error": "notFound",
                    "message": format!("no partial writer for handle {handle}"),
                }),
            }
        }
        "partial.digest" => {
            let handle = arg_i("handle") as u64;
            let locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get(&handle) {
                Some(w) => json!({ "digest": w.digest_hex() }),
                None => json!({
                    "error": "notFound",
                    "message": format!("no partial writer for handle {handle}"),
                }),
            }
        }
        "partial.close" => {
            let handle = arg_i("handle") as u64;
            partial_writers()
                .lock()
                .expect("partial writer registry poisoned")
                .remove(&handle);
            json!({ "ok": true })
        }

        // ---- Fase 6: Server + syscalls leaves ----
        "server.quote" => json!({ "quoted": server::shell_quote(arg_str("text")) }),
        "server.displayCommand" => {
            let env = pairs_from_value(args.get("env").unwrap_or(&Value::Null));
            let spec = server::LaunchSpec {
                executable_path: arg_str("executablePath").to_string(),
                arguments: strings_from_value(args.get("arguments").unwrap_or(&Value::Null)),
                extra_arg_count: args
                    .get("extraArgCount")
                    .and_then(Value::as_i64)
                    .unwrap_or(0) as usize,
                env,
            };
            json!({ "command": spec.display_command() })
        }
        "server.hostArgument" => {
            let build = args
                .get("versionBuild")
                .and_then(Value::as_i64)
                .map(|b| LlamaVersion { build: b });
            json!({
                "host": server::host_argument(arg_str("bindAddress"), build)
            })
        }
        "server.isTailscaleAddress" => json!({
            "isTailscale": server::is_tailscale_address(arg_str("address"))
        }),
        "server.pickIpAddress" => {
            let addrs = interface_adresses(args);
            json!({
                "address": server::pick_ip_address(&addrs, args.get("primary").and_then(Value::as_str))
            })
        }
        "server.tailscaleAddress" => {
            let addrs = interface_adresses(args);
            json!({ "address": server::tailscale_address(&addrs) })
        }
        "server.effectiveBindAddress" => json!({
            "address": server::effective_bind_address(
                args.get("configured").and_then(Value::as_str),
                &strings_from_value(args.get("localAddrs").unwrap_or(&Value::Null)),
            )
        }),
        "server.effectiveWebuiDirectory" => json!({
            "directory": server::effective_webui_directory(
                args.get("configured").and_then(Value::as_str),
            )
        }),
        "server.localHost" => json!({
            "host": server::local_host(args.get("effectiveBind").and_then(Value::as_str))
        }),
        "server.resolvedHost" => {
            let addrs = interface_adresses(args);
            json!({
                "host": server::resolved_host(
                    args.get("effectiveBind").and_then(Value::as_str),
                    &addrs,
                    args.get("primary").and_then(Value::as_str),
                )
            })
        }
        "server.fitTargetMb" => json!({
            "mb": server::fit_target_mb(
                arg_i("gpuWorkingSetMb"),
                arg_i("memoryMb"),
                args.get("useFullWorkingSet").and_then(Value::as_bool).unwrap_or(false),
            )
        }),
        "server.presetRejectionOption" => json!({
            "option": server::preset_rejection_option(arg_str("message"))
        }),
        "server.parseModelStatuses" => {
            match server::parse_model_statuses(arg_str("body").as_bytes()) {
                Some(statuses) => json!({
                    "statuses": statuses
                        .iter()
                        .map(|(id, status)| json!([id, status]))
                        .collect::<Vec<_>>(),
                }),
                None => json!({ "error": "malformed" }),
            }
        }
        "server.extractPids" => json!({
            "pids": server::extract_pids(arg_str("lsofOutput"))
        }),
        "server.isLlamaProcess" => json!({
            "isLlama": server::is_llama_process(arg_str("comm"))
        }),
        "server.lastPathComponent" => json!({
            "name": server::last_path_component(arg_str("path"))
        }),
        "server.buildLaunchSpec" => {
            let addrs = interface_adresses(args);
            let extra_args = strings_from_value(args.get("extraArgs").unwrap_or(&Value::Null));
            let version_build = args
                .get("currentVersionBuild")
                .and_then(Value::as_i64)
                .map(|b| LlamaVersion { build: b });
            let input = server::LaunchSpecInput {
                llama_path: args.get("llamaPath").and_then(Value::as_str),
                presets_path: arg_str("presetsPath"),
                log_file_path: arg_str("logFilePath"),
                hf_cache_directory: arg_str("hfCacheDirectory"),
                port: arg_i("port"),
                fit_target_mb: arg_i("fitTargetMb"),
                configured_bind_address: args.get("configuredBindAddress").and_then(Value::as_str),
                address_by_interface: &addrs,
                primary_interface: args.get("primaryInterface").and_then(Value::as_str),
                current_version: version_build,
                configured_webui_dir: args.get("configuredWebUiDirectory").and_then(Value::as_str),
                sleep_idle_seconds: args.get("sleepIdleSeconds").and_then(Value::as_i64),
                agent_mode: args
                    .get("agentMode")
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                extra_args: &extra_args,
            };
            match server::build_launch_spec(&input) {
                Some(spec) => json!({
                    "spec": {
                        "executablePath": spec.executable_path,
                        "arguments": spec.arguments,
                        "extraArgCount": spec.extra_arg_count,
                        "env": spec.env
                            .iter()
                            .map(|(k, v)| json!([k, v]))
                            .collect::<Vec<_>>(),
                    }
                }),
                None => json!({ "error": "noBinary" }),
            }
        }

        // Smoke test for the event channel.
        "emit.test" => {
            let payload = args.get("payload").cloned().unwrap_or(Value::Null);
            emit(&json!({ "event": "test", "payload": payload }));
            json!({ "ok": true })
        }

        other => json!({ "error": format!("unknown method: {other}") }),
    }
}
