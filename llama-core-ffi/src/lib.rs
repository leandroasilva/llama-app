//! C ABI for `llama-core`.
//!
//! Contract (see `MIGRACAO_RUST_REVIEW.md`):
//! - Rust allocates every returned string; Swift releases it with
//!   [`llama_core_free`].
//! - Inputs are UTF-8, NUL-terminated, and only need to live for the call.
//! - Every entry point catches Rust panics so a panic never crosses the FFI
//!   boundary (which would be undefined behavior).
//! - Complex payloads travel as JSON; this keeps the ABI small and stable.

use llama_core::config::core_config::CoreConfig;
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
            match serde_json::from_str(s) {
                Ok(v) => v,
                Err(_) => return -2,
            }
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
            match serde_json::from_str(args_s) {
                Ok(v) => v,
                Err(e) => {
                    return json!({
                        "error": "invalidArgs",
                        "message": format!("failed to parse args JSON: {e}"),
                    })
                }
            }
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

    match method.split('.').next().unwrap_or("") {
        "ping" => dispatch_ping(method, args),
        "version" => dispatch_version(method, args),
        "repo" => dispatch_repo(method, args),
        "quant" => dispatch_quant(method, args),
        "gguf" => dispatch_gguf(method, args),
        "hfCache" => dispatch_hf_cache(method, args),
        "models" => dispatch_models_scan(method, args),
        "draftHead" => dispatch_draft_head(method, args),
        "sidecar" => dispatch_sidecar(method, args),
        "modelId" => dispatch_model_id(method, args),
        "model" => dispatch_model(method, args),
        "contextTier" => dispatch_context_tier(method, args),
        "resolvedPaths" => dispatch_resolved_paths(method, args),
        "compat" => dispatch_compat(method, args),
        "ini" => dispatch_ini(method, args),
        "aliases" => dispatch_aliases(method, args),
        "net" => dispatch_net(method, args),
        "catalog" => dispatch_catalog(method, args),
        "hfRequest" => dispatch_hf_request(method, args),
        "engine" => dispatch_engine(method, args),
        "download" => dispatch_download(method, args),
        "partial" => dispatch_partial(method, args),
        "server" => dispatch_server(method, args),
        "emit" => dispatch_emit(method, args),
        "core" => dispatch_core(method, args),
        "event" => dispatch_event(method, args),
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn arg_str<'a>(args: &'a Value, key: &str) -> &'a str {
    args.get(key).and_then(Value::as_str).unwrap_or("")
}

fn arg_f(args: &Value, key: &str) -> f64 {
    args.get(key).and_then(Value::as_f64).unwrap_or(0.0)
}

fn arg_i(args: &Value, key: &str) -> i64 {
    args.get(key).and_then(Value::as_i64).unwrap_or(0)
}

fn dispatch_ping(method: &str, _args: &Value) -> Value {
    match method {
        "ping" => json!({ "pong": true }),
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_version(method: &str, args: &Value) -> Value {
    match method {
        "version.parse" => {
            json!({ "build": LlamaVersion::parse(arg_str(args, "output")).map(|v| v.build) })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_repo(method: &str, args: &Value) -> Value {
    match method {
        "repo.parse" => match HFRepoParser::parse(arg_str(args, "repoDir")) {
            Some((org, repo)) => json!({ "org": org, "repo": repo }),
            None => Value::Null,
        },
        "repo.parseQuant" => json!(HFRepoParser::parse_quant(arg_str(args, "filename"))),
        "repo.isSplitShard" => json!(HFRepoParser::is_split_shard(arg_str(args, "filename"))),
        "repo.isFirstShard" => json!(HFRepoParser::is_first_shard(arg_str(args, "filename"))),
        "repo.shardTotal" => json!(HFRepoParser::shard_total(arg_str(args, "filename"))),
        "repo.splitShardBaseName" => {
            json!(HFRepoParser::split_shard_base_name(arg_str(
                args, "filename"
            )))
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_quant(method: &str, args: &Value) -> Value {
    match method {
        "quant.parseLabel" => {
            json!({ "label": GGUFQuant::parse_label(arg_str(args, "path")) })
        }
        "quant.tagForPath" => {
            let requested = args.get("requested").and_then(Value::as_str);
            json!({ "tag": GGUFQuant::tag_for_path(arg_str(args, "path"), requested) })
        }
        "quant.quantBitsForPath" => {
            json!({ "bits": GGUFQuant::quant_bits_for_path(arg_str(args, "path")) })
        }
        "quant.quantBitsForTag" => {
            json!({ "bits": GGUFQuant::quant_bits_for_tag(arg_str(args, "tag")) })
        }
        "quant.canonicalTag" => {
            json!({ "tag": GGUFQuant::canonical_tag(arg_str(args, "label")) })
        }
        "quant.matches" => {
            json!({ "value": GGUFQuant::matches(arg_str(args, "a"), arg_str(args, "b")) })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_gguf(method: &str, args: &Value) -> Value {
    match method {
        "gguf.contextLength" => match GGUFMetadata::context_length(arg_str(args, "path")) {
            None => Value::Null,
            Some(length) => json!({ "parsed": true, "length": length }),
        },
        "gguf.hasEmbeddedMTPHead" => {
            match GGUFMetadata::has_embedded_mtp_head(arg_str(args, "path")) {
                None => Value::Null,
                Some(value) => json!({ "value": value }),
            }
        }
        "gguf.isDecisionModel" => match GGUFMetadata::is_decision_model(arg_str(args, "path")) {
            None => Value::Null,
            Some(value) => json!({ "value": value }),
        },
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_hf_cache(method: &str, args: &Value) -> Value {
    match method {
        "hfCache.repoDirName" => {
            json!({ "name": HFCache::repo_dir_name_from_url(arg_str(args, "url")) })
        }
        "hfCache.repoRelativePath" => {
            json!({ "path": HFCache::repo_relative_path_from_url(arg_str(args, "url")) })
        }
        "hfCache.partialDir" => json!({
            "path": HFCache::partial_dir(arg_str(args, "cacheDir"), arg_str(args, "modelId"))
                .to_string_lossy()
        }),
        "hfCache.partialPath" => json!({
            "path": HFCache::partial_path(arg_str(args, "cacheDir"), arg_str(args, "modelId"), arg_str(args, "filename"))
                .to_string_lossy()
        }),
        "hfCache.placeholderURL" => json!({
            "path": HFCache::placeholder_url(arg_str(args, "cacheDir"), arg_str(args, "modelId"))
                .to_string_lossy()
        }),
        "hfCache.removePartials" => {
            HFCache::remove_partials(arg_str(args, "cacheDir"), arg_str(args, "modelId"));
            json!({ "ok": true })
        }
        "hfCache.partialBytes" => json!({
            "bytes": HFCache::partial_bytes(arg_str(args, "cacheDir"), arg_str(args, "modelId"))
        }),
        "hfCache.scanPlaceholders" => {
            let paths: Vec<String> = HFCache::scan_placeholders(arg_str(args, "cacheDir"))
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
            HFCache::clean_installed_partials(arg_str(args, "cacheDir"), &ids);
            json!({ "ok": true })
        }
        "hfCache.snapshotFileExists" => json!({
            "value": HFCache::snapshot_file_exists(
                arg_str(args, "cacheDir"), arg_str(args, "repoDir"), arg_str(args, "filename")
            )
        }),
        "hfCache.writeBlobAndLink" => {
            match HFCache::write_blob_and_link(
                arg_str(args, "cacheDir"),
                arg_str(args, "repoDir"),
                arg_str(args, "commit"),
                arg_str(args, "blobHash"),
                arg_str(args, "filename"),
                arg_str(args, "tempFile"),
            ) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        "hfCache.deleteModelFiles" => {
            let paths = resolved_paths_from_value(args.get("paths").unwrap_or(&Value::Null));
            match HFCache::delete_model_files(
                arg_str(args, "cacheDir"),
                arg_str(args, "repoDir"),
                &paths,
            ) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.to_string() }),
            }
        }
        "hfCache.scanForSideloaded" => {
            let entries: Vec<Value> = HFCache::scan_for_sideloaded(arg_str(args, "cacheDir"))
                .iter()
                .map(sideloaded_entry_to_value)
                .collect();
            json!({ "entries": entries })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_models_scan(method: &str, args: &Value) -> Value {
    match method {
        "models.scan" => {
            let scan = llama_core::models::scan::scan(
                arg_str(args, "cacheDir"),
                arg_str(args, "memCacheDir"),
            );
            let models: Vec<Value> = scan
                .models
                .iter()
                .map(|m| {
                    let mut v = sideloaded_entry_to_value(&m.entry);
                    if let Some(obj) = v.as_object_mut() {
                        obj.insert(
                            "ctxBytesPer1kTokens".to_string(),
                            json!(m.ctx_bytes_per_1k_tokens),
                        );
                        obj.insert("residentBytes".to_string(), json!(m.resident_bytes));
                    }
                    v
                })
                .collect();
            let needs_profile: Vec<Value> = scan
                .needs_profile
                .iter()
                .map(|(id, path)| json!([id, path]))
                .collect();
            json!({ "models": models, "needsProfile": needs_profile })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_draft_head(method: &str, args: &Value) -> Value {
    match method {
        "draftHead.fromSidecarPath" => {
            json!({ "kind": DraftHead::from_sidecar_path(arg_str(args, "path")).map(DraftHead::as_str) })
        }
        "draftHead.describe" => match DraftHead::from_str_name(arg_str(args, "kind")) {
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
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_sidecar(method: &str, args: &Value) -> Value {
    match method {
        "sidecar.isMmproj" => json!({ "value": SidecarPicker::is_mmproj(arg_str(args, "path")) }),
        "sidecar.isDraftHead" => {
            json!({ "value": SidecarPicker::is_draft_head(arg_str(args, "path")) })
        }
        "sidecar.mmproj" => {
            let names = strings_from_value(args.get("names").unwrap_or(&Value::Null));
            json!({ "path": SidecarPicker::mmproj(&names, arg_str(args, "mainPath")) })
        }
        "sidecar.draftHead" => {
            let names = strings_from_value(args.get("names").unwrap_or(&Value::Null));
            let tag = args.get("tag").and_then(Value::as_str);
            json!({ "path": SidecarPicker::draft_head(&names, arg_str(args, "mainPath"), tag) })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_model_id(method: &str, args: &Value) -> Value {
    match method {
        "modelId.parse" => parsed_to_value(&ModelIdParser::parse(arg_str(args, "id"))),
        "modelId.plainName" => {
            let show_tags = args
                .get("showTags")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({ "name": ModelIdParser::parse(arg_str(args, "id")).plain_name(show_tags) })
        }
        "modelId.displayKey" => {
            json!({ "key": ModelIdParser::parse(arg_str(args, "id")).display_key() })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_model(method: &str, args: &Value) -> Value {
    match method {
        "model.makeId" => json!({
            "id": model::make_id(arg_str(args, "org"), arg_str(args, "repo"), arg_str(args, "tag"))
        }),
        "model.idBaseOf" => json!({ "base": model::id_base_of(arg_str(args, "id")) }),
        "model.sortKey" => json!({ "key": model::sort_key(arg_str(args, "id")) }),
        "model.displayOrder" => {
            let order = model::display_order(arg_str(args, "lhsId"), arg_str(args, "rhsId"));
            json!({ "order": order as i32 })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_context_tier(method: &str, _args: &Value) -> Value {
    match method {
        "contextTier.list" => {
            let tiers: Vec<Value> = ContextTier::ALL
                .iter()
                .map(|t| json!({ "rawValue": t.raw_value(), "shortLabel": t.short_label(), "label": t.label(), "suffix": t.suffix() }))
                .collect();
            Value::Array(tiers)
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_resolved_paths(method: &str, args: &Value) -> Value {
    match method {
        "resolvedPaths.allPaths" => {
            let rp = ResolvedPaths {
                model_file: arg_str(args, "modelFile").to_string(),
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
                hf_repo_dir_name: arg_str(args, "hfRepoDirName").to_string(),
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
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_compat(method: &str, args: &Value) -> Value {
    match method {
        "compat.deviceBudget" => {
            let use_full = args
                .get("useFullWorkingSet")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            json!({ "mb": llama_core::models::compat::device_memory_budget_mb(arg_f(args, "workingSetMb"), arg_f(args, "physicalMb"), use_full) })
        }
        "compat.fitTarget" => {
            json!({ "mb": llama_core::models::compat::fit_target_mb(arg_f(args, "workingSetMb"), arg_f(args, "budgetMb")) })
        }
        "compat.estimatedWeightFits" => {
            let bytes = args.get("bytes").and_then(Value::as_i64);
            json!({ "value": llama_core::models::compat::estimated_weight_fits(bytes, arg_f(args, "budgetMb")) })
        }
        "compat.info" => {
            let mem = ModelMem {
                ctx_window: arg_i(args, "ctxWindow"),
                file_size: arg_i(args, "fileSize"),
                ctx_bytes_per_1k_tokens: arg_i(args, "ctxBytesPer1kTokens"),
                resident_bytes: arg_i(args, "residentBytes"),
            };
            let info =
                mem.compatibility_info(arg_f(args, "ctxWindowTokens"), arg_f(args, "budgetMb"));
            json!({ "isCompatible": info.is_compatible, "incompatibilitySummary": info.incompatibility_summary })
        }
        "compat.runtimeMemoryMb" => {
            let mem = ModelMem {
                ctx_window: arg_i(args, "ctxWindow"),
                file_size: arg_i(args, "fileSize"),
                ctx_bytes_per_1k_tokens: arg_i(args, "ctxBytesPer1kTokens"),
                resident_bytes: arg_i(args, "residentBytes"),
            };
            json!({ "mb": mem.runtime_memory_usage_mb(arg_f(args, "ctxWindowTokens")) })
        }
        "compat.supportedTiers" => {
            let mem = ModelMem {
                ctx_window: arg_i(args, "ctxWindow"),
                file_size: arg_i(args, "fileSize"),
                ctx_bytes_per_1k_tokens: arg_i(args, "ctxBytesPer1kTokens"),
                resident_bytes: arg_i(args, "residentBytes"),
            };
            json!({ "tiers": tiers_to_value(&mem.supported_context_tiers(arg_f(args, "budgetMb"))) })
        }
        "compat.displayTiers" => {
            let mem = ModelMem {
                ctx_window: arg_i(args, "ctxWindow"),
                file_size: arg_i(args, "fileSize"),
                ctx_bytes_per_1k_tokens: arg_i(args, "ctxBytesPer1kTokens"),
                resident_bytes: arg_i(args, "residentBytes"),
            };
            json!({ "tiers": tiers_to_value(&mem.display_context_tiers()) })
        }
        "compat.nativeMaxTier" => {
            let mem = ModelMem {
                ctx_window: arg_i(args, "ctxWindow"),
                file_size: 0,
                ctx_bytes_per_1k_tokens: 0,
                resident_bytes: 0,
            };
            json!({ "tier": mem.native_max_tier().map(|t| t.raw_value()) })
        }
        "compat.effectiveTier" => {
            let mem = ModelMem {
                ctx_window: arg_i(args, "ctxWindow"),
                file_size: arg_i(args, "fileSize"),
                ctx_bytes_per_1k_tokens: arg_i(args, "ctxBytesPer1kTokens"),
                resident_bytes: arg_i(args, "residentBytes"),
            };
            let selected = args
                .get("selected")
                .and_then(Value::as_i64)
                .and_then(ContextTier::from_raw);
            json!({ "tier": mem.effective_ctx_tier(arg_f(args, "budgetMb"), selected).map(|t| t.raw_value()) })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_ini(method: &str, args: &Value) -> Value {
    match method {
        "ini.parse" => {
            let sections = ini::parse(arg_str(args, "text"));
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
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_aliases(method: &str, args: &Value) -> Value {
    match method {
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
            json!({ "error": aliases::validation_error(arg_str(args, "name"), &aliases, replacing.as_ref(), &override_names) })
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
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_net(method: &str, args: &Value) -> Value {
    match method {
        "net.resolveFromSiblings" => {
            let requested = args.get("requestedQuant").and_then(Value::as_str);
            let siblings = siblings_from_value(args.get("siblings").unwrap_or(&Value::Null));
            match resolve_from_siblings(
                arg_str(args, "repo"),
                requested,
                &siblings,
                arg_f(args, "budgetMb"),
            ) {
                Ok(r) => resolved_to_value(&r),
                Err(e) => resolve_error_to_value(&e),
            }
        }
        "net.resolveUrl" => {
            json!({ "url": resolve_url(arg_str(args, "repo"), arg_str(args, "path")) })
        }
        "net.blobHash" => json!({ "hash": blob_hash(&response_from_value(args)) }),
        "net.commitHash" => json!({ "hash": commit_hash(&response_from_value(args)) }),
        "net.fileMetadata" => {
            let meta = file_metadata(&response_from_value(args))
                .map(|m| json!({ "blobHash": m.blob_hash, "commitHash": m.commit_hash }));
            json!({ "metadata": meta })
        }
        "net.sameHostRedirect" => json!({
            "value": same_host_redirect(args.get("from").and_then(Value::as_str), args.get("to").and_then(Value::as_str))
        }),
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_catalog(method: &str, args: &Value) -> Value {
    match method {
        "catalog.parseBytes" => {
            let label = args.get("label").and_then(Value::as_str);
            json!({ "bytes": catalog::parse_bytes(label) })
        }
        "catalog.featuredSuggestions" => {
            let families = families_from_value(args.get("families").unwrap_or(&Value::Null));
            let suggestions = catalog::featured_suggestions(
                &families,
                arg_f(args, "budgetMb"),
                arg_i(args, "systemMemoryMb") as u64,
            );
            json!({ "suggestions": suggestions_to_value(&suggestions) })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_hf_request(method: &str, args: &Value) -> Value {
    match method {
        "hfRequest.userAgent" => json!({
            "ua": hf_request::user_agent(arg_str(args, "shortVersion"), arg_i(args, "osMajor"), arg_i(args, "osMinor"))
        }),
        "hfRequest.make" => {
            let token = args.get("token").and_then(Value::as_str);
            let headers: Vec<Value> =
                hf_request::make(arg_str(args, "url"), token, arg_str(args, "userAgent"))
                    .iter()
                    .map(|(k, v)| json!([k, v]))
                    .collect();
            json!({ "headers": headers })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_engine(method: &str, args: &Value) -> Value {
    match method {
        "engine.managedPath" => json!({ "path": binaries::managed_path(arg_str(args, "home")) }),
        "engine.metalConfig" => match installer::metal_config(arg_str(args, "brand")) {
            Ok(config) => json!({ "config": config }),
            Err(e) => json!({ "error": e.kind(), "message": e.message() }),
        },
        "engine.artifactUrl" => {
            json!({ "url": installer::artifact_url(arg_str(args, "version"), arg_str(args, "config")) })
        }
        "engine.unzstdUrl" => json!({ "url": installer::unzstd_url(arg_str(args, "version")) }),
        "engine.binariesResolve" => {
            let unmanaged_dirs: Vec<String> =
                match args.get("unmanagedDirs").and_then(Value::as_array) {
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
            match binaries::resolve(arg_str(args, "home"), &unmanaged_dirs, ignore) {
                Some((path, origin)) => json!({ "origin": origin.as_str(), "path": path }),
                None => Value::Null,
            }
        }
        "engine.readVersion" => {
            json!({ "build": binaries::read_version(arg_str(args, "path")).map(|v| v.build) })
        }
        "engine.decompress" => match installer::decompress(
            arg_str(args, "unzstd"),
            arg_str(args, "zst"),
            arg_str(args, "dest"),
        ) {
            Ok(()) => json!({ "ok": true }),
            Err(e) => json!({ "error": e.kind(), "message": e.message() }),
        },
        "engine.atomicInstall" => {
            match installer::atomic_install(arg_str(args, "src"), arg_str(args, "dest")) {
                Ok(()) => json!({ "ok": true }),
                Err(e) => json!({ "error": e.kind(), "message": e.message() }),
            }
        }
        "engine.ensurePathSymlink" => {
            match installer::ensure_path_symlink(arg_str(args, "home"), arg_str(args, "llamaPath"))
            {
                installer::PathSymlinkOutcome::Linked => json!({ "outcome": "linked" }),
                installer::PathSymlinkOutcome::LeftExisting => json!({ "outcome": "leftExisting" }),
                installer::PathSymlinkOutcome::Failed(message) => {
                    json!({ "outcome": "failed", "message": message })
                }
            }
        }
        "engine.promoteStaged" => {
            let target = LlamaVersion {
                build: arg_i(args, "targetBuild"),
            };
            match installer::promote_staged(
                arg_str(args, "staged"),
                arg_str(args, "live"),
                target,
                arg_str(args, "home"),
            ) {
                installer::PromoteOutcome::NothingStaged => json!({ "outcome": "nothingStaged" }),
                installer::PromoteOutcome::DiscardedStale => json!({ "outcome": "discardedStale" }),
                installer::PromoteOutcome::Failed(message) => {
                    json!({ "outcome": "failed", "message": message })
                }
                installer::PromoteOutcome::Promoted => json!({ "outcome": "promoted" }),
            }
        }
        "engine.fit" => match mem_profile::fit(
            arg_i(args, "loMib"),
            arg_i(args, "hiMib"),
            arg_i(args, "ctxLo"),
            arg_i(args, "ctxHi"),
        ) {
            Some(p) => {
                json!({ "profile": { "ctxBytesPer1kTokens": p.ctx_bytes_per_1k_tokens, "residentBytes": p.resident_bytes, "schemaVersion": p.schema_version } })
            }
            None => Value::Null,
        },
        "engine.parseFitp" => {
            json!({ "sum": mem_profile::parse_fitp_output(arg_str(args, "output")) })
        }
        "engine.memCacheFileName" => {
            json!({ "hash": mem_profile::cache_file_name(arg_str(args, "modelId")) })
        }
        "engine.memCachePath" => {
            json!({ "path": mem_profile::cache_path(arg_str(args, "cacheDir"), arg_str(args, "modelId")) })
        }
        "engine.memCacheGet" => {
            match mem_profile::cache_get(arg_str(args, "cacheDir"), arg_str(args, "modelId")) {
                Some(p) => {
                    json!({ "profile": { "ctxBytesPer1kTokens": p.ctx_bytes_per_1k_tokens, "residentBytes": p.resident_bytes, "schemaVersion": p.schema_version } })
                }
                None => json!({ "profile": Value::Null }),
            }
        }
        "engine.memCacheSet" => {
            let profile = mem_profile::MemProfile::new(
                arg_i(args, "ctxBytesPer1kTokens"),
                arg_i(args, "residentBytes"),
            );
            match mem_profile::cache_set(
                arg_str(args, "cacheDir"),
                arg_str(args, "modelId"),
                &profile,
            ) {
                Ok(()) => json!({ "ok": true }),
                Err(message) => json!({ "error": message }),
            }
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_download(method: &str, args: &Value) -> Value {
    match method {
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
            json!({ "size": download::extract_full_size(arg_i(args, "status"), &headers) })
        }
        "download.userMessage" => {
            json!({ "message": download::user_message(arg_i(args, "status")) })
        }
        "download.isTransient" => {
            json!({ "transient": download::is_transient_error(arg_i(args, "code")) })
        }
        "download.retryDelay" => {
            let base = args
                .get("base")
                .and_then(Value::as_f64)
                .unwrap_or(download::BASE_RETRY_DELAY);
            json!({ "delay": download::retry_delay_seconds(arg_i(args, "attempts"), base) })
        }
        "download.transientDecision" => {
            let decision = download::transient_decision(
                args.get("isNetworkAvailable")
                    .and_then(Value::as_bool)
                    .unwrap_or(true),
                arg_i(args, "attempts"),
                args.get("maxAttempts")
                    .and_then(Value::as_i64)
                    .unwrap_or(download::MAX_RETRY_ATTEMPTS),
            );
            match decision {
                download::TransientDecision::Retry => json!({ "outcome": "retry" }),
                download::TransientDecision::Park {
                    resume_on_reconnect,
                } => json!({ "outcome": "park", "resumeOnReconnect": resume_on_reconnect }),
            }
        }
        "download.refreshProgress" => {
            let s = download::refresh_progress(
                arg_i(args, "totalUnitCount"),
                arg_i(args, "completedFilesBytes"),
                arg_i(args, "activeBytes"),
                arg_i(args, "expectedActiveBytes"),
            );
            json!({ "totalUnitCount": s.total_unit_count, "completedUnitCount": s.completed_unit_count })
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
            let window = args.get("window").and_then(Value::as_f64).unwrap_or(5.0);
            json!({ "rate": download::rate_bytes_per_second(&samples, window) })
        }
        "download.errorMessage" => {
            let e = if arg_str(args, "kind") == "notCompatible" {
                download::DownloadError::NotCompatible(arg_str(args, "reason").to_string())
            } else {
                download::DownloadError::NotEnoughDiskSpace {
                    required: arg_str(args, "required").to_string(),
                    available: arg_str(args, "available").to_string(),
                }
            };
            json!({ "description": e.description(), "recovery": e.recovery_suggestion() })
        }
        "download.verifyHash" => {
            json!({ "matches": download::expected_hash_matches(arg_str(args, "computed"), arg_str(args, "expected")) })
        }
        "download.isTooSmall" => json!({ "tooSmall": download::is_too_small(arg_i(args, "size")) }),
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_partial(method: &str, args: &Value) -> Value {
    match method {
        "partial.sha256" => {
            json!({ "hex": partial_writer::sha256_hex(arg_str(args, "text").as_bytes()) })
        }
        "partial.feedHash" => match partial_writer::feed_sha256(arg_str(args, "path")) {
            Ok(digest) => json!({ "digest": digest }),
            Err(e) => json!({ "error": e.kind(), "message": e.message() }),
        },
        "partial.open" => {
            let expected = args
                .get("expectedBlobHash")
                .and_then(Value::as_str)
                .map(str::to_string);
            match partial_writer::PartialWriter::open(arg_str(args, "path"), expected) {
                Ok(w) => {
                    let bytes = w.bytes_written();
                    let digest = w.digest_hex();
                    let handle = next_partial_handle();
                    partial_writers()
                        .lock()
                        .expect("partial writer registry poisoned")
                        .insert(handle, w);
                    json!({ "handle": handle, "bytesWritten": bytes, "digest": digest })
                }
                Err(e) => json!({ "error": e.kind(), "message": e.message() }),
            }
        }
        "partial.append" => {
            let handle = arg_i(args, "handle") as u64;
            let mut locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get_mut(&handle) {
                Some(w) => match w.append(arg_str(args, "text").as_bytes()) {
                    Ok(()) => {
                        json!({ "ok": true, "bytesWritten": w.bytes_written(), "digest": w.digest_hex() })
                    }
                    Err(e) => json!({ "error": e.kind(), "message": e.message() }),
                },
                None => {
                    json!({ "error": "notFound", "message": format!("no partial writer for handle {handle}") })
                }
            }
        }
        "partial.restart" => {
            let handle = arg_i(args, "handle") as u64;
            let mut locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get_mut(&handle) {
                Some(w) => match w.restart() {
                    Ok(()) => {
                        json!({ "ok": true, "bytesWritten": w.bytes_written(), "digest": w.digest_hex() })
                    }
                    Err(e) => json!({ "error": e.kind(), "message": e.message() }),
                },
                None => {
                    json!({ "error": "notFound", "message": format!("no partial writer for handle {handle}") })
                }
            }
        }
        "partial.setTotalExpected" => {
            let handle = arg_i(args, "handle") as u64;
            let mut locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get_mut(&handle) {
                Some(w) => {
                    w.set_total_expected(arg_i(args, "size"));
                    json!({ "ok": true, "totalExpected": w.total_expected() })
                }
                None => {
                    json!({ "error": "notFound", "message": format!("no partial writer for handle {handle}") })
                }
            }
        }
        "partial.digest" => {
            let handle = arg_i(args, "handle") as u64;
            let locked = partial_writers()
                .lock()
                .expect("partial writer registry poisoned");
            match locked.get(&handle) {
                Some(w) => json!({ "digest": w.digest_hex() }),
                None => {
                    json!({ "error": "notFound", "message": format!("no partial writer for handle {handle}") })
                }
            }
        }
        "partial.close" => {
            let handle = arg_i(args, "handle") as u64;
            partial_writers()
                .lock()
                .expect("partial writer registry poisoned")
                .remove(&handle);
            json!({ "ok": true })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_server(method: &str, args: &Value) -> Value {
    match method {
        "server.quote" => json!({ "quoted": server::shell_quote(arg_str(args, "text")) }),
        "server.displayCommand" => {
            let env = pairs_from_value(args.get("env").unwrap_or(&Value::Null));
            let spec = server::LaunchSpec {
                executable_path: arg_str(args, "executablePath").to_string(),
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
            json!({ "host": server::host_argument(arg_str(args, "bindAddress"), build) })
        }
        "server.isTailscaleAddress" => {
            json!({ "isTailscale": server::is_tailscale_address(arg_str(args, "address")) })
        }
        "server.pickIpAddress" => {
            let addrs = interface_adresses(args);
            json!({ "address": server::pick_ip_address(&addrs, args.get("primary").and_then(Value::as_str)) })
        }
        "server.tailscaleAddress" => {
            let addrs = interface_adresses(args);
            json!({ "address": server::tailscale_address(&addrs) })
        }
        "server.effectiveBindAddress" => json!({
            "address": server::effective_bind_address(args.get("configured").and_then(Value::as_str), &strings_from_value(args.get("localAddrs").unwrap_or(&Value::Null)))
        }),
        "server.effectiveWebuiDirectory" => json!({
            "directory": server::effective_webui_directory(args.get("configured").and_then(Value::as_str))
        }),
        "server.localHost" => {
            json!({ "host": server::local_host(args.get("effectiveBind").and_then(Value::as_str)) })
        }
        "server.resolvedHost" => {
            let addrs = interface_adresses(args);
            json!({ "host": server::resolved_host(args.get("effectiveBind").and_then(Value::as_str), &addrs, args.get("primary").and_then(Value::as_str)) })
        }
        "server.fitTargetMb" => json!({
            "mb": server::fit_target_mb(arg_i(args, "gpuWorkingSetMb"), arg_i(args, "memoryMb"), args.get("useFullWorkingSet").and_then(Value::as_bool).unwrap_or(false))
        }),
        "server.presetRejectionOption" => {
            json!({ "option": server::preset_rejection_option(arg_str(args, "message")) })
        }
        "server.parseModelStatuses" => {
            match server::parse_model_statuses(arg_str(args, "body").as_bytes()) {
                Some(statuses) => {
                    json!({ "statuses": statuses.iter().map(|(id, status)| json!([id, status])).collect::<Vec<_>>() })
                }
                None => json!({ "error": "malformed" }),
            }
        }
        "server.extractPids" => {
            json!({ "pids": server::extract_pids(arg_str(args, "lsofOutput")) })
        }
        "server.isLlamaProcess" => {
            json!({ "isLlama": server::is_llama_process(arg_str(args, "comm")) })
        }
        "server.lastPathComponent" => {
            json!({ "name": server::last_path_component(arg_str(args, "path")) })
        }
        "server.buildLaunchSpec" => {
            let addrs = interface_adresses(args);
            let extra_args = strings_from_value(args.get("extraArgs").unwrap_or(&Value::Null));
            let version_build = args
                .get("currentVersionBuild")
                .and_then(Value::as_i64)
                .map(|b| LlamaVersion { build: b });
            let input = server::LaunchSpecInput {
                llama_path: args.get("llamaPath").and_then(Value::as_str),
                presets_path: arg_str(args, "presetsPath"),
                log_file_path: arg_str(args, "logFilePath"),
                hf_cache_directory: arg_str(args, "hfCacheDirectory"),
                port: arg_i(args, "port"),
                fit_target_mb: arg_i(args, "fitTargetMb"),
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
                Some(spec) => {
                    json!({ "spec": { "executablePath": spec.executable_path, "arguments": spec.arguments, "extraArgCount": spec.extra_arg_count, "env": spec.env.iter().map(|(k, v)| json!([k, v])).collect::<Vec<_>>() } })
                }
                None => json!({ "error": "noBinary" }),
            }
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_emit(method: &str, args: &Value) -> Value {
    match method {
        "emit.test" => {
            let payload = args.get("payload").cloned().unwrap_or(Value::Null);
            emit(&json!({ "event": "test", "payload": payload }));
            json!({ "ok": true })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_core(method: &str, _args: &Value) -> Value {
    match method {
        "core.snapshot" => {
            let cfg = core()
                .lock()
                .ok()
                .map(|c| CoreConfig::from_value(&c.config))
                .unwrap_or_default();
            json!({ "schemaVersion": llama_core::config::CORE_CONFIG_SCHEMA_VERSION, "config": cfg.to_value() })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}

fn dispatch_event(method: &str, args: &Value) -> Value {
    match method {
        "event.emit" => {
            let kind = arg_str(args, "kind");
            let payload = args.get("payload").cloned().unwrap_or(Value::Null);
            emit(&llama_core::events::event_value(kind, payload));
            json!({ "ok": true, "known": llama_core::events::is_known_event(kind) })
        }
        _ => json!({ "error": format!("unknown method: {method}") }),
    }
}
