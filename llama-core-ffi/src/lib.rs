//! C ABI for `llama-core`.
//!
//! Contract (see `MIGRACAO_RUST_REVIEW.md`):
//! - Rust allocates every returned string; Swift releases it with
//!   [`llama_core_free`].
//! - Inputs are UTF-8, NUL-terminated, and only need to live for the call.
//! - Every entry point catches Rust panics so a panic never crosses the FFI
//!   boundary (which would be undefined behavior).
//! - Complex payloads travel as JSON; this keeps the ABI small and stable.

use llama_core::engine::LlamaVersion;
use llama_core::hf::HFRepoParser;
use serde_json::{json, Value};
use std::ffi::{c_char, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
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

fn dispatch(method: &str, args: &Value) -> Value {
    // Development-only escape hatch used by tests to prove that a panic is
    // caught at the boundary rather than unwinding into the host.
    #[cfg(debug_assertions)]
    if method == "panic.test" {
        panic!("intentional panic for tests");
    }

    let arg_str = |key: &str| args.get(key).and_then(Value::as_str).unwrap_or("");
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

        // Smoke test for the event channel.
        "emit.test" => {
            let payload = args.get("payload").cloned().unwrap_or(Value::Null);
            emit(&json!({ "event": "test", "payload": payload }));
            json!({ "ok": true })
        }

        other => json!({ "error": format!("unknown method: {other}") }),
    }
}
