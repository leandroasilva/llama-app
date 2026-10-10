//! Exercises the C ABI directly: JSON round-trip, config push, the event
//! channel, and panic containment.

use llama_core_ffi::{
    llama_core_call, llama_core_free, llama_core_init, llama_core_set_config,
    llama_core_set_event_cb,
};
use std::ffi::{c_char, CStr, CString};
use std::sync::{Mutex, OnceLock};

/// Returns the response body, or `None` if the call returned NULL (e.g. panic).
fn call(method: &str, args: &str) -> Option<String> {
    let m = CString::new(method).unwrap();
    let a = CString::new(args).unwrap();
    let p = unsafe { llama_core_call(m.as_ptr(), a.as_ptr()) };
    if p.is_null() {
        return None;
    }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { llama_core_free(p) };
    Some(s)
}

#[test]
fn dispatch_returns_json() {
    assert_eq!(call("ping", "{}").unwrap(), r#"{"pong":true}"#);
    assert_eq!(
        call("version.parse", r#"{"output":"b9370-aa50b2c2a"}"#).unwrap(),
        r#"{"build":9370}"#
    );
    assert_eq!(
        call("version.parse", r#"{"output":"garbage"}"#).unwrap(),
        r#"{"build":null}"#
    );
    assert_eq!(
        call("repo.parse", r#"{"repoDir":"models--org--repo--extra"}"#).unwrap(),
        r#"{"org":"org","repo":"repo--extra"}"#
    );
    assert_eq!(call("repo.parse", r#"{"repoDir":"nope"}"#).unwrap(), "null");
    assert_eq!(
        call(
            "repo.parseQuant",
            r#"{"filename":"Llama-3.2-1B-Q4_K_M.gguf"}"#
        )
        .unwrap(),
        r#""Q4_K_M""#
    );
    assert!(call("no.such.method", "{}")
        .unwrap()
        .contains("unknown method"));
}

#[test]
fn config_push_succeeds() {
    assert_eq!(unsafe { llama_core_init(std::ptr::null()) }, 0);
    let cfg = CString::new(r#"{"threads":4}"#).unwrap();
    assert_eq!(unsafe { llama_core_set_config(cfg.as_ptr()) }, 0);
    assert_eq!(unsafe { llama_core_init(cfg.as_ptr()) }, 0);
}

static EVENTS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();

extern "C" fn on_event(p: *const c_char) {
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    EVENTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .push(s);
}

#[test]
fn event_channel_delivers() {
    EVENTS
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .unwrap()
        .clear();
    unsafe { llama_core_set_event_cb(Some(on_event)) };
    assert_eq!(
        call("emit.test", r#"{"payload":{"n":1}}"#).unwrap(),
        r#"{"ok":true}"#
    );
    unsafe { llama_core_set_event_cb(None) };

    let events = EVENTS.get().unwrap().lock().unwrap();
    assert_eq!(events.len(), 1);
    assert!(events[0].contains(r#""event":"test""#));
    assert!(events[0].contains(r#""n":1"#));
}

#[test]
fn panic_is_contained() {
    assert_eq!(call("panic.test", "{}"), None);
    // The core is still usable after a contained panic.
    assert_eq!(call("ping", "{}").unwrap(), r#"{"pong":true}"#);
}

#[test]
fn fase1_quant_and_sidecar() {
    assert_eq!(
        call(
            "quant.tagForPath",
            r#"{"path":"unsloth/x-UD-Q4_K_XL.gguf"}"#
        )
        .unwrap(),
        r#"{"tag":"Q4_K_XL"}"#
    );
    assert_eq!(
        call("quant.parseLabel", r#"{"path":"x-Q4_0_4_4.gguf"}"#).unwrap(),
        r#"{"label":"Q4_0"}"#
    );
    assert_eq!(
        call("quant.parseLabel", r#"{"path":"noquant.gguf"}"#).unwrap(),
        r#"{"label":null}"#
    );
    assert_eq!(
        call("quant.quantBitsForPath", r#"{"path":"x-Q4_K_M.gguf"}"#).unwrap(),
        r#"{"bits":4}"#
    );
    assert_eq!(
        call("quant.canonicalTag", r#"{"label":"UD-Q4_K_XL"}"#).unwrap(),
        r#"{"tag":"Q4_K_XL"}"#
    );
    assert_eq!(
        call(
            "sidecar.draftHead",
            r#"{"names":["mtp-Q8_0.gguf","dflash-Q8_0.gguf"],"mainPath":"m-Q8_0.gguf","tag":"Q8_0"}"#
        )
        .unwrap(),
        r#"{"path":"dflash-Q8_0.gguf"}"#
    );
    assert_eq!(
        call("draftHead.fromSidecarPath", r#"{"path":"eagle3-x.gguf"}"#).unwrap(),
        r#"{"kind":null}"#
    );
}

#[test]
fn fase1_model_and_compat() {
    assert_eq!(
        call(
            "modelId.parse",
            r#"{"id":"ggml-org/gemma-3-4b-it-GGUF:Q4_K_M"}"#
        )
        .unwrap(),
        r#"{"displayOrg":null,"name":"gemma-3","org":"ggml-org","params":"4B","quant":"Q4_K_M","tags":["it"]}"#
    );
    assert_eq!(
        call(
            "modelId.plainName",
            r#"{"id":"unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M:Q4_K_M","showTags":false}"#
        )
        .unwrap(),
        r#"{"name":"unsloth/Qwen3-Coder 30B-A3B Q4_K_M"}"#
    );
    assert_eq!(
        call("model.makeId", r#"{"org":"a","repo":"b","tag":"Q4_K_M"}"#).unwrap(),
        r#"{"id":"a/b:Q4_K_M"}"#
    );
    assert_eq!(
        call(
            "compat.deviceBudget",
            r#"{"workingSetMb":24576,"physicalMb":32768,"useFullWorkingSet":false}"#
        )
        .unwrap(),
        r#"{"mb":23552.0}"#
    );
    assert!(call("contextTier.list", "{}").unwrap().starts_with(
        r#"[{"label":"4k ctx","rawValue":4096,"shortLabel":"4k","suffix":":4k ctx"}"#
    ));
}

#[test]
fn fase1_ini_and_aliases() {
    assert_eq!(
        call("ini.parse", r#"{"text":"[a]\nk = v\n\n[b]\nj = w\n"}"#).unwrap(),
        r#"{"sections":[{"name":"a","pairs":[{"key":"k","value":"v"}]},{"name":"b","pairs":[{"key":"j","value":"w"}]}]}"#
    );
    assert_eq!(
        call(
            "aliases.validationError",
            r#"{"name":"code","aliases":[{"name":"code","modelId":"m1"}],"replacing":null,"overrideNames":[]}"#
        )
        .unwrap(),
        r#"{"error":"There's already an alias named code."}"#
    );
    assert_eq!(
        call(
            "aliases.setAlias",
            r#"{"section":{"name":"M","pairs":[]},"names":["a","b"]}"#
        )
        .unwrap(),
        r#"{"section":{"name":"M","pairs":[{"key":"alias","value":"a,b"}]}}"#
    );
}
