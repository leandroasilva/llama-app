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
