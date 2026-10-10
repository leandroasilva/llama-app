//! Fase 6 equivalence harness (Server + syscalls leaves). Shares
//! `Fixtures/server.json` with the Swift `LlamaTests` target. Filesystem state
//! the leaves read (an `index.html` next to a custom web UI folder) is staged
//! in temp dirs under the fixture's `<TMP>` placeholder, and both test suites
//! substitute the same token -- so the expected values stay in one place.

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::Value;

use llama_core::engine::server::{
    build_launch_spec, effective_bind_address, effective_webui_directory, extract_pids,
    fit_target_mb, host_argument, is_llama_process, is_tailscale_address, last_path_component,
    local_host, parse_model_statuses, pick_ip_address, preset_rejection_option, resolved_host,
    shell_quote, tailscale_address, LaunchSpec, LaunchSpecInput,
};
use llama_core::engine::version::LlamaVersion;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Fixtures")
}

fn load(name: &str) -> Value {
    let path = fixtures_dir().join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

/// Substitutes the fixture `<TMP>` placeholder with a scratch dir that the test
/// stages files into, mirroring what the Swift test does with its temp dir.
fn fixture_dir() -> PathBuf {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let n = SEQ.fetch_add(1, Ordering::SeqCst);
    let path = std::env::temp_dir().join(format!("llama-core-phase6-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}

fn normalize(value: &Value, tmp: &str) -> String {
    value.as_str().unwrap().replacen("<TMP>", tmp, usize::MAX)
}

fn pids(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|n| n.as_i64().unwrap())
        .collect()
}

fn pairs(v: &Value) -> Vec<(String, String)> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            (
                a[0].as_str().unwrap().to_string(),
                a[1].as_str().unwrap().to_string(),
            )
        })
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap().to_string())
        .collect()
}

fn version(build: &Value) -> Option<LlamaVersion> {
    build.as_i64().map(|b| LlamaVersion { build: b })
}

#[test]
fn quote_fixture() {
    let f = load("server.json");
    for c in f["quoteCases"].as_array().unwrap() {
        assert_eq!(
            shell_quote(c["text"].as_str().unwrap()),
            c["quoted"].as_str().unwrap(),
            "quote {:?}",
            c["name"]
        );
    }
}

#[test]
fn display_command_fixture() {
    let f = load("server.json");
    for c in f["displayCommandCases"].as_array().unwrap() {
        let spec = LaunchSpec {
            executable_path: c["executablePath"].as_str().unwrap().to_string(),
            arguments: strings(&c["arguments"]),
            extra_arg_count: c["extraArgCount"].as_u64().unwrap() as usize,
            env: pairs(&c["env"]),
        };
        assert_eq!(
            spec.display_command(),
            c["command"].as_str().unwrap(),
            "display {:?}",
            c["name"]
        );
    }
}

#[test]
fn host_argument_fixture() {
    let f = load("server.json");
    for c in f["hostArgumentCases"].as_array().unwrap() {
        assert_eq!(
            host_argument(
                c["bindAddress"].as_str().unwrap(),
                version(&c["versionBuild"])
            ),
            c["expected"].as_str().unwrap(),
            "host {:?}",
            c["name"]
        );
    }
}

#[test]
fn tailscale_fixture() {
    let f = load("server.json");
    for c in f["tailscaleCases"].as_array().unwrap() {
        assert_eq!(
            is_tailscale_address(c["ip"].as_str().unwrap()),
            c["expected"].as_bool().unwrap(),
            "tailscale {:?}",
            c["name"]
        );
    }
    for c in f["tailscaleAddressCases"].as_array().unwrap() {
        let addrs = pairs(&c["addresses"]);
        assert_eq!(
            tailscale_address(&addrs),
            c["expected"].as_str().map(str::to_string),
            "tailscaleAddr {:?}",
            c["name"]
        );
    }
}

#[test]
fn ip_pick_and_hosts_fixture() {
    let f = load("server.json");
    for c in f["pickIpCases"].as_array().unwrap() {
        let addrs = pairs(&c["addresses"]);
        assert_eq!(
            pick_ip_address(&addrs, c["primary"].as_str()),
            c["expected"].as_str().map(str::to_string),
            "pickIp {:?}",
            c["name"]
        );
    }
    for c in f["localHostCases"].as_array().unwrap() {
        assert_eq!(
            local_host(c["effectiveBind"].as_str()),
            c["expected"].as_str().unwrap(),
            "localHost {:?}",
            c["name"]
        );
    }
    for c in f["resolvedHostCases"].as_array().unwrap() {
        let addrs = pairs(&c["addresses"]);
        assert_eq!(
            resolved_host(c["effectiveBind"].as_str(), &addrs, c["primary"].as_str()),
            c["expected"].as_str().unwrap(),
            "resolvedHost {:?}",
            c["name"]
        );
    }
}

#[test]
fn effective_and_fit_fixture() {
    let f = load("server.json");
    let tmp = fixture_dir();

    for c in f["effectiveBindCases"].as_array().unwrap() {
        let addrs = strings(&c["localAddrs"]);
        assert_eq!(
            effective_bind_address(c["configured"].as_str(), &addrs),
            c["expected"].as_str().map(str::to_string),
            "effectiveBind {:?}",
            c["name"]
        );
    }

    for c in f["effectiveWebuiCases"].as_array().unwrap() {
        let dir = normalize(&c["dir"], tmp.to_str().unwrap());
        std::fs::create_dir_all(&dir).unwrap();
        if c["indexed"].as_bool().unwrap() {
            std::fs::write(
                std::path::Path::new(&dir).join("index.html"),
                "<html></html>",
            )
            .unwrap();
        }
        assert_eq!(
            effective_webui_directory(Some(&dir)),
            c["expected"].as_str().map(|_| dir.clone()),
            "effectiveWebui {:?}",
            c["name"]
        );
    }

    for c in f["fitTargetCases"].as_array().unwrap() {
        assert_eq!(
            fit_target_mb(
                c["gpuWorkingSetMb"].as_i64().unwrap(),
                c["memoryMb"].as_i64().unwrap(),
                c["useFullWorkingSet"].as_bool().unwrap(),
            ),
            c["expected"].as_i64().unwrap(),
            "fitTarget {:?}",
            c["name"]
        );
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[test]
fn preset_and_status_fixture() {
    let f = load("server.json");
    for c in f["presetRejectionCases"].as_array().unwrap() {
        assert_eq!(
            preset_rejection_option(c["message"].as_str().unwrap()),
            c["expected"].as_str().map(str::to_string),
            "presetRejection {:?}",
            c["name"]
        );
    }
    for c in f["modelsStatusCases"].as_array().unwrap() {
        let got = parse_model_statuses(c["body"].as_str().unwrap().as_bytes());
        let expected = c["statuses"].as_array().map(|list| {
            list.iter()
                .map(|p| {
                    let a = p.as_array().unwrap();
                    (
                        a[0].as_str().unwrap().to_string(),
                        a[1].as_str().unwrap().to_string(),
                    )
                })
                .collect::<Vec<_>>()
        });
        assert_eq!(got, expected, "modelsStatus {:?}", c["name"]);
    }
}

#[test]
fn reclaim_fixture() {
    let f = load("server.json");
    for c in f["reclaimCases"].as_array().unwrap() {
        if c.get("lsofOutput").is_some() {
            assert_eq!(
                extract_pids(c["lsofOutput"].as_str().unwrap()),
                pids(&c["pids"]),
                "pids {:?}",
                c["name"]
            );
        }
        if let Some(comms) = c.get("comms") {
            for comm in comms.as_array().unwrap() {
                assert_eq!(
                    is_llama_process(comm["comm"].as_str().unwrap()),
                    comm["isLlama"].as_bool().unwrap(),
                    "comm {:?}",
                    c["name"]
                );
            }
        }
        if let Some(paths) = c.get("paths") {
            for path in paths.as_array().unwrap() {
                assert_eq!(
                    last_path_component(path["path"].as_str().unwrap()),
                    path["name"].as_str().unwrap(),
                    "path {:?}",
                    c["name"]
                );
            }
        }
    }
}

#[test]
fn build_launch_spec_fixture() {
    let f = load("server.json");
    let tmp = fixture_dir();
    let tmp = tmp.to_str().unwrap();

    for c in f["buildLaunchSpecCases"].as_array().unwrap() {
        // Stage the web UI folder every case points at.
        let mut staged: Vec<String> = Vec::new();
        let stage = |s: &str, tmp: &str, staged: &mut Vec<String>| {
            let absolute = s.replacen("<TMP>", tmp, usize::MAX);
            std::fs::create_dir_all(&absolute).unwrap();
            std::fs::write(
                std::path::Path::new(&absolute).join("index.html"),
                "<html></html>",
            )
            .unwrap();
            staged.push(absolute)
        };
        if let Some(dir) = c["configuredWebUiDirectory"].as_str() {
            stage(dir, tmp, &mut staged);
        }

        let has_llama = c["llamaPath"].is_string();
        let address_by_interface: Vec<(String, String)> = c["addresses"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let a = p.as_array().unwrap();
                (
                    a[0].as_str().unwrap().to_string(),
                    a[1].as_str().unwrap().to_string(),
                )
            })
            .collect();
        let extra_args: Vec<String> = c["extraArgs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap().to_string())
            .collect();
        let held: Vec<(String, String)> = [
            "llamaPath",
            "presetsPath",
            "logFilePath",
            "hfCacheDirectory",
            "configuredBindAddress",
            "configuredWebUiDirectory",
        ]
        .iter()
        .filter_map(|k| {
            c.get(*k)
                .and_then(Value::as_str)
                .map(|s| ((*k).to_string(), s.replacen("<TMP>", tmp, usize::MAX)))
        })
        .collect();
        let get = |k: &str| {
            held.iter()
                .find(|(key, _)| key == k)
                .map(|(_, value)| value.as_str())
        };

        let input = LaunchSpecInput {
            llama_path: get("llamaPath").filter(|_| has_llama),
            presets_path: get("presetsPath").unwrap(),
            log_file_path: get("logFilePath").unwrap(),
            hf_cache_directory: get("hfCacheDirectory").unwrap(),
            port: c["port"].as_i64().unwrap(),
            fit_target_mb: c["fitTargetMb"].as_i64().unwrap(),
            configured_bind_address: get("configuredBindAddress"),
            address_by_interface: &address_by_interface,
            primary_interface: c["primaryInterface"].as_str(),
            current_version: version(&c["versionBuild"]),
            configured_webui_dir: get("configuredWebUiDirectory"),
            sleep_idle_seconds: c["sleepIdleSeconds"].as_i64(),
            agent_mode: c["agentMode"].as_bool().unwrap(),
            extra_args: &extra_args,
        };

        let got = build_launch_spec(&input);
        let expected = c.get("spec").filter(|s| !s.is_null());
        match (got, expected) {
            (None, None) => {}
            (Some(spec), Some(expected)) => {
                assert_eq!(
                    spec.executable_path,
                    normalize(&expected["executablePath"], tmp),
                    "spec exe {:?}",
                    c["name"]
                );
                assert_eq!(
                    spec.arguments,
                    expected["arguments"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|s| s.as_str().unwrap().replacen("<TMP>", tmp, usize::MAX))
                        .collect::<Vec<_>>(),
                    "spec args {:?}",
                    c["name"]
                );
                assert_eq!(
                    spec.extra_arg_count,
                    expected["extraArgCount"].as_u64().unwrap() as usize,
                    "spec extra {:?}",
                    c["name"]
                );
                let expected_env: Vec<(String, String)> = expected["env"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| {
                        let a = p.as_array().unwrap();
                        (
                            a[0].as_str().unwrap().to_string(),
                            a[1].as_str().unwrap().replacen("<TMP>", tmp, usize::MAX),
                        )
                    })
                    .collect();
                assert_eq!(spec.env, expected_env, "spec env {:?}", c["name"]);
            }
            (Some(_), None) => panic!("case {:?}: Rust built a spec, fixture says none", c["name"]),
            (None, Some(_)) => panic!(
                "case {:?}: Rust returned none, fixture has a spec",
                c["name"]
            ),
        }
    }
}
