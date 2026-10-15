//! Port of `Llama/Server` leaves: `LlamaServer.buildLaunchSpec` /
//! `LaunchSpec.displayCommand` / `hostArgument`, the bind/webui fallbacks, and
//! `LlamaServerAPI.fetchModelStatuses` parsing, plus `TailscaleInterface`,
//! `LocalNetwork.ipAddress`, and the pure parsers behind `reclaimPort`.
//!
//! The syscalls and orchestration stay in Swift -- getifaddrs, SCDynamicStore,
//! the ARP `sysctl`, `Process`/`URLSession`, `lsof`/`ps`/`kill`, and the
//! `socket` bind probe. The values they produce are *injected* as inputs here
//! (the same pattern as the SysCtl brand in Fase 4 and `DiskSpace` in Fase 2),
//! so every decision and string renderer is provable on both sides from the
//! shared fixture.

use super::binaries::multi_host_version;
use super::version::LlamaVersion;
use std::path::Path;

/// `LaunchSpec` from `LlamaServer`, minus the live `Process` state: the fully
/// resolved invocation plus the presentation-only `extraArgCount`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchSpec {
    pub executable_path: String,
    pub arguments: Vec<String>,
    pub extra_arg_count: usize,
    pub env: Vec<(String, String)>,
}

impl LaunchSpec {
    /// Renders the launch spec as a runnable shell command: env vars as bare
    /// `export` statements up top, then the binary/subcommand, then each
    /// `--flag` grouped with its value, hanging-indented. Mirrors
    /// `LaunchSpec.displayCommand`.
    pub fn display_command(&self) -> String {
        // Group app-managed args so a `--flag` carries its following value(s)
        // on one line; bare positionals (`serve`) stand alone.
        let managed = &self.arguments[..self.arguments.len().saturating_sub(self.extra_arg_count)];
        let mut lines: Vec<String> = Vec::new();
        let mut idx = 0;
        while idx < managed.len() {
            let arg = &managed[idx];
            let next = idx + 1;
            if arg.starts_with('-') && next < managed.len() && !managed[next].starts_with('-') {
                lines.push(format!("{} {}", arg, shell_quote(&managed[next])));
                idx = next + 1;
            } else {
                lines.push(shell_quote(arg));
                idx = next;
            }
        }

        // The binary leads; leading positionals (the subcommand) ride that line.
        let mut first_line = shell_quote(&self.executable_path);
        while lines.first().is_some_and(|head| !head.starts_with('-')) {
            let head = lines.remove(0);
            first_line.push(' ');
            first_line.push_str(&head);
        }

        // User extra args: all on one trailing, commented line -- safe only as
        // the *last* line, since a `#` would swallow a trailing `\`.
        if self.extra_arg_count > 0 {
            let extra: Vec<String> = self
                .arguments
                .iter()
                .rev()
                .take(self.extra_arg_count)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .map(|a| shell_quote(a))
                .collect();
            lines.push(format!("{}  # custom arguments", extra.join(" ")));
        }

        let export_lines: Vec<String> = self
            .env
            .iter()
            .map(|(k, v)| format!("export {}={}", k, shell_quote(v)))
            .collect();

        let mut cmd_lines = Vec::with_capacity(lines.len() + 1);
        cmd_lines.push(first_line);
        cmd_lines.append(&mut lines);
        let cmd_block = cmd_lines
            .iter()
            .enumerate()
            .map(|(i, line)| {
                let prefix = if i == 0 { "" } else { "  " };
                let suffix = if i == cmd_lines.len() - 1 { "" } else { " \\" };
                format!("{prefix}{line}{suffix}")
            })
            .collect::<Vec<_>>()
            .join("\n");

        if export_lines.is_empty() {
            cmd_block
        } else {
            let mut out = export_lines;
            out.push(String::new());
            out.push(cmd_block);
            out.join("\n")
        }
    }
}

/// Minimal shell quoting, mirroring `LaunchSpec.quote`: wrap in single quotes
/// only when the token holds a character the shell treats specially, escaping
/// embedded single quotes.
pub fn shell_quote(s: &str) -> String {
    let safe = s.chars().all(|c| {
        c.is_alphabetic() || c.is_numeric() || matches!(c, '-' | '_' | '.' | '/' | '=' | ':')
    });
    if safe {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// The `--host` value for a network bind address (`LlamaServer.hostArgument`).
///
/// Engines at or above `multi_host_version` can bind a list, so loopback rides
/// along with the specific address -- keeping `localhost:<port>` answering even
/// when only the specific address is up. `0.0.0.0` already covers loopback and
/// repeating `127.0.0.1` would fail the second bind, so both pass through.
pub fn host_argument(bind_address: &str, current_version: Option<LlamaVersion>) -> String {
    if bind_address != "0.0.0.0"
        && bind_address != "127.0.0.1"
        && current_version.is_some_and(|v| v >= multi_host_version())
    {
        return format!("127.0.0.1,{bind_address}");
    }
    bind_address.to_string()
}

/// `TailscaleInterface.isTailscaleAddress`: the CGNAT range 100.64.0.0/10.
pub fn is_tailscale_address(ip: &str) -> bool {
    let parts: Vec<i64> = ip.split('.').filter_map(|p| p.parse().ok()).collect();
    parts.len() == 4 && parts[0] == 100 && (64..=127).contains(&parts[1])
}

/// `LocalNetwork.ipAddress`: the interface carrying the default route wins,
/// otherwise the smallest non-loopback IPv4 (sorted for a stable pick).
pub fn pick_ip_address(
    address_by_interface: &[(String, String)],
    primary_interface: Option<&str>,
) -> Option<String> {
    if let Some(primary) = primary_interface {
        if let Some((_, ip)) = address_by_interface.iter().find(|(k, _)| k == primary) {
            return Some(ip.clone());
        }
    }
    address_by_interface.iter().map(|(_, v)| v).min().cloned()
}

/// `TailscaleInterface.address`: requires both the CGNAT range and a `utun`
/// interface; sorted for a stable pick.
pub fn tailscale_address(address_by_interface: &[(String, String)]) -> Option<String> {
    address_by_interface
        .iter()
        .filter(|(k, v)| k.starts_with("utun") && is_tailscale_address(v))
        .map(|(_, v)| v)
        .min()
        .cloned()
}

/// `LlamaServer.effectiveBindAddress`: a specific address only while the
/// machine still holds it (falls back to loopback otherwise); `0.0.0.0` is not
/// an interface address and passes as-is.
pub fn effective_bind_address(configured: Option<&str>, local_addrs: &[String]) -> Option<String> {
    let configured = configured?;
    if configured == "0.0.0.0" {
        return Some(configured.to_string());
    }
    local_addrs
        .contains(&configured.to_string())
        .then(|| configured.to_string())
}

/// `LlamaServer.effectiveWebUIDirectory`: a custom web app folder only while it
/// still has the `index.html` served for `/`.
pub fn effective_webui_directory(configured: Option<&str>) -> Option<String> {
    configured.and_then(|dir| {
        Path::new(dir)
            .join("index.html")
            .is_file()
            .then(|| dir.to_string())
    })
}

/// `LlamaServer.localHost`: where the app itself reaches the server. A specific
/// address is bound either way, so it wins; `0.0.0.0` includes loopback.
pub fn local_host(effective_bind: Option<&str>) -> String {
    match effective_bind {
        Some(bind) if bind != "0.0.0.0" => bind.to_string(),
        _ => "localhost".to_string(),
    }
}

/// `LlamaServer.resolvedHost`: the host in URLs for *other* devices. A specific
/// bind passes through; `0.0.0.0` resolves to an actual local IP.
pub fn resolved_host(
    effective_bind: Option<&str>,
    address_by_interface: &[(String, String)],
    primary_interface: Option<&str>,
) -> String {
    match effective_bind {
        Some("0.0.0.0") => pick_ip_address(address_by_interface, primary_interface)
            .unwrap_or_else(|| "0.0.0.0".to_string()),
        Some(bind) => bind.to_string(),
        None => "localhost".to_string(),
    }
}

/// `Model.fitTargetMb` (SystemMemory values injected): the margin handed to
/// `serve` as `--fit-target`, so its runtime fit lands on the same number the
/// app's predictions use.
pub fn fit_target_mb(
    gpu_working_set_mb: i64,
    physical_memory_mb: i64,
    use_full_working_set: bool,
) -> i64 {
    let ws = gpu_working_set_mb as f64;
    let physical = physical_memory_mb as f64;
    let os_floor = 4096.0;
    let fit_slack = 1024.0;
    let ram_limb = physical - os_floor;
    let gpu_limb = if use_full_working_set {
        ws
    } else {
        ws - fit_slack
    };
    let budget = gpu_limb.min(ram_limb).max(0.0);
    (ws - budget).max(0.0) as i64
}

/// All inputs `LlamaServer.buildLaunchSpec` reads, resolved by the host. The
/// effective bind/webui values are computed here (via `effective_bind_address`
/// / `effective_webui_directory`), so the rendered command matches Swift's
/// exactly -- including the fallbacks to loopback and the built-in UI.
#[derive(Debug, Clone)]
pub struct LaunchSpecInput<'a> {
    pub llama_path: Option<&'a str>,
    pub presets_path: &'a str,
    pub log_file_path: &'a str,
    pub hf_cache_directory: &'a str,
    pub port: i64,
    pub fit_target_mb: i64,
    pub configured_bind_address: Option<&'a str>,
    pub address_by_interface: &'a [(String, String)],
    pub primary_interface: Option<&'a str>,
    pub current_version: Option<LlamaVersion>,
    pub configured_webui_dir: Option<&'a str>,
    pub sleep_idle_seconds: Option<i64>,
    pub agent_mode: bool,
    pub extra_args: &'a [String],
}

/// `LlamaServer.buildLaunchSpec`. Returns `None` exactly when no llama binary
/// is installed.
pub fn build_launch_spec(input: &LaunchSpecInput<'_>) -> Option<LaunchSpec> {
    let llama_path = input.llama_path?;

    let env = vec![
        ("GGML_METAL_NO_RESIDENCY".to_string(), "1".to_string()),
        (
            "HF_HUB_CACHE".to_string(),
            input.hf_cache_directory.to_string(),
        ),
        (
            "LLAMA_CACHE".to_string(),
            "/tmp/llama-empty-cache".to_string(),
        ),
    ];

    let mut arguments = vec![
        "serve".to_string(),
        "--models-preset".to_string(),
        input.presets_path.to_string(),
        "--log-file".to_string(),
        input.log_file_path.to_string(),
    ];

    if let Some(webui) = effective_webui_directory(input.configured_webui_dir) {
        arguments.extend(["--path".to_string(), webui]);
    }

    arguments.extend([
        "--port".to_string(),
        input.port.to_string(),
        "--models-max".to_string(),
        "1".to_string(),
        "--fit-target".to_string(),
        input.fit_target_mb.to_string(),
    ]);

    // Always pass an explicit `--host`, defaulting to loopback. Omitting it when
    // exposure is Off left the bind to llama.cpp's own resolution, where a stray
    // `LLAMA_ARG_HOST` in the inherited environment or a `host=` line in
    // `~/.config/llama.cpp/config.ini` could expose the server on the network
    // while the UI -- and the consent guard -- still reported "Off". A present
    // CLI flag wins over both, so making loopback explicit closes that gap.
    let local_addrs: Vec<String> = input
        .address_by_interface
        .iter()
        .map(|(_, v)| v.clone())
        .collect();
    let bind = effective_bind_address(input.configured_bind_address, &local_addrs)
        .unwrap_or_else(|| "127.0.0.1".to_string());
    arguments.extend([
        "--host".to_string(),
        host_argument(&bind, input.current_version),
    ]);

    if let Some(seconds) = input.sleep_idle_seconds {
        arguments.extend(["--sleep-idle-seconds".to_string(), seconds.to_string()]);
    }

    arguments.extend(["--jinja".to_string(), "--spec-default".to_string()]);

    if input.agent_mode {
        arguments.push("--agent".to_string());
    }

    let extra_args: Vec<String> = input.extra_args.to_vec();
    let extra_arg_count = extra_args.len();
    arguments.extend(extra_args);

    Some(LaunchSpec {
        executable_path: llama_path.to_string(),
        arguments,
        extra_arg_count,
        env,
    })
}

/// `LlamaServer.notePresetRejection`: last line carrying `option 'x' not
/// recognized in preset 'y'` wins, matching Swift's overwrite-per-line.
pub fn preset_rejection_option(message: &str) -> Option<String> {
    let mut last = None;
    for line in message.lines() {
        if !line.contains("not recognized in preset") {
            continue;
        }
        let start = line.find("option '")?;
        let after = &line[start + "option '".len()..];
        let end = after.find('\'')?;
        last = Some(after[..end].to_string());
    }
    last
}

/// `LlamaServerAPI.fetchModelStatuses` body parsing: the `data[].id` +
/// `status.value` flat list, with missing/unknown states normalized to
/// `unloaded`. `None` when the body can't be decoded as `ModelsResponse`.
pub fn parse_model_statuses(bytes: &[u8]) -> Option<Vec<(String, String)>> {
    let root: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let data = root.get("data")?.as_array()?;
    let mut out = Vec::with_capacity(data.len());
    for item in data {
        let id = item.get("id")?.as_str()?.to_string();
        let state = match item.get("status") {
            None => "unloaded".to_string(),
            Some(status) => {
                let value = status.get("value")?.as_str()?;
                match value {
                    "loaded" | "loading" | "sleeping" | "unloaded" => value.to_string(),
                    _ => "unloaded".to_string(),
                }
            }
        };
        out.push((id, state));
    }
    Some(out)
}

/// `reclaimPort`'s `lsof -ti` output → PIDs, as Swift splits by newline and
/// parses each token.
pub fn extract_pids(lsof_output: &str) -> Vec<i64> {
    lsof_output
        .split(['\n', '\r'])
        .filter_map(|s| s.parse::<i64>().ok())
        .collect()
}

/// Whether `ps -o comm=` output names a `llama` executable
/// (`comm.hasSuffix("llama")`).
pub fn is_llama_process(comm: &str) -> bool {
    comm.ends_with("llama")
}

/// `reclaimPort` blockk-er name: `URL(fileURLWithPath:).lastPathComponent`,
/// with URL's trailing-slash normalization.
pub fn last_path_component(path: &str) -> String {
    let trimmed = path.trim_end_matches('/');
    if trimmed.is_empty() {
        return String::new();
    }
    trimmed.rsplit('/').next().unwrap_or("").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(exe: &str, args: &[&str], extra: usize, env: &[(&str, &str)]) -> LaunchSpec {
        LaunchSpec {
            executable_path: exe.to_string(),
            arguments: args.iter().map(|s| s.to_string()).collect(),
            extra_arg_count: extra,
            env: env
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    #[test]
    fn display_command_reads_like_a_rendered_invocation() {
        let s = spec(
            "/Users/x/llama",
            &[
                "serve",
                "--models-preset",
                "/Library/Application Support/Llama/models.ini",
                "--log-file",
                "/Users/x/Library/Logs/Llama/llama-server.log",
                "--port",
                "9931",
                "--sleep-idle-seconds",
                "300",
                "--jinja",
                "serve", // user extra (commented line)
            ],
            1,
            &[
                ("GGML_METAL_NO_RESIDENCY", "1"),
                ("LLAMA_CACHE", "/tmp/llama-empty-cache"),
            ],
        );
        let rendered = s.display_command();
        assert_eq!(
            rendered,
            "export GGML_METAL_NO_RESIDENCY=1\nexport LLAMA_CACHE=/tmp/llama-empty-cache\n\n/Users/x/llama serve \\\n  --models-preset '/Library/Application Support/Llama/models.ini' \\\n  --log-file /Users/x/Library/Logs/Llama/llama-server.log \\\n  --port 9931 \\\n  --sleep-idle-seconds 300 \\\n  --jinja \\\n  serve  # custom arguments"
        );
        assert!(rendered.contains("serve  # custom arguments"));
    }

    #[test]
    fn shell_quote_only_wraps_unsafe_tokens() {
        assert_eq!(
            shell_quote("/opt/homebrew/bin/llama"),
            "/opt/homebrew/bin/llama"
        );
        assert_eq!(shell_quote("--fit-target=512"), "--fit-target=512");
        assert_eq!(
            shell_quote("Library/Application Support"),
            "'Library/Application Support'"
        );
        assert_eq!(shell_quote("a'b"), "'a'\\''b'");
        assert_eq!(shell_quote("ünïcode"), "ünïcode");
    }

    #[test]
    fn host_argument_respects_gate_versions() {
        let ver = LlamaVersion { build: 11_104 };
        let older = LlamaVersion { build: 11_103 };
        assert_eq!(
            host_argument("100.64.0.1", Some(ver)),
            "127.0.0.1,100.64.0.1"
        );
        assert_eq!(host_argument("100.64.0.1", Some(older)), "100.64.0.1");
        assert_eq!(host_argument("100.64.0.1", None), "100.64.0.1");
        assert_eq!(host_argument("0.0.0.0", Some(ver)), "0.0.0.0");
        assert_eq!(host_argument("127.0.0.1", Some(ver)), "127.0.0.1");
    }

    #[test]
    fn tailscale_classification_and_lookup() {
        assert!(is_tailscale_address("100.64.0.1"));
        assert!(is_tailscale_address("100.127.255.255"));
        assert!(!is_tailscale_address("100.128.0.1"));
        assert!(!is_tailscale_address("192.168.1.1"));
        assert!(!is_tailscale_address("100.64.0.1.9"));
        assert!(!is_tailscale_address("x.64.0.1"));

        let addrs = vec![
            ("en0".to_string(), "192.168.1.5".to_string()),
            ("utun4".to_string(), "100.64.0.9".to_string()),
            ("utun5".to_string(), "100.64.0.3".to_string()),
            ("utun6".to_string(), "10.0.0.1".to_string()), // not CGNAT
        ];
        assert_eq!(tailscale_address(&addrs).as_deref(), Some("100.64.0.3"));
    }

    #[test]
    fn ip_pick_prefers_primary_interface_then_sorted() {
        let addrs = vec![
            ("en0".to_string(), "192.168.1.5".to_string()),
            ("en1".to_string(), "192.168.2.9".to_string()),
        ];
        assert_eq!(
            pick_ip_address(&addrs, Some("en1")).as_deref(),
            Some("192.168.2.9")
        );
        assert_eq!(
            pick_ip_address(&addrs, Some("missing")).as_deref(),
            Some("192.168.1.5")
        );
        assert_eq!(
            pick_ip_address(&addrs, None).as_deref(),
            Some("192.168.1.5")
        );
        assert_eq!(pick_ip_address(&[], None), None);
    }

    #[test]
    fn effective_bind_and_webui_fallbacks() {
        let addrs = ["192.168.1.5".to_string()];
        assert_eq!(
            effective_bind_address(Some("192.168.1.5"), &addrs).as_deref(),
            Some("192.168.1.5")
        );
        assert_eq!(
            effective_bind_address(Some("0.0.0.0"), &addrs).as_deref(),
            Some("0.0.0.0")
        );
        assert_eq!(effective_bind_address(Some("10.0.0.2"), &addrs), None);
        assert_eq!(effective_bind_address(None, &addrs), None);

        let dir = std::env::temp_dir().join("llama-server-webui-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), "<html></html>").unwrap();
        assert_eq!(
            effective_webui_directory(Some(dir.to_str().unwrap())).as_deref(),
            Some(dir.to_str().unwrap())
        );
        std::fs::remove_file(dir.join("index.html")).unwrap();
        assert_eq!(effective_webui_directory(Some(dir.to_str().unwrap())), None);
        assert_eq!(effective_webui_directory(None), None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn local_and_resolved_host() {
        let addrs = vec![("en0".to_string(), "192.168.1.5".to_string())];
        assert_eq!(local_host(None), "localhost");
        assert_eq!(local_host(Some("0.0.0.0")), "localhost");
        assert_eq!(local_host(Some("100.64.0.1")), "100.64.0.1");
        assert_eq!(resolved_host(None, &addrs, None), "localhost");
        assert_eq!(
            resolved_host(Some("100.64.0.1"), &addrs, None),
            "100.64.0.1"
        );
        assert_eq!(
            resolved_host(Some("0.0.0.0"), &addrs, Some("en0")),
            "192.168.1.5"
        );
        assert_eq!(resolved_host(Some("0.0.0.0"), &[], None), "0.0.0.0");
    }

    #[test]
    fn fit_target_math() {
        // 8 GB Mac: the OS floor (4096) binds over the working-set slack.
        assert_eq!(fit_target_mb(6_144, 8_192, false), 2_048);
        assert_eq!(fit_target_mb(6_144, 8_192, true), 2_048);
        // Roomier Mac: the working-set limb binds; full-working-set spends the
        // slack, so the margin is zero.
        assert_eq!(fit_target_mb(98_304, 131_072, false), 1_024);
        assert_eq!(fit_target_mb(98_304, 131_072, true), 0);
    }

    #[test]
    fn build_launch_spec_composes_arguments_in_order() {
        let dir = std::env::temp_dir().join("llama-server-launch-test");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.html"), "<html></html>").unwrap();

        let spec = build_launch_spec(&LaunchSpecInput {
            llama_path: Some("/Users/x/llama"),
            presets_path: "/Users/x/Library/Application Support/Llama/models.ini",
            log_file_path: "/Users/x/Library/Logs/Llama/llama-server.log",
            hf_cache_directory: "/Users/x/Library/Caches/Llama",
            port: 9931,
            fit_target_mb: 1024,
            configured_bind_address: Some("100.64.0.1"),
            address_by_interface: &[("utun4".to_string(), "100.64.0.1".to_string())],
            primary_interface: None,
            current_version: Some(LlamaVersion { build: 11_200 }),
            configured_webui_dir: Some(dir.to_str().unwrap()),
            sleep_idle_seconds: Some(300),
            agent_mode: true,
            extra_args: &["--ctx-size".to_string(), "8192".to_string()],
        })
        .unwrap();

        assert_eq!(
            spec.arguments,
            vec![
                "serve",
                "--models-preset",
                "/Users/x/Library/Application Support/Llama/models.ini",
                "--log-file",
                "/Users/x/Library/Logs/Llama/llama-server.log",
                "--path",
                dir.to_str().unwrap(),
                "--port",
                "9931",
                "--models-max",
                "1",
                "--fit-target",
                "1024",
                "--host",
                "127.0.0.1,100.64.0.1",
                "--sleep-idle-seconds",
                "300",
                "--jinja",
                "--spec-default",
                "--agent",
                "--ctx-size",
                "8192",
            ]
        );
        assert_eq!(spec.extra_arg_count, 2);
        assert_eq!(
            spec.env[0],
            ("GGML_METAL_NO_RESIDENCY".to_string(), "1".to_string())
        );
        assert_eq!(
            spec.env[2],
            (
                "LLAMA_CACHE".to_string(),
                "/tmp/llama-empty-cache".to_string()
            )
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn preset_rejection_extracts_last_option() {
        assert_eq!(
            preset_rejection_option(
                "llama-server: error: failed to parse preset\noption 'temperture' not recognized in preset 'models.user.ini'\n"
            )
            .as_deref(),
            Some("temperture")
        );
        assert_eq!(
            preset_rejection_option(
                "first line\noption 'alpha' not recognized in preset 'x'\noption 'beta' not recognized in preset 'x'\n"
            )
            .as_deref(),
            Some("beta")
        );
        assert_eq!(
            preset_rejection_option("all quiet on the preset front"),
            None
        );
    }

    #[test]
    fn model_statuses_parse_with_unloaded_fallback() {
        let body = br#"{"data":[{"id":"org/Model:Q4_0","status":{"value":"loaded"}},{"id":"org/Other","status":{"value":"sleeping"}},{"id":"org/Missing"}]}"#;
        assert_eq!(
            parse_model_statuses(body),
            Some(vec![
                ("org/Model:Q4_0".to_string(), "loaded".to_string()),
                ("org/Other".to_string(), "sleeping".to_string()),
                ("org/Missing".to_string(), "unloaded".to_string()),
            ])
        );
        // Unknown state normalizes to unloaded.
        assert_eq!(
            parse_model_statuses(br#"{"data":[{"id":"m","status":{"value":"bogus"}}]}"#),
            Some(vec![("m".to_string(), "unloaded".to_string())])
        );
        // Malformed bodies fail like Swift's JSONDecoder.
        assert_eq!(parse_model_statuses(br#"{"data":[{"id":7}]}"#), None);
        assert_eq!(parse_model_statuses(b"not json"), None);
        assert_eq!(parse_model_statuses(br#"{"data":5}"#), None);
    }

    #[test]
    fn reclaim_port_parsers() {
        assert_eq!(extract_pids("123\n456\n"), vec![123, 456]);
        assert_eq!(extract_pids("\n\r1\n2\r\n3\n"), vec![1, 2, 3]);
        assert_eq!(extract_pids("wat\n7\n"), vec![7]);
        assert_eq!(extract_pids(""), Vec::<i64>::new());

        assert!(is_llama_process("/Users/x/llama"));
        assert!(is_llama_process("llama"));
        assert!(!is_llama_process("/usr/sbin/lsof"));
        assert!(!is_llama_process("nothing"));

        assert_eq!(last_path_component("llama"), "llama");
        assert_eq!(last_path_component("/opt/homebrew/bin/llama"), "llama");
        assert_eq!(last_path_component("/usr/sbin/lsof"), "lsof");
        assert_eq!(last_path_component("app/"), "app");
    }
}
