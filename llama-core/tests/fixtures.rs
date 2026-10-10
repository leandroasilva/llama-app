//! Equivalence harness: both this crate and the Swift `LlamaTests` target read
//! the same `Fixtures/*.json` corpus. A behavior change on either side that
//! diverges from the pinned expectation fails one of the two suites -- that is
//! what keeps the Rust port byte-compatible with the Swift original.

use llama_core::engine::LlamaVersion;
use llama_core::hf::HFRepoParser;
use serde::Deserialize;
use std::path::PathBuf;

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("Fixtures")
        .join(name);
    let text =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

// MARK: - LlamaVersion

#[derive(Deserialize)]
struct VersionCase {
    input: String,
    build: Option<i64>,
}

#[derive(Deserialize)]
struct VersionFixture {
    cases: Vec<VersionCase>,
}

#[test]
fn llama_version_fixtures() {
    let f: VersionFixture = load("llama_version.json");
    for c in &f.cases {
        let got = LlamaVersion::parse(&c.input).map(|v| v.build);
        assert_eq!(got, c.build, "LlamaVersion.parse({:?})", c.input);
    }
}

// MARK: - HFRepoParser

#[derive(Deserialize)]
struct ParseCase {
    input: String,
    org: Option<String>,
    repo: Option<String>,
}

#[derive(Deserialize)]
struct StrCase {
    input: String,
    expect: Option<String>,
}

#[derive(Deserialize)]
struct BoolCase {
    input: String,
    expect: bool,
}

#[derive(Deserialize)]
struct IntCase {
    input: String,
    expect: Option<i64>,
}

#[derive(Deserialize)]
struct RepoFixture {
    parse: Vec<ParseCase>,
    parse_quant: Vec<StrCase>,
    is_split_shard: Vec<BoolCase>,
    is_first_shard: Vec<BoolCase>,
    shard_total: Vec<IntCase>,
    split_shard_base_name: Vec<StrCase>,
}

#[test]
fn hf_repo_parser_fixtures() {
    let f: RepoFixture = load("hf_repo_parser.json");

    for c in &f.parse {
        let want = c.org.clone().zip(c.repo.clone());
        assert_eq!(HFRepoParser::parse(&c.input), want, "parse({:?})", c.input);
    }
    for c in &f.parse_quant {
        assert_eq!(
            HFRepoParser::parse_quant(&c.input),
            c.expect,
            "parse_quant({:?})",
            c.input
        );
    }
    for c in &f.is_split_shard {
        assert_eq!(
            HFRepoParser::is_split_shard(&c.input),
            c.expect,
            "is_split_shard({:?})",
            c.input
        );
    }
    for c in &f.is_first_shard {
        assert_eq!(
            HFRepoParser::is_first_shard(&c.input),
            c.expect,
            "is_first_shard({:?})",
            c.input
        );
    }
    for c in &f.shard_total {
        assert_eq!(
            HFRepoParser::shard_total(&c.input),
            c.expect,
            "shard_total({:?})",
            c.input
        );
    }
    for c in &f.split_shard_base_name {
        assert_eq!(
            HFRepoParser::split_shard_base_name(&c.input),
            c.expect,
            "split_shard_base_name({:?})",
            c.input
        );
    }
}
