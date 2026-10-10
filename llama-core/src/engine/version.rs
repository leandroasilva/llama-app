//! Port of `Llama/Engine/LlamaVersion.swift`.

/// A llama.cpp build version. Comparison is by build number -- the monotonic
/// `bXXXX` counter llama.cpp bumps each release; commit shas and semantic
/// versions are ignored.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct LlamaVersion {
    /// The build number, e.g. `9370` for `b9370`.
    pub build: i64,
}

/// Mirrors Swift's `Int(substring.prefix(while: { $0.isNumber }))`: collect the
/// leading characters that Swift's `Character.isNumber` accepts, then parse. A
/// non-ASCII digit survives the filter but fails the integer parse -- the same
/// nil result Swift produces.
fn leading_number(s: &str) -> Option<i64> {
    let digits: String = s.chars().take_while(|c| c.is_numeric()).collect();
    digits.parse::<i64>().ok()
}

impl LlamaVersion {
    /// Parses a pinned tag (`b9370`) or `llama version` output, which comes in
    /// two formats:
    /// - `b9370-aa50b2c2a` -- builds before b10398
    /// - `version: 0.5.0-dev (build 11200, commit 81bc6b83f)` -- b10398 on
    ///
    /// Returns `None` if no build number can be read.
    pub fn parse(output: &str) -> Option<LlamaVersion> {
        // Newer format: take the number after `(build `. Checked first because
        // the leading token (`version:`) would otherwise fail the older parse.
        if let Some(idx) = output.find("(build ") {
            let build = leading_number(&output[idx + "(build ".len()..])?;
            return Some(LlamaVersion { build });
        }

        // Older format (and pinned tags): an optional leading `b`, the build
        // digits, then optionally `-<sha>`.
        let token = output.split_whitespace().next()?;
        let digits = match token.chars().next() {
            Some('b') | Some('B') => &token[1..],
            _ => token,
        };
        let build = leading_number(digits)?;
        Some(LlamaVersion { build })
    }

    /// The build tag, e.g. `b9370`. For display.
    pub fn tag(&self) -> String {
        format!("b{}", self.build)
    }
}
