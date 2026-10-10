//! Port of `Llama/Models/Model+Compatibility.swift`.
//!
//! The struct carries the four measured values the math needs; settings and
//! device memory (`UserSettings`, `SystemMemory`) are injected so this stays
//! pure.

use crate::models::ContextTier;

/// We evaluate compatibility assuming a 4k-token context, the default
/// llama.cpp launches with when no explicit value is provided.
pub const COMPATIBILITY_CTX_WINDOW_TOKENS: f64 = 4096.0;

/// Slack (MB) left inside the GPU working set for what the estimate can't see.
pub const FIT_SLACK_MB: f64 = 1024.0;

/// Physical RAM (MB) kept out of a model's reach for macOS and other apps.
pub const OS_FLOOR_MB: f64 = 4096.0;

/// Overhead multiplier applied to file size when estimating weight memory
/// without a MemProfile measurement.
pub const WEIGHT_OVERHEAD_MULTIPLIER: f64 = 1.05;

/// The memory a model may use on this device.
pub fn device_memory_budget_mb(
    working_set_mb: f64,
    physical_mb: f64,
    use_full_working_set: bool,
) -> f64 {
    let ram_limb_mb = physical_mb - OS_FLOOR_MB;
    let gpu_limb_mb = if use_full_working_set {
        working_set_mb
    } else {
        working_set_mb - FIT_SLACK_MB
    };
    gpu_limb_mb.min(ram_limb_mb).max(0.0)
}

/// The margin to hand `serve` as `--fit-target`.
pub fn fit_target_mb(working_set_mb: f64, budget_mb: f64) -> f64 {
    (working_set_mb - budget_mb).max(0.0)
}

/// Rough pre-download fit check: weight memory ≈ `bytes × 1.05` within budget.
pub fn estimated_weight_fits(bytes: Option<i64>, budget_mb: f64) -> bool {
    let Some(bytes) = bytes else { return true };
    if bytes <= 0 {
        return true;
    }
    let weight_mb = bytes as f64 / 1_048_576.0 * WEIGHT_OVERHEAD_MULTIPLIER;
    weight_mb <= budget_mb
}

fn bytes_to_mb(bytes: i64) -> f64 {
    bytes as f64 / 1_048_576.0
}

/// Compatibility result for a model at a given context length.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompatibilityInfo {
    pub is_compatible: bool,
    pub incompatibility_summary: Option<String>,
}

/// The measured/spec values compatibility math reads from a model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelMem {
    pub ctx_window: i64,
    pub file_size: i64,
    pub ctx_bytes_per_1k_tokens: i64,
    pub resident_bytes: i64,
}

impl ModelMem {
    /// Weight memory (MB): prefers measured `resident_bytes`, else
    /// `file_size × 1.05`.
    pub fn weight_memory_mb(&self) -> f64 {
        if self.resident_bytes > 0 {
            return self.resident_bytes as f64 / 1_048_576.0;
        }
        bytes_to_mb(self.file_size) * WEIGHT_OVERHEAD_MULTIPLIER
    }

    /// Estimated runtime memory (MB), rounded up.
    pub fn runtime_memory_usage_mb(&self, ctx_window_tokens: f64) -> u64 {
        let weight_mb = self.weight_memory_mb();
        let ctx_multiplier = ctx_window_tokens / 1000.0;
        let ctx_bytes = self.ctx_bytes_per_1k_tokens as f64 * ctx_multiplier;
        let ctx_mb = ctx_bytes / 1_048_576.0;
        let total_mb = weight_mb + ctx_mb;
        total_mb.ceil() as u64
    }

    /// The largest standard tier within the model's native context window.
    pub fn native_max_tier(&self) -> Option<ContextTier> {
        ContextTier::ALL
            .into_iter()
            .rev()
            .find(|t| t.raw_value() <= self.ctx_window)
    }

    /// Full compatibility info at `ctx_window_tokens`.
    pub fn compatibility_info(&self, ctx_window_tokens: f64, budget_mb: f64) -> CompatibilityInfo {
        if (self.ctx_window as f64) < COMPATIBILITY_CTX_WINDOW_TOKENS {
            return CompatibilityInfo {
                is_compatible: false,
                incompatibility_summary: Some("requires models with ≥4k context".to_string()),
            };
        }

        if ctx_window_tokens > 0.0 && ctx_window_tokens > self.ctx_window as f64 {
            let max_label = self
                .native_max_tier()
                .map(|t| t.label())
                .unwrap_or_else(|| format!("{}k", self.ctx_window / 1024));
            return CompatibilityInfo {
                is_compatible: false,
                incompatibility_summary: Some(format!("model max is {max_label}")),
            };
        }

        let estimated_memory_usage_mb = self.runtime_memory_usage_mb(ctx_window_tokens);

        let memory_requirement_summary = || -> String {
            let needed_mb = estimated_memory_usage_mb as f64;
            let from_working_set = (needed_mb + FIT_SLACK_MB) / 0.75;
            let from_os_floor = needed_mb + OS_FLOOR_MB;
            let required_total_mb = from_working_set.max(from_os_floor);
            let gb = (required_total_mb / 1024.0).ceil();

            let common_sizes: [f64; 11] = [
                8.0, 16.0, 18.0, 24.0, 32.0, 36.0, 48.0, 64.0, 96.0, 128.0, 192.0,
            ];
            let display_gb = common_sizes
                .iter()
                .copied()
                .find(|s| *s >= gb)
                .unwrap_or(gb);

            format!("Requires a Mac with {display_gb:.0} GB+ of memory")
        };

        if budget_mb <= 0.0 {
            return CompatibilityInfo {
                is_compatible: false,
                incompatibility_summary: Some(memory_requirement_summary()),
            };
        }

        let is_compatible = estimated_memory_usage_mb <= budget_mb as u64;
        CompatibilityInfo {
            is_compatible,
            incompatibility_summary: if is_compatible {
                None
            } else {
                Some(memory_requirement_summary())
            },
        }
    }

    pub fn is_compatible(&self, ctx_window_tokens: f64, budget_mb: f64) -> bool {
        self.compatibility_info(ctx_window_tokens, budget_mb)
            .is_compatible
    }

    pub fn incompatibility_summary(
        &self,
        ctx_window_tokens: f64,
        budget_mb: f64,
    ) -> Option<String> {
        self.compatibility_info(ctx_window_tokens, budget_mb)
            .incompatibility_summary
    }

    /// Context tiers this device can actually run.
    pub fn supported_context_tiers(&self, budget_mb: f64) -> Vec<ContextTier> {
        if self.ctx_bytes_per_1k_tokens <= 0 {
            return vec![ContextTier::K4];
        }

        let mut tiers: Vec<ContextTier> = ContextTier::STANDARD_TIERS
            .into_iter()
            .filter(|t| self.is_compatible(t.raw_value() as f64, budget_mb))
            .collect();

        if self.is_compatible(ContextTier::K256.raw_value() as f64, budget_mb)
            && !tiers.contains(&ContextTier::K256)
        {
            tiers.push(ContextTier::K256);
        }

        tiers.sort();
        tiers
    }

    /// All tiers the model itself supports, regardless of this device.
    pub fn display_context_tiers(&self) -> Vec<ContextTier> {
        if self.ctx_bytes_per_1k_tokens <= 0 {
            return vec![ContextTier::K4];
        }

        let mut tiers: Vec<ContextTier> = ContextTier::STANDARD_TIERS
            .into_iter()
            .filter(|t| t.raw_value() <= self.ctx_window)
            .collect();

        if ContextTier::K256.raw_value() <= self.ctx_window {
            tiers.push(ContextTier::K256);
        }

        tiers.sort();
        tiers
    }

    /// The effective context tier: the still-compatible saved selection, else 4K.
    pub fn effective_ctx_tier(
        &self,
        budget_mb: f64,
        selected: Option<ContextTier>,
    ) -> Option<ContextTier> {
        let supported = self.supported_context_tiers(budget_mb);
        if supported.is_empty() {
            return None;
        }
        if let Some(sel) = selected {
            if supported.contains(&sel) {
                return Some(sel);
            }
        }
        Some(ContextTier::K4)
    }
}
