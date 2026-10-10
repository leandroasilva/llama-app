//! Port of `Llama/Models/ContextTier.swift`.

/// A context length tier for running models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ContextTier {
    K4,
    K8,
    K16,
    K32,
    K64,
    K128,
    K256,
}

impl ContextTier {
    /// `CaseIterable` order.
    pub const ALL: [ContextTier; 7] = [
        ContextTier::K4,
        ContextTier::K8,
        ContextTier::K16,
        ContextTier::K32,
        ContextTier::K64,
        ContextTier::K128,
        ContextTier::K256,
    ];

    /// The standard tiers shown in the UI (when compatible).
    pub const STANDARD_TIERS: [ContextTier; 6] = [
        ContextTier::K4,
        ContextTier::K8,
        ContextTier::K16,
        ContextTier::K32,
        ContextTier::K64,
        ContextTier::K128,
    ];

    /// The tier's raw value (context length in tokens).
    pub fn raw_value(self) -> i64 {
        match self {
            ContextTier::K4 => 4096,
            ContextTier::K8 => 8192,
            ContextTier::K16 => 16384,
            ContextTier::K32 => 32768,
            ContextTier::K64 => 65536,
            ContextTier::K128 => 131_072,
            ContextTier::K256 => 262_144,
        }
    }

    pub fn from_raw(value: i64) -> Option<ContextTier> {
        ContextTier::ALL
            .into_iter()
            .find(|t| t.raw_value() == value)
    }

    /// Number-only label without the `ctx` suffix (e.g. `"4k"`).
    pub fn short_label(self) -> String {
        format!("{}k", self.raw_value() / 1024)
    }

    pub fn label(self) -> String {
        format!("{} ctx", self.short_label())
    }

    pub fn suffix(self) -> String {
        format!(":{}", self.label())
    }
}
