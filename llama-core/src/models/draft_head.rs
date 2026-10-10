//! Port of `Llama/Models/DraftHead.swift`.

use crate::pathutil::last_path_component;

/// A speculative-decoding draft head the app drives. Both kinds borrow the
/// target model's embeddings and output projection, so neither runs as a model
/// on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DraftHead {
    /// Multi-token prediction: the head predicts a few tokens ahead.
    Mtp,
    /// Block-diffusion drafting: the head fills a whole block of tokens at once.
    Dflash,
}

impl DraftHead {
    /// Kinds in order of preference, for repos that ship more than one head.
    /// DFlash wins over MTP.
    pub const PREFERENCE_ORDER: [DraftHead; 2] = [DraftHead::Dflash, DraftHead::Mtp];

    /// The sidecar's filename prefix.
    pub fn sidecar_prefix(self) -> &'static str {
        match self {
            DraftHead::Mtp => "mtp-",
            DraftHead::Dflash => "dflash-",
        }
    }

    /// The `spec-type` value that tells llama-server to draft with this head.
    pub fn spec_type(self) -> &'static str {
        match self {
            DraftHead::Mtp => "draft-mtp",
            DraftHead::Dflash => "draft-dflash",
        }
    }

    /// Cap on tokens drafted per step (`spec-draft-n-max`).
    pub fn draft_n_max(self) -> i32 {
        match self {
            DraftHead::Mtp => 3,
            DraftHead::Dflash => 7,
        }
    }

    /// Name shown in the model row's chip.
    pub fn label(self) -> &'static str {
        match self {
            DraftHead::Mtp => "MTP",
            DraftHead::Dflash => "DFlash",
        }
    }

    /// The stable string form used in JSON (matches the Swift `RawValue`).
    pub fn as_str(self) -> &'static str {
        match self {
            DraftHead::Mtp => "mtp",
            DraftHead::Dflash => "dflash",
        }
    }

    /// Parses the Swift raw value (`"mtp"` / `"dflash"`).
    pub fn from_str_name(s: &str) -> Option<DraftHead> {
        match s {
            "mtp" => Some(DraftHead::Mtp),
            "dflash" => Some(DraftHead::Dflash),
            _ => None,
        }
    }

    /// The kind of the sidecar at `path`, judged by its basename; `None` when
    /// it isn't a draft-head sidecar we drive.
    pub fn from_sidecar_path(path: &str) -> Option<DraftHead> {
        let name = last_path_component(path).to_lowercase();
        if !name.ends_with(".gguf") {
            return None;
        }
        DraftHead::PREFERENCE_ORDER
            .into_iter()
            .find(|k| name.starts_with(k.sidecar_prefix()))
    }
}
