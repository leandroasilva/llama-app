//! Port of `Llama/Models/ResolvedPaths.swift`.

use crate::models::DraftHead;

/// Resolved file paths for a downloaded model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedPaths {
    pub model_file: String,
    pub additional_parts: Vec<String>,
    pub mmproj_file: Option<String>,
    pub uses_mtp: bool,
    pub draft_sidecar_file: Option<String>,
    pub hf_repo_dir_name: String,
}

impl ResolvedPaths {
    /// The kind of `draft_sidecar_file`, read from its filename prefix.
    pub fn draft_sidecar_kind(&self) -> Option<DraftHead> {
        self.draft_sidecar_file
            .as_deref()
            .and_then(DraftHead::from_sidecar_path)
    }

    /// All file paths this model occupies on disk.
    pub fn all_paths(&self) -> Vec<String> {
        let mut paths = vec![self.model_file.clone()];
        paths.extend(self.additional_parts.iter().cloned());
        if let Some(mmproj) = &self.mmproj_file {
            paths.push(mmproj.clone());
        }
        if let Some(draft) = &self.draft_sidecar_file {
            paths.push(draft.clone());
        }
        paths
    }
}
