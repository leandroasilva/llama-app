//! Platform-specific llama.cpp release artifacts.
//!
//! The engine release is one GitHub tag with per-OS/arch/backend archives. The
//! macOS app pins the Apple Silicon archive; the Linux frontend picks a variant
//! from the host architecture and the user's selected backend. Digests for the
//! pinned release live in `release.json`, so an install never trusts an archive
//! without one when the tag is known.

use super::binaries::target_release;
use super::installer::RELEASE_BASE;

/// Host operating system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetOs {
    MacOS,
    Linux,
}

impl TargetOs {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::MacOS => "macos",
            Self::Linux => "linux",
        }
    }
}

/// Host CPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    Aarch64,
    X86_64,
}

impl TargetArch {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Aarch64 => "aarch64",
            Self::X86_64 => "x86_64",
        }
    }

    /// Detect the running process architecture (`aarch64` or `x86_64`; other
    /// architectures are unsupported by the published archives we consume).
    pub fn current() -> Option<Self> {
        match std::env::consts::ARCH {
            "aarch64" => Some(Self::Aarch64),
            "x86_64" => Some(Self::X86_64),
            _ => None,
        }
    }
}

/// Compute backend variant requested for Linux.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinuxBackend {
    Cpu,
    Vulkan,
}

impl LinuxBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Vulkan => "vulkan",
        }
    }
}

/// One published release archive.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    MacOSAarch64,
    MacOSX86_64,
    LinuxCpuAarch64,
    LinuxCpuX86_64,
    LinuxVulkanAarch64,
    LinuxVulkanX86_64,
}

impl Target {
    pub fn key(self) -> &'static str {
        match self {
            Self::MacOSAarch64 => "macos-aarch64",
            Self::MacOSX86_64 => "macos-x86_64",
            Self::LinuxCpuAarch64 => "linux-cpu-aarch64",
            Self::LinuxCpuX86_64 => "linux-cpu-x86_64",
            Self::LinuxVulkanAarch64 => "linux-vulkan-aarch64",
            Self::LinuxVulkanX86_64 => "linux-vulkan-x86_64",
        }
    }

    pub fn os(self) -> TargetOs {
        match self {
            Self::MacOSAarch64 | Self::MacOSX86_64 => TargetOs::MacOS,
            _ => TargetOs::Linux,
        }
    }

    pub fn arch(self) -> TargetArch {
        match self {
            Self::MacOSAarch64 | Self::LinuxCpuAarch64 | Self::LinuxVulkanAarch64 => {
                TargetArch::Aarch64
            }
            _ => TargetArch::X86_64,
        }
    }

    pub fn linux_backend(self) -> Option<LinuxBackend> {
        match self {
            Self::LinuxCpuAarch64 | Self::LinuxCpuX86_64 => Some(LinuxBackend::Cpu),
            Self::LinuxVulkanAarch64 | Self::LinuxVulkanX86_64 => Some(LinuxBackend::Vulkan),
            _ => None,
        }
    }

    /// The archive's top-level directory, e.g. `llama-b12005`.
    pub fn bundle_name(self, version: &str) -> String {
        format!("llama-{version}")
    }

    fn asset_stem(self) -> &'static str {
        match self {
            Self::MacOSAarch64 => "macos-arm64",
            Self::MacOSX86_64 => "macos-x64",
            Self::LinuxCpuAarch64 => "ubuntu-arm64",
            Self::LinuxCpuX86_64 => "ubuntu-x64",
            Self::LinuxVulkanAarch64 => "ubuntu-vulkan-arm64",
            Self::LinuxVulkanX86_64 => "ubuntu-vulkan-x64",
        }
    }
}

/// Select a published target from the host OS/arch and a Linux backend.
pub fn target_for(os: TargetOs, arch: TargetArch, linux_backend: LinuxBackend) -> Option<Target> {
    match (os, arch, linux_backend) {
        (TargetOs::MacOS, TargetArch::Aarch64, _) => Some(Target::MacOSAarch64),
        (TargetOs::MacOS, TargetArch::X86_64, _) => Some(Target::MacOSX86_64),
        (TargetOs::Linux, TargetArch::Aarch64, LinuxBackend::Cpu) => Some(Target::LinuxCpuAarch64),
        (TargetOs::Linux, TargetArch::X86_64, LinuxBackend::Cpu) => Some(Target::LinuxCpuX86_64),
        (TargetOs::Linux, TargetArch::Aarch64, LinuxBackend::Vulkan) => {
            Some(Target::LinuxVulkanAarch64)
        }
        (TargetOs::Linux, TargetArch::X86_64, LinuxBackend::Vulkan) => {
            Some(Target::LinuxVulkanX86_64)
        }
    }
}

/// A downloadable engine archive and its pinned digest (when known).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artifact {
    pub target: Target,
    pub url: String,
    pub file_name: String,
    pub sha256: Option<String>,
    pub bundle_name: String,
}

/// The archive URL for a target/tag. Mirrors upstream's asset naming.
pub fn artifact_url(version: &str, target: Target) -> String {
    let file = artifact_file_name(version, target);
    format!("{RELEASE_BASE}/{version}/{file}")
}

pub fn artifact_file_name(version: &str, target: Target) -> String {
    format!("llama-{version}-bin-{}.tar.gz", target.asset_stem())
}

/// Resolve a target's archive, attaching the pinned digest from the release
/// manifest when `version` matches the app's pinned tag.
pub fn artifact_for(version: &str, target: Target) -> Artifact {
    let release = target_release();
    let sha256 = (release.version == version)
        .then(|| {
            release
                .artifacts
                .as_ref()
                .and_then(|m| m.get(target.key()).cloned())
                .or_else(|| (target == Target::MacOSAarch64).then(|| release.sha256.clone()))
        })
        .flatten();
    Artifact {
        target,
        url: artifact_url(version, target),
        file_name: artifact_file_name(version, target),
        sha256,
        bundle_name: target.bundle_name(version),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_cpu_assets_match_upstream_names() {
        assert_eq!(
            artifact_url("b12005", Target::LinuxCpuX86_64),
            "https://github.com/ggml-org/llama.cpp/releases/download/b12005/llama-b12005-bin-ubuntu-x64.tar.gz"
        );
        assert_eq!(
            artifact_file_name("b12005", Target::LinuxCpuAarch64),
            "llama-b12005-bin-ubuntu-arm64.tar.gz"
        );
    }

    #[test]
    fn linux_vulkan_assets_match_upstream_names() {
        assert_eq!(
            artifact_file_name("b12005", Target::LinuxVulkanX86_64),
            "llama-b12005-bin-ubuntu-vulkan-x64.tar.gz"
        );
        assert_eq!(
            artifact_file_name("b12005", Target::LinuxVulkanAarch64),
            "llama-b12005-bin-ubuntu-vulkan-arm64.tar.gz"
        );
    }

    #[test]
    fn pinned_release_carries_linux_digests() {
        let release = target_release();
        let x64 = artifact_for(&release.version, Target::LinuxCpuX86_64);
        assert_eq!(
            x64.sha256.as_deref(),
            Some("fca5e36156c4b6d0eebd98ae807d8baad7684f50c14425cca67cccc94de722f1")
        );
        let arm = artifact_for(&release.version, Target::LinuxVulkanAarch64);
        assert_eq!(
            arm.sha256.as_deref(),
            Some("dbb907a07bbc8eee05442002b2626bfb0fee20d816463cda2f4f8d24c5034d69")
        );
        assert_eq!(x64.bundle_name, format!("llama-{}", release.version));
    }

    #[test]
    fn unpinned_version_has_no_digest() {
        let a = artifact_for("b1", Target::LinuxCpuX86_64);
        assert_eq!(a.sha256, None);
    }

    #[test]
    fn mac_arm_artifact_stays_the_legacy_url() {
        assert_eq!(
            artifact_url("b12005", Target::MacOSAarch64),
            super::super::installer::artifact_url("b12005")
        );
    }
}
