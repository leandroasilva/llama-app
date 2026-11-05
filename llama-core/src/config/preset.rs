//! Pure `models.ini` generation shared by the macOS and Linux hosts.
//!
//! The Swift host still owns the scan and file write, but the content shape is
//! pinned here: a `[*]` performance section, one section per installed model,
//! then the user's `models.user.ini` overrides. Aliases are written on each
//! model section and conflict-checked at the end, matching
//! `ModelManager.generatedModelSections`.

use crate::config::aliases::{self, Alias};
use crate::config::ini::Section;
use crate::models::DraftHead;

/// A KV cache element type understood by llama.cpp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvCacheType {
    F16,
    Q8_0,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Iq4Nl,
}

impl KvCacheType {
    pub const ALL: [KvCacheType; 7] = [
        Self::F16,
        Self::Q8_0,
        Self::Q4_0,
        Self::Q4_1,
        Self::Q5_0,
        Self::Q5_1,
        Self::Iq4Nl,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Iq4Nl => "iq4_nl",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|t| t.as_str() == value)
    }
}

/// How the app picks a context length when the user has no per-model choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextLengthPolicy {
    LargestFitting,
    Minimum,
}

impl ContextLengthPolicy {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "largestFitting" => Some(Self::LargestFitting),
            "minimum" => Some(Self::Minimum),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::LargestFitting => "largestFitting",
            Self::Minimum => "minimum",
        }
    }
}

/// Memory-related llama.cpp preset keys (`PerformanceConfig` in Swift).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerformanceConfig {
    pub flash_attention: bool,
    pub kv_cache_type_k: KvCacheType,
    pub kv_cache_type_v: KvCacheType,
    pub parallel: i64,
    pub batch_size: i64,
    pub ubatch_size: i64,
    pub no_kv_offload: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            flash_attention: true,
            kv_cache_type_k: KvCacheType::Q8_0,
            kv_cache_type_v: KvCacheType::Q8_0,
            parallel: 1,
            batch_size: 512,
            ubatch_size: 256,
            no_kv_offload: false,
        }
    }
}

impl PerformanceConfig {
    pub fn normalized(self) -> Self {
        let mut copy = self;
        copy.parallel = copy.parallel.clamp(1, 8);
        if ![128, 256, 512, 1024, 2048].contains(&copy.batch_size) {
            copy.batch_size = 512;
        }
        if ![64, 128, 256, 512, 1024, 2048].contains(&copy.ubatch_size) {
            copy.ubatch_size = 256;
        }
        if copy.ubatch_size > copy.batch_size {
            copy.ubatch_size = copy.batch_size;
        }
        copy
    }

    pub fn ini_pairs(self) -> Vec<(String, String)> {
        let c = self.normalized();
        vec![
            (
                "flash-attn".into(),
                if c.flash_attention { "on" } else { "off" }.into(),
            ),
            ("cache-type-k".into(), c.kv_cache_type_k.as_str().into()),
            ("cache-type-v".into(), c.kv_cache_type_v.as_str().into()),
            ("parallel".into(), c.parallel.to_string()),
            ("batch-size".into(), c.batch_size.to_string()),
            ("ubatch-size".into(), c.ubatch_size.to_string()),
            (
                "no-kv-offload".into(),
                if c.no_kv_offload { "true" } else { "false" }.into(),
            ),
        ]
    }
}

/// Everything one generated model section needs. Paths are absolute, as the
/// server reads them from `models.ini`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelPreset {
    pub id: String,
    pub model_path: String,
    pub ctx_size: i64,
    pub mmproj_path: Option<String>,
    pub draft_sidecar_kind: Option<DraftHead>,
    pub draft_sidecar_path: Option<String>,
    pub uses_mtp: bool,
    pub is_decision_model: bool,
    /// Per-model performance keys, when they differ from `[*]`.
    pub performance_override: Option<PerformanceConfig>,
    pub aliases: Vec<String>,
}

/// The generated sections before user overrides are applied.
pub fn generated_sections(
    global_performance: &PerformanceConfig,
    models: &[ModelPreset],
) -> Vec<Section> {
    let mut sections = Vec::with_capacity(models.len() + 1);

    let mut global = Section::new("*");
    for (key, value) in global_performance.ini_pairs() {
        global.set(&key, &value);
    }
    sections.push(global);

    for model in models {
        let mut section = Section::new(&model.id);
        section.set("model", &model.model_path);
        section.set("ctx-size", &model.ctx_size.to_string());
        if let Some(mmproj) = &model.mmproj_path {
            section.set("mmproj", mmproj);
        }

        let draft_kind = model
            .draft_sidecar_kind
            .or_else(|| model.uses_mtp.then_some(DraftHead::Mtp));
        if let Some(kind) = draft_kind {
            section.set("spec-type", kind.spec_type());
            if let Some(sidecar) = &model.draft_sidecar_path {
                section.set("spec-draft-model", sidecar);
            }
            section.set("spec-draft-n-max", &kind.draft_n_max().to_string());
        }

        if let Some(perf) = &model.performance_override {
            if perf.normalized() != global_performance.normalized() {
                for (key, value) in perf.ini_pairs() {
                    section.set(&key, &value);
                }
            }
        }

        section.set(
            "tags",
            if model.is_decision_model {
                "decision"
            } else {
                "chat"
            },
        );
        aliases::set_alias(&mut section, &model.aliases);
        sections.push(section);
    }

    sections
}

/// The comment block prepended to the generated file. `user_config_path` may be
/// shortened with `~` by the host; it is only display text.
pub fn generated_file_header(user_config_path: &str) -> String {
    format!(
        "; Generated by Llama from your installed models -- edits here are lost on\n\
         ; the next launch. To change these settings, or add your own, edit:\n\
         ;   {user_config_path}\n\n\n"
    )
}

/// Seeded, comment-only `models.user.ini`. Mirrors the macOS template so the
/// escape hatch is discoverable on both hosts.
pub fn user_ini_template() -> &'static str {
    "; Custom settings for your models, applied on top of the ones the app picks.\n\
     ;\n\
     ; The app regenerates its own config from a scan of your model cache every\n\
     launch. This file is yours -- the app only ever reads it, so your edits\n\
     survive. Anything you set here wins, including context size.\n\
     ;\n\
     ; Use the same section headers the app uses (org/repo:QUANT). You only need\n\
     the keys you want to change:\n\
     ;\n\
     ;   [ggml-org/gemma-4-E4B-it-GGUF:Q8_0]\n\
     ;   temp = 0.7\n\
     ;   top-p = 0.8\n\
     ;   ctx-size = 32768\n\
     ;\n\
     ; A section for a model the app didn't find is passed through as-is, which\n\
     is how you point at weights or a draft model from another repo:\n\
     ;\n\
     ;   [unsloth/DeepSeek-V4-Flash-0731-GGUF:UD-Q2_K_XL]\n\
     ;   model = /path/to/DeepSeek-V4-Flash-UD-Q2_K_XL.gguf\n\
     ;   spec-type = draft-dspark\n\
     ;   spec-draft-model = /path/to/draft.gguf\n\
     ;\n\
     ; Keys are `llama serve` options without the leading dashes. Changes apply\n\
     the next time the model list is refreshed -- restart the app if unsure.\n\
     ;\n\
     ; Setting ctx-size here takes over from the context picker in the app, which\n\
     shows the pinned value instead. Remove the key to get the picker back.\n\n"
}

/// Merge generated sections with overrides, drop alias conflicts, and render
/// the complete `models.ini` text. `suspended` mirrors the host's recovery from
/// an unknown key in the user file.
pub fn render_models_ini(
    user_config_path: &str,
    global_performance: &PerformanceConfig,
    models: &[ModelPreset],
    overrides: &[Section],
    suspended: bool,
) -> String {
    let generated = generated_sections(global_performance, models);
    let merged =
        aliases::remove_conflicts(&crate::config::ini::apply(&generated, overrides, suspended));
    let mut out = generated_file_header(user_config_path);
    for section in merged {
        out.push_str(&section.serialized());
    }
    out
}

/// The aliases that belong on a model's section, in stored order.
pub fn aliases_for(aliases: &[Alias], model_id: &str) -> Vec<String> {
    aliases
        .iter()
        .filter(|a| a.model_id == model_id)
        .map(|a| a.name.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model() -> ModelPreset {
        ModelPreset {
            id: "org/Model:Q4_K_M".into(),
            model_path: "/cache/model.gguf".into(),
            ctx_size: 32768,
            mmproj_path: None,
            draft_sidecar_kind: None,
            draft_sidecar_path: None,
            uses_mtp: false,
            is_decision_model: false,
            performance_override: None,
            aliases: vec![],
        }
    }

    #[test]
    fn global_section_uses_defaults_in_stable_order() {
        let sections = generated_sections(&PerformanceConfig::default(), &[]);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].name, "*");
        assert_eq!(
            sections[0].pairs,
            vec![
                ("flash-attn".into(), "on".into()),
                ("cache-type-k".into(), "q8_0".into()),
                ("cache-type-v".into(), "q8_0".into()),
                ("parallel".into(), "1".into()),
                ("batch-size".into(), "512".into()),
                ("ubatch-size".into(), "256".into()),
                ("no-kv-offload".into(), "false".into()),
            ]
        );
    }

    #[test]
    fn model_section_carries_path_ctx_tags_and_spec() {
        let mut m = model();
        m.mmproj_path = Some("/cache/mmproj.gguf".into());
        m.draft_sidecar_kind = Some(DraftHead::Dflash);
        m.draft_sidecar_path = Some("/cache/draft.gguf".into());
        m.is_decision_model = true;
        m.aliases = vec!["fast".into()];
        let sections = generated_sections(&PerformanceConfig::default(), &[m]);
        let s = &sections[1];
        assert_eq!(s.name, "org/Model:Q4_K_M");
        assert_eq!(s.value("model"), Some("/cache/model.gguf"));
        assert_eq!(s.value("ctx-size"), Some("32768"));
        assert_eq!(s.value("mmproj"), Some("/cache/mmproj.gguf"));
        assert_eq!(s.value("spec-type"), Some("draft-dflash"));
        assert_eq!(s.value("spec-draft-model"), Some("/cache/draft.gguf"));
        assert_eq!(s.value("spec-draft-n-max"), Some("7"));
        assert_eq!(s.value("tags"), Some("decision"));
        assert_eq!(s.value("alias"), Some("fast"));
    }

    #[test]
    fn per_model_performance_only_when_different() {
        let m = model();
        let sections = generated_sections(&PerformanceConfig::default(), &[m]);
        assert!(sections[1].value("flash-attn").is_none());

        let mut m = model();
        m.performance_override = Some(PerformanceConfig {
            parallel: 2,
            ..PerformanceConfig::default()
        });
        let sections = generated_sections(&PerformanceConfig::default(), &[m]);
        assert_eq!(sections[1].value("parallel"), Some("2"));
        assert_eq!(sections[1].value("flash-attn"), Some("on"));
    }

    #[test]
    fn mtp_head_gets_spec_type_without_sidecar_path() {
        let mut m = model();
        m.uses_mtp = true;
        let sections = generated_sections(&PerformanceConfig::default(), &[m]);
        assert_eq!(sections[1].value("spec-type"), Some("draft-mtp"));
        assert_eq!(sections[1].value("spec-draft-model"), None);
    }

    #[test]
    fn overrides_are_merged_and_alias_conflicts_removed() {
        let m = ModelPreset {
            aliases: vec!["org/Model:Q4_K_M".into()],
            ..model()
        };
        let overrides = crate::config::ini::parse("[org/Model:Q4_K_M]\ntemp = 0.7\n");
        let text = render_models_ini(
            "~/.config/llama/models.user.ini",
            &PerformanceConfig::default(),
            &[m],
            &overrides,
            false,
        );
        assert!(text.contains("temp = 0.7"));
        assert!(!text.contains("alias = org/Model:Q4_K_M"));
        assert!(text.contains(";   ~/.config/llama/models.user.ini"));
    }

    #[test]
    fn aliases_for_filters_by_model() {
        let aliases = vec![
            Alias {
                name: "fast".into(),
                model_id: "a".into(),
            },
            Alias {
                name: "other".into(),
                model_id: "b".into(),
            },
        ];
        assert_eq!(aliases_for(&aliases, "a"), vec!["fast"]);
    }
}
