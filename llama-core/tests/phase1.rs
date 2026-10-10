//! Fase 1 equivalence harness. Shares `Fixtures/*.json` with the Swift
//! `LlamaTests` target; both sides must reproduce the pinned expectations.

use llama_core::config::{aliases, ini};
use llama_core::hf::{GGUFQuant, SidecarPicker};
use llama_core::models::{compat, model, ContextTier, DraftHead, ModelIdParser, ModelMem};
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

// MARK: - GGUFQuant

#[derive(Deserialize)]
struct LabelCase {
    path: String,
    label: Option<String>,
}

#[derive(Deserialize)]
struct TagCase {
    path: String,
    requested: Option<String>,
    tag: String,
}

#[derive(Deserialize)]
struct BitsPathCase {
    path: String,
    bits: i64,
}

#[derive(Deserialize)]
struct BitsTagCase {
    tag: String,
    bits: i64,
}

#[derive(Deserialize)]
struct CanonicalCase {
    label: String,
    tag: String,
}

#[derive(Deserialize)]
struct QuantFixture {
    #[serde(rename = "labelCases")]
    label_cases: Vec<LabelCase>,
    #[serde(rename = "tagCases")]
    tag_cases: Vec<TagCase>,
    #[serde(rename = "bitsPathCases")]
    bits_path_cases: Vec<BitsPathCase>,
    #[serde(rename = "bitsTagCases")]
    bits_tag_cases: Vec<BitsTagCase>,
    #[serde(rename = "canonicalCases")]
    canonical_cases: Vec<CanonicalCase>,
}

#[test]
fn gguf_quant_fixtures() {
    let f: QuantFixture = load("gguf_quant.json");
    for c in &f.label_cases {
        assert_eq!(
            GGUFQuant::parse_label(&c.path),
            c.label,
            "parseLabel({})",
            c.path
        );
    }
    for c in &f.tag_cases {
        assert_eq!(
            GGUFQuant::tag_for_path(&c.path, c.requested.as_deref()),
            c.tag,
            "tag(forPath: {})",
            c.path
        );
    }
    for c in &f.bits_path_cases {
        assert_eq!(
            GGUFQuant::quant_bits_for_path(&c.path),
            c.bits,
            "quantBits(forPath: {})",
            c.path
        );
    }
    for c in &f.bits_tag_cases {
        assert_eq!(
            GGUFQuant::quant_bits_for_tag(&c.tag),
            c.bits,
            "quantBits(forTag: {:?})",
            c.tag
        );
    }
    for c in &f.canonical_cases {
        assert_eq!(
            GGUFQuant::canonical_tag(&c.label),
            c.tag,
            "canonicalTag({})",
            c.label
        );
    }
}

// MARK: - SidecarPicker

#[derive(Deserialize)]
struct BoolCase {
    path: String,
    value: bool,
}

#[derive(Deserialize)]
struct MmprojCase {
    names: Vec<String>,
    #[serde(rename = "mainPath")]
    main_path: String,
    expected: Option<String>,
}

#[derive(Deserialize)]
struct DraftSelectCase {
    names: Vec<String>,
    #[serde(rename = "mainPath")]
    main_path: String,
    tag: Option<String>,
    expected: Option<String>,
}

#[derive(Deserialize)]
struct SidecarFixture {
    #[serde(rename = "isMmproj")]
    is_mmproj: Vec<BoolCase>,
    #[serde(rename = "isDraftHead")]
    is_draft_head: Vec<BoolCase>,
    mmproj: Vec<MmprojCase>,
    #[serde(rename = "draftHead")]
    draft_head: Vec<DraftSelectCase>,
}

#[test]
fn sidecar_picker_fixtures() {
    let f: SidecarFixture = load("sidecar_picker.json");
    for c in &f.is_mmproj {
        assert_eq!(
            SidecarPicker::is_mmproj(&c.path),
            c.value,
            "isMmproj({})",
            c.path
        );
    }
    for c in &f.is_draft_head {
        assert_eq!(
            SidecarPicker::is_draft_head(&c.path),
            c.value,
            "isDraftHead({})",
            c.path
        );
    }
    for c in &f.mmproj {
        assert_eq!(
            SidecarPicker::mmproj(&c.names, &c.main_path),
            c.expected,
            "mmproj(mainPath: {})",
            c.main_path
        );
    }
    for c in &f.draft_head {
        assert_eq!(
            SidecarPicker::draft_head(&c.names, &c.main_path, c.tag.as_deref()),
            c.expected,
            "draftHead(mainPath: {})",
            c.main_path
        );
    }
}

// MARK: - DraftHead

#[derive(Deserialize)]
struct FromSidecarCase {
    path: String,
    kind: Option<String>,
}

#[derive(Deserialize)]
struct DescribeCase {
    kind: String,
    prefix: String,
    #[serde(rename = "specType")]
    spec_type: String,
    #[serde(rename = "draftNMax")]
    draft_n_max: i32,
    label: String,
}

#[derive(Deserialize)]
struct DraftHeadFixture {
    #[serde(rename = "fromSidecarPath")]
    from_sidecar_path: Vec<FromSidecarCase>,
    describe: Vec<DescribeCase>,
    #[serde(rename = "preferenceOrder")]
    preference_order: Vec<String>,
}

#[test]
fn draft_head_fixtures() {
    let f: DraftHeadFixture = load("draft_head.json");
    for c in &f.from_sidecar_path {
        let got = DraftHead::from_sidecar_path(&c.path).map(DraftHead::as_str);
        assert_eq!(got, c.kind.as_deref(), "fromSidecarPath({})", c.path);
    }
    for c in &f.describe {
        let k = DraftHead::from_str_name(&c.kind).expect("known kind");
        assert_eq!(k.sidecar_prefix(), c.prefix);
        assert_eq!(k.spec_type(), c.spec_type);
        assert_eq!(k.draft_n_max(), c.draft_n_max);
        assert_eq!(k.label(), c.label);
    }
    let got: Vec<&str> = DraftHead::PREFERENCE_ORDER
        .iter()
        .map(|k| k.as_str())
        .collect();
    let want: Vec<&str> = f.preference_order.iter().map(String::as_str).collect();
    assert_eq!(got, want);
}

// MARK: - ModelIdParser + Model helpers

#[derive(Deserialize)]
struct ParsedCase {
    id: String,
    org: Option<String>,
    #[serde(rename = "displayOrg")]
    display_org: Option<String>,
    name: String,
    params: Option<String>,
    quant: Option<String>,
    tags: Vec<String>,
}

#[derive(Deserialize)]
struct KeyCase {
    id: String,
    key: String,
}

#[derive(Deserialize)]
struct PlainNameCase {
    id: String,
    #[serde(rename = "showTags")]
    show_tags: bool,
    expected: String,
}

#[derive(Deserialize)]
struct MakeIdCase {
    org: String,
    repo: String,
    tag: String,
    id: String,
}

#[derive(Deserialize)]
struct IdBaseCase {
    id: String,
    base: String,
}

#[derive(Deserialize)]
struct OrderCase {
    #[serde(rename = "lhsId")]
    lhs_id: String,
    #[serde(rename = "rhsId")]
    rhs_id: String,
    order: i32,
}

#[derive(Deserialize)]
struct ModelIdFixture {
    parse: Vec<ParsedCase>,
    #[serde(rename = "displayKey")]
    display_key: Vec<KeyCase>,
    #[serde(rename = "plainName")]
    plain_name: Vec<PlainNameCase>,
    #[serde(rename = "makeId")]
    make_id: Vec<MakeIdCase>,
    #[serde(rename = "idBaseOf")]
    id_base_of: Vec<IdBaseCase>,
    #[serde(rename = "sortKey")]
    sort_key: Vec<KeyCase>,
    #[serde(rename = "displayOrder")]
    display_order: Vec<OrderCase>,
}

#[test]
fn model_id_fixtures() {
    let f: ModelIdFixture = load("model_id.json");
    for c in &f.parse {
        let p = ModelIdParser::parse(&c.id);
        assert_eq!(p.org, c.org, "parse({}).org", c.id);
        assert_eq!(
            p.display_org(),
            c.display_org.as_deref(),
            "parse({}).displayOrg",
            c.id
        );
        assert_eq!(p.name, c.name, "parse({}).name", c.id);
        assert_eq!(p.params, c.params, "parse({}).params", c.id);
        assert_eq!(p.quant, c.quant, "parse({}).quant", c.id);
        assert_eq!(p.tags, c.tags, "parse({}).tags", c.id);
    }
    for c in &f.display_key {
        assert_eq!(
            ModelIdParser::parse(&c.id).display_key(),
            c.key,
            "displayKey({})",
            c.id
        );
    }
    for c in &f.plain_name {
        assert_eq!(
            ModelIdParser::parse(&c.id).plain_name(c.show_tags),
            c.expected,
            "plainName({}, showTags: {})",
            c.id,
            c.show_tags
        );
    }
    for c in &f.make_id {
        assert_eq!(
            model::make_id(&c.org, &c.repo, &c.tag),
            c.id,
            "makeId({})",
            c.id
        );
    }
    for c in &f.id_base_of {
        assert_eq!(model::id_base_of(&c.id), c.base, "idBaseOf({})", c.id);
    }
    for c in &f.sort_key {
        assert_eq!(model::sort_key(&c.id), c.key, "sortKey({})", c.id);
    }
    for c in &f.display_order {
        assert_eq!(
            model::display_order(&c.lhs_id, &c.rhs_id) as i32,
            c.order,
            "displayOrder({}, {})",
            c.lhs_id,
            c.rhs_id
        );
    }
}

// MARK: - ContextTier

#[derive(Deserialize)]
struct TierCase {
    #[serde(rename = "rawValue")]
    raw_value: i64,
    #[serde(rename = "shortLabel")]
    short_label: String,
    label: String,
    suffix: String,
}

#[derive(Deserialize)]
struct TierFixture {
    tiers: Vec<TierCase>,
}

#[test]
fn context_tier_fixtures() {
    let f: TierFixture = load("context_tier.json");
    assert_eq!(f.tiers.len(), ContextTier::ALL.len());
    for (tier, c) in ContextTier::ALL.iter().zip(&f.tiers) {
        assert_eq!(tier.raw_value(), c.raw_value);
        assert_eq!(tier.short_label(), c.short_label);
        assert_eq!(tier.label(), c.label);
        assert_eq!(tier.suffix(), c.suffix);
    }
}

// MARK: - Model+Compatibility

#[derive(Deserialize)]
struct DeviceBudgetCase {
    #[serde(rename = "workingSetMb")]
    working_set_mb: f64,
    #[serde(rename = "physicalMb")]
    physical_mb: f64,
    #[serde(rename = "useFullWorkingSet")]
    use_full_working_set: bool,
    mb: f64,
}

#[derive(Deserialize)]
struct FitTargetCase {
    #[serde(rename = "workingSetMb")]
    working_set_mb: f64,
    #[serde(rename = "budgetMb")]
    budget_mb: f64,
    mb: f64,
}

#[derive(Deserialize)]
struct WeightFitsCase {
    bytes: Option<i64>,
    #[serde(rename = "budgetMb")]
    budget_mb: f64,
    value: bool,
}

fn mem(v: &serde_json::Value) -> ModelMem {
    ModelMem {
        ctx_window: v["ctxWindow"].as_i64().unwrap_or(0),
        file_size: v["fileSize"].as_i64().unwrap_or(0),
        ctx_bytes_per_1k_tokens: v["ctxBytesPer1kTokens"].as_i64().unwrap_or(0),
        resident_bytes: v["residentBytes"].as_i64().unwrap_or(0),
    }
}

fn tiers(v: &serde_json::Value) -> Vec<i64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

#[derive(Deserialize)]
struct CompatFixture {
    #[serde(rename = "deviceBudget")]
    device_budget: Vec<DeviceBudgetCase>,
    #[serde(rename = "fitTarget")]
    fit_target: Vec<FitTargetCase>,
    #[serde(rename = "estimatedWeightFits")]
    estimated_weight_fits: Vec<WeightFitsCase>,
    info: Vec<serde_json::Value>,
    #[serde(rename = "runtimeMemoryMb")]
    runtime_memory_mb: Vec<serde_json::Value>,
    #[serde(rename = "supportedTiers")]
    supported_tiers: Vec<serde_json::Value>,
    #[serde(rename = "displayTiers")]
    display_tiers: Vec<serde_json::Value>,
    #[serde(rename = "nativeMaxTier")]
    native_max_tier: Vec<serde_json::Value>,
    #[serde(rename = "effectiveTier")]
    effective_tier: Vec<serde_json::Value>,
}

#[test]
fn compat_fixtures() {
    let f: CompatFixture = load("compat.json");

    for c in &f.device_budget {
        assert_eq!(
            compat::device_memory_budget_mb(
                c.working_set_mb,
                c.physical_mb,
                c.use_full_working_set
            ),
            c.mb
        );
    }
    for c in &f.fit_target {
        assert_eq!(compat::fit_target_mb(c.working_set_mb, c.budget_mb), c.mb);
    }
    for c in &f.estimated_weight_fits {
        assert_eq!(compat::estimated_weight_fits(c.bytes, c.budget_mb), c.value);
    }
    for c in &f.info {
        let got = mem(c).compatibility_info(
            c["ctxWindowTokens"].as_f64().unwrap(),
            c["budgetMb"].as_f64().unwrap(),
        );
        assert_eq!(
            got.is_compatible,
            c["isCompatible"].as_bool().unwrap(),
            "info {c}"
        );
        assert_eq!(
            got.incompatibility_summary,
            c["incompatibilitySummary"].as_str().map(str::to_string),
            "info {c}"
        );
    }
    for c in &f.runtime_memory_mb {
        assert_eq!(
            mem(c).runtime_memory_usage_mb(c["ctxWindowTokens"].as_f64().unwrap()),
            c["mb"].as_u64().unwrap()
        );
    }
    for c in &f.supported_tiers {
        let got: Vec<i64> = mem(c)
            .supported_context_tiers(c["budgetMb"].as_f64().unwrap())
            .iter()
            .map(|t| t.raw_value())
            .collect();
        assert_eq!(got, tiers(&c["tiers"]));
    }
    for c in &f.display_tiers {
        let got: Vec<i64> = mem(c)
            .display_context_tiers()
            .iter()
            .map(|t| t.raw_value())
            .collect();
        assert_eq!(got, tiers(&c["tiers"]));
    }
    for c in &f.native_max_tier {
        assert_eq!(
            mem(c).native_max_tier().map(|t| t.raw_value()),
            c["tier"].as_i64()
        );
    }
    for c in &f.effective_tier {
        let selected = c["selected"].as_i64().and_then(ContextTier::from_raw);
        assert_eq!(
            mem(c)
                .effective_ctx_tier(c["budgetMb"].as_f64().unwrap(), selected)
                .map(|t| t.raw_value()),
            c["tier"].as_i64()
        );
    }
}

// MARK: - UserModelOverrides (INI)

#[derive(Deserialize, Clone, Debug, PartialEq)]
struct TestPair {
    key: String,
    value: String,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
struct TestSection {
    name: String,
    pairs: Vec<TestPair>,
}

impl TestSection {
    fn to_ini(&self) -> ini::Section {
        let mut s = ini::Section::new(self.name.clone());
        for p in &self.pairs {
            s.set(&p.key, &p.value);
        }
        s
    }
    fn from_ini(s: &ini::Section) -> TestSection {
        TestSection {
            name: s.name.clone(),
            pairs: s
                .pairs
                .iter()
                .map(|(k, v)| TestPair {
                    key: k.clone(),
                    value: v.clone(),
                })
                .collect(),
        }
    }
    fn assert_eq(&self, other: &ini::Section) {
        assert_eq!(&TestSection::from_ini(other), self);
    }
}

#[derive(Deserialize)]
struct ParseCase {
    text: String,
    sections: Vec<TestSection>,
}

#[derive(Deserialize)]
struct SerializeCase {
    sections: Vec<TestSection>,
    text: String,
}

#[derive(Deserialize)]
struct ApplyCase {
    generated: Vec<TestSection>,
    overrides: Vec<TestSection>,
    suspended: bool,
    expected: Vec<TestSection>,
}

#[derive(Deserialize)]
struct OverridesFixture {
    #[serde(rename = "parseCases")]
    parse_cases: Vec<ParseCase>,
    #[serde(rename = "serializeCases")]
    serialize_cases: Vec<SerializeCase>,
    #[serde(rename = "applyCases")]
    apply_cases: Vec<ApplyCase>,
}

fn to_ini_all(v: &[TestSection]) -> Vec<ini::Section> {
    v.iter().map(TestSection::to_ini).collect()
}

#[test]
fn user_model_overrides_fixtures() {
    let f: OverridesFixture = load("user_model_overrides.json");
    for c in &f.parse_cases {
        let got = ini::parse(&c.text);
        assert_eq!(got.len(), c.sections.len());
        for (want, g) in c.sections.iter().zip(&got) {
            want.assert_eq(g);
        }
    }
    for c in &f.serialize_cases {
        let text: String = to_ini_all(&c.sections)
            .iter()
            .map(ini::Section::serialized)
            .collect();
        assert_eq!(text, c.text);
    }
    for c in &f.apply_cases {
        let got = ini::apply(
            &to_ini_all(&c.generated),
            &to_ini_all(&c.overrides),
            c.suspended,
        );
        assert_eq!(got.len(), c.expected.len());
        for (want, g) in c.expected.iter().zip(&got) {
            want.assert_eq(g);
        }
    }
}

// MARK: - ModelAliases

#[derive(Deserialize)]
struct AliasCase {
    name: String,
    #[serde(rename = "modelId")]
    model_id: String,
}

#[derive(Deserialize)]
struct ValidationCase {
    name: String,
    aliases: Vec<AliasCase>,
    replacing: Option<AliasCase>,
    #[serde(rename = "overrideNames")]
    override_names: Vec<String>,
    error: Option<String>,
}

#[derive(Deserialize)]
struct RemoveConflictsCase {
    sections: Vec<TestSection>,
    expected: Vec<TestSection>,
}

#[derive(Deserialize)]
struct SetAliasCase {
    section: TestSection,
    names: Vec<String>,
    expected: TestSection,
}

#[derive(Deserialize)]
struct AliasesFixture {
    #[serde(rename = "validationCases")]
    validation_cases: Vec<ValidationCase>,
    #[serde(rename = "removeConflictsCases")]
    remove_conflicts_cases: Vec<RemoveConflictsCase>,
    #[serde(rename = "setAliasCases")]
    set_alias_cases: Vec<SetAliasCase>,
}

fn to_alias(a: &AliasCase) -> aliases::Alias {
    aliases::Alias {
        name: a.name.clone(),
        model_id: a.model_id.clone(),
    }
}

#[test]
fn model_aliases_fixtures() {
    let f: AliasesFixture = load("model_aliases.json");
    for c in &f.validation_cases {
        let aliases: Vec<aliases::Alias> = c.aliases.iter().map(to_alias).collect();
        let replacing = c.replacing.as_ref().map(to_alias);
        assert_eq!(
            aliases::validation_error(&c.name, &aliases, replacing.as_ref(), &c.override_names),
            c.error,
            "validationError({})",
            c.name
        );
    }
    for c in &f.remove_conflicts_cases {
        let got = aliases::remove_conflicts(&to_ini_all(&c.sections));
        assert_eq!(got.len(), c.expected.len());
        for (want, g) in c.expected.iter().zip(&got) {
            want.assert_eq(g);
        }
    }
    for c in &f.set_alias_cases {
        let mut section = c.section.to_ini();
        aliases::set_alias(&mut section, &c.names);
        c.expected.assert_eq(&section);
    }
}
