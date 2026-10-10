import Foundation
import XCTest

@testable import Llama

/// Fase 1 equivalence: the Swift originals and the shared `Fixtures/*.json`
/// corpus must agree. Where the Swift API is pure and injectable the test
/// asserts Swift == fixture directly; the cases that depend on live device
/// memory or the `UserSettings` store (not injectable yet) are pinned on the
/// Rust side by `cargo test` and spot-checked through the FFI here.
final class Phase1FixtureTests: XCTestCase {

  // MARK: - Fixture loading

  private static let fixturesDir = Fixtures.directory

  private func load<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
    let url = Self.fixturesDir.appendingPathComponent(name)
    return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
  }

  private func model(
    id: String = "x", ctxWindow: Int = 131_072, fileSize: Int64 = 0,
    ctxBytesPer1kTokens: Int = 0, residentBytes: Int = 0
  ) -> Model {
    Model(
      id: id, ctxWindow: ctxWindow, fileSize: fileSize,
      ctxBytesPer1kTokens: ctxBytesPer1kTokens, residentBytes: residentBytes,
      downloadUrl: URL(fileURLWithPath: "/tmp/model.gguf"))
  }

  // MARK: - GGUFQuant

  private struct QuantFixture: Decodable {
    struct LabelCase: Decodable { let path: String; let label: String? }
    struct TagCase: Decodable { let path: String; let requested: String?; let tag: String }
    struct BitsPathCase: Decodable { let path: String; let bits: Int }
    struct BitsTagCase: Decodable { let tag: String; let bits: Int }
    struct CanonicalCase: Decodable { let label: String; let tag: String }
    let labelCases: [LabelCase]
    let tagCases: [TagCase]
    let bitsPathCases: [BitsPathCase]
    let bitsTagCases: [BitsTagCase]
    let canonicalCases: [CanonicalCase]
  }

  func testGGUFQuantFixtures() throws {
    let f = try load("gguf_quant.json", as: QuantFixture.self)
    for c in f.labelCases {
      XCTAssertEqual(GGUFQuant.parseLabel(c.path), c.label, "parseLabel(\(c.path))")
    }
    for c in f.tagCases {
      XCTAssertEqual(GGUFQuant.tag(forPath: c.path, requested: c.requested), c.tag, "tag(\(c.path))")
    }
    for c in f.bitsPathCases {
      XCTAssertEqual(GGUFQuant.quantBits(forPath: c.path), c.bits, "quantBits(forPath: \(c.path))")
    }
    for c in f.bitsTagCases {
      XCTAssertEqual(GGUFQuant.quantBits(forTag: c.tag), c.bits, "quantBits(forTag: \(c.tag))")
    }
    for c in f.canonicalCases {
      XCTAssertEqual(GGUFQuant.canonicalTag(c.label), c.tag, "canonicalTag(\(c.label))")
    }
  }

  // MARK: - SidecarPicker

  private struct SidecarFixture: Decodable {
    struct BoolCase: Decodable { let path: String; let value: Bool }
    struct MmprojCase: Decodable { let names: [String]; let mainPath: String; let expected: String? }
    struct DraftCase: Decodable {
      let names: [String]; let mainPath: String; let tag: String?; let expected: String?
    }
    let isMmproj: [BoolCase]
    let isDraftHead: [BoolCase]
    let mmproj: [MmprojCase]
    let draftHead: [DraftCase]
  }

  func testSidecarPickerFixtures() throws {
    let f = try load("sidecar_picker.json", as: SidecarFixture.self)
    for c in f.isMmproj {
      XCTAssertEqual(SidecarPicker.isMmproj(c.path), c.value, "isMmproj(\(c.path))")
    }
    for c in f.isDraftHead {
      XCTAssertEqual(SidecarPicker.isDraftHead(c.path), c.value, "isDraftHead(\(c.path))")
    }
    for c in f.mmproj {
      XCTAssertEqual(
        SidecarPicker.mmproj(among: c.names, mainPath: c.mainPath), c.expected, "mmproj")
    }
    for c in f.draftHead {
      XCTAssertEqual(
        SidecarPicker.draftHead(among: c.names, mainPath: c.mainPath, tag: c.tag), c.expected,
        "draftHead")
    }
  }

  // MARK: - DraftHead

  private struct DraftHeadFixture: Decodable {
    struct FromSidecarCase: Decodable { let path: String; let kind: String? }
    struct DescribeCase: Decodable {
      let kind: String; let prefix: String; let specType: String; let draftNMax: Int; let label: String
    }
    let fromSidecarPath: [FromSidecarCase]
    let describe: [DescribeCase]
    let preferenceOrder: [String]
  }

  func testDraftHeadFixtures() throws {
    let f = try load("draft_head.json", as: DraftHeadFixture.self)
    for c in f.fromSidecarPath {
      XCTAssertEqual(DraftHead(sidecarPath: c.path)?.rawValue, c.kind, "fromSidecarPath(\(c.path))")
    }
    for c in f.describe {
      let k = try XCTUnwrap(DraftHead(rawValue: c.kind))
      XCTAssertEqual(k.sidecarPrefix, c.prefix)
      XCTAssertEqual(k.specType, c.specType)
      XCTAssertEqual(k.draftNMax, c.draftNMax)
      XCTAssertEqual(k.label, c.label)
    }
    XCTAssertEqual(DraftHead.preferenceOrder.map(\.rawValue), f.preferenceOrder)
  }

  // MARK: - ModelIdParser + Model helpers

  private struct ModelIdFixture: Decodable {
    struct ParsedCase: Decodable {
      let id: String; let org: String?; let displayOrg: String?; let name: String
      let params: String?; let quant: String?; let tags: [String]
    }
    struct KeyCase: Decodable { let id: String; let key: String }
    struct PlainNameCase: Decodable { let id: String; let showTags: Bool; let expected: String }
    struct MakeIdCase: Decodable { let org: String; let repo: String; let tag: String; let id: String }
    struct IdBaseCase: Decodable { let id: String; let base: String }
    struct OrderCase: Decodable { let lhsId: String; let rhsId: String; let order: Int }
    let parse: [ParsedCase]
    let displayKey: [KeyCase]
    let plainName: [PlainNameCase]
    let makeId: [MakeIdCase]
    let idBaseOf: [IdBaseCase]
    let sortKey: [KeyCase]
    let displayOrder: [OrderCase]
  }

  func testModelIdFixtures() throws {
    let f = try load("model_id.json", as: ModelIdFixture.self)
    for c in f.parse {
      let p = ModelIdParser.parse(c.id)
      XCTAssertEqual(p.org, c.org, "parse(\(c.id)).org")
      XCTAssertEqual(p.displayOrg, c.displayOrg, "parse(\(c.id)).displayOrg")
      XCTAssertEqual(p.name, c.name, "parse(\(c.id)).name")
      XCTAssertEqual(p.params, c.params, "parse(\(c.id)).params")
      XCTAssertEqual(p.quant, c.quant, "parse(\(c.id)).quant")
      XCTAssertEqual(p.tags, c.tags, "parse(\(c.id)).tags")
    }
    for c in f.displayKey {
      XCTAssertEqual(ModelIdParser.displayKey(c.id), c.key, "displayKey(\(c.id))")
    }
    for c in f.plainName {
      XCTAssertEqual(ModelIdParser.plainName(c.id, showTags: c.showTags), c.expected)
    }
    for c in f.makeId {
      XCTAssertEqual(Model.makeId(org: c.org, repo: c.repo, tag: c.tag), c.id)
    }
    for c in f.idBaseOf {
      XCTAssertEqual(model(id: c.id).idBase, c.base, "idBase(\(c.id))")
    }
    for c in f.displayOrder {
      let lhs = model(id: c.lhsId)
      let rhs = model(id: c.rhsId)
      switch c.order {
      case -1: XCTAssertTrue(Model.displayOrder(lhs, rhs), "\(c.lhsId) should sort before \(c.rhsId)")
      case 1: XCTAssertFalse(Model.displayOrder(lhs, rhs), "\(c.lhsId) should sort after \(c.rhsId)")
      default:
        XCTAssertFalse(Model.displayOrder(lhs, rhs))
        XCTAssertFalse(Model.displayOrder(rhs, lhs))
      }
    }
  }

  // MARK: - ContextTier

  private struct TierFixture: Decodable {
    struct TierCase: Decodable {
      let rawValue: Int; let shortLabel: String; let label: String; let suffix: String
    }
    let tiers: [TierCase]
  }

  func testContextTierFixtures() throws {
    let f = try load("context_tier.json", as: TierFixture.self)
    XCTAssertEqual(ContextTier.allCases.count, f.tiers.count)
    for (tier, c) in zip(ContextTier.allCases, f.tiers) {
      XCTAssertEqual(tier.rawValue, c.rawValue)
      XCTAssertEqual(tier.shortLabel, c.shortLabel)
      XCTAssertEqual(tier.label, c.label)
      XCTAssertEqual(tier.suffix, c.suffix)
    }
  }

  // MARK: - Compatibility (deterministic subset)

  private struct CompatFixture: Decodable {
    struct WeightFitsCase: Decodable { let bytes: Int64?; let budgetMb: Double; let value: Bool }
    struct RuntimeCase: Decodable {
      let ctxWindow: Int; let fileSize: Int64; let ctxBytesPer1kTokens: Int
      let residentBytes: Int; let ctxWindowTokens: Double; let mb: UInt64
    }
    struct DisplayTiersCase: Decodable {
      let ctxWindow: Int; let ctxBytesPer1kTokens: Int; let tiers: [Int]
    }
    struct NativeTierCase: Decodable { let ctxWindow: Int; let tier: Int? }
    let estimatedWeightFits: [WeightFitsCase]
    let runtimeMemoryMb: [RuntimeCase]
    let displayTiers: [DisplayTiersCase]
    let nativeMaxTier: [NativeTierCase]
  }

  func testCompatFixtures() throws {
    let f = try load("compat.json", as: CompatFixture.self)
    for c in f.estimatedWeightFits {
      XCTAssertEqual(
        Model.estimatedWeightFits(bytes: c.bytes, budgetMb: c.budgetMb), c.value)
    }
    for c in f.runtimeMemoryMb {
      let m = model(
        ctxWindow: c.ctxWindow, fileSize: c.fileSize,
        ctxBytesPer1kTokens: c.ctxBytesPer1kTokens, residentBytes: c.residentBytes)
      XCTAssertEqual(m.runtimeMemoryUsageMb(ctxWindowTokens: c.ctxWindowTokens), c.mb)
    }
    for c in f.displayTiers {
      let m = model(ctxWindow: c.ctxWindow, ctxBytesPer1kTokens: c.ctxBytesPer1kTokens)
      XCTAssertEqual(m.displayContextTiers.map(\.rawValue), c.tiers)
    }
    for c in f.nativeMaxTier {
      let m = model(ctxWindow: c.ctxWindow)
      XCTAssertEqual(m.nativeMaxTier?.rawValue, c.tier)
    }
  }

  // MARK: - UserModelOverrides + ModelAliases

  private struct TestPair: Codable, Equatable { let key: String; let value: String }
  private struct TestSection: Codable, Equatable { let name: String; let pairs: [TestPair] }

  private func toSection(_ t: TestSection) -> UserModelOverrides.Section {
    UserModelOverrides.Section(name: t.name, pairs: t.pairs.map { (key: $0.key, value: $0.value) })
  }
  private func toTest(_ s: UserModelOverrides.Section) -> TestSection {
    TestSection(name: s.name, pairs: s.pairs.map { TestPair(key: $0.key, value: $0.value) })
  }

  private struct OverridesFixture: Decodable {
    struct ParseCase: Decodable { let text: String; let sections: [TestSection] }
    struct SerializeCase: Decodable { let sections: [TestSection]; let text: String }
    struct ApplyCase: Decodable {
      let generated: [TestSection]; let overrides: [TestSection]
      let suspended: Bool; let expected: [TestSection]
    }
    let parseCases: [ParseCase]
    let serializeCases: [SerializeCase]
    let applyCases: [ApplyCase]
  }

  private struct AliasesFixture: Decodable {
    struct RemoveConflictsCase: Decodable {
      let sections: [TestSection]; let expected: [TestSection]
    }
    let removeConflictsCases: [RemoveConflictsCase]
  }

  func testOverridesAndAliasesFixtures() throws {
    let of = try load("user_model_overrides.json", as: OverridesFixture.self)
    // The parser now lives in the core (`ini.parse`); pin it to the fixture.
    for c in of.parseCases {
      let out = try XCTUnwrap(LlamaCore.callJSON("ini.parse", ["text": c.text]))
      let sections = try XCTUnwrap(out["sections"] as? [[String: Any]])
      XCTAssertEqual(sections.map(UserModelOverrides.Section.fromFFI).map(toTest), c.sections)
    }
    for c in of.serializeCases {
      let text = c.sections.map(toSection).map { $0.serialized() }.joined()
      XCTAssertEqual(text, c.text)
    }
    // The Swift wrapper reads the global suspension flag, so its non-suspended
    // path is pinned here; the suspended branch is pinned through `ini.apply`
    // directly (the wrapper delegates to it) below.
    for c in of.applyCases where !c.suspended {
      let got = UserModelOverrides.apply(
        to: c.generated.map(toSection), overrides: c.overrides.map(toSection))
      XCTAssertEqual(got.map(toTest), c.expected)
    }
    for c in of.applyCases {
      let out = try XCTUnwrap(
        LlamaCore.callJSON("ini.apply", [
          "generated": c.generated.map(toSection).map(\.ffiValue),
          "overrides": c.overrides.map(toSection).map(\.ffiValue),
          "suspended": c.suspended,
        ]))
      let sections = try XCTUnwrap(out["sections"] as? [[String: Any]])
      XCTAssertEqual(sections.map(UserModelOverrides.Section.fromFFI).map(toTest), c.expected)
    }

    let af = try load("model_aliases.json", as: AliasesFixture.self)
    for c in af.removeConflictsCases {
      let got = ModelAliases.removeConflicts(from: c.sections.map(toSection))
      XCTAssertEqual(got.map(toTest), c.expected)
    }
  }

  // MARK: - FFI spot checks for the Rust-pinned (non-injectable) cases

  func testRustAgreesOnDeviceBoundCompat() throws {
    let json = try XCTUnwrap(LlamaCore.call(
      "compat.deviceBudget",
      #"{"workingSetMb":24576,"physicalMb":32768,"useFullWorkingSet":false}"#))
    XCTAssertEqual(json, #"{"mb":23552.0}"#)

    let info = try XCTUnwrap(LlamaCore.call(
      "compat.info",
      #"{"ctxWindow":2048,"fileSize":4294967296,"ctxBytesPer1kTokens":0,"residentBytes":0,"ctxWindowTokens":4096,"budgetMb":23552}"#))
    XCTAssertEqual(info, #"{"incompatibilitySummary":"requires models with ≥4k context","isCompatible":false}"#)
  }

  func testRustAgreesOnPrivateParsers() throws {
    let parsed = try XCTUnwrap(LlamaCore.call(
      "modelId.parse", #"{"id":"unsloth/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M:Q4_K_M"}"#))
    XCTAssertTrue(parsed.contains(#""name":"Qwen3-Coder""#))
    XCTAssertTrue(parsed.contains(#""params":"30B-A3B""#))

    let ini = try XCTUnwrap(LlamaCore.call(
      "ini.parse", #"{"text":"[a]\nk = v\n\n[b]\nj = w\n"}"#))
    XCTAssertEqual(
      ini,
      #"{"sections":[{"name":"a","pairs":[{"key":"k","value":"v"}]},{"name":"b","pairs":[{"key":"j","value":"w"}]}]}"#)
  }
}
