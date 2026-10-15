import Foundation
import XCTest

@testable import Llama

/// Reads the same `Fixtures/*.json` corpus as the Rust `cargo test` suites, so
/// the Swift surface and the Rust core are pinned to identical expectations.
/// If either side drifts from the corpus, one of the two test runs fails.
final class FixtureEquivalenceTests: XCTestCase {

  // MARK: - Fixture loading

  private static let fixturesDir = Fixtures.directory

  private func load<T: Decodable>(_ name: String, as type: T.Type) throws -> T {
    let url = Self.fixturesDir.appendingPathComponent(name)
    let data = try Data(contentsOf: url)
    return try JSONDecoder().decode(T.self, from: data)
  }

  // MARK: - LlamaVersion

  private struct VersionFixture: Decodable {
    struct Case: Decodable {
      let input: String
      let build: Int?
    }
    let cases: [Case]
  }

  func testLlamaVersionFixtures() throws {
    let fixture = try load("llama_version.json", as: VersionFixture.self)
    for c in fixture.cases {
      let got = LlamaVersion(parsing: c.input)?.build
      XCTAssertEqual(got, c.build, "LlamaVersion(parsing: \(c.input.debugDescription))")
    }
  }

  // MARK: - HFRepoParser

  private struct RepoFixture: Decodable {
    struct ParseCase: Decodable {
      let input: String
      let org: String?
      let repo: String?
    }
    struct StrCase: Decodable {
      let input: String
      let expect: String?
    }
    struct BoolCase: Decodable {
      let input: String
      let expect: Bool
    }
    struct IntCase: Decodable {
      let input: String
      let expect: Int?
    }

    let parse: [ParseCase]
    let parse_quant: [StrCase]
    let is_split_shard: [BoolCase]
    let is_first_shard: [BoolCase]
    let shard_total: [IntCase]
    let split_shard_base_name: [StrCase]
  }

  func testHFRepoParserFixtures() throws {
    let f = try load("hf_repo_parser.json", as: RepoFixture.self)

    for c in f.parse {
      let out = LlamaCore.callJSON("repo.parse", ["repoDir": c.input])
      if let org = c.org, let repo = c.repo {
        XCTAssertEqual(out?["org"] as? String, org, "parse(\(c.input.debugDescription)) org")
        XCTAssertEqual(out?["repo"] as? String, repo, "parse(\(c.input.debugDescription)) repo")
      } else {
        XCTAssertNil(out, "parse(\(c.input.debugDescription)) expected nil")
      }
    }
    for c in f.parse_quant {
      let got: String? = LlamaCore.callScalar("repo.parseQuant", ["filename": c.input])
      XCTAssertEqual(got, c.expect, "parseQuant(\(c.input.debugDescription))")
    }
    for c in f.is_split_shard {
      let got: Bool? = LlamaCore.callScalar("repo.isSplitShard", ["filename": c.input])
      XCTAssertEqual(got, c.expect, "isSplitShard(\(c.input.debugDescription))")
    }
    for c in f.is_first_shard {
      let got: Bool? = LlamaCore.callScalar("repo.isFirstShard", ["filename": c.input])
      XCTAssertEqual(got, c.expect, "isFirstShard(\(c.input.debugDescription))")
    }
    for c in f.shard_total {
      let got: Int? = LlamaCore.callScalar("repo.shardTotal", ["filename": c.input])
      XCTAssertEqual(got, c.expect, "shardTotal(\(c.input.debugDescription))")
    }
    for c in f.split_shard_base_name {
      let got: String? = LlamaCore.callScalar("repo.splitShardBaseName", ["filename": c.input])
      XCTAssertEqual(got, c.expect, "splitShardBaseName(\(c.input.debugDescription))")
    }
  }
}
