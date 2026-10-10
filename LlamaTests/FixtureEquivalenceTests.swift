import Foundation
import XCTest

@testable import Llama

/// Reads the same `Fixtures/*.json` corpus as the Rust `cargo test` suites, so
/// the Swift original and the Rust port are pinned to identical expectations.
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
      let got = HFRepoParser.parse(repoDir: c.input)
      if let org = c.org, let repo = c.repo {
        XCTAssertEqual(got?.org, org, "parse(\(c.input.debugDescription)) org")
        XCTAssertEqual(got?.repo, repo, "parse(\(c.input.debugDescription)) repo")
      } else {
        XCTAssertNil(got, "parse(\(c.input.debugDescription)) expected nil")
      }
    }
    for c in f.parse_quant {
      XCTAssertEqual(
        HFRepoParser.parseQuant(filename: c.input), c.expect,
        "parseQuant(\(c.input.debugDescription))")
    }
    for c in f.is_split_shard {
      XCTAssertEqual(
        HFRepoParser.isSplitShard(c.input), c.expect, "isSplitShard(\(c.input.debugDescription))")
    }
    for c in f.is_first_shard {
      XCTAssertEqual(
        HFRepoParser.isFirstShard(c.input), c.expect, "isFirstShard(\(c.input.debugDescription))")
    }
    for c in f.shard_total {
      XCTAssertEqual(
        HFRepoParser.shardTotal(c.input), c.expect, "shardTotal(\(c.input.debugDescription))")
    }
    for c in f.split_shard_base_name {
      XCTAssertEqual(
        HFRepoParser.splitShardBaseName(c.input), c.expect,
        "splitShardBaseName(\(c.input.debugDescription))")
    }
  }
}
