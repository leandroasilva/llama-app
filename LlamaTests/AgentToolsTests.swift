import XCTest

@testable import Llama

final class AgentToolsTests: XCTestCase {
  private var root: URL!

  override func setUpWithError() throws {
    root = FileManager.default.temporaryDirectory
      .appendingPathComponent("llama-agent-tests-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
  }

  override func tearDownWithError() throws {
    try? FileManager.default.removeItem(at: root)
  }

  func testWorkspaceRejectsPathsOutsideTheRoot() throws {
    let workspace = AgentWorkspace(root: root)

    XCTAssertNoThrow(try workspace.resolve("Sources/main.swift"))
    XCTAssertNoThrow(try workspace.resolve(root.appendingPathComponent("file.txt").path))
    XCTAssertThrowsError(try workspace.resolve("../secrets.txt"))
    XCTAssertThrowsError(try workspace.resolve("/etc/passwd"))

    let link = root.appendingPathComponent("outside")
    try FileManager.default.createSymbolicLink(
      at: link, withDestinationURL: URL(fileURLWithPath: "/etc"))
    XCTAssertThrowsError(try workspace.resolve("outside/passwd"))
  }

  func testWriteReadAndEditRoundTrip() async throws {
    let workspace = AgentWorkspace(root: root)

    let write = await AgentToolRunner.execute(
      call("write", ["path": "notes.txt", "content": "hello world\n"]), workspace: workspace)
    XCTAssertFalse(write.isError)

    let read = await AgentToolRunner.execute(
      call("read", ["path": "notes.txt"]), workspace: workspace)
    XCTAssertFalse(read.isError)
    XCTAssertTrue(read.text.contains("hello world"))

    let edit = await AgentToolRunner.execute(
      call("edit", ["path": "notes.txt", "oldString": "world", "newString": "there"]),
      workspace: workspace)
    XCTAssertFalse(edit.isError)

    let contents = try String(
      contentsOf: root.appendingPathComponent("notes.txt"), encoding: .utf8)
    XCTAssertEqual(contents, "hello there\n")
  }

  func testEditRequiresAUniqueMatchUnlessReplaceAllIsSet() async throws {
    let workspace = AgentWorkspace(root: root)
    _ = await AgentToolRunner.execute(
      call("write", ["path": "dupes.txt", "content": "one one\n"]), workspace: workspace)

    let ambiguous = await AgentToolRunner.execute(
      call("edit", ["path": "dupes.txt", "oldString": "one", "newString": "two"]),
      workspace: workspace)
    XCTAssertTrue(ambiguous.isError)

    let replaceAll = await AgentToolRunner.execute(
      call(
        "edit",
        [
          "path": "dupes.txt", "oldString": "one", "newString": "two",
          "replaceAll": true,
        ]), workspace: workspace)
    XCTAssertFalse(replaceAll.isError)

    let contents = try String(
      contentsOf: root.appendingPathComponent("dupes.txt"), encoding: .utf8)
    XCTAssertEqual(contents, "two two\n")
  }

  func testEditPreservesTheExecutableBit() async throws {
    let workspace = AgentWorkspace(root: root)
    let script = root.appendingPathComponent("script.sh")
    try "#!/bin/sh\necho old\n".write(to: script, atomically: true, encoding: .utf8)
    try FileManager.default.setAttributes(
      [.posixPermissions: 0o755], ofItemAtPath: script.path)

    _ = await AgentToolRunner.execute(
      call("edit", ["path": "script.sh", "oldString": "old", "newString": "new"]),
      workspace: workspace)

    let attributes = try FileManager.default.attributesOfItem(atPath: script.path)
    XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o755)
  }

  func testGlobFindsFilesAndSkipsIgnoredDirectories() async throws {
    let sourceDirectory = root.appendingPathComponent("Sources")
    let ignoredDirectory = root.appendingPathComponent("node_modules")
    try FileManager.default.createDirectory(at: sourceDirectory, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: ignoredDirectory, withIntermediateDirectories: true)
    try "let a = 1\n".write(
      to: sourceDirectory.appendingPathComponent("main.swift"), atomically: true, encoding: .utf8)
    try "ignored\n".write(
      to: ignoredDirectory.appendingPathComponent("dependency.swift"), atomically: true,
      encoding: .utf8)

    let result = await AgentToolRunner.execute(
      call("glob", ["pattern": "**/*.swift"]), workspace: AgentWorkspace(root: root))

    XCTAssertTrue(result.text.contains("Sources/main.swift"))
    XCTAssertFalse(result.text.contains("node_modules"))
  }

  func testGlobMatching() throws {
    let regex = try AgentToolRunner.globRegex("**/*.swift")
    func matches(_ path: String) -> Bool {
      regex.firstMatch(in: path, range: NSRange(path.startIndex..<path.endIndex, in: path)) != nil
    }

    XCTAssertTrue(matches("main.swift"))
    XCTAssertTrue(matches("Sources/main.swift"))
    XCTAssertFalse(matches("Sources/main.txt"))
  }

  func testBashRunsInTheWorkspace() async throws {
    let workspace = AgentWorkspace(root: root)
    let result = await AgentToolRunner.execute(
      call("bash", ["command": "pwd && printf 'ok'"]), workspace: workspace)

    XCTAssertFalse(result.isError)
    XCTAssertTrue(result.text.contains(root.resolvingSymlinksInPath().path))
    XCTAssertTrue(result.text.contains("ok"))
  }

  private func call(_ name: String, _ arguments: [String: Any]) -> AgentToolCall {
    let data = try! JSONSerialization.data(withJSONObject: arguments)
    return AgentToolCall(
      id: UUID().uuidString, name: name, arguments: String(decoding: data, as: UTF8.self))
  }
}
