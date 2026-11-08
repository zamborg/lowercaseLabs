import Foundation
import Testing
@testable import WERCore

private func withManifest(_ contents: String, _ body: (URL, URL) throws -> Void) throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let manifest = directory.appendingPathComponent("reference.jsonl")
    try contents.write(to: manifest, atomically: true, encoding: .utf8)
    try body(manifest, directory)
}

@Test func readsMinimalManifestAndSyntheticProvenance() throws {
    try withManifest("""
    {"file":"one.aiff","text":"Hello"}

    {"file":"two.aiff","text":"","source":"synthetic"}
    """) { manifest, root in
        let references = try Reference.load(from: manifest, audioDirectory: root)
        #expect(references.count == 2)
        #expect(references[0].source == nil)
        #expect(references[1].source == .synthetic)
    }
}

@Test func rejectsMalformedDuplicateAndEscapingPaths() throws {
    for contents in ["", "not json", "{\"file\":\"one.aiff\"}",
                     "{\"file\":\"../outside.aiff\",\"text\":\"hi\"}",
                     "{\"file\":\"/outside.aiff\",\"text\":\"hi\"}",
                     "{\"file\":\"one.aiff\",\"text\":\"hi\"}\n{\"file\":\"./one.aiff\",\"text\":\"hi\"}"] {
        try withManifest(contents) { manifest, root in
            #expect(throws: (any Error).self) { try Reference.load(from: manifest, audioDirectory: root) }
        }
    }
}

@Test func rejectsSymlinkEscape() throws {
    try withManifest("{\"file\":\"escape\",\"text\":\"hi\"}") { manifest, root in
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("escape"),
                                                  withDestinationURL: root.deletingLastPathComponent())
        #expect(throws: (any Error).self) { try Reference.load(from: manifest, audioDirectory: root) }
    }
}
