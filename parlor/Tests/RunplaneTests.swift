import XCTest
@testable import ParlorCore

final class RunplaneTests: XCTestCase {
    private func install(_ home: ParlorHome, _ repo: String, kind: ModelKind, format: ModelFormat, file: String) throws {
        let dir = home.modelsDir.appendingPathComponent(repo)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(count: 8).write(to: dir.appendingPathComponent(file))
        try JSONEncoder().encode(ModelLibrary.Marker(repo: repo, kind: kind, format: format, file: file))
            .write(to: dir.appendingPathComponent(".parlor-model.json"))
    }

    private func catalog(_ home: ParlorHome, explicit: [HostSpec] = []) -> Catalog {
        EngineSeeds.seed(home: home)
        let engines = NounStore<EngineManifest>(dir: home.enginesDir, layout: .folder("engine.json")).list()
        let lib = ModelLibrary(home: home)
        var dirs: [String: URL] = [:]
        for e in engines { dirs[e.id] = home.enginesDir.appendingPathComponent(e.id) }
        return Catalog(models: lib.installed, engines: engines, hosts: HostRegistry.build(explicit: explicit, installed: lib.installed, engines: engines), engineDirs: dirs)
    }

    func testEveryFormatResolvesToAHostWithTheRightStatus() throws {
        setenv("PARLOR_BIN_LLAMA_SERVER", "/usr/bin/true", 1); setenv("PARLOR_BIN_SD_CLI", "/nonexistent/sd-cli", 1)
        defer { unsetenv("PARLOR_BIN_LLAMA_SERVER"); unsetenv("PARLOR_BIN_SD_CLI") }
        let home = tempHome(); try home.ensure()
        try install(home, "mlx-community/Llama-3", kind: .language, format: .mlxLM, file: "w.safetensors")
        try install(home, "Org/Stheno-GGUF", kind: .language, format: .gguf, file: "s.gguf")
        try install(home, "Org/Wai-XL", kind: .image, format: .sdCheckpoint, file: "m.safetensors")
        try install(home, "mlx-community/Z-Image-Turbo", kind: .image, format: .zImage, file: "w.safetensors")
        let cat = catalog(home)
        XCTAssertEqual(cat.host(forModel: "mlx-community/Llama-3")?.spec.engine, "mlx")
        XCTAssertEqual(cat.host(forModel: "Org/Stheno-GGUF")?.spec.engine, "llama.cpp")
        XCTAssertEqual(cat.host(forModel: "Org/Wai-XL")?.spec.engine, "stable-diffusion.cpp")
        XCTAssertEqual(cat.host(forModel: "mlx-community/Z-Image-Turbo")?.spec.engine, "z-image")
        XCTAssertEqual(cat.status(host: "llama-3"), .ready)
        XCTAssertEqual(cat.status(host: "stheno-gguf"), .ready)
        if case .missingBinary = cat.status(host: "wai-xl") {} else { XCTFail("sd-cli is missing in this test") }
        XCTAssertEqual(cat.hosts(capability: "image").map(\.id).sorted(), ["wai-xl", "z-image-turbo"])
        XCTAssertEqual(cat.hosts(capability: "chat").map(\.id).sorted(), ["llama-3", "stheno-gguf"])
    }

    func testAutoHostIdsAreStableAndExplicitHostsWin() throws {
        let home = tempHome(); try home.ensure()
        try install(home, "a/Same", kind: .language, format: .mlxLM, file: "w.safetensors")
        try install(home, "b/Same", kind: .language, format: .mlxLM, file: "w.safetensors")
        var cat = catalog(home)
        XCTAssertEqual(cat.hosts.map(\.id), ["same", "same-2"])
        XCTAssertEqual(cat.host("same")?.spec.model, "a/Same", "collisions are numbered in sorted repo order")
        cat = catalog(home, explicit: [HostSpec(id: "same", engine: "mlx", model: "b/Same", defaults: ["temperature": .double(0.5)])])
        XCTAssertFalse(cat.host("same")!.isAuto)
        XCTAssertEqual(cat.host("same")?.spec.model, "b/Same")
        XCTAssertEqual(cat.hosts.count, 2)
    }

    func testMissingModelAndOffAndUnknownEngine() throws {
        let home = tempHome(); try home.ensure()
        let cat = catalog(home, explicit: [
            HostSpec(id: "ghost", engine: "llama.cpp", model: "Org/Nope-GGUF"),
            HostSpec(id: "bad", engine: "nonsense"),
            HostSpec(id: "lm", engine: "openai-remote", enabled: false, url: "http://127.0.0.1:1234/v1")])
        XCTAssertEqual(cat.status(host: "ghost"), .missingModel)
        XCTAssertEqual(cat.status(host: "bad"), .unknownEngine)
        XCTAssertEqual(cat.status(host: "lm"), .off)
        XCTAssertNotNil(cat.problem(host: "ghost"))
    }

    func testNounStoreRoundTripKeepsUnknownKeysAndMergePatches() throws {
        let home = tempHome(); try home.ensure()
        let store = NounStore<HostSpec>(dir: home.hostsDir, layout: .file)
        _ = try store.create(.object(["id": .string("h1"), "engine": .string("mlx"), "model": .string("a/b"), "note": .string("mine")]))
        var h = try XCTUnwrap(store.read("h1"))
        h.defaults["temperature"] = .double(0.9)
        try store.save(h)
        XCTAssertEqual(store.raw("h1")?["note"]?.stringValue, "mine", "unknown keys survive a save")
        let patched = try store.update("h1", patch: .object(["settings": .object(["context": .int(4096)]), "note": .null]))
        XCTAssertEqual(patched.settings["context"], .int(4096))
        XCTAssertEqual(patched.defaults["temperature"], .double(0.9))
        XCTAssertNil(store.raw("h1")?["note"], "null removes a key")
        XCTAssertThrowsError(try store.create(.object(["id": .string("h1"), "engine": .string("mlx")]))) { XCTAssertTrue("\($0)".contains("exists") || ($0 as? ValidationError)?.issues.first?.code == "exists") }
        // A draft reports every problem at once.
        let issues = store.validate(.object(["id": .string("bad id!"), "engine": .int(3)]))
        XCTAssertTrue(issues.contains { $0.path == "id" })
        XCTAssertTrue(issues.contains { $0.path == "engine" })
        XCTAssertEqual(store.ids(), ["h1"])
    }

    func testSeededEnginesDecodeAndTemplatesExpand() throws {
        let home = tempHome(); try home.ensure()
        EngineSeeds.seed(home: home)
        let store = NounStore<EngineManifest>(dir: home.enginesDir, layout: .folder("engine.json"))
        XCTAssertEqual(store.ids(), ["llama.cpp", "mlx", "openai-remote", "stable-diffusion.cpp", "z-image"])
        for id in store.ids() { XCTAssertTrue(store.validate(.object(store.raw(id)!)).errors.isEmpty, "\(id): \(store.validate(.object(store.raw(id)!)).summary)") }
        let sd = try XCTUnwrap(store.read("stable-diffusion.cpp"))
        let vals: [String: JSON] = ["model.file": .string("/m"), "prompt": .string("a cat"), "negative": .string("bad"), "width": .int(512), "height": .int(512),
                                    "steps": .int(20), "seed": .int(1), "out": .string("/o.png"), "settings.cfg": .int(6), "settings.sampler": .string("euler_a"),
                                    "settings.scheduler": .string("discrete"), "settings.clip_skip": .int(2), "settings.args": .array([.string("--vae-tiling")])]
        let (args, missing) = ArgTemplate.expand(sd.launch!.args, values: vals)
        XCTAssertEqual(missing, [])
        XCTAssertFalse(args.contains("--prediction"), "an unset optional drops its flag too")
        XCTAssertFalse(args.contains("--vae"))
        XCTAssertTrue(args.contains("--vae-tiling"), "lists splice")
        XCTAssertEqual(args.last, "/o.png")
        // Reseeding rewrites managed files and leaves edited ones.
        let f = home.enginesDir.appendingPathComponent("mlx/engine.json")
        try "{ \"id\": \"mlx\" }".write(to: f, atomically: true, encoding: .utf8)
        XCTAssertFalse(EngineSeeds.seed(home: home).contains("mlx"), "a file without the marker is the user's")
    }

    func testSchemaCheckReportsEverythingWithPaths() {
        let schema: JSON = .object(["type": .string("object"), "properties": .object([
            "images": .object(["enum": .array([.string("auto"), .string("off")])]),
            "n": .object(["type": .string("integer"), "minimum": .int(1)])])])
        let issues = SchemaCheck.validate(.object(["images": .string("sometimes"), "n": .int(0), "typo": .int(1)]), schema: schema)
        XCTAssertEqual(issues.filter { $0.severity == .error }.map(\.path).sorted(), ["images", "n"])
        XCTAssertEqual(issues.filter { $0.severity == .warning }.map(\.path), ["typo"], "unknown keys warn, never block")
    }

    /// Invariant 1: only the Catalog (and the library it reads) decides what is installed.
    func testNobodyElseCallsInstalledDirectory() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
        var offenders: [String] = []
        for sub in ["Packages/ParlorCore/Sources", "App", "CLI"] {
            guard let e = FileManager.default.enumerator(at: root.appendingPathComponent(sub), includingPropertiesForKeys: nil) else { continue }
            for case let u as URL in e where u.pathExtension == "swift" {
                if (try? String(contentsOf: u, encoding: .utf8))?.contains("installedDirectory") == true { offenders.append(u.lastPathComponent) }
            }
        }
        XCTAssertEqual(offenders, [])
    }
}
