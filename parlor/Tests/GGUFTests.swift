import XCTest
@testable import ParlorCore

final class GGUFTests: XCTestCase {
    private func info(weights: Int64 = 0, gguf: [(String, Int64)]) -> ModelLibrary.RepoInfo {
        ModelLibrary.RepoInfo(bytes: gguf.reduce(0) { $0 + $1.1 }, weightBytes: weights, files: gguf.map(\.0), gguf: gguf.map { ($0.0, $0.1) })
    }

    func testChoosesFourBitKQuantAndIgnoresProjector() {
        let files: [(path: String, size: Int64)] = [("m-q8_0.gguf", 800), ("m-q4_k_m.gguf", 400), ("m-q2_k.gguf", 200), ("mmproj-f16.gguf", 50)]
        XCTAssertEqual(ModelLibrary.chooseGGUF(files).map(\.path), ["m-q4_k_m.gguf"])
        XCTAssertEqual(ModelLibrary.chooseGGUF([("only-q4_1.gguf", 9)]).map(\.path), ["only-q4_1.gguf"])
        XCTAssertTrue(ModelLibrary.chooseGGUF([("mmproj.gguf", 1)]).isEmpty)
    }

    func testSplitShardsAreFetchedTogether() {
        let files: [(path: String, size: Int64)] = [
            ("big-q4_k_m-00001-of-00002.gguf", 100), ("big-q4_k_m-00002-of-00002.gguf", 100), ("big-q8_0.gguf", 300)]
        XCTAssertEqual(ModelLibrary.chooseGGUF(files).map(\.path).sorted(), ["big-q4_k_m-00001-of-00002.gguf", "big-q4_k_m-00002-of-00002.gguf"])
    }

    func testGGUFOnlyRepoIsSupportedWhenLlamaServerExists() {
        let r = ModelLibrary.unsupportedReason(info(gguf: [("a-q4_0.gguf", 5)]), kind: .language, repo: "x/a-gguf")
        if ModelLibrary.llamaServerPath != nil { XCTAssertNil(r) } else { XCTAssertTrue(r?.contains("llama-server") == true) }
        // Image repos and repos with nothing runnable stay refused.
        XCTAssertNotNil(ModelLibrary.unsupportedReason(info(gguf: [("a.gguf", 5)]), kind: .image, repo: "x/z-image-gguf"))
        XCTAssertNotNil(ModelLibrary.unsupportedReason(info(gguf: []), kind: .language, repo: "x/torch"))
    }

    /// An installed GGUF model gets an auto host on llama.cpp, a derived (never written) provider, and no settings churn.
    @MainActor func testInstalledGGUFGetsAnAutoHostAndDerivedProvider() async throws {
        setenv("PARLOR_BIN_LLAMA_SERVER", "/usr/bin/true", 1)
        defer { unsetenv("PARLOR_BIN_LLAMA_SERVER") }
        let home = tempHome(); try home.ensure()
        let dir = home.modelsDir.appendingPathComponent("Org/Flint-Q4-GGUF")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(count: 8).write(to: dir.appendingPathComponent("flint it's.gguf"))
        try JSONEncoder().encode(ModelLibrary.Marker(repo: "Org/Flint-Q4-GGUF", kind: .language, format: .gguf, file: "flint it's.gguf"))
            .write(to: dir.appendingPathComponent(".parlor-model.json"))
        let engine = ParlorEngine(home: home, runtime: DemoRuntime())
        await engine.start()

        XCTAssertEqual(engine.library.installed.first?.format, .gguf)
        let host = try XCTUnwrap(engine.catalog.host(forModel: "Org/Flint-Q4-GGUF"))
        XCTAssertTrue(host.isAuto)
        XCTAssertEqual(host.id, "flint-q4-gguf")
        XCTAssertEqual(host.spec.engine, "llama.cpp")
        XCTAssertEqual(engine.catalog.status(host: host.id), .ready)
        XCTAssertTrue(engine.settings.providers.isEmpty, "derived state is never stored as settings")
        let p = try XCTUnwrap(engine.router.provider("flint-q4-gguf"))
        XCTAssertTrue(p.config.service!.command.contains(#"'\''"#), "quotes in the path are escaped for the shell")
        XCTAssertTrue(p.config.service!.command.contains("'--port'"))
        XCTAssertNil(engine.modelProblem(for: { var a = Seeds.mabel(); a.model = "Org/Flint-Q4-GGUF"; return a }()))
        await engine.shutdown()
    }
}
