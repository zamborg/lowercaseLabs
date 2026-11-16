import XCTest
@testable import ParlorCore

final class SDCppTests: XCTestCase {
    func testVPredictionIsRecognisedFromTheFileName() {
        XCTAssertEqual(SDCppImage.isVPrediction(file: "WAI-NSFW-illustrious-SDXL-V14.0-V-Prediction.safetensors"), true)
        XCTAssertTrue(SDCppImage.isVPrediction(file: "model_vpred_fp16.safetensors"))
        XCTAssertFalse(SDCppImage.isVPrediction(file: "waiNSFWIllustrious_v140.safetensors"))
    }

    func testCommandLineCarriesTheRecommendedSettings() {
        var c = SDCppImage.Config(); c.prediction = "v"
        c.vae = "/v.safetensors"; c.extraArgs = ["--vae-tiling"]
        let a = SDCppImage.arguments(config: c, model: "/m.safetensors", prompt: "1girl, red hair", negative: nil,
                                     width: 832, height: 1216, seed: 7, steps: 28, output: "/o.png")
        func value(_ flag: String) -> String? { a.firstIndex(of: flag).map { a[$0 + 1] } }
        XCTAssertEqual(value("--sampling-method"), "euler_a")
        XCTAssertEqual(value("--steps"), "28")
        XCTAssertEqual(value("--cfg-scale"), "6.0")
        XCTAssertEqual(value("--clip-skip"), "2")
        XCTAssertEqual(value("--prediction"), "v")
        XCTAssertEqual(value("-W"), "832"); XCTAssertEqual(value("-H"), "1216")
        XCTAssertEqual(value("--seed"), "7")
        XCTAssertEqual(value("--vae"), "/v.safetensors")
        XCTAssertEqual(a.last, "--vae-tiling")
        XCTAssertTrue(value("-p")!.hasPrefix("masterpiece, best quality"))
        XCTAssertTrue(value("-n")!.contains("worst quality"))
        // An ordinary checkpoint passes no prediction override, and a caller's negative wins.
        let b = SDCppImage.arguments(config: SDCppImage.Config(), model: "/m", prompt: "masterpiece, 1boy", negative: "blurry",
                                     width: 1, height: 1, seed: 1, steps: 1, output: "/o")
        XCTAssertFalse(b.contains("--prediction"))
        XCTAssertEqual(b[b.firstIndex(of: "-n")! + 1], "blurry")
        XCTAssertEqual(b[b.firstIndex(of: "-p")! + 1], "masterpiece, 1boy")
    }

    func testShapesComeFromTheConfig() {
        let c = SDCppImage.Config()
        XCTAssertEqual(c.size(shape: "portrait").0, 832); XCTAssertEqual(c.size(shape: "portrait").1, 1216)
        XCTAssertEqual(c.size(shape: "landscape").0, 1216)
        XCTAssertEqual(c.size(shape: "square").0, 1024)
    }

    /// An SD host's settings reach the sd-cli command line, the auto host defaults V-prediction from the file name,
    /// a caller scoped away from the host is refused, and no parlor-image.json is read or written.
    @MainActor func testHostSettingsReachSdCliThroughTheGateway() async throws {
        let home = tempHome(); try home.ensure()
        try #"{"gateway":{"enabled":true,"port":0}}"#.write(to: home.settingsFile, atomically: true, encoding: .utf8)
        let dir = home.modelsDir.appendingPathComponent("Aaa/sdxl-model-vpred")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data(count: 4).write(to: dir.appendingPathComponent("m-vpred.safetensors"))
        try JSONEncoder().encode(ModelLibrary.Marker(repo: "Aaa/sdxl-model-vpred", kind: .image, format: .sdCheckpoint, file: "m-vpred.safetensors"))
            .write(to: dir.appendingPathComponent(".parlor-model.json"))
        let record = home.root.appendingPathComponent("args.txt")
        let fake = home.root.appendingPathComponent("fake-sd-cli")
        try "#!/bin/sh\necho \"$@\" > '\(record.path)'\nwhile [ $# -gt 0 ]; do if [ \"$1\" = -o ]; then printf x > \"$2\"; fi; shift; done\n"
            .write(to: fake, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: fake.path)
        setenv("PARLOR_BIN_SD_CLI", fake.path, 1)
        defer { unsetenv("PARLOR_BIN_SD_CLI") }

        let e = ParlorEngine(home: home, runtime: DemoRuntime())
        await e.start()
        let auto = try XCTUnwrap(e.catalog.hosts.first { $0.spec.model == "Aaa/sdxl-model-vpred" })
        XCTAssertEqual(auto.spec.settings["prediction"]?.stringValue, "v", "the auto host carries the V-prediction default")

        // The explicit host's settings win and reach the command line.
        try #"{"id":"sd","engine":"stable-diffusion.cpp","model":"Aaa/sdxl-model-vpred","settings":{"steps":7,"fast_steps":7,"cfg":3.5}}"#
            .write(to: home.hostsDir.appendingPathComponent("sd.json"), atomically: true, encoding: .utf8)
        e.reloadRunplane()
        var r = try await e.gateway.generateImage(ImageRequest(prompt: "a tree", detailed: true), host: "sd", caller: .local)
        XCTAssertEqual(r.host, "sd")
        let line = try String(contentsOf: record, encoding: .utf8)
        XCTAssertTrue(line.contains("--steps 7") && line.contains("--cfg-scale 3.5"), line)
        XCTAssertTrue(line.contains("--prediction v"), "the file name still says V-prediction: " + line)

        // A caller scoped to other hosts is refused, and the same call through the tool reports it.
        let scoped = Caller(kind: .agent, name: "mabel", tier: .none, hosts: ["other"])
        do { r = try await e.gateway.generateImage(ImageRequest(prompt: "x"), host: "sd", caller: scoped); XCTFail("should be refused") }
        catch { XCTAssertTrue(error.localizedDescription.contains("may not use host"), error.localizedDescription) }

        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("parlor-image.json").path), "no settings file is written beside the weights")
        await e.shutdown()
    }

    func testChoosesTheCheckpointFile() {
        let gb: Int64 = 1_000_000_000
        XCTAssertEqual(ModelLibrary.chooseCheckpoint([("model.safetensors", 7 * gb), ("vae.safetensors", gb / 5)])?.path, "model.safetensors")
        XCTAssertEqual(ModelLibrary.chooseCheckpoint([("m-fp32.safetensors", 14 * gb), ("m-fp16.safetensors", 7 * gb), ("m-pruned.safetensors", 6 * gb)])?.path, "m-fp16.safetensors")
        XCTAssertNil(ModelLibrary.chooseCheckpoint([("my-lora.safetensors", 2 * gb), ("tiny.safetensors", gb / 2)]))
    }

    func testCheckpointReposAreSupportedAndImageListsIncludeThem() throws {
        let gb: Int64 = 1_000_000_000
        let info = ModelLibrary.RepoInfo(bytes: 7 * gb, weightBytes: 7 * gb, files: ["m.safetensors"], rootSafetensors: [("m.safetensors", 7 * gb)])
        let r = ModelLibrary.unsupportedReason(info, kind: .image, repo: "x/some-sdxl")
        if SDCppImage.binaryURL() != nil { XCTAssertNil(r) } else { XCTAssertTrue(r?.contains("sd-cli") == true) }
        // A language request for the same repo, and a repo of nothing but a LoRA, stay refused for image use.
        let lora = ModelLibrary.RepoInfo(bytes: gb, weightBytes: gb, files: ["l.safetensors"], rootSafetensors: [("l.safetensors", gb)])
        XCTAssertNotNil(ModelLibrary.unsupportedReason(lora, kind: .image, repo: "x/lora"))

        let home = tempHome()
        func install(_ repo: String, format: ModelFormat?, file: String) throws {
            let dir = home.modelsDir.appendingPathComponent(repo)
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(count: 4).write(to: dir.appendingPathComponent(file))
            try JSONEncoder().encode(ModelLibrary.Marker(repo: repo, kind: .image, format: format, file: format == .sdCheckpoint ? file : nil))
                .write(to: dir.appendingPathComponent(".parlor-model.json"))
        }
        try install("Aaa/sdxl-model", format: .sdCheckpoint, file: "m.safetensors")
        try install("mlx-community/Z-Image-Turbo", format: nil, file: "w.safetensors")
        let lib = ModelLibrary(home: home)
        XCTAssertEqual(lib.installed(kind: .image).count, 2)
        XCTAssertEqual(lib.installed.first { $0.repo == "mlx-community/Z-Image-Turbo" }?.format, .zImage, "an MLX image model is Z-Image")
        XCTAssertEqual(lib.installed.first { $0.repo == "Aaa/sdxl-model" }?.format, .sdCheckpoint)
    }
}
