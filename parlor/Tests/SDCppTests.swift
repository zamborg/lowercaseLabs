import XCTest
@testable import ParlorCore

final class SDCppTests: XCTestCase {
    func testVPredictionIsRecognisedFromTheFileName() {
        XCTAssertEqual(SDCppImage.Config.defaults(forFile: "WAI-NSFW-illustrious-SDXL-V14.0-V-Prediction.safetensors").prediction, "v")
        XCTAssertEqual(SDCppImage.Config.defaults(forFile: "model_vpred_fp16.safetensors").prediction, "v")
        XCTAssertNil(SDCppImage.Config.defaults(forFile: "waiNSFWIllustrious_v140.safetensors").prediction)
    }

    func testCommandLineCarriesTheRecommendedSettings() {
        var c = SDCppImage.Config.defaults(forFile: "x-vpred.safetensors")
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
        let b = SDCppImage.arguments(config: .defaults(forFile: "a.safetensors"), model: "/m", prompt: "masterpiece, 1boy", negative: "blurry",
                                     width: 1, height: 1, seed: 1, steps: 1, output: "/o")
        XCTAssertFalse(b.contains("--prediction"))
        XCTAssertEqual(b[b.firstIndex(of: "-n")! + 1], "blurry")
        XCTAssertEqual(b[b.firstIndex(of: "-p")! + 1], "masterpiece, 1boy")
    }

    func testShapesAndConfigFileRoundTrip() throws {
        let dir = tempHome().root
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        SDCppImage.Config.writeDefault(in: dir, file: "m-v-pred.safetensors")
        var c = SDCppImage.Config.load(dir: dir, file: "m-v-pred.safetensors")
        XCTAssertEqual(c.prediction, "v")
        XCTAssertEqual(c.size(shape: "portrait").0, 832); XCTAssertEqual(c.size(shape: "portrait").1, 1216)
        XCTAssertEqual(c.size(shape: "landscape").0, 1216)
        XCTAssertEqual(c.size(shape: "square").0, 1024)
        // An edited file is kept, and unknown or missing keys fall back to defaults.
        try Data(#"{"steps": 12, "cfg": 4.5}"#.utf8).write(to: SDCppImage.Config.url(in: dir))
        SDCppImage.Config.writeDefault(in: dir, file: "m-v-pred.safetensors")
        c = SDCppImage.Config.load(dir: dir, file: "m-v-pred.safetensors")
        XCTAssertEqual(c.steps, 12); XCTAssertEqual(c.cfg, 4.5); XCTAssertEqual(c.sampler, "euler_a")
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
            try JSONEncoder().encode(ModelLibrary.Marker(repo: repo, kind: .image, format: format, file: format == .sdcpp ? file : nil))
                .write(to: dir.appendingPathComponent(".parlor-model.json"))
        }
        try install("Aaa/sdxl-model", format: .sdcpp, file: "m.safetensors")
        try install("mlx-community/Z-Image-Turbo", format: nil, file: "w.safetensors")
        let lib = ModelLibrary(home: home)
        XCTAssertEqual(lib.installed(kind: .image).count, 2)
        XCTAssertEqual(lib.defaultImage?.repo, "mlx-community/Z-Image-Turbo", "the built-in model stays the default")
        XCTAssertEqual(lib.installed.first { $0.repo == "Aaa/sdxl-model" }?.format, .sdcpp)
    }
}
