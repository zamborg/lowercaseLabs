// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "TranscriptionEval",
    platforms: [.macOS(.v26)],
    products: [.executable(name: "wer-eval", targets: ["WEREval"])],
    targets: [
        .target(name: "WERCore"),
        .executableTarget(name: "WEREval", dependencies: ["WERCore"]),
        .testTarget(name: "WERCoreTests", dependencies: ["WERCore"]),
    ]
)
