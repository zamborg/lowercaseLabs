// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "ParlorCore",
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "ParlorCore", targets: ["ParlorCore"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", .upToNextMajor(from: "3.31.3")),
        .package(url: "https://github.com/huggingface/swift-huggingface", from: "1.0.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
        .package(url: "https://github.com/modelcontextprotocol/swift-sdk", from: "0.12.0"),
    ],
    targets: [
        .target(
            name: "ParlorCore",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "MCP", package: "swift-sdk"),
            ],
            swiftSettings: [.swiftLanguageMode(.v5)]
        ),
    ]
)
