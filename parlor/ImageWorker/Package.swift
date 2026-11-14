// swift-tools-version: 6.2
import PackageDescription

// Separate package on purpose: z-image-swift and the app's mlx-swift-lm can resolve different
// MLX versions over time. A helper executable keeps the two dependency graphs independent.
let package = Package(
    name: "parlor-imagegen",
    platforms: [.macOS(.v26)],
    dependencies: [
        .package(url: "https://github.com/xocialize/z-image-swift", from: "0.7.1"),
    ],
    targets: [
        .executableTarget(
            name: "parlor-imagegen",
            dependencies: [
                .product(name: "MLXZImage", package: "z-image-swift"),
            ]
        ),
    ]
)
