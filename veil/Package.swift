// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Veil",
    platforms: [.macOS(.v13)],
    products: [.executable(name: "Veil", targets: ["Veil"])],
    targets: [
        .target(name: "VeilCore"),
        .executableTarget(name: "Veil", dependencies: ["VeilCore"]),
        .testTarget(name: "VeilCoreTests", dependencies: ["VeilCore"])
    ]
)
