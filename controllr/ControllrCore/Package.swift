// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "ControllrCore",
    platforms: [.iOS(.v17), .macOS(.v14)],
    products: [
        .library(name: "ControllrCore", targets: ["ControllrCore"]),
        .executable(name: "controllr-eval", targets: ["controllr-eval"]),
    ],
    targets: [
        .target(name: "ControllrCore", resources: [.copy("Actions")]),
        .executableTarget(name: "controllr-eval", dependencies: ["ControllrCore"]),
        .testTarget(name: "ControllrCoreTests", dependencies: ["ControllrCore"]),
    ]
)
