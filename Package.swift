// swift-tools-version: 6.1

import PackageDescription

let package = Package(
    name: "OAuthLanguageModels",
    platforms: [
        .macOS(.v14),
        .macCatalyst(.v17),
        .iOS(.v17),
        .tvOS(.v17),
        .watchOS(.v10),
        .visionOS(.v1)
    ],
    products: [
        .library(
            name: "OAuthLanguageModels",
            targets: ["OAuthLanguageModels"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/finnvoor/AnyLanguageModel", branch: "restore-instructions-from-transcript")
    ],
    targets: [
        .target(
            name: "OAuthLanguageModels",
            dependencies: [
                .product(name: "AnyLanguageModel", package: "AnyLanguageModel")
            ]
        )
    ],
    swiftLanguageModes: [.v6]
)
