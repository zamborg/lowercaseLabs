import Foundation
import WERCore

struct Options {
    let audio: URL
    let reference: URL
    let output: URL
    let installAssets: Bool
    let timeout: Double

    static let usage = """
    Usage: wer-eval --audio <directory> --ref <reference.jsonl> [options]

      --output <directory>  Write wer.md and results.json here (default: current directory).
      --no-install-assets   Fail if asset setup requires installation (default: ensure/install).
      --timeout <seconds>   Maximum analysis time per clip (default: 120).
      --help                Show this help.

    Manifest: one JSON object per line: {"file":"clip.m4a","text":"Corrected transcript"}
    Optional source: "synthetic", "recorded", or "unspecified". Files are relative to --audio.
    Only listed files are evaluated, sequentially, in manifest order. Locale is fixed to en-US.
    """

    static func parse(_ arguments: [String]) throws -> Self {
        var values: [String: String] = [:]
        var installAssets = true
        var index = 0
        while index < arguments.count {
            let flag = arguments[index]
            if flag == "--no-install-assets" {
                guard installAssets else { throw EvalError("Duplicate option: \(flag)") }
                installAssets = false
                index += 1
                continue
            }
            guard ["--audio", "--ref", "--output", "--timeout"].contains(flag) else {
                throw EvalError("Unknown option: \(flag)\n\(usage)")
            }
            guard values[flag] == nil, index + 1 < arguments.count,
                  !arguments[index + 1].hasPrefix("--") else {
                throw EvalError("Missing value or duplicate option: \(flag)")
            }
            values[flag] = arguments[index + 1]
            index += 2
        }
        guard let audio = values["--audio"], let reference = values["--ref"] else {
            throw EvalError(usage)
        }
        let timeout = Double(values["--timeout"] ?? "120") ?? .nan
        guard timeout.isFinite, timeout > 0, timeout <= 86_400 else {
            throw EvalError("--timeout must be a finite number greater than zero and at most 86400.")
        }
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: audio, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw EvalError("--audio is not an existing directory: \(audio)")
        }
        return Self(audio: URL(fileURLWithPath: audio, isDirectory: true),
                    reference: URL(fileURLWithPath: reference),
                    output: URL(fileURLWithPath: values["--output"] ?? FileManager.default.currentDirectoryPath,
                                isDirectory: true),
                    installAssets: installAssets, timeout: timeout)
    }
}
