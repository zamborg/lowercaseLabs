import Darwin
import Foundation
import WERCore

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

@main
struct WEREval {
    static func main() async {
        let arguments = Array(CommandLine.arguments.dropFirst())
        if arguments == ["--help"] || arguments == ["-h"] { print(Options.usage); return }
        do {
            let options = try Options.parse(arguments)
            let references = try Reference.load(from: options.reference, audioDirectory: options.audio)
            let setup: SpeechSetup
            do {
                log("Checking on-device en-US SpeechTranscriber assets...")
                setup = try await SpeechSetup.prepare(installAssets: options.installAssets)
            } catch {
                let message = error.localizedDescription
                let report = Report(options: options, setup: nil, runError: message,
                                    files: references.map { FileResult(reference: $0, error: "Speech setup failed: \(message)") })
                try report.write(to: options.output)
                print("Aggregate WER: N/A (0/\(references.count) clips succeeded). Reports: \(options.output.path)")
                throw error
            }
            var files: [FileResult] = []
            for (index, reference) in references.enumerated() {
                log("[\(index + 1)/\(references.count)] \(reference.file) [\((reference.source ?? .unspecified).rawValue)]")
                do {
                    let transcription = try await transcribe(url: reference.audioURL(in: options.audio),
                                                           locale: setup.locale, timeout: options.timeout)
                    let result = FileResult(reference: reference, transcription: transcription)
                    files.append(result)
                    log("  WER \(percent(result.score?.wer)); final \(number(result.timeToFinalSeconds)) s; RTF \(number(result.realTimeFactor))")
                } catch {
                    files.append(FileResult(reference: reference, error: error.localizedDescription))
                    log("  FAILED: \(error.localizedDescription)")
                }
            }
            await setup.release()
            let report = Report(options: options, setup: setup, runError: nil, files: files)
            try report.write(to: options.output)
            print("Aggregate WER: \(percent(report.summary.score.wer)) (\(report.summary.successfulFiles)/\(references.count) clips succeeded)")
            print("Mean time-to-final: \(number(report.summary.meanTimeToFinalSeconds)) s; corpus RTF: \(number(report.summary.realTimeFactor))")
            if report.summary.syntheticFiles > 0 { print("SYNTHETIC AUDIO included; see provenance in reports.") }
            print("Reports: \(options.output.appendingPathComponent("wer.md").path), \(options.output.appendingPathComponent("results.json").path)")
            if report.summary.failedFiles > 0 { exit(1) }
        } catch {
            log("wer-eval: \(error.localizedDescription)")
            exit(1)
        }
    }
}
