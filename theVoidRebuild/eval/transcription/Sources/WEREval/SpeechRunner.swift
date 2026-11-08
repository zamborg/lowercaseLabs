import AVFoundation
import Foundation
import Speech
import WERCore

func seconds(_ duration: Duration) -> Double {
    let parts = duration.components
    return Double(parts.seconds) + Double(parts.attoseconds) / 1e18
}

struct SpeechSetup {
    let locale: Locale
    let initialAssetStatus: String
    let setupSeconds: Double
    let ownsReservation: Bool

    static func prepare(installAssets: Bool) async throws -> Self {
        let clock = ContinuousClock()
        let start = clock.now
        guard SpeechTranscriber.isAvailable else {
            throw EvalError("SpeechTranscriber is unavailable on this hardware/OS.")
        }
        guard let locale = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US")) else {
            throw EvalError("SpeechTranscriber did not return a supported en-US locale. Check model service availability and system restrictions.")
        }
        let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
        let status = await AssetInventory.status(forModules: [transcriber])
        let reserved = await AssetInventory.reservedLocales
        let ownsReservation = !reserved.contains(locale)
        do {
            guard status != .unsupported else { throw EvalError("en-US transcription assets are unsupported.") }
            if ownsReservation {
                guard try await AssetInventory.reserve(locale: locale) else {
                    throw EvalError("Could not reserve en-US assets; the system's locale reservation limit may be reached.")
                }
            }
            // Ask even if the initial status says installed: the system manages
            // asset subscriptions and readiness for this executable's identity.
            if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                guard installAssets else {
                    throw EvalError("en-US asset setup requires installation (initial status: \(status)). Rerun without --no-install-assets to ensure the on-device model is installed.")
                }
                log("Ensuring en-US speech assets are installed (excluded from clip timing)...")
                try await request.downloadAndInstall()
            }
            guard await AssetInventory.status(forModules: [transcriber]) == .installed else {
                throw EvalError("en-US assets are still not installed after setup.")
            }
            return Self(locale: locale, initialAssetStatus: String(describing: status),
                        setupSeconds: seconds(start.duration(to: clock.now)), ownsReservation: ownsReservation)
        } catch {
            if ownsReservation { await AssetInventory.release(reservedLocale: locale) }
            throw error
        }
    }

    func release() async {
        if ownsReservation { await AssetInventory.release(reservedLocale: locale) }
    }
}

struct Transcription: Sendable {
    let text: String
    let audioDurationSeconds: Double
    let timeToFinalSeconds: Double
    let finalResultCount: Int
}

func transcribe(url: URL, locale: Locale, timeout: Double) async throws -> Transcription {
    let clock = ContinuousClock()
    let start = clock.now
    let transcriber = SpeechTranscriber(locale: locale, preset: .transcription)
    let analyzer = SpeechAnalyzer(modules: [transcriber])

    // The consumer must run while input is analyzed. Do not cancel it at EOF:
    // finishing the analyzer terminates the sequence after all final results.
    return try await withThrowingTaskGroup(of: Transcription.self) { group in
        group.addTask {
            let consumer = Task {
                var segments: [(start: Double, text: String)] = []
                for try await result in transcriber.results {
                    guard result.isFinal else { continue }
                    segments.append((result.range.start.seconds, String(result.text.characters)))
                }
                return segments.sorted { $0.start < $1.start }
            }
            do {
                let audio = try AVAudioFile(forReading: url)
                let duration = Double(audio.length) / audio.processingFormat.sampleRate
                guard duration.isFinite, duration > 0 else { throw EvalError("Audio has no positive duration.") }
                // The AVAudioFile API handles decoding/format conversion for the modules.
                if let end = try await analyzer.analyzeSequence(from: audio) {
                    try await analyzer.finalizeAndFinish(through: end)
                } else {
                    await analyzer.cancelAndFinishNow()
                    throw EvalError("Analyzer consumed no audio samples.")
                }
                let segments = try await consumer.value
                return Transcription(
                    text: segments.map { $0.text.trimmingCharacters(in: .whitespacesAndNewlines) }
                        .filter { !$0.isEmpty }.joined(separator: " "),
                    audioDurationSeconds: duration,
                    timeToFinalSeconds: seconds(start.duration(to: clock.now)),
                    finalResultCount: segments.count
                )
            } catch {
                await analyzer.cancelAndFinishNow()
                consumer.cancel()
                _ = await consumer.result
                throw error
            }
        }
        group.addTask {
            try await Task.sleep(for: .seconds(timeout))
            throw EvalError("Transcription timed out after \(timeout) seconds.")
        }
        do {
            let result = try await group.next()!
            group.cancelAll()
            return result
        } catch {
            group.cancelAll()
            await analyzer.cancelAndFinishNow()
            throw error
        }
    }
}
