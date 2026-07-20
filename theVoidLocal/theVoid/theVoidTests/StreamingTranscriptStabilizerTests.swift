import AVFoundation
import Foundation
import Testing
@testable import theVoid

struct StreamingTranscriptStabilizerTests {
    @Test func confirmsStableWordsWithoutDuplicatingThem() {
        var stabilizer = StreamingTranscriptStabilizer()

        let first = makeHypothesis(["one", "two", "three", "four", "five", "six"])
        let second = makeHypothesis(["one", "two", "three", "four", "five", "six", "seven"])
        let third = makeHypothesis(
            ["five", "six", "seven", "eight", "nine"],
            startingAt: 4
        )

        #expect(stabilizer.incorporate(first) == "one two three four five six")
        #expect(stabilizer.incorporate(second) == "one two three four five six seven")
        #expect(stabilizer.context.clipTimestamp == 4)
        #expect(stabilizer.context.prefixTokens == [5, 6])
        #expect(stabilizer.incorporate(third) == "one two three four five six seven eight nine")
        #expect(stabilizer.context.prefixTokens == [6, 7])
    }

    @Test func letsOnlyTheUnstableTailRevise() {
        var stabilizer = StreamingTranscriptStabilizer()

        let first = makeHypothesis(["I", "think", "this", "is", "good"])
        let revised = makeHypothesis(["I", "think", "this", "is", "great", "actually"])

        _ = stabilizer.incorporate(first)
        let updated = stabilizer.incorporate(revised)

        #expect(updated == "I think this is great actually")
        #expect(stabilizer.context.prefixTokens == [3, 4])
    }

    @Test func displaysFastPartialTextWithoutWordTimings() {
        var stabilizer = StreamingTranscriptStabilizer()
        let hypothesis = StreamingTranscriptionHypothesis(
            text: "This appears while I am speaking",
            words: []
        )

        #expect(stabilizer.incorporate(hypothesis) == "This appears while I am speaking")
    }

    private func makeHypothesis(
        _ words: [String],
        startingAt start: Int = 0
    ) -> StreamingTranscriptionHypothesis {
        let timedWords = words.enumerated().map { index, word in
            let position = start + index
            return StreamingTranscriptionWord(
                text: position == 0 ? word : " \(word)",
                start: Float(position),
                end: Float(position) + 0.8,
                tokens: [position + 1]
            )
        }
        return StreamingTranscriptionHypothesis(
            text: words.joined(separator: " "),
            words: timedWords
        )
    }
}

struct StreamingWhisperSessionTests {
    @Test func publishesAfterFourTenthsOfAudio() async throws {
        let transcriber = ImmediateStreamingTranscriber()
        let session = StreamingWhisperSession(transcriber: transcriber)
        let receivedText = LockedText()
        session.onUpdate = { receivedText.set($0) }

        let format = try #require(AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ))
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 6_400
        ))
        buffer.frameLength = buffer.frameCapacity
        try session.append(buffer)

        for _ in 0..<100 where receivedText.get() == nil {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(receivedText.get() == "Words while speaking")
        session.cancel()
    }

    @Test func boundsPartialInferenceToFifteenSeconds() async throws {
        let transcriber = RecordingStreamingTranscriber()
        let session = StreamingWhisperSession(transcriber: transcriber)
        let format = try #require(AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 16_000,
            channels: 1,
            interleaved: false
        ))
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 16_000
        ))
        buffer.frameLength = buffer.frameCapacity

        for _ in 0..<20 {
            try session.append(buffer)
        }
        for _ in 0..<100 where transcriber.partialCallCount < 2 {
            try await Task.sleep(for: .milliseconds(10))
        }

        #expect(transcriber.partialCallCount >= 2)
        #expect(transcriber.maximumPartialSampleCount == 240_000)
        session.cancel()
    }
}

struct RollingTranscriptAssemblerTests {
    @Test func replacesGrowingHypothesesBeforeTheRollingWindow() {
        var assembler = RollingTranscriptAssembler()

        #expect(assembler.incorporate("one two three", usesRollingWindow: false) == "one two three")
        #expect(assembler.incorporate("one two three four", usesRollingWindow: false) == "one two three four")
    }

    @Test func mergesOnlyTheNewTailFromARollingWindow() {
        var assembler = RollingTranscriptAssembler()
        _ = assembler.incorporate("one two three four five", usesRollingWindow: false)

        let merged = assembler.incorporate(
            "three four five six seven",
            usesRollingWindow: true
        )

        #expect(merged == "one two three four five six seven")
    }

    @Test func keepsStableTextWhenAWindowHasNoReliableOverlap() {
        var assembler = RollingTranscriptAssembler()
        _ = assembler.incorporate("one two three four", usesRollingWindow: false)

        #expect(
            assembler.incorporate("unrelated new window", usesRollingWindow: true)
                == "one two three four"
        )
    }
}

struct TranscriptionLanguageTests {
    @Test func usesTheEnglishOptimizedWhisperModelForEnglish() {
        #expect(WhisperEngineConfiguration(language: .englishUS).modelName == "tiny.en")
        #expect(WhisperEngineConfiguration(language: .englishUK).languageCode == "en")
    }

    @Test func usesTheMultilingualWhisperModelForOtherLanguages() {
        #expect(WhisperEngineConfiguration(language: .spanish).modelName == "tiny")
        #expect(WhisperEngineConfiguration(language: .spanish).languageCode == "es")
    }

    @Test func attributesAcceptedTextToTheSelectedEngineAndLanguage() {
        let configuration = TranscriptionConfiguration(
            engine: .apple,
            language: .spanish
        )

        #expect(configuration.acceptedTranscriptStrategy == "apple_speech_es_accepted")
    }
}

struct M4AAudioTranscoderTests {
    @Test func convertsLinearPCMRecordingToPlayableM4A() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("M4AAudioTranscoderTests-\(UUID().uuidString)")
        let inputURL = directory.appendingPathComponent("capture.caf")
        let outputURL = directory.appendingPathComponent("recording.m4a")
        try FileManager.default.createDirectory(
            at: directory,
            withIntermediateDirectories: true
        )
        defer { try? FileManager.default.removeItem(at: directory) }

        let format = try #require(AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 48_000,
            channels: 1,
            interleaved: false
        ))
        let audioFile = try AVAudioFile(
            forWriting: inputURL,
            settings: format.settings,
            commonFormat: format.commonFormat,
            interleaved: format.isInterleaved
        )
        let buffer = try #require(AVAudioPCMBuffer(
            pcmFormat: format,
            frameCapacity: 48_000
        ))
        buffer.frameLength = buffer.frameCapacity
        let samples = try #require(buffer.floatChannelData?[0])
        for frame in 0..<Int(buffer.frameLength) {
            samples[frame] = Float(sin(2 * Double.pi * 440 * Double(frame) / format.sampleRate)) * 0.1
        }
        try audioFile.write(from: buffer)
        audioFile.close()

        try await M4AAudioTranscoder.export(inputURL: inputURL, outputURL: outputURL)

        let exportedFile = try AVAudioFile(forReading: outputURL)
        #expect(exportedFile.length > 0)
        #expect((try? Data(contentsOf: outputURL).isEmpty) == false)
    }
}

private final class ImmediateStreamingTranscriber: StreamingWhisperTranscribing {
    func transcribe(samples _: [Float]) async throws -> String {
        "Words while speaking"
    }

    func transcribePartial(
        samples _: [Float],
        context _: StreamingTranscriptionContext
    ) async throws -> StreamingTranscriptionHypothesis {
        StreamingTranscriptionHypothesis(text: "Words while speaking", words: [])
    }
}

private final class RecordingStreamingTranscriber: StreamingWhisperTranscribing, @unchecked Sendable {
    private let lock = NSLock()
    private var partialSampleCounts: [Int] = []

    var partialCallCount: Int {
        lock.withLock { partialSampleCounts.count }
    }

    var maximumPartialSampleCount: Int {
        lock.withLock { partialSampleCounts.max() ?? 0 }
    }

    func transcribe(samples _: [Float]) async throws -> String {
        "Final words"
    }

    func transcribePartial(
        samples: [Float],
        context _: StreamingTranscriptionContext
    ) async throws -> StreamingTranscriptionHypothesis {
        lock.withLock {
            partialSampleCounts.append(samples.count)
        }
        return StreamingTranscriptionHypothesis(
            text: "Words while speaking",
            words: []
        )
    }
}

private final class LockedText: @unchecked Sendable {
    private let lock = NSLock()
    private var value: String?

    func set(_ text: String) {
        lock.lock()
        value = text
        lock.unlock()
    }

    func get() -> String? {
        lock.lock()
        defer { lock.unlock() }
        return value
    }
}
