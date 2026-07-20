import AVFoundation
import Foundation
import SwiftUI

// MARK: - Local draft store

final class DraftStore {
    private let directory: URL
    private let manifestURL: URL
    private let fileManager = FileManager.default

    init() {
        let root = fileManager.urls(for: .documentDirectory, in: .userDomainMask).first!
        self.directory = root.appendingPathComponent("VoidDrafts", isDirectory: true)
        self.manifestURL = directory.appendingPathComponent("drafts.json")

        if !fileManager.fileExists(atPath: directory.path) {
            try? fileManager.createDirectory(
                at: directory,
                withIntermediateDirectories: true,
                attributes: nil
            )
        }

        reconcileManifestWithFilesystem()
    }

    func makeDraftURL() -> URL {
        let draftID = UUID().uuidString
        let fileName = "\(draftID).m4a"
        var manifest = loadManifest()
        let now = ICloudSyncTimestamp.nowString()
        manifest.removeAll { $0.draftID == draftID }
        manifest.append(
            StoredLocalDraft(
                draftID: draftID,
                createdAt: now,
                updatedAt: now,
                audioFileName: fileName
            )
        )
        saveManifest(manifest)
        return directory.appendingPathComponent(fileName)
    }

    func listDrafts() -> [URL] {
        reconcileManifestWithFilesystem()
        let manifest = loadManifest()
            .filter { !$0.isDeleted }
            .sorted { left, right in
                let leftDate = ICloudSyncTimestamp.date(from: left.updatedAt) ?? .distantPast
                let rightDate = ICloudSyncTimestamp.date(from: right.updatedAt) ?? .distantPast
                if leftDate != rightDate {
                    return leftDate > rightDate
                }
                return left.draftID > right.draftID
            }

        return manifest.compactMap { draft in
            let url = directory.appendingPathComponent(draft.audioFileName)
            return fileManager.fileExists(atPath: url.path) ? url : nil
        }
    }

    func delete(_ url: URL) {
        try? fileManager.removeItem(at: url)
        var manifest = loadManifest()
        let fileName = url.lastPathComponent
        let draftID = url.deletingPathExtension().lastPathComponent
        manifest.removeAll { $0.audioFileName == fileName || $0.draftID == draftID }
        saveManifest(manifest)
    }

    func storedDrafts() -> [StoredLocalDraft] {
        reconcileManifestWithFilesystem()
        return loadManifest()
    }

    func storedDraft(draftID: String) -> StoredLocalDraft? {
        loadManifest().first(where: { $0.draftID == draftID })
    }

    func draftData(draftID: String) throws -> Data {
        guard let draft = storedDraft(draftID: draftID), !draft.isDeleted else {
            throw LocalReflectionError.missingAudioForEntry
        }
        let url = directory.appendingPathComponent(draft.audioFileName)
        guard fileManager.fileExists(atPath: url.path) else {
            throw LocalReflectionError.missingAudioForEntry
        }
        return try Data(contentsOf: url)
    }

    func upsertDraftFromCloud(_ draft: StoredLocalDraft, audioData: Data) throws {
        let targetURL = directory.appendingPathComponent(draft.audioFileName)
        try audioData.write(to: targetURL, options: [.atomic])

        var manifest = loadManifest()
        manifest.removeAll { $0.draftID == draft.draftID }
        manifest.append(draft)
        saveManifest(manifest)
    }

    func deleteDraft(draftID: String) {
        var manifest = loadManifest()
        guard let draft = manifest.first(where: { $0.draftID == draftID }) else {
            return
        }
        let targetURL = directory.appendingPathComponent(draft.audioFileName)
        try? fileManager.removeItem(at: targetURL)
        manifest.removeAll { $0.draftID == draftID }
        saveManifest(manifest)
    }

    func updateCloudChangeTag(draftID: String, changeTag: String?) {
        var manifest = loadManifest()
        guard let index = manifest.firstIndex(where: { $0.draftID == draftID }) else {
            return
        }
        let current = manifest[index]
        manifest[index] = StoredLocalDraft(
            draftID: current.draftID,
            createdAt: current.createdAt,
            updatedAt: current.updatedAt,
            audioFileName: current.audioFileName,
            isDeleted: current.isDeleted,
            cloudRecordChangeTag: changeTag
        )
        saveManifest(manifest)
    }

    private func reconcileManifestWithFilesystem() {
        let files = ((try? fileManager.contentsOfDirectory(
            at: directory,
            includingPropertiesForKeys: nil,
            options: [.skipsHiddenFiles]
        )) ?? [])
        .filter { $0.pathExtension.lowercased() == "m4a" }

        var manifest = loadManifest()
        let knownIDs = Set(manifest.map(\.draftID))
        let now = ICloudSyncTimestamp.nowString()

        for fileURL in files {
            let draftID = fileURL.deletingPathExtension().lastPathComponent
            if knownIDs.contains(draftID) {
                continue
            }
            manifest.append(
                StoredLocalDraft(
                    draftID: draftID,
                    createdAt: now,
                    updatedAt: now,
                    audioFileName: fileURL.lastPathComponent
                )
            )
        }

        let validFileNames = Set(files.map(\.lastPathComponent))
        manifest.removeAll { !validFileNames.contains($0.audioFileName) }
        saveManifest(manifest)
    }

    private func loadManifest() -> [StoredLocalDraft] {
        guard fileManager.fileExists(atPath: manifestURL.path),
              let data = try? Data(contentsOf: manifestURL) else {
            return []
        }
        return (try? JSONDecoder().decode([StoredLocalDraft].self, from: data)) ?? []
    }

    private func saveManifest(_ manifest: [StoredLocalDraft]) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        guard let data = try? encoder.encode(manifest) else {
            return
        }
        try? data.write(to: manifestURL, options: [.atomic])
    }
}

// MARK: - Recorder

struct RecordingResult {
    let url: URL
    let durationSeconds: Int
    let transcript: String?
    let audioErrorDescription: String?
}

struct RecordingFinalization {
    let url: URL
    let durationSeconds: Int
    let initialTranscript: String
    let task: Task<RecordingResult, Never>
}

enum MicrophonePermissionStatus {
    case granted
    case denied
    case undetermined
}

@MainActor
final class RecorderEngine: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var isFinalizing = false
    @Published var elapsed: TimeInterval = 0
    @Published var amplitude: CGFloat = 0.05
    @Published var liveTranscript = ""

    var onWarning: ((TimeInterval) -> Void)?
    var onAutoStop: (() -> Void)?

    private let audioEngine = AVAudioEngine()
    private var captureSink: RecordingCaptureSink?
    private var transcriptionSession: (any LiveTranscriptionSession)?
    private var meterTimer: Timer?
    private var startDate: Date?
    private var currentURL: URL?
    private var currentCaptureURL: URL?
    private var warned430 = false
    private var warned455 = false
    private var autoStopTriggered = false

    func recordPermissionStatus() -> MicrophonePermissionStatus {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            return .granted
        case .denied:
            return .denied
        case .undetermined:
            return .undetermined
        @unknown default:
            return .undetermined
        }
    }

    func requestPermission() async -> Bool {
        switch AVAudioApplication.shared.recordPermission {
        case .granted:
            return true
        case .denied:
            return false
        case .undetermined:
            break
        @unknown default:
            break
        }

        return await withCheckedContinuation { continuation in
            AVAudioApplication.requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording(
        at url: URL,
        sessionFactory: (AVAudioFormat) throws -> any LiveTranscriptionSession
    ) throws {
        guard !isRecording, !isFinalizing else {
            throw StreamingTranscriptionError.recorderBusy
        }

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(
            .playAndRecord,
            mode: .measurement,
            options: [.defaultToSpeaker, .allowBluetoothHFP]
        )
        try session.setPreferredSampleRate(48_000)
        try session.setPreferredIOBufferDuration(0.02)
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        audioEngine.stop()
        audioEngine.reset()
        let inputNode = audioEngine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw StreamingTranscriptionError.audioCaptureFailed
        }

        let transcriptionSession: any LiveTranscriptionSession
        do {
            transcriptionSession = try sessionFactory(inputFormat)
        } catch {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
        transcriptionSession.onUpdate = { [weak self] text in
            self?.liveTranscript = text
        }
        let captureURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("void-recording-\(UUID().uuidString)")
            .appendingPathExtension("caf")
        let sink: RecordingCaptureSink
        do {
            sink = try RecordingCaptureSink(
                url: captureURL,
                inputFormat: inputFormat,
                streamingSession: transcriptionSession
            )
        } catch {
            transcriptionSession.cancel()
            try? FileManager.default.removeItem(at: captureURL)
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }

        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 4_096, format: inputFormat) { buffer, _ in
            sink.receive(buffer)
        }

        do {
            audioEngine.prepare()
            try audioEngine.start()
        } catch {
            inputNode.removeTap(onBus: 0)
            sink.finish()
            transcriptionSession.cancel()
            try? FileManager.default.removeItem(at: captureURL)
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }

        captureSink = sink
        self.transcriptionSession = transcriptionSession
        currentURL = url
        currentCaptureURL = captureURL
        startDate = Date()
        warned430 = false
        warned455 = false
        autoStopTriggered = false
        elapsed = 0
        amplitude = 0.05
        liveTranscript = ""
        isRecording = true

        meterTimer?.invalidate()
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.tick()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    func stopRecording() -> RecordingFinalization? {
        guard isRecording,
              let currentURL,
              let currentCaptureURL,
              let captureSink,
              let transcriptionSession else {
            return nil
        }

        meterTimer?.invalidate()
        meterTimer = nil
        audioEngine.inputNode.removeTap(onBus: 0)
        audioEngine.stop()

        let captureError = captureSink.finish()
        let duration = max(1, Int(elapsed.rounded()))
        let initialTranscript = liveTranscript.trimmingCharacters(in: .whitespacesAndNewlines)

        isRecording = false
        isFinalizing = true
        self.currentURL = nil
        self.currentCaptureURL = nil
        self.captureSink = nil
        self.transcriptionSession = nil
        startDate = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)

        let finalizationTask = Task { [weak self] in
            let transcriptionTask = Task {
                try? await transcriptionSession.finish()
            }
            var audioError = captureError

            if audioError == nil {
                do {
                    try await M4AAudioTranscoder.export(
                        inputURL: currentCaptureURL,
                        outputURL: currentURL
                    )
                } catch {
                    audioError = error
                }
            }

            let finalTranscript = await transcriptionTask.value
            try? FileManager.default.removeItem(at: currentCaptureURL)
            self?.isFinalizing = false
            return RecordingResult(
                url: currentURL,
                durationSeconds: duration,
                transcript: finalTranscript,
                audioErrorDescription: audioError?.localizedDescription
            )
        }

        return RecordingFinalization(
            url: currentURL,
            durationSeconds: duration,
            initialTranscript: initialTranscript,
            task: finalizationTask
        )
    }

    private func tick() {
        guard isRecording, let startDate else {
            return
        }

        elapsed = Date().timeIntervalSince(startDate)
        amplitude = captureSink?.currentAmplitude ?? 0.05

        if elapsed >= 120, !warned430 {
            warned430 = true
            onWarning?(120)
        }

        if elapsed >= 145, !warned455 {
            warned455 = true
            onWarning?(145)
        }

        if elapsed >= 150, !autoStopTriggered {
            autoStopTriggered = true
            onAutoStop?()
        }
    }
}

private final class RecordingCaptureSink {
    private let fileLock = NSLock()
    private let stateLock = NSLock()
    private let transcriptionSession: any LiveTranscriptionSession
    private var audioFile: AVAudioFile?
    private var audioError: Error?
    private var streamingFailed = false
    private var smoothedAmplitude: CGFloat = 0.05

    var currentAmplitude: CGFloat {
        stateLock.lock()
        defer { stateLock.unlock() }
        return smoothedAmplitude
    }

    init(
        url: URL,
        inputFormat: AVAudioFormat,
        streamingSession: any LiveTranscriptionSession
    ) throws {
        transcriptionSession = streamingSession
        audioFile = try AVAudioFile(
            forWriting: url,
            settings: inputFormat.settings,
            commonFormat: inputFormat.commonFormat,
            interleaved: inputFormat.isInterleaved
        )
    }

    func receive(_ buffer: AVAudioPCMBuffer) {
        fileLock.lock()
        if audioError == nil, let audioFile {
            do {
                try audioFile.write(from: buffer)
            } catch {
                audioError = error
            }
        }
        fileLock.unlock()

        if !streamingFailed {
            do {
                try transcriptionSession.append(buffer)
            } catch {
                streamingFailed = true
                transcriptionSession.cancel()
            }
        }

        updateAmplitude(from: buffer)
    }

    @discardableResult
    func finish() -> Error? {
        fileLock.lock()
        defer { fileLock.unlock() }
        audioFile?.close()
        audioFile = nil
        return audioError
    }

    private func updateAmplitude(from buffer: AVAudioPCMBuffer) {
        guard let channels = buffer.floatChannelData else { return }
        let frameCount = Int(buffer.frameLength)
        let channelCount = Int(buffer.format.channelCount)
        guard frameCount > 0, channelCount > 0 else { return }

        var sumOfSquares: Float = 0
        for channel in 0..<channelCount {
            let samples = channels[channel]
            for frame in 0..<frameCount {
                let sample = samples[frame]
                sumOfSquares += sample * sample
            }
        }

        let rms = sqrt(sumOfSquares / Float(frameCount * channelCount))
        let decibels = 20 * log10(max(rms, 0.000_001))
        let linear = max(0, min(1, (decibels + 55) / 55))
        let normalized = max(0.04, CGFloat(pow(linear, 0.72)))

        stateLock.lock()
        smoothedAmplitude = (smoothedAmplitude * 0.68) + (normalized * 0.32)
        stateLock.unlock()
    }
}

enum M4AAudioTranscoderError: LocalizedError {
    case exportSessionUnavailable

    var errorDescription: String? {
        "The recording could not be converted to M4A."
    }
}

enum M4AAudioTranscoder {
    static func export(inputURL: URL, outputURL: URL) async throws {
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: outputURL.path) {
            try fileManager.removeItem(at: outputURL)
        }

        let asset = AVURLAsset(url: inputURL)
        guard let exportSession = AVAssetExportSession(
            asset: asset,
            presetName: AVAssetExportPresetAppleM4A
        ) else {
            throw M4AAudioTranscoderError.exportSessionUnavailable
        }

        do {
            try await exportSession.export(to: outputURL, as: .m4a)
        } catch {
            try? fileManager.removeItem(at: outputURL)
            throw error
        }
    }
}

// MARK: - Audio playback

@MainActor
final class AudioPlaybackController: NSObject, ObservableObject, AVAudioPlayerDelegate {
    @Published var isLoading = false
    @Published var isReady = false
    @Published var isPlaying = false
    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0

    private var player: AVAudioPlayer?
    private var progressTimer: Timer?
    private var isScrubbing = false

    func load(fetchAudio: () async throws -> Data, forceReload: Bool = false) async throws {
        if isLoading {
            return
        }
        if isReady && !forceReload {
            return
        }

        isLoading = true
        defer { isLoading = false }

        let data = try await fetchAudio()
        try configurePlayer(with: data)
    }

    func togglePlayback() {
        guard let player else {
            return
        }

        if player.isPlaying {
            player.pause()
            isPlaying = false
            stopTimer()
            return
        }

        player.currentTime = currentTime
        if player.play() {
            isPlaying = true
            startTimer()
        }
    }

    func restart() {
        guard let player else {
            return
        }
        player.currentTime = 0
        currentTime = 0
        if !player.isPlaying {
            _ = player.play()
        }
        isPlaying = player.isPlaying
        startTimer()
    }

    func stop() {
        guard let player else {
            return
        }
        player.stop()
        player.currentTime = 0
        currentTime = 0
        isPlaying = false
        stopTimer()
    }

    func beginScrubbing() {
        isScrubbing = true
    }

    func scrub(to time: TimeInterval) {
        currentTime = min(max(time, 0), max(duration, 0))
    }

    func endScrubbing() {
        guard let player else {
            isScrubbing = false
            return
        }
        player.currentTime = currentTime
        isScrubbing = false
    }

    func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        isPlaying = false
        currentTime = duration
        stopTimer()
    }

    private func configurePlayer(with data: Data) throws {
        stopTimer()
        player?.stop()

        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default, options: [])
        try? session.setActive(true, options: .notifyOthersOnDeactivation)

        let nextPlayer = try AVAudioPlayer(data: data)
        nextPlayer.delegate = self
        nextPlayer.prepareToPlay()

        player = nextPlayer
        duration = nextPlayer.duration
        currentTime = 0
        isPlaying = false
        isReady = true
    }

    private func startTimer() {
        stopTimer()
        let timer = Timer(timeInterval: 0.2, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer, forMode: .common)
        progressTimer = timer
    }

    private func stopTimer() {
        progressTimer?.invalidate()
        progressTimer = nil
    }

    private func tick() {
        guard let player else {
            return
        }
        if !isScrubbing {
            currentTime = player.currentTime
        }
        isPlaying = player.isPlaying
    }
}
