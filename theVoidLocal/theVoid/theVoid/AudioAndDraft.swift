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

@MainActor
final class RecorderEngine: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var elapsed: TimeInterval = 0
    @Published var amplitude: CGFloat = 0.05

    var onWarning: ((TimeInterval) -> Void)?
    var onAutoStop: ((URL, Int) -> Void)?

    private var recorder: AVAudioRecorder?
    private var meterTimer: Timer?
    private var startDate: Date?
    private var currentURL: URL?
    private var warned430 = false
    private var warned455 = false

    func recordPermissionStatus() -> AVAudioSession.RecordPermission {
        AVAudioSession.sharedInstance().recordPermission
    }

    func requestPermission() async -> Bool {
        switch AVAudioSession.sharedInstance().recordPermission {
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
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
    }

    func startRecording(at url: URL) throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatMPEG4AAC),
            AVSampleRateKey: 12_000,
            AVNumberOfChannelsKey: 1,
            AVEncoderAudioQualityKey: AVAudioQuality.high.rawValue,
        ]

        recorder = try AVAudioRecorder(url: url, settings: settings)
        recorder?.isMeteringEnabled = true
        recorder?.record()

        currentURL = url
        startDate = Date()
        warned430 = false
        warned455 = false
        elapsed = 0
        amplitude = 0.05
        isRecording = true

        meterTimer?.invalidate()
        let timer = Timer(timeInterval: 0.1, repeats: true) { [weak self] _ in
            self?.tick()
        }
        RunLoop.main.add(timer, forMode: .common)
        meterTimer = timer
    }

    func stopRecording() -> URL? {
        meterTimer?.invalidate()
        meterTimer = nil
        recorder?.stop()
        isRecording = false
        let finishedURL = currentURL
        currentURL = nil
        return finishedURL
    }

    private func tick() {
        guard let recorder, let startDate else {
            return
        }

        recorder.updateMeters()
        let power = recorder.averagePower(forChannel: 0)
        let normalized = max(0.04, CGFloat((power + 160.0) / 160.0))

        elapsed = Date().timeIntervalSince(startDate)
        amplitude = normalized

        if elapsed >= 270, !warned430 {
            warned430 = true
            onWarning?(270)
        }

        if elapsed >= 295, !warned455 {
            warned455 = true
            onWarning?(295)
        }

        if elapsed >= 300 {
            let duration = max(1, Int(elapsed.rounded()))
            if let finishedURL = stopRecording() {
                onAutoStop?(finishedURL, duration)
            }
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
