import AVFoundation
import Foundation
import UIKit

struct Capture {
    let id: UUID
    let audioURL: URL
    let transcript: String
    let duration: TimeInterval
    let timings: Timings
}

/// Latency marks, so we measure the targets instead of guessing.
struct Timings {
    var pressAt: CFAbsoluteTime = 0
    var capturingAt: CFAbsoluteTime = 0   // file + transcriber attached, held audio flushed
    var releaseAt: CFAbsoluteTime = 0
    var finalAt: CFAbsoluteTime = 0
    var preRollMs: Double = 0
    var pressToCapturingMs: Double { (capturingAt - pressAt) * 1000 }
    var releaseToFinalMs: Double { (finalAt - releaseAt) * 1000 }
}

struct Session { let id: UUID; let url: URL; let startedAt: Date }

/// One AVAudioEngine tap feeds BOTH the file writer and the transcriber. Single source of audio.
///
/// The engine runs while the app is in the foreground (mic hardware startup is ~150 ms, far over
/// budget). While idle, audio is discarded except a short rolling pre-roll, so a press captures
/// the moments just before it. Nothing is stored or transcribed until a press.
final class Recorder: @unchecked Sendable {
    private let engine = AVAudioEngine()
    private var warm: LiveTranscription.Warm?

    private let lock = NSLock()
    private let engineLock = NSLock()
    private var isRecording = false
    private var file: AVAudioFile?
    private var live: LiveTranscription?
    private var current: Session?
    private var timings = Timings()
    private var pending: [AVAudioPCMBuffer] = []
    private var ring: [AVAudioPCMBuffer] = []
    private var ringFrames = 0
    private var ringSampleRate = 48_000.0
    private let preRollSeconds = 0.3

    init() {
        // A call, Siri, or a route change stops the engine under us. Come back when we can, only in the foreground.
        let nc = NotificationCenter.default
        nc.addObserver(forName: AVAudioSession.interruptionNotification, object: nil, queue: .main) { [weak self] note in
            let raw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt ?? 0
            if AVAudioSession.InterruptionType(rawValue: raw) == .ended { self?.recoverListening() }
        }
        nc.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) { [weak self] _ in
            self?.recoverListening()
        }
    }

    private func recoverListening() {
        guard warm != nil, UIApplication.shared.applicationState == .active else { return }
        stopListening()
        try? startListening()
    }

    var onText: (@Sendable (String) -> Void)?
    /// 0...1 voice level, ~30 Hz, on the main queue, only while recording.
    var onLevel: (@Sendable (Double) -> Void)?
    private var lastLevelAt: CFAbsoluteTime = 0   // audio thread only
    private var smoothedLevel = 0.0               // audio thread only
    var isWarm: Bool { warm != nil }

    /// Everything slow happens here, at app open, so press-down does almost nothing.
    func prewarm() async throws {
        guard await AVAudioApplication.requestRecordPermission() else { throw CaptureError.micDenied }
        let session = AVAudioSession.sharedInstance()
        // playAndRecord so entries can be played back while the mic stays warm for the next press.
        try session.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker, .allowBluetoothHFP])
        try session.setActive(true)
        warm = try await LiveTranscription.warm()
        try startListening()
    }

    /// Mic on, audio discarded except the pre-roll. Call again after returning to foreground.
    func startListening() throws {
        // prewarm (background executor) and the foreground hook (main) can both land here at launch;
        // AVAudioEngine asserts if two taps are installed, so this is serialized.
        engineLock.lock(); defer { engineLock.unlock() }
        guard !engine.isRunning else { return }
        // The session may have been deactivated (interruption, lock, call). Reactivate, and refuse to touch the
        // engine with a dead input format: installTap on one raises an Objective-C exception we can't catch.
        try AVAudioSession.sharedInstance().setActive(true)
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else { throw CaptureError.micUnavailable }
        ringSampleRate = format.sampleRate
        input.removeTap(onBus: 0)
        input.installTap(onBus: 0, bufferSize: 512, format: format) { [weak self] buffer, _ in
            self?.handle(buffer)
        }
        engine.prepare()
        try engine.start()
    }

    /// Mic off (app backgrounded). Never leave the mic running in the background.
    func stopListening() {
        engineLock.lock(); defer { engineLock.unlock() }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        lock.withLock { ring = []; ringFrames = 0 }
    }

    func start() throws {
        guard let warm else { throw CaptureError.notWarm }
        if !engine.isRunning { try startListening() }
        var t = Timings()
        t.pressAt = CFAbsoluteTimeGetCurrent()

        let held = lock.withLock { () -> [AVAudioPCMBuffer] in
            self.isRecording = true
            self.file = nil; self.live = nil
            let held = ring
            ring = []; ringFrames = 0
            pending = held
            return held
        }
        t.preRollMs = Double(held.reduce(0) { $0 + Int($1.frameLength) }) / ringSampleRate * 1000

        let format = engine.inputNode.outputFormat(forBus: 0)
        let id = UUID()
        let url = Entry.audioDirectory().appendingPathComponent("\(id.uuidString).caf")
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        let live = try LiveTranscription(warm: warm, inputFormat: format)
        live.onText = onText

        lock.lock()
        current = Session(id: id, url: url, startedAt: Date())
        self.file = file
        self.live = live
        for buffer in pending {
            try? file.write(from: buffer)
            live.append(buffer)
        }
        pending = []
        t.capturingAt = CFAbsoluteTimeGetCurrent()
        timings = t
        lock.unlock()
    }

    /// Audio render thread.
    private func handle(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        if !isRecording {
            let copy = buffer.copy() ?? buffer
            ring.append(copy)
            ringFrames += Int(copy.frameLength)
            let maxFrames = Int(preRollSeconds * ringSampleRate)
            while ring.count > 1, ringFrames - Int(ring[0].frameLength) >= maxFrames {
                ringFrames -= Int(ring[0].frameLength)
                ring.removeFirst()
            }
            lock.unlock()
            return
        }
        guard let file, let live else {
            pending.append(buffer.copy() ?? buffer)
            lock.unlock()
            return
        }
        lock.unlock()
        try? file.write(from: buffer)
        live.append(buffer)
        meter(buffer)
    }

    /// Cheap RMS to dB, smoothed, throttled to ~30 Hz. Runs after the audio is already written and queued.
    private func meter(_ buffer: AVAudioPCMBuffer) {
        let now = CFAbsoluteTimeGetCurrent()
        guard now - lastLevelAt > 1.0 / 30, let data = buffer.floatChannelData?[0], buffer.frameLength > 0 else { return }
        lastLevelAt = now
        var sum: Float = 0
        for i in 0..<Int(buffer.frameLength) { sum += data[i] * data[i] }
        let db = 20 * log10(max(Double((sum / Float(buffer.frameLength)).squareRoot()), 1e-6))
        let target = min(max((db + 50) / 40, 0), 1)       // -50 dB silence ... -10 dB loud
        smoothedLevel += (target - smoothedLevel) * (target > smoothedLevel ? 0.6 : 0.25)
        let level = smoothedLevel
        DispatchQueue.main.async { [weak self] in self?.onLevel?(level) }
    }

    func stop() async -> Capture? {
        let release = CFAbsoluteTimeGetCurrent()
        let (live, current, stopTimings) = takeSession(release: release)
        guard let live, let current else { return nil }
        let transcript = await live.finish()
        var timings = stopTimings
        timings.finalAt = CFAbsoluteTimeGetCurrent()
        return Capture(
            id: current.id, audioURL: current.url, transcript: transcript,
            duration: Date().timeIntervalSince(current.startedAt), timings: timings
        )
    }

    private func takeSession(release: CFAbsoluteTime) -> (LiveTranscription?, Session?, Timings) {
        lock.lock(); defer { lock.unlock() }
        isRecording = false
        file = nil // closes the file
        var t = timings
        t.releaseAt = release
        let result = (live, current, t)
        live = nil; current = nil
        return result
    }
}

private extension AVAudioPCMBuffer {
    /// Tap buffers may be recycled, so held audio is copied.
    func copy() -> AVAudioPCMBuffer? {
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameLength),
              let src = floatChannelData, let dst = out.floatChannelData else { return nil }
        out.frameLength = frameLength
        for ch in 0..<Int(format.channelCount) {
            memcpy(dst[ch], src[ch], Int(frameLength) * MemoryLayout<Float>.size)
        }
        return out
    }
}
