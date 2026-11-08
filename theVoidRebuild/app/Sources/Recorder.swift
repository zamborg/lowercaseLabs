import AVFoundation
import Foundation

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

    var onText: (@Sendable (String) -> Void)?
    var isWarm: Bool { warm != nil }

    /// Everything slow happens here, at app open, so press-down does almost nothing.
    func prewarm() async throws {
        guard await AVAudioApplication.requestRecordPermission() else { throw CaptureError.micDenied }
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.record, mode: .default)
        try session.setActive(true)
        warm = try await LiveTranscription.warm()
        try startListening()
    }

    /// Mic on, audio discarded except the pre-roll. Call again after returning to foreground.
    func startListening() throws {
        guard !engine.isRunning else { return }
        let input = engine.inputNode
        let format = input.outputFormat(forBus: 0)
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
