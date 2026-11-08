import SwiftData
import SwiftUI
import UIKit

@MainActor
@Observable
final class CaptureModel {
    enum Phase { case warming, ready, recording, finishing, failed(String) }
    var phase: Phase = .warming
    var liveText = ""
    var level = 0.0
    var locked = false
    var last: Capture?
    var lastEntry: Entry?
    var taggingMs: Double?
    var note = ""
    /// Private until the user says otherwise, then whatever they chose last (D5).
    private(set) var visibility: Visibility = Visibility(rawValue: UserDefaults.standard.string(forKey: "lastVisibility") ?? "") ?? .private

    private let recorder = Recorder()
    let journal: Journal
    private var tagger: Tagger { journal.tagger }

    init(journal: Journal) {
        self.journal = journal
        recorder.onText = { [weak self] text in
            Task { @MainActor in self?.liveText = text }
        }
        recorder.onLevel = { [weak self] level in
            Task { @MainActor in self?.level = level }
        }
    }

    func setVisibility(_ visibility: Visibility) {
        self.visibility = visibility
        UserDefaults.standard.set(visibility.rawValue, forKey: "lastVisibility")
    }

    func warmUp() async {
        do {
            try await recorder.prewarm()
            phase = .ready          // capture is usable as soon as audio + speech are ready
        } catch {
            phase = .failed("\(error)")
            return
        }
        do {
            try await tagger.warm()  // model load happens in the background
            await journal.enrichPending()
        } catch {
            note = "tagger unavailable: \(error)"
        }
        await journal.reconcileDots()
        if ProcessInfo.processInfo.arguments.contains("-autotest") { await autotest() }
        if ProcessInfo.processInfo.arguments.contains("-cloudtest") { await cloudTest() }
        if let i = ProcessInfo.processInfo.arguments.firstIndex(of: "-longtest"),
           let secs = Double(ProcessInfo.processInfo.arguments.dropFirst(i + 1).first ?? "") { await longTest(seconds: secs) }
    }

    /// `-longtest <seconds>`: one long recording, then tagging of a long transcript. Run on the phone with auto-lock
    /// on: the screen must stay awake while recording, and nothing may crash.
    private func longTest(seconds: Double) async {
        func line(_ s: String) { print("LONG \(s)"); fflush(stdout) }
        press()
        guard case .recording = phase else { line("FAILED could not start: \(phase)"); return }
        line("recording \(Int(seconds)) s, idleTimerDisabled=\(UIApplication.shared.isIdleTimerDisabled)")
        try? await Task.sleep(for: .seconds(seconds))
        if case .recording = phase {
            phase = .finishing
            let entry = await finishRecording()
            line("saved duration=\(Int(entry?.duration ?? -1)) s, kept=\(entry != nil), idleTimerDisabled=\(UIApplication.shared.isIdleTimerDisabled)")
        } else { line("phase changed during recording: \(phase) (app was backgrounded?)") }
        let long = String(repeating: "I felt a bit overwhelmed at work today but then a walk with a friend helped me feel calmer and more grateful. ", count: 40)
        let t = CFAbsoluteTimeGetCurrent()
        let r = try? await tagger.tag(transcript: long)
        line(String(format: "tag %d words ms=%.0f ok=%@", long.split(separator: " ").count, (CFAbsoluteTimeGetCurrent() - t) * 1000, "\(r != nil)"))
        line("done")
    }

    /// `-cloudtest` launch argument: F6 against the real CloudKit zone (this account, development environment).
    private func cloudTest() async {
        func line(_ s: String) { print("CLOUD \(s)"); fflush(stdout) }
        let dots = DotStore.shared
        guard DotStore.enabled else { line("FAILED iCloud off in this build"); return }
        let e = journal.save(audioFileName: "none.caf", duration: 1, transcript: "cloud test", visibility: .shared)
        journal.setTags([.highPleasant], on: e)
        func present() async -> Bool? { try? await dots.ownDotIDs().contains(e.id.uuidString) }
        var t = CFAbsoluteTimeGetCurrent()
        await journal.reconcileDots()
        line("shared+tagged -> dot present=\(String(describing: await present())) ms=\(Int((CFAbsoluteTimeGetCurrent() - t) * 1000)) status=\(dots.status)")
        journal.setVisibility(.private, on: e); t = CFAbsoluteTimeGetCurrent()
        await journal.reconcileDots()
        line("flipped private -> dot present=\(String(describing: await present())) ms=\(Int((CFAbsoluteTimeGetCurrent() - t) * 1000))")
        journal.setVisibility(.shared, on: e)
        await journal.reconcileDots()
        line("flipped shared -> dot present=\(String(describing: await present()))")
        journal.delete(e); await journal.reconcileDots()
        line("deleted -> dot present=\(String(describing: await present())) status=\(dots.status)")
        line("done")
    }

    /// Foreground/background. `.inactive` (Control Center, notification, screen locking) is ignored for the mic: only a
    /// real background stops it, and a recording in flight is saved first, never dropped or torn down mid-write.
    func scenePhaseChanged(_ phase: ScenePhase) {
        guard recorder.isWarm else { return }
        switch phase {
        case .active:
            do { try recorder.startListening(); if case .failed = self.phase { self.phase = .ready } }
            catch { note = "mic unavailable: \(error)" }
        case .background:
            if case .recording = self.phase {
                self.phase = .finishing
                Task {
                    let entry = await finishRecording()
                    recorder.stopListening()
                    if let entry { await enrich(entry) }   // pauses itself: no GPU in the background
                }
            } else {
                recorder.stopListening()
            }
        default: break
        }
    }

    func press() {
        guard case .ready = phase else { return }
        liveText = ""
        lastEntry = nil
        taggingMs = nil
        locked = false
        do {
            try recorder.start()
            phase = .recording
            UIApplication.shared.isIdleTimerDisabled = true   // the screen must not lock mid-recording
        } catch {
            phase = .failed("\(error)")
        }
    }

    /// Slid up past the threshold while holding: keep recording hands-free until a tap.
    func lock() {
        guard case .recording = phase else { return }
        locked = true
    }

    func release() {
        guard case .recording = phase else { return }
        phase = .finishing
        Task {
            if let entry = await finishRecording() { await enrich(entry) }
        }
    }

    /// Stops, then saves before anything else can fail. Returns the saved entry.
    private func finishRecording() async -> Entry? {
        let capture = await recorder.stop()
        last = capture
        phase = .ready
        level = 0
        locked = false
        UIApplication.shared.isIdleTimerDisabled = false
        guard let capture else { return nil }
        liveText = capture.transcript
        // A real recording is never thrown away because speech failed; only accidental taps are.
        guard capture.duration >= 1 || !capture.transcript.isEmpty else {
            try? FileManager.default.removeItem(at: capture.audioURL)
            return nil
        }
        let entry = journal.save(id: capture.id, audioFileName: capture.audioURL.lastPathComponent,
                                 duration: capture.duration, transcript: capture.transcript, visibility: visibility)
        lastEntry = entry
        return entry
    }

    /// Back in the foreground: enrich whatever the model couldn't finish while we were away, and settle dots.
    func resumeEnrichment() async {
        guard tagger.isWarm else { return }
        await journal.enrichPending()
        await journal.reconcileDots()
    }

    private func enrich(_ entry: Entry) async {
        guard tagger.isWarm else { return }   // enrichPending picks it up once the model is ready
        let started = CFAbsoluteTimeGetCurrent()
        do {
            try await journal.enrich(entry)
            taggingMs = (CFAbsoluteTimeGetCurrent() - started) * 1000
        } catch TaggerError.paused {
            // App left the foreground mid-tagging: the entry is saved and gets enriched on return.
        } catch {
            note = "tagging failed: \(error)"
        }
    }

    /// `-autotest` launch argument: press/release programmatically and print timings to the console.
    private func autotest() async {
        print("TIMING start"); fflush(stdout)
        for i in 1...6 {
            try? await Task.sleep(for: .seconds(1))
            press()
            try? await Task.sleep(for: .seconds(1.5))
            guard case .recording = phase else { print("TIMING run=\(i) FAILED \(phase)"); continue }
            phase = .finishing
            if let entry = await finishRecording() { await enrich(entry) }
            if let t = last?.timings {
                print(String(format: "TIMING run=%d press_to_capturing_ms=%.1f release_to_final_ms=%.1f tagging_ms=%.0f",
                             i, t.pressToCapturingMs, t.releaseToFinalMs, taggingMs ?? -1)); fflush(stdout)
            }
        }
        for i in 1...3 {
            let t = CFAbsoluteTimeGetCurrent()
            let r = try? await tagger.tag(transcript: "I'm so excited, I just got the job offer and I can't stop smiling.")
            print(String(format: "TIMING tag_run=%d ms=%.0f tags=%@", i, (CFAbsoluteTimeGetCurrent() - t) * 1000,
                         "\(r?.tags.map(\.rawValue) ?? [])")); fflush(stdout)
        }
        print("TIMING done"); fflush(stdout)
    }

    /// `-seed` launch argument: sample entries for exercising the UI where mic/speech don't exist.
    func seedIfRequested() {
        guard ProcessInfo.processInfo.arguments.contains("-seed") else { return }
        // Deterministic: a seeded launch always starts from exactly these entries.
        for old in (try? journal.context.fetch(FetchDescriptor<Entry>())) ?? [] { journal.context.delete(old) }
        let samples: [(String, String, Quadrant, Visibility)] = [
            ("Big launch day", "I can't believe we shipped it, I'm buzzing, so much energy today.", .highPleasant, .shared),
            ("Quiet evening", "Just sitting on the porch, tea in hand, feeling really calm and content.", .lowPleasant, .private),
            ("Deadline dread", "Everything is due tomorrow and my chest is tight, I can't stop spiraling.", .highUnpleasant, .private),
            ("Running on empty", "I'm so tired. Nothing feels worth doing and I just want to lie down.", .lowUnpleasant, .shared),
        ]
        for (i, s) in samples.enumerated() {
            let e = journal.save(audioFileName: "none.caf", duration: 8 + Double(i) * 21, transcript: s.1, visibility: s.3,
                                 createdAt: Date().addingTimeInterval(Double(-i) * 7200))
            e.title = s.0; e.titleSourceValue = Source.model.rawValue
            e.tagValues = [s.2.rawValue]; e.tagsSourceValue = Source.model.rawValue
        }
        try? journal.context.save()
        let now = Date()
        func dot(_ hours: Double, _ tags: [Quadrant]) -> FriendDot {
            FriendDot(id: UUID().uuidString, ownerID: "x", createdAt: now.addingTimeInterval(-hours * 3600), tags: tags)
        }
        DotStore.shared.seedFriends([
            Friend(id: "a", name: "Maya", dots: [dot(2, [.lowPleasant]), dot(20, [.lowPleasant, .highPleasant]), dot(30, [.highPleasant]), dot(55, [.lowPleasant]), dot(100, [.highUnpleasant])]),
            Friend(id: "b", name: "Dev", dots: [dot(1, [.lowUnpleasant]), dot(5, [.lowUnpleasant]), dot(26, [.lowUnpleasant, .highUnpleasant]), dot(75, [.lowPleasant])], sharesBack: false),
        ])
    }
}

struct CaptureView: View {
    let model: CaptureModel
    @State private var pressed = false
    @State private var ignoreTouch = false
    @State private var dragUp: CGFloat = 0

    private let lockThreshold: CGFloat = 90
    private var debug: Bool { ProcessInfo.processInfo.arguments.contains("-debug") }

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 18) {
                Text(model.liveText)
                    .font(.system(.title3, design: .rounded))
                    .foregroundStyle(.white.opacity(0.85))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(.horizontal)

                if let entry = model.lastEntry, !entry.title.isEmpty {
                    HStack(spacing: 8) {
                        ForEach(entry.tags, id: \.self) { Circle().fill($0.color).frame(width: 14, height: 14) }
                        Text(entry.title).font(.headline).foregroundStyle(.white)
                    }
                    .transition(.opacity)
                }
                status
                visibilityToggle
                circle
            }
            .animation(.easeOut(duration: 0.2), value: model.lastEntry?.title)
        }
        .task {
            await model.warmUp()
        }
    }

    private var status: some View {
        VStack(spacing: 2) {
            if debug, let c = model.last {
                Text(String(format: "press→capturing %.0f ms · release→final %.0f ms%@",
                            c.timings.pressToCapturingMs, c.timings.releaseToFinalMs,
                            model.taggingMs.map { String(format: " · tagged %.0f ms", $0) } ?? ""))
            }
            if !model.note.isEmpty { Text(model.note).foregroundStyle(.orange) }
            if case .failed(let message) = model.phase { Text(message).foregroundStyle(.red) }
        }
        .font(.caption2.monospaced())
        .foregroundStyle(.white.opacity(0.4))
        .multilineTextAlignment(.center)
        .padding(.horizontal)
    }

    private var visibilityToggle: some View {
        let shared = model.visibility == .shared
        return Button {
            model.setVisibility(shared ? .private : .shared)
        } label: {
            Label(shared ? "shared with friends" : "private", systemImage: shared ? "person.2.fill" : "lock.fill")
                .font(.footnote.weight(.medium))
                .foregroundStyle(shared ? Color.black : .white.opacity(0.7))
                .padding(.horizontal, 14).padding(.vertical, 8)
                .background(shared ? Color.white : .white.opacity(0.12), in: Capsule())
        }
        .buttonStyle(.plain)
        .opacity(isRecording ? 0.4 : 1)
        .accessibilityIdentifier("visibilityToggle")
    }

    private var isRecording: Bool { if case .recording = model.phase { true } else { false } }

    private var circle: some View {
        let willLock = dragUp > lockThreshold
        return ZStack {
            // Lock affordance: appears above the circle while holding, fills when the slide is far enough.
            if pressed && !model.locked {
                Image(systemName: willLock ? "lock.fill" : "lock.open")
                    .font(.title3)
                    .foregroundStyle(willLock ? .black : .white.opacity(0.7))
                    .frame(width: 44, height: 44)
                    .background(willLock ? Color.white : .white.opacity(0.15), in: Circle())
                    .offset(y: -150)
                    .transition(.opacity)
            }
            Circle()
                .fill(model.locked ? Color.red.opacity(0.9) : .white.opacity(pressed ? 1 : 0.85))
                .frame(width: (pressed || model.locked) ? 120 : 96, height: (pressed || model.locked) ? 120 : 96)
                .scaleEffect(1 + (isRecording ? model.level * 0.45 : 0))
                .animation(.linear(duration: 0.08), value: model.level)
                .animation(.spring(duration: 0.18, bounce: 0.3), value: pressed)
                .overlay { if model.locked { Image(systemName: "stop.fill").font(.title).foregroundStyle(.white) } }
                .opacity(isReady ? 1 : 0.3)
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { drag in
                            if ignoreTouch { return }
                            if !pressed {
                                if model.locked {           // tap while locked = stop
                                    ignoreTouch = true
                                    DispatchQueue.main.async { model.release() }
                                    return
                                }
                                pressed = true
                                // Let the circle draw this frame; recording setup follows on the next main-queue turn.
                                DispatchQueue.main.async { model.press() }
                            }
                            dragUp = max(0, -drag.translation.height)
                            if dragUp > lockThreshold && !model.locked { model.lock() }
                        }
                        .onEnded { _ in
                            let wasIgnored = ignoreTouch
                            ignoreTouch = false
                            dragUp = 0
                            pressed = false
                            if wasIgnored || model.locked { return }
                            DispatchQueue.main.async { model.release() }
                        }
                )
        }
        .frame(height: 120)
        .padding(.bottom, 24)
        .animation(.easeOut(duration: 0.15), value: willLock)
        .accessibilityIdentifier("captureCircle")
    }

    private var isReady: Bool {
        switch model.phase { case .ready, .recording: true; default: false }
    }
}
