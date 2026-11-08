import SwiftData
import SwiftUI

@MainActor
@Observable
final class CaptureModel {
    enum Phase { case warming, ready, recording, finishing, failed(String) }
    var phase: Phase = .warming
    var liveText = ""
    var last: Capture?
    var lastTitle = ""
    var lastTags: [Quadrant] = []
    var taggingMs: Double?
    var note = ""

    private let recorder = Recorder()
    private let tagger = Tagger()
    var context: ModelContext?

    init() {
        recorder.onText = { [weak self] text in
            Task { @MainActor in self?.liveText = text }
        }
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
            try await tagger.warm()  // model load/download happens in the background
        } catch {
            note = "tagger unavailable: \(error)"
        }
        if ProcessInfo.processInfo.arguments.contains("-autotest") { await autotest() }
    }

    /// Background/foreground: the mic only runs while the app is visible.
    func setActive(_ active: Bool) {
        guard recorder.isWarm else { return }
        if active { try? recorder.startListening() } else { recorder.stopListening() }
    }

    func press() {
        guard case .ready = phase else { return }
        liveText = ""
        lastTitle = ""
        lastTags = []
        taggingMs = nil
        do {
            try recorder.start()
            phase = .recording
        } catch {
            phase = .failed("\(error)")
        }
    }

    func release() {
        guard case .recording = phase else { return }
        phase = .finishing
        Task { await finishRecording() }
    }

    private func finishRecording() async {
        let capture = await recorder.stop()
        last = capture
        phase = .ready
        guard let capture else { return }
        liveText = capture.transcript
        let transcript = capture.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else { return }

        // Save immediately so nothing is ever lost, then enrich with title + tags.
        let memo = Memo(
            id: capture.id, createdAt: Date(), duration: capture.duration,
            audioFileName: capture.audioURL.lastPathComponent, transcript: transcript
        )
        context?.insert(memo)
        try? context?.save()

        guard tagger.isWarm else { return }
        let started = CFAbsoluteTimeGetCurrent()
        do {
            let tagging = try await tagger.tag(transcript: transcript)
            memo.title = tagging.title
            memo.tagValues = tagging.tags.map(\.rawValue)
            try? context?.save()
            lastTitle = tagging.title
            lastTags = tagging.tags
            do {
                try await DotStore.shared.publish(memoID: memo.id, tags: tagging.tags, createdAt: memo.createdAt)
            } catch {
                note = "dot not published: \(error.localizedDescription)"
            }
            taggingMs = (CFAbsoluteTimeGetCurrent() - started) * 1000
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
            await finishRecording()
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

    /// `-seed` launch argument: sample memos for exercising the list UI where mic/speech don't exist.
    func seedIfRequested() {
        guard ProcessInfo.processInfo.arguments.contains("-seed"), let context else { return }
        let samples: [(String, String, Quadrant)] = [
            ("Big launch day", "I can't believe we shipped it, I'm buzzing, so much energy today.", .highPleasant),
            ("Quiet evening", "Just sitting on the porch, tea in hand, feeling really calm and content.", .lowPleasant),
            ("Deadline dread", "Everything is due tomorrow and my chest is tight, I can't stop spiraling.", .highUnpleasant),
            ("Running on empty", "I'm so tired. Nothing feels worth doing and I just want to lie down.", .lowUnpleasant),
        ]
        for (i, s) in samples.enumerated() {
            context.insert(Memo(id: UUID(), createdAt: Date().addingTimeInterval(Double(-i) * 3600), duration: 8,
                                audioFileName: "none.caf", transcript: s.1, title: s.0, tagValues: [s.2.rawValue]))
        }
        try? context.save()
    }
}

struct CaptureView: View {
    @State private var model = CaptureModel()
    @State private var pressed = false
    @State private var showMemos = false
    @State private var showFriends = false
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.modelContext) private var modelContext

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            VStack(spacing: 20) {
                HStack {
                    Button { showFriends = true } label: {
                        Image(systemName: "person.2").font(.title3).foregroundStyle(.white.opacity(0.5))
                    }
                    Spacer()
                    Button { showMemos = true } label: {
                        Image(systemName: "list.bullet").font(.title3).foregroundStyle(.white.opacity(0.5))
                    }
                }
                .padding(.horizontal)

                Text(model.liveText)
                    .font(.system(.title3, design: .rounded))
                    .foregroundStyle(.white.opacity(0.85))
                    .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                    .padding(.horizontal)

                if !model.lastTitle.isEmpty {
                    HStack(spacing: 8) {
                        ForEach(model.lastTags, id: \.self) { Circle().fill($0.color).frame(width: 14, height: 14) }
                        Text(model.lastTitle).font(.headline).foregroundStyle(.white)
                    }
                }
                VStack(spacing: 2) {
                    if let c = model.last {
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

                Circle()
                    .fill(.white.opacity(pressed ? 1 : 0.85))
                    .frame(width: pressed ? 120 : 96, height: pressed ? 120 : 96)
                    .animation(.spring(duration: 0.18, bounce: 0.3), value: pressed)
                    .opacity(isReady ? 1 : 0.3)
                    .gesture(
                        DragGesture(minimumDistance: 0)
                            .onChanged { _ in
                                guard !pressed else { return }
                                pressed = true
                                // Let the circle draw this frame; recording setup follows on the next main-queue turn.
                                DispatchQueue.main.async { model.press() }
                            }
                            .onEnded { _ in
                                pressed = false
                                DispatchQueue.main.async { model.release() }
                            }
                    )
                    .padding(.bottom, 48)
            }
        }
        .sheet(isPresented: $showMemos) { MemoListView() }
        .sheet(isPresented: $showFriends) { FriendsView() }
        .task {
            model.context = modelContext
            model.seedIfRequested()
            if ProcessInfo.processInfo.arguments.contains("-showmemos") { showMemos = true }
            await model.warmUp()
        }
        .onChange(of: scenePhase) { _, new in model.setActive(new == .active) }
    }

    private var isReady: Bool {
        switch model.phase { case .ready, .recording: true; default: false }
    }
}
