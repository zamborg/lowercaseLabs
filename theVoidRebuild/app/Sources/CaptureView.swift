import SwiftData
import SwiftUI
import ImageIO
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

    // Photo mode (D15, D16). One camera, owned here so the Capture page and the Void card's "add photo" share it.
    enum Page { case voice, camera }
    var page: Page = .voice
    let camera = Camera()
    /// Set while the camera is open to add a photo to an existing message instead of starting a new one.
    var addPhotoTarget: Entry?
    /// The photo message being made right now: its photo is saved, its voice (if you keep holding) is still coming.
    private var cameraEntry: Entry?
    private var snapTask: Task<Void, Never>?
    var cameraShowing: Bool { page == .camera || addPhotoTarget != nil }

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
        if ProcessInfo.processInfo.arguments.contains("-cameratest") { await cameraTest() }
    }

    // MARK: Camera

    /// Touch on the circle while the camera is showing. The mic starts first (nothing slows the press); the photo
    /// snaps at the same moment and is saved on its own, so the photo exists even if you let go at once.
    func pressInCamera() {
        if let target = addPhotoTarget {      // adding to a message: a snap and nothing else
            snapTask = Task { await snap(attachingTo: target) }
            return
        }
        press()
        snapTask = Task { await snap(attachingTo: nil) }
    }

    /// Let go on the camera. Tap: a photo-only message. Held: the voice joins that same message.
    func releaseInCamera() {
        if addPhotoTarget != nil { return }
        if case .recording = phase { release(); return }
        Task {   // the mic wasn't recording (still warming, or unavailable): the photo stands alone
            await snapTask?.value; snapTask = nil
            if let entry = cameraEntry { cameraEntry = nil; lastEntry = entry; await enrich(entry) }
        }
    }

    /// The ✕ after a snap: the only way a photo is thrown away, and it is explicit.
    func discardLast() {
        if let entry = lastEntry { journal.delete(entry) }
        lastEntry = nil
    }

    func openCamera(adding entry: Entry? = nil) {
        addPhotoTarget = entry
        if entry == nil { page = .camera }
    }

    private func snap(attachingTo target: Entry?) async {
        guard let shot = await camera.snap() else { note = "camera not ready"; return }
        do {
            if let target {
                try journal.attachPhoto(back: shot.back, front: shot.front, to: target)
                lastEntry = target
                addPhotoTarget = nil; page = .voice; camera.stop()
                await enrich(target)
            } else {
                cameraEntry = try journal.savePhoto(back: shot.back, front: shot.front, visibility: visibility)
            }
        } catch {
            note = "couldn't save the photo: \(error)"
        }
    }

    /// `-cameratest`: on the phone, with the real cameras. Arrive-to-live, snap-to-both-images, save, enrich, and then
    /// the regression that matters: voice capture still starts within a frame after the camera has run.
    private func cameraTest() async {
        func line(_ s: String) { print("CAMERA \(s)"); fflush(stdout) }
        page = .camera
        await camera.start()
        line("state=\(camera.state) front=\(camera.hasFront) arrive_ms=\(Int(camera.arriveMs ?? -1))")
        guard camera.state == .running else { return }
        try? await Task.sleep(for: .seconds(1))
        for i in 1...3 {
            var t = CFAbsoluteTimeGetCurrent()
            guard let shot = await camera.snap() else { line("run=\(i) FAILED snap returned nothing"); continue }
            let snapMs = (CFAbsoluteTimeGetCurrent() - t) * 1000
            t = CFAbsoluteTimeGetCurrent()
            guard let entry = try? journal.savePhoto(back: shot.back, front: shot.front, visibility: .private) else { line("run=\(i) FAILED save"); continue }
            let saveMs = (CFAbsoluteTimeGetCurrent() - t) * 1000
            t = CFAbsoluteTimeGetCurrent()
            try? await journal.enrich(entry)
            line(String(format: "run=%d snap_ms=%.0f both_images=%@ gap_ms=%.0f save_ms=%.0f enrich_ms=%.0f bytes=%d/%d labels=\"%@\" title=\"%@\"",
                        i, snapMs, "\(shot.front != nil)", camera.snapGapMs ?? -1, saveMs, (CFAbsoluteTimeGetCurrent() - t) * 1000,
                        shot.back.count, shot.front?.count ?? 0, entry.photoLabels, entry.title))
            journal.delete(entry)
        }
        camera.stop(); page = .voice
        try? await Task.sleep(for: .seconds(1))
        press()
        if case .recording = phase {
            try? await Task.sleep(for: .seconds(1))
            phase = .finishing
            _ = await finishRecording()
            line(String(format: "voice after camera: press_to_capturing_ms=%.1f", last?.timings.pressToCapturingMs ?? -1))
        } else { line("voice after camera FAILED: \(phase)") }
        line("done")
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
        print("SCENE \(phase) at \(Date().formatted(date: .omitted, time: .standard)) recording=\(isRecordingNow)"); fflush(stdout)
        // The camera only runs while its page is showing and the app is active.
        if phase == .background { camera.stop() }
        if phase == .active, cameraShowing { Task { await camera.start() } }
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

    private var isRecordingNow: Bool { if case .recording = phase { true } else { false } }

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
        if let snap = snapTask { await snap.value; snapTask = nil }   // after stopping the mic, so the photo never delays it
        if let photoEntry = cameraEntry {
            // Held on the camera: the voice joins the photo message. A tap leaves a photo-only message.
            cameraEntry = nil
            if let capture {
                if capture.duration >= 0.6 || !capture.transcript.isEmpty {
                    liveText = capture.transcript
                    journal.attachVoice(audioFileName: capture.audioURL.lastPathComponent, duration: capture.duration,
                                        transcript: capture.transcript, to: photoEntry)
                } else {
                    try? FileManager.default.removeItem(at: capture.audioURL)
                }
            }
            lastEntry = photoEntry
            return photoEntry
        }
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
        if ProcessInfo.processInfo.arguments.contains("-camerapage") { page = .camera }
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

/// Capture: two pages side by side. Voice (the circle records) and the camera (the same circle is the shutter).
/// Swipe left for the camera and right to come back; the glyph does the same for a tap.
struct CaptureView: View {
    let model: CaptureModel

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            switch model.page {
            case .voice: VoicePage(model: model)
            case .camera: CameraPage(model: model, attaching: false)
            }
        }
        // The circle's own gesture wins on the circle; anywhere else, a sideways swipe changes page.
        .gesture(DragGesture(minimumDistance: 30).onEnded { drag in
            guard abs(drag.translation.height) < 80 else { return }
            if drag.translation.width < -60 { model.page = .camera }
            else if drag.translation.width > 60 { model.page = .voice }
        })
        .task { await model.warmUp() }
    }
}

struct VoicePage: View {
    let model: CaptureModel

    var body: some View {
        VStack(spacing: 18) {
            HStack {
                Spacer()
                Button { model.page = .camera } label: {
                    Image(systemName: "camera").font(.title3).foregroundStyle(.white.opacity(0.7))
                        .frame(width: 44, height: 44)
                }
                .accessibilityIdentifier("cameraGlyph")
            }
            .padding(.horizontal, 8)

            Text(model.liveText)
                .font(.system(.title3, design: .rounded))
                .foregroundStyle(.white.opacity(0.85))
                .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                .padding(.horizontal)

            LastMessage(model: model, offerPhoto: true)
            StatusLine(model: model)
            VisibilityToggle(model: model)
            CircleControl(model: model, inCamera: false)
        }
    }
}

/// The message just made: title and tag dots, plus (on the voice page) a way to add a photo to it.
struct LastMessage: View {
    let model: CaptureModel
    let offerPhoto: Bool

    var body: some View {
        if let entry = model.lastEntry {
            VStack(spacing: 8) {
                if !entry.title.isEmpty {
                    HStack(spacing: 8) {
                        ForEach(entry.tags, id: \.self) { Circle().fill($0.color).frame(width: 14, height: 14) }
                        Text(entry.title).font(.headline).foregroundStyle(.white)
                    }
                    .transition(.opacity)
                }
                if offerPhoto, !entry.hasPhoto {
                    Button { model.openCamera(adding: entry) } label: {
                        Label("add photo", systemImage: "camera.badge.plus").font(.footnote.weight(.medium))
                            .foregroundStyle(.white.opacity(0.7))
                    }
                    .accessibilityIdentifier("addPhotoButton")
                }
            }
            .animation(.easeOut(duration: 0.2), value: entry.title)
        }
    }
}

struct StatusLine: View {
    let model: CaptureModel

    var body: some View {
        VStack(spacing: 2) {
            if Settings.showTimings {
                if let c = model.last {
                    Text(String(format: "press→capturing %.0f ms · release→final %.0f ms%@",
                                c.timings.pressToCapturingMs, c.timings.releaseToFinalMs,
                                model.taggingMs.map { String(format: " · tagged %.0f ms", $0) } ?? ""))
                }
                if let ms = model.camera.arriveMs, model.page == .camera || model.addPhotoTarget != nil {
                    Text(String(format: "camera live %.0f ms%@", ms, model.camera.snapGapMs.map { String(format: " · front/back gap %.0f ms", $0) } ?? ""))
                }
            }
            if !model.note.isEmpty { Text(model.note).foregroundStyle(.orange) }
            if case .failed(let message) = model.phase { Text(message).foregroundStyle(.red) }
        }
        .font(.caption2.monospaced())
        .foregroundStyle(.white.opacity(0.4))
        .multilineTextAlignment(.center)
        .padding(.horizontal)
    }
}

struct VisibilityToggle: View {
    let model: CaptureModel

    var body: some View {
        let shared = model.visibility == .shared
        let recording: Bool = { if case .recording = model.phase { true } else { false } }()
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
        .opacity(recording ? 0.4 : 1)
        .accessibilityIdentifier("visibilityToggle")
    }
}

/// The camera page: both cameras live, and the circle is the shutter. Tap = photo only. Hold = the photo snaps on
/// touch and your voice joins the same message. `attaching` is the "add photo" case: a snap joins an existing message.
struct CameraPage: View {
    let model: CaptureModel
    let attaching: Bool

    var body: some View {
        ZStack {
            CameraPreview(camera: model.camera).ignoresSafeArea()
            if case .unavailable(let why) = model.camera.state {
                Text(why).font(.footnote).foregroundStyle(.white.opacity(0.7)).multilineTextAlignment(.center).padding(40)
            }
            VStack(spacing: 14) {
                HStack {
                    if attaching {
                        Button("Cancel") { model.addPhotoTarget = nil; model.camera.stop() }
                            .foregroundStyle(.white).padding(14)
                            .accessibilityIdentifier("cancelCamera")
                    }
                    Spacer()
                }
                Spacer()
                if !attaching { SnappedStrip(model: model) }
                StatusLine(model: model)
                if !attaching { VisibilityToggle(model: model) }
                CircleControl(model: model, inCamera: true)
            }
        }
        .onAppear { Task { await model.camera.start() } }
        .onDisappear { model.camera.stop() }
    }
}

/// After a snap: the two images stay on screen with the message's title, and an ✕ that discards the message.
struct SnappedStrip: View {
    let model: CaptureModel

    var body: some View {
        if let entry = model.lastEntry, entry.hasPhoto {
            HStack(spacing: 10) {
                PhotoThumb(url: entry.photoBackURL, width: 54, height: 72).accessibilityIdentifier("thumbBack")
                if let front = entry.photoFrontURL { PhotoThumb(url: front, width: 54, height: 72).accessibilityIdentifier("thumbFront") }
                if !entry.title.isEmpty { Text(entry.title).font(.subheadline.weight(.medium)).foregroundStyle(.white) }
                ForEach(entry.tags, id: \.self) { Circle().fill($0.color).frame(width: 12, height: 12) }
                Button { model.discardLast() } label: {
                    Image(systemName: "xmark").font(.footnote.weight(.bold)).foregroundStyle(.white)
                        .frame(width: 32, height: 32).background(.white.opacity(0.2), in: Circle())
                }
                .accessibilityIdentifier("discardPhoto")
            }
            .padding(10)
            .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 16))
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("snappedStrip")
        }
    }
}

/// The circle: one control for both pages. Press, hold, slide up to lock, tap to stop a locked recording.
struct CircleControl: View {
    let model: CaptureModel
    let inCamera: Bool
    @State private var pressed = false
    @State private var ignoreTouch = false
    @State private var dragUp: CGFloat = 0

    private let lockThreshold: CGFloat = 90
    private var isRecording: Bool { if case .recording = model.phase { true } else { false } }
    private var isReady: Bool {
        if inCamera { return model.camera.state == .running }   // a snap needs a live camera; the mic is not required
        switch model.phase { case .ready, .recording: return true; default: return false }
    }

    private func press() { inCamera ? model.pressInCamera() : model.press() }
    private func release() { inCamera ? model.releaseInCamera() : model.release() }

    var body: some View {
        let willLock = dragUp > lockThreshold
        return ZStack {
            // Lock affordance: appears above the circle while holding, fills when the slide is far enough.
            if pressed && !model.locked && model.addPhotoTarget == nil {
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
                                    DispatchQueue.main.async { release() }
                                    return
                                }
                                pressed = true
                                // Let the circle draw this frame; recording setup follows on the next main-queue turn.
                                DispatchQueue.main.async { press() }
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
                            DispatchQueue.main.async { release() }
                        }
                )
        }
        .frame(height: 120)
        .padding(.bottom, 24)
        .animation(.easeOut(duration: 0.15), value: willLock)
        .accessibilityIdentifier(inCamera ? "shutterCircle" : "captureCircle")
    }
}

/// A photo from the app's own folder, decoded small. Never loads the full file into a list.
struct PhotoThumb: View {
    let url: URL
    var width: CGFloat
    var height: CGFloat
    @State private var image: UIImage?

    var body: some View {
        Group {
            if let image { Image(uiImage: image).resizable().scaledToFill() } else { Color.white.opacity(0.08) }
        }
        .frame(width: width, height: height)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .task(id: url) {
            let side = max(width, height) * 3
            image = await Task.detached { () -> UIImage? in
                guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
                      let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                          kCGImageSourceCreateThumbnailFromImageAlways: true,
                          kCGImageSourceThumbnailMaxPixelSize: side,
                      ] as CFDictionary) else { return nil }
                return UIImage(cgImage: cg)
            }.value
        }
    }
}
