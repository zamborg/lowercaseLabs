import AVFoundation
import SwiftData
import SwiftUI

/// Tab 2: every message, newest first.
struct VoidView: View {
    let journal: Journal
    let capture: CaptureModel
    @Query(sort: \Entry.createdAt, order: .reverse) private var entries: [Entry]
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            List {
                ForEach(entries) { entry in
                    NavigationLink(value: entry.id) { EntryRow(entry: entry) }
                        .listRowBackground(Color.white.opacity(0.06))
                }
                .onDelete { offsets in offsets.map { entries[$0] }.forEach(journal.delete) }
            }
            .scrollContentBackground(.hidden)
            .background(Color.black)
            .overlay { if entries.isEmpty { Text("nothing here yet").foregroundStyle(.secondary) } }
            .navigationTitle("Void")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: { Image(systemName: "gearshape") }
                        .accessibilityIdentifier("settingsGear")
                }
            }
            .sheet(isPresented: $showSettings) { SettingsView(journal: journal) }
            .navigationDestination(for: UUID.self) { id in
                if let entry = entries.first(where: { $0.id == id }) { EntryCard(entry: entry, journal: journal, capture: capture) }
            }
        }
    }
}

struct EntryRow: View {
    let entry: Entry

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            if entry.hasPhoto { PhotoThumb(url: entry.photoBackURL, width: 56, height: 74) }
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 8) {
                    Text(entry.title.isEmpty ? (entry.needsEnrichment || entry.hasPhoto && entry.photoLabels.isEmpty ? "…" : "untitled") : entry.title)
                        .font(.headline)
                    Spacer()
                    ForEach(entry.tags, id: \.self) { Circle().fill($0.color).frame(width: 10, height: 10) }
                }
                if entry.hasVoice { Text(entry.transcript).font(.subheadline).foregroundStyle(.secondary).lineLimit(2) }
                HStack(spacing: 6) {
                    Text(entry.createdAt.formatted(date: .abbreviated, time: .shortened))
                    if entry.hasVoice {
                        Text("·")
                        Text(Duration.seconds(entry.duration).formatted(.time(pattern: .minuteSecond)))
                    }
                    if entry.isShared { Image(systemName: "person.2.fill") }
                }
                .font(.caption2).foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
        .accessibilityElement(children: .combine)
        .accessibilityIdentifier("entryRow")
    }
}

/// A message: playback, editable transcript, title, tags, visibility, delete.
struct EntryCard: View {
    @Bindable var entry: Entry
    let journal: Journal
    let capture: CaptureModel
    @State private var player = Player()
    @State private var confirmDelete = false
    @State private var title = ""
    @State private var transcript = ""
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 22) {
                TextField("title", text: $title, axis: .vertical)
                    .font(.title2.weight(.semibold))
                    .onSubmit { journal.setTitle(title.trimmingCharacters(in: .whitespacesAndNewlines), on: entry) }
                    .accessibilityIdentifier("titleField")

                tagPicker
                photos
                if entry.hasVoice {
                    playback

                    VStack(alignment: .leading, spacing: 6) {
                        Text("transcript").font(.caption).foregroundStyle(.secondary)
                        TextEditor(text: $transcript)
                            .scrollContentBackground(.hidden)
                            .frame(minHeight: 160)
                            .padding(8)
                            .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))
                            .accessibilityIdentifier("transcriptEditor")
                    }
                }

                Toggle(isOn: Binding(
                    get: { entry.isShared },
                    set: { journal.setVisibility($0 ? .shared : .private, on: entry) }
                )) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(entry.isShared ? "shared with friends" : "private")
                        Text("Friends only ever see your tag colors and the time.").font(.caption).foregroundStyle(.secondary)
                    }
                }

                Button("Delete message", role: .destructive) { confirmDelete = true }
                    .frame(maxWidth: .infinity)
            }
            .padding()
        }
        .scrollDismissesKeyboard(.interactively)
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog("Delete this message?", isPresented: $confirmDelete, titleVisibility: .visible) {
            Button("Delete", role: .destructive) { player.stop(); journal.delete(entry); dismiss() }
        } message: { Text("Its audio, photos and dot are removed. This can't be undone.") }
        .onAppear { title = entry.title; transcript = entry.transcript }
        .onChange(of: entry.title) { _, new in if new != title { title = new } }   // enrichment landing
        .onDisappear {
            player.stop()
            let t = title.trimmingCharacters(in: .whitespacesAndNewlines)
            if t != entry.title { journal.setTitle(t, on: entry) }
            if transcript != entry.transcript { journal.setTranscript(transcript, on: entry) }
        }
    }

    /// Both images, plainly (the view is a placeholder for the real UI), or a way to add one.
    @ViewBuilder private var photos: some View {
        if entry.hasPhoto {
            HStack(alignment: .top, spacing: 10) {
                PhotoThumb(url: entry.photoBackURL, width: 190, height: 250)
                if let front = entry.photoFrontURL { PhotoThumb(url: front, width: 110, height: 146) }
            }
            .accessibilityElement(children: .contain)
            .accessibilityIdentifier("entryPhotos")
        } else {
            Button { capture.openCamera(adding: entry) } label: {
                Label("add photo", systemImage: "camera.badge.plus")
            }
            .accessibilityIdentifier("addPhotoButton")
        }
    }

    private var tagPicker: some View {
        HStack(spacing: 12) {
            ForEach(Quadrant.allCases, id: \.self) { q in
                let on = entry.tags.contains(q)
                Button { toggle(q) } label: {
                    VStack(spacing: 6) {
                        Circle().fill(q.color.opacity(on ? 1 : 0.25)).frame(width: 34, height: 34)
                            .overlay { if on { Circle().stroke(.white, lineWidth: 2) } }
                        Text(q.shortLabel).font(.caption2).foregroundStyle(on ? .white : .secondary)
                    }
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("tag_\(q.rawValue)")
            }
        }
    }

    /// Tap to add (up to two, first is dominant) or remove (at least one stays).
    private func toggle(_ q: Quadrant) {
        var tags = entry.tags
        if let i = tags.firstIndex(of: q) {
            guard tags.count > 1 else { return }
            tags.remove(at: i)
        } else if tags.count >= 2 {
            tags[1] = q
        } else {
            tags.append(q)
        }
        journal.setTags(tags, on: entry)
    }

    @ViewBuilder private var playback: some View {
        if FileManager.default.fileExists(atPath: entry.audioURL.path) {
            HStack(spacing: 14) {
                Button { player.toggle(entry.audioURL) } label: {
                    Image(systemName: player.isPlaying ? "pause.fill" : "play.fill")
                        .font(.title3).frame(width: 44, height: 44)
                        .background(.white.opacity(0.12), in: Circle())
                }
                .buttonStyle(.plain)
                .accessibilityIdentifier("playButton")
                Slider(value: Binding(get: { player.progress }, set: { player.seek($0) }))
                    .tint(.white)
                Text(Duration.seconds(entry.duration).formatted(.time(pattern: .minuteSecond)))
                    .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
        } else {
            Text("Audio isn't on this device.").font(.footnote).foregroundStyle(.secondary)
        }
    }
}

extension Quadrant {
    var shortLabel: String {
        switch self {
        case .highPleasant: "energized"
        case .lowPleasant: "calm"
        case .highUnpleasant: "tense"
        case .lowUnpleasant: "low"
        }
    }
}

@MainActor @Observable
final class Player: NSObject, AVAudioPlayerDelegate {
    var isPlaying = false
    var progress = 0.0
    private var audio: AVAudioPlayer?
    private var timer: Timer?

    func toggle(_ url: URL) {
        if audio == nil { audio = try? AVAudioPlayer(contentsOf: url); audio?.delegate = self }
        guard let audio else { return }
        if audio.isPlaying {
            audio.pause(); isPlaying = false; timer?.invalidate()
        } else {
            audio.play(); isPlaying = true
            timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
                MainActor.assumeIsolated { self?.tick() }
            }
        }
    }

    func seek(_ fraction: Double) {
        progress = fraction
        if let audio { audio.currentTime = fraction * audio.duration }
    }

    func stop() { audio?.stop(); timer?.invalidate(); isPlaying = false }

    private func tick() {
        guard let audio, audio.duration > 0 else { return }
        progress = audio.currentTime / audio.duration
    }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        Task { @MainActor in isPlaying = false; progress = 0; timer?.invalidate(); audio?.currentTime = 0 }
    }
}
