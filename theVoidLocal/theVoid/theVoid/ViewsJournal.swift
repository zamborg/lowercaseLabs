import SwiftUI

// MARK: - Journal

struct EmotionAtlasScreen: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    EmotionAtlasView(entries: model.entries, height: 500)
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Recent Tags")
                            .font(.headline)
                        let recent = recentTags(from: model.entries)
                        if recent.isEmpty {
                            Text("Record a reflection to populate your map.")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        } else {
                            WrapTags(tags: recent)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Emotion Atlas")
            .refreshable {
                await model.refreshEntries()
            }
        }
    }

    private func recentTags(from entries: [APIEntry]) -> [String] {
        let sevenDaysAgo = Calendar.current.date(byAdding: .day, value: -7, to: Date()) ?? Date()
        let filtered = entries.filter { entry in
            guard let date = DateFormatter.localDate.date(from: entry.localDate) else {
                return false
            }
            return date >= sevenDaysAgo
        }
        let tags = filtered.flatMap { $0.insight?.moodTags ?? [] }
        var ordered: [String] = []
        for tag in tags {
            let canonical = EmotionTaxonomy.canonicalTag(for: tag)
            if !ordered.contains(canonical) {
                ordered.append(canonical)
            }
            if ordered.count >= 24 {
                break
            }
        }
        return ordered
    }
}

struct JournalView: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    Text("Dot Stream")
                        .font(.headline)
                    MoodHeatmap(entries: model.entries)

                    Text("Timeline")
                        .font(.headline)

                    LazyVStack(spacing: 12) {
                        ForEach(model.entries) { entry in
                            NavigationLink(value: entry) {
                                EntryCard(entry: entry)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                }
                .padding()
            }
            .navigationTitle("Journal")
            .navigationDestination(for: APIEntry.self) { entry in
                EntryDetailView(entry: entry)
            }
            .refreshable {
                await model.refreshEntries()
            }
        }
    }
}

private enum EmotionAtlasMode: String, CaseIterable, Identifiable {
    case recent
    case all

    var id: String { rawValue }

    var title: String {
        switch self {
        case .recent:
            return "Your Tags"
        case .all:
            return "All Tags"
        }
    }
}

struct EmotionAtlasView: View {
    let entries: [APIEntry]
    var height: CGFloat = 440

    @State private var mode: EmotionAtlasMode = .recent

    private var tagCounts: [String: Int] {
        var counts: [String: Int] = [:]
        for entry in entries {
            guard let tags = entry.insight?.moodTags else { continue }
            for raw in tags {
                let canonical = EmotionTaxonomy.canonicalTag(for: raw)
                counts[canonical, default: 0] += 1
            }
        }
        return counts
    }

    private var hasRecentTags: Bool {
        !tagCounts.isEmpty
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Pleasantness × Energy")
                        .font(.headline)
                    Text("Top right is high-energy pleasant. Bottom left is low-energy unpleasant.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            if hasRecentTags {
                Picker("Scope", selection: $mode) {
                    ForEach(EmotionAtlasMode.allCases) { option in
                        Text(option.title).tag(option)
                    }
                }
                .pickerStyle(.segmented)
            }

            EmotionAtlasPlane(
                allTags: EmotionTaxonomy.tags,
                tagCounts: tagCounts,
                mode: hasRecentTags ? mode : .all
            )
            .frame(height: height)
            .clipped()

            Text("Inspired by How We Feel's emotional axes. Colors come from each tag's position on the map.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }
}

private struct EmotionAtlasPlane: View {
    let allTags: [EmotionTagDefinition]
    let tagCounts: [String: Int]
    let mode: EmotionAtlasMode

    private var visibleTags: [EmotionTagDefinition] {
        switch mode {
        case .all:
            return allTags
        case .recent:
            let filtered = allTags.filter { tagCounts[$0.id, default: 0] > 0 }
            return filtered.isEmpty ? allTags : filtered
        }
    }

    var body: some View {
        GeometryReader { geometry in
            let size = geometry.size
            let plotInset: CGFloat = 28
            let plotRect = CGRect(
                x: plotInset,
                y: plotInset,
                width: max(1, size.width - (plotInset * 2)),
                height: max(1, size.height - (plotInset * 2))
            )

            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(
                        LinearGradient(
                            colors: [
                                Color(red: 0.72, green: 0.20, blue: 0.27),
                                Color(red: 0.95, green: 0.77, blue: 0.35),
                                Color(red: 0.43, green: 0.73, blue: 0.58),
                                Color(red: 0.21, green: 0.45, blue: 0.61),
                            ],
                            startPoint: .topLeading,
                            endPoint: .bottomTrailing
                        )
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .fill(Color.black.opacity(0.28))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 24)
                            .stroke(Color.white.opacity(0.18), lineWidth: 1)
                    )

                gridLines(in: plotRect)
                    .stroke(Color.white.opacity(0.14), lineWidth: 0.8)

                centerAxes(in: plotRect)
                    .stroke(Color.white.opacity(0.28), lineWidth: 1.2)

                ForEach(visibleTags) { tag in
                    let count = tagCounts[tag.id, default: 0]
                    let isRecent = count > 0
                    let point = position(for: tag, in: plotRect)
                    EmotionTagNode(
                        tag: tag,
                        count: count,
                        emphasized: isRecent,
                        isDimmed: mode == .all && !isRecent
                    )
                    .position(point)
                }
            }
            .overlay(alignment: .top) {
                Text("High Energy")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .padding(.top, 6)
            }
            .overlay(alignment: .bottom) {
                Text("Low Energy")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .padding(.bottom, 6)
            }
            .overlay(alignment: .leading) {
                Text("Unpleasant")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .rotationEffect(.degrees(-90))
                    .padding(.leading, -8)
            }
            .overlay(alignment: .trailing) {
                Text("Pleasant")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.white.opacity(0.88))
                    .rotationEffect(.degrees(90))
                    .padding(.trailing, -8)
            }
        }
    }

    private func gridLines(in rect: CGRect) -> Path {
        var path = Path()
        let steps = 4
        for step in 1..<steps {
            let ratio = CGFloat(step) / CGFloat(steps)
            let x = rect.minX + (rect.width * ratio)
            let y = rect.minY + (rect.height * ratio)
            path.move(to: CGPoint(x: x, y: rect.minY))
            path.addLine(to: CGPoint(x: x, y: rect.maxY))
            path.move(to: CGPoint(x: rect.minX, y: y))
            path.addLine(to: CGPoint(x: rect.maxX, y: y))
        }
        return path
    }

    private func centerAxes(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.midX, y: rect.minY))
        path.addLine(to: CGPoint(x: rect.midX, y: rect.maxY))
        path.move(to: CGPoint(x: rect.minX, y: rect.midY))
        path.addLine(to: CGPoint(x: rect.maxX, y: rect.midY))
        return path
    }

    private func position(for tag: EmotionTagDefinition, in rect: CGRect) -> CGPoint {
        let normalizedX = CGFloat((tag.pleasantness + 1.0) / 2.0)
        let normalizedY = CGFloat(1.0 - ((tag.energy + 1.0) / 2.0))
        let baseX = rect.minX + (rect.width * normalizedX)
        let baseY = rect.minY + (rect.height * normalizedY)
        let jitter = jitterOffset(for: tag.id)
        return CGPoint(
            x: min(rect.maxX, max(rect.minX, baseX + jitter.width)),
            y: min(rect.maxY, max(rect.minY, baseY + jitter.height))
        )
    }

    private func jitterOffset(for text: String) -> CGSize {
        var hash: UInt64 = 1469598103934665603
        for byte in text.utf8 {
            hash ^= UInt64(byte)
            hash &*= 1099511628211
        }
        let xRaw = Double(hash % 10_000) / 10_000.0
        let yRaw = Double((hash / 10_000) % 10_000) / 10_000.0
        let maxJitter: CGFloat = 10
        return CGSize(
            width: CGFloat((xRaw * 2.0) - 1.0) * maxJitter,
            height: CGFloat((yRaw * 2.0) - 1.0) * maxJitter
        )
    }
}

private struct EmotionTagNode: View {
    let tag: EmotionTagDefinition
    let count: Int
    let emphasized: Bool
    let isDimmed: Bool

    var body: some View {
        let color = EmotionPalette.color(forTag: tag.id)
        let label = EmotionTaxonomy.displayName(for: tag.id)
        let annotated = count > 1 ? "\(label) ×\(count)" : label

        Text(annotated)
            .font(.system(size: emphasized ? 10.5 : 9.5, weight: emphasized ? .bold : .medium, design: .rounded))
            .foregroundStyle(Color.white.opacity(isDimmed ? 0.7 : 0.98))
            .lineLimit(1)
            .padding(.horizontal, 7)
            .padding(.vertical, 4)
            .background(
                Capsule()
                    .fill(color.opacity(isDimmed ? 0.24 : 0.82))
            )
            .overlay(
                Capsule()
                    .stroke(color.opacity(isDimmed ? 0.35 : 0.95), lineWidth: 0.8)
            )
            .scaleEffect(emphasized ? 1.05 : 1.0)
            .shadow(color: color.opacity(0.22), radius: emphasized ? 6 : 2, x: 0, y: 1)
            .opacity(isDimmed ? 0.45 : 1.0)
    }
}

struct MoodHeatmap: View {
    let entries: [APIEntry]

    private let columns = [GridItem(.adaptive(minimum: 18), spacing: 8)]

    var body: some View {
        let ordered = entries.sorted { lhs, rhs in
            if lhs.localDate != rhs.localDate {
                return lhs.localDate > rhs.localDate
            }
            return lhs.createdAt > rhs.createdAt
        }

        LazyVGrid(columns: columns, spacing: 5) {
            ForEach(ordered) { entry in
                EmotionMixedCircle(
                    tags: entry.insight?.moodTags ?? [],
                    diameter: 16,
                    borderOpacity: 0.24
                )
            }
        }
        .padding(10)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 12))
    }
}

struct EntryCard: View {
    let entry: APIEntry

    private var statusLabel: String {
        entry.status
            .replacingOccurrences(of: "_", with: " ")
            .capitalized
    }

    var body: some View {
        let tags = entry.insight?.moodTags ?? []
        let timeLabel = entry.createdAtTimeLabel

        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                EmotionMixedCircle(
                    tags: tags,
                    diameter: 18,
                    borderOpacity: 0.28
                )
                VStack(alignment: .leading, spacing: 4) {
                    Text(entry.displayTitle)
                        .font(.headline)
                        .lineLimit(1)

                    HStack(spacing: 6) {
                        Text(entry.localDate)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        if let timeLabel {
                            EntryTimeBadge(timeLabel: timeLabel)
                        }
                    }
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 6) {
                    Text(statusLabel)
                        .font(.caption)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 4)
                        .background(Color.secondary.opacity(0.15), in: Capsule())
                    if let healthSnapshot = entry.healthSnapshot {
                        HealthScorePill(snapshot: healthSnapshot)
                    }
                }
            }

            if let tags = entry.insight?.moodTags, !tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(tags, id: \.self) { tag in
                            TagChip(tag: tag)
                        }
                    }
                }
            }

            if let transcript = entry.transcript?.text, !transcript.isEmpty {
                Text(transcript)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                    .lineLimit(3)
            }
        }
        .padding(14)
        .background(Color.secondary.opacity(0.08), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct EntryTimeBadge: View {
    let timeLabel: String

    var body: some View {
        Text(timeLabel)
            .font(.caption2.weight(.semibold))
            .foregroundStyle(.secondary)
            .padding(.horizontal, 7)
            .padding(.vertical, 3)
            .background(Color.secondary.opacity(0.14), in: Capsule())
    }
}

struct EntryDetailView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.dismiss) private var dismiss
    let entry: APIEntry

    @State private var currentEntry: APIEntry
    @StateObject private var audioPlayback = AudioPlaybackController()
    @State private var showDeleteConfirmation = false
    @State private var showTagEditor = false
    @State private var showTitleEditor = false
    @State private var isDeletingEntry = false
    @State private var isSavingTags = false
    @State private var isSavingTitle = false
    @State private var isRetranscribing = false

    init(entry: APIEntry) {
        self.entry = entry
        _currentEntry = State(initialValue: entry)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    Text(currentEntry.displayTitle)
                        .font(.title2.bold())
                        .lineLimit(2)
                        .minimumScaleFactor(0.85)

                    HStack(spacing: 8) {
                        Text(currentEntry.localDate)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                        if let timeLabel = currentEntry.createdAtTimeLabel {
                            EntryTimeBadge(timeLabel: timeLabel)
                        }
                    }
                }

                if let insight = currentEntry.insight {
                    VStack(alignment: .leading, spacing: 10) {
                        EmotionMixedCircle(
                            tags: insight.moodTags,
                            diameter: 186,
                            borderOpacity: 0.34
                        )
                        .frame(maxWidth: .infinity, alignment: .center)
                        .padding(.vertical, 4)

                        HStack(spacing: 8) {
                            Text("Tags")
                                .font(.headline)
                            Spacer()
                            Button("Edit Tags") {
                                showTagEditor = true
                            }
                            .buttonStyle(.bordered)
                            .disabled(isSavingTags || isDeletingEntry)
                        }
                        WrapTags(tags: insight.moodTags)

                        if isSavingTags {
                            ProgressView("Saving tags...")
                                .font(.footnote)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                if let transcript = currentEntry.transcript {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Transcript")
                            .font(.headline)
                        Text(transcript.text)
                            .font(.body)
                    }
                } else {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Transcript")
                            .font(.headline)
                        Text("No transcript yet. Use Retranscribe to process this audio note.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                EntryHealthSnapshotSection(
                    snapshot: currentEntry.healthSnapshot,
                    capturedAtLabel: currentEntry.healthSnapshot.flatMap { healthSnapshot in
                        formattedHealthTimestamp(healthSnapshot.capturedAtISO8601)
                    }
                )

                VStack(alignment: .leading, spacing: 10) {
                    Text("Audio")
                        .font(.headline)

                    HStack(spacing: 10) {
                        Button(audioPlayback.isPlaying ? "Pause" : "Play") {
                            Task {
                                await togglePlayback()
                            }
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(audioPlayback.isLoading)
                    }

                    Button(isRetranscribing ? "Retranscribing..." : "Retranscribe") {
                        Task {
                            await retranscribeEntry()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(isRetranscribing || isDeletingEntry)

                    if audioPlayback.isLoading {
                        ProgressView("Loading audio...")
                    }

                    Slider(
                        value: Binding(
                            get: { audioPlayback.currentTime },
                            set: { audioPlayback.scrub(to: $0) }
                        ),
                        in: 0...max(audioPlayback.duration, 1),
                        onEditingChanged: { editing in
                            if editing {
                                audioPlayback.beginScrubbing()
                            } else {
                                audioPlayback.endScrubbing()
                            }
                        }
                    )
                    .disabled(!audioPlayback.isReady)

                    HStack {
                        Text(formatDuration(audioPlayback.currentTime))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                        Spacer()
                        Text(formatDuration(audioPlayback.duration))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .padding()
        }
        .navigationTitle(currentEntry.displayTitle)
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button {
                    showTitleEditor = true
                } label: {
                    Image(systemName: "pencil")
                }
                .disabled(isDeletingEntry || isSavingTitle)

                Button(role: .destructive) {
                    showDeleteConfirmation = true
                } label: {
                    Image(systemName: "trash")
                }
                .disabled(isDeletingEntry)
            }
        }
        .alert("Delete reflection?", isPresented: $showDeleteConfirmation) {
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                Task { await deleteEntry() }
            }
            .disabled(isDeletingEntry)
        } message: {
            Text("This removes the local entry, transcript, audio, and social dot for this day.")
        }
        .sheet(isPresented: $showTagEditor) {
            if let insight = currentEntry.insight {
                EntryTagEditorSheet(
                    initialTags: insight.moodTags,
                    isSaving: isSavingTags,
                    onSave: { tags in
                        Task {
                            await saveTags(tags)
                        }
                    }
                )
            } else {
                VStack(spacing: 12) {
                    Text("No tags available for this entry.")
                        .font(.headline)
                    Button("Close") {
                        showTagEditor = false
                    }
                    .buttonStyle(.bordered)
                }
                .padding(24)
            }
        }
        .sheet(isPresented: $showTitleEditor) {
            EntryTitleEditorSheet(
                initialTitle: currentEntry.displayTitle,
                isSaving: isSavingTitle,
                onSave: { newTitle in
                    Task {
                        await saveTitle(newTitle)
                    }
                }
            )
        }
        .task {
            if let refreshed = model.entries.first(where: { $0.id == entry.id }) {
                currentEntry = refreshed
            }
            await loadAudio(forceReload: false)
        }
        .onDisappear {
            audioPlayback.stop()
        }
    }

    private func loadAudio(forceReload: Bool) async {
        do {
            try await audioPlayback.load(
                fetchAudio: { try await model.fetchAudio(entryID: currentEntry.id) },
                forceReload: forceReload
            )
        } catch {
            model.errorMessage = error.localizedDescription
        }
    }

    private func togglePlayback() async {
        if !audioPlayback.isReady {
            await loadAudio(forceReload: true)
        }
        guard audioPlayback.isReady else { return }
        audioPlayback.togglePlayback()
    }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let total = max(0, Int(seconds.rounded()))
        let mins = total / 60
        let secs = total % 60
        return String(format: "%02d:%02d", mins, secs)
    }

    private func formattedHealthTimestamp(_ raw: String) -> String? {
        let parsedDate = Self.iso8601WithFractional.date(from: raw) ?? Self.iso8601Basic.date(from: raw)
        guard let parsedDate else {
            return nil
        }
        return DateFormatter.clock.string(from: parsedDate)
    }

    private static let iso8601WithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    private static let iso8601Basic: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private func deleteEntry() async {
        guard !isDeletingEntry else {
            return
        }
        isDeletingEntry = true
        await model.deleteEntry(entryID: currentEntry.id)
        isDeletingEntry = false
        if !model.entries.contains(where: { $0.id == currentEntry.id }) {
            dismiss()
        }
    }

    private func saveTags(_ tags: [String]) async {
        guard !isSavingTags else { return }
        isSavingTags = true
        if let updated = await model.updateEntryTags(
            entryID: currentEntry.id,
            moodTags: tags
        ) {
            currentEntry = updated
            showTagEditor = false
        }
        isSavingTags = false
    }

    private func saveTitle(_ title: String) async {
        guard !isSavingTitle else { return }
        isSavingTitle = true
        if let updated = await model.updateEntryTitle(
            entryID: currentEntry.id,
            title: title
        ) {
            currentEntry = updated
            showTitleEditor = false
        }
        isSavingTitle = false
    }

    private func retranscribeEntry() async {
        guard !isRetranscribing else { return }
        isRetranscribing = true
        if let updated = await model.retranscribeEntry(entryID: currentEntry.id) {
            currentEntry = updated
        }
        isRetranscribing = false
    }
}

private struct HealthScorePill: View {
    let snapshot: EntryHealthSnapshot

    var body: some View {
        Group {
            if let readinessScore = snapshot.readinessScore {
                Text("Health \(readinessScore)")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.mint.opacity(0.2), in: Capsule())
            } else {
                Text("Health --")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.secondary.opacity(0.18), in: Capsule())
            }
        }
    }
}

private struct EntryHealthSnapshotSection: View {
    let snapshot: EntryHealthSnapshot?
    let capturedAtLabel: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Health Snapshot")
                .font(.headline)

            if let snapshot {
                HStack(alignment: .center, spacing: 14) {
                    HealthScoreDial(score: snapshot.readinessScore)

                    VStack(alignment: .leading, spacing: 6) {
                        if let readinessScore = snapshot.readinessScore {
                            Text("Readiness \(readinessScore)")
                                .font(.title3.weight(.bold))
                        } else {
                            Text("Readiness unavailable")
                                .font(.subheadline.weight(.semibold))
                                .foregroundStyle(.secondary)
                        }

                        if snapshot.confidence < 0.5 {
                            Text("Low confidence (\(snapshot.confidencePercent)%)")
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.orange)
                        } else {
                            Text("Confidence \(snapshot.confidencePercent)%")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        if let capturedAtLabel {
                            Text("Captured \(capturedAtLabel)")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                    Spacer()
                }

                if !snapshot.sortedComponents.isEmpty {
                    HealthComponentScoreChart(components: snapshot.sortedComponents)

                    ForEach(snapshot.sortedComponents, id: \.type) { component in
                        HealthComponentRow(component: component)
                    }
                }
            } else {
                Text("No health snapshot was captured for this entry.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }

            Text("Wellness estimate only, not medical advice.")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
    }
}

private struct HealthScoreDial: View {
    let score: Int?

    private var clampedScore: Double {
        Double(max(0, min(100, score ?? 0)))
    }

    var body: some View {
        ZStack {
            Circle()
                .stroke(Color.secondary.opacity(0.2), lineWidth: 8)

            Circle()
                .trim(from: 0, to: clampedScore / 100)
                .stroke(
                    AngularGradient(
                        gradient: Gradient(colors: [Color.teal, Color.cyan, Color.mint]),
                        center: .center
                    ),
                    style: StrokeStyle(lineWidth: 8, lineCap: .round)
                )
                .rotationEffect(.degrees(-90))

            VStack(spacing: 1) {
                Text(score.map(String.init) ?? "--")
                    .font(.title3.weight(.bold))
                Text("Readiness")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: 92, height: 92)
    }
}

private struct HealthComponentScoreChart: View {
    let components: [HealthComponentSnapshot]

    private let chartHeight: CGFloat = 72

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Component Scores")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            HStack(alignment: .bottom, spacing: 12) {
                ForEach(components, id: \.type) { component in
                    VStack(spacing: 5) {
                        ZStack(alignment: .bottom) {
                            RoundedRectangle(cornerRadius: 6)
                                .fill(Color.secondary.opacity(0.14))
                                .frame(width: 28, height: chartHeight)

                            RoundedRectangle(cornerRadius: 6)
                                .fill(component.isStale ? Color.orange.opacity(0.82) : Color.mint.opacity(0.85))
                                .frame(
                                    width: 28,
                                    height: max(4, chartHeight * CGFloat(max(0, min(100, component.componentScore))) / 100)
                                )
                        }

                        Text(shortLabel(for: component.type))
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
        .padding(.vertical, 2)
    }

    private func shortLabel(for type: HealthMetricType) -> String {
        switch type {
        case .sleepHours:
            return "Sleep"
        case .hrvSdnnMs:
            return "HRV"
        case .restingHeartRateBpm:
            return "RHR"
        case .stepsToday:
            return "Steps"
        }
    }
}

private struct HealthComponentRow: View {
    let component: HealthComponentSnapshot

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(component.type.displayName)
                    .font(.subheadline.weight(.semibold))
                Spacer()
                Text(component.formattedRawValue)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                Text("\(component.scorePercent)")
                    .font(.caption.weight(.bold))
                    .foregroundStyle(.secondary)
            }
            ProgressView(value: component.componentScore / 100)
                .tint(component.isStale ? .orange : .mint)
            if component.isStale {
                Text("Stale sample")
                    .font(.caption2)
                    .foregroundStyle(.orange)
            }
        }
    }
}

struct WrapTags: View {
    let tags: [String]

    var body: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 8)], spacing: 8) {
            ForEach(tags, id: \.self) { tag in
                TagChip(tag: tag)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
}

private struct EntryTitleEditorSheet: View {
    @Environment(\.dismiss) private var dismiss

    let isSaving: Bool
    let onSave: (String) -> Void

    @State private var titleValue: String

    init(
        initialTitle: String,
        isSaving: Bool,
        onSave: @escaping (String) -> Void
    ) {
        self.isSaving = isSaving
        self.onSave = onSave
        _titleValue = State(initialValue: APIEntry.sanitizeTitle(initialTitle))
    }

    var body: some View {
        NavigationStack {
            VStack(alignment: .leading, spacing: 14) {
                Text("Title")
                    .font(.headline)
                TextField("Entry", text: $titleValue, axis: .vertical)
                    .textInputAutocapitalization(.sentences)
                    .autocorrectionDisabled(false)
                    .lineLimit(2)
                    .padding(12)
                    .background(Color.secondary.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
                Text("Shown on the journal timeline and this entry.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
                Spacer(minLength: 0)
            }
            .padding(16)
            .navigationTitle("Edit Title")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .disabled(isSaving)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        onSave(titleValue)
                    }
                    .disabled(isSaving)
                }
            }
        }
    }
}

private struct EntryTagEditorSheet: View {
    @Environment(\.dismiss) private var dismiss

    let isSaving: Bool
    let onSave: ([String]) -> Void

    @State private var selectedTags: [String]

    init(
        initialTags: [String],
        isSaving: Bool,
        onSave: @escaping ([String]) -> Void
    ) {
        let normalized = EntryTagEditorSheet.sanitized(initialTags)
        self.isSaving = isSaving
        self.onSave = onSave
        _selectedTags = State(initialValue: normalized)
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Select up to 4 tags")
                        .font(.headline)
                    Text("\(selectedTags.count)/4 selected")
                        .font(.footnote)
                        .foregroundStyle(.secondary)

                    LazyVGrid(
                        columns: [GridItem(.adaptive(minimum: 128), spacing: 8)],
                        spacing: 8
                    ) {
                        ForEach(EmotionTaxonomy.tags) { definition in
                            Button {
                                toggle(definition.id)
                            } label: {
                                SelectableTagChip(
                                    tag: definition.id,
                                    isSelected: selectedTags.contains(definition.id)
                                )
                            }
                            .buttonStyle(.plain)
                            .disabled(isSaving)
                        }
                    }

                    Text("Tags update your journal entry and social dot for this day if the entry was shared.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                        .padding(.top, 4)
                }
                .padding(16)
            }
            .navigationTitle("Edit Tags")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                    .disabled(isSaving)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Save") {
                        onSave(selectedTags)
                    }
                    .disabled(selectedTags.isEmpty || isSaving)
                }
            }
        }
    }

    private func toggle(_ tag: String) {
        let canonical = EmotionTaxonomy.canonicalTag(for: tag)
        if let existingIndex = selectedTags.firstIndex(of: canonical) {
            selectedTags.remove(at: existingIndex)
            return
        }
        guard selectedTags.count < 4 else {
            return
        }
        selectedTags.append(canonical)
    }

    private static func sanitized(_ tags: [String]) -> [String] {
        var ordered: [String] = []
        for tag in tags {
            let canonical = EmotionTaxonomy.canonicalTag(for: tag)
            if canonical.isEmpty || ordered.contains(canonical) {
                continue
            }
            ordered.append(canonical)
            if ordered.count >= 4 {
                break
            }
        }
        return ordered
    }
}

private struct SelectableTagChip: View {
    let tag: String
    let isSelected: Bool

    var body: some View {
        let color = EmotionPalette.color(forTag: tag)

        HStack(spacing: 6) {
            Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                .font(.caption.weight(.semibold))
            Text(EmotionTaxonomy.displayName(for: tag))
                .font(.caption.weight(.semibold))
                .lineLimit(1)
                .minimumScaleFactor(0.85)
        }
        .foregroundStyle(isSelected ? Color.primary : Color.primary.opacity(0.86))
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            LinearGradient(
                colors: [
                    color.opacity(isSelected ? 0.44 : 0.26),
                    color.opacity(isSelected ? 0.22 : 0.14),
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: Capsule()
        )
        .overlay(
            Capsule()
                .stroke(color.opacity(isSelected ? 0.88 : 0.42), lineWidth: isSelected ? 1.3 : 1.0)
        )
    }
}

enum EmotionPalette {
    static func color(forTag tag: String) -> Color {
        if let definition = EmotionTaxonomy.definition(for: tag) {
            return color(
                pleasantness: definition.pleasantness,
                energy: definition.energy
            )
        }

        var hash: UInt64 = 5381
        for byte in EmotionTaxonomy.normalize(tag).utf8 {
            hash = ((hash << 5) &+ hash) &+ UInt64(byte)
        }
        let hue = Double(hash % 360) / 360.0
        return Color(hue: hue, saturation: 0.55, brightness: 0.82)
    }

    private static func color(pleasantness: Double, energy: Double) -> Color {
        let clampedPleasantness = max(-1.0, min(1.0, pleasantness))
        let clampedEnergy = max(-1.0, min(1.0, energy))
        let hue = 0.01 + ((clampedPleasantness + 1.0) / 2.0) * 0.47
        let saturation = min(
            0.92,
            0.45
            + abs(clampedPleasantness) * 0.28
            + max(0.0, clampedEnergy) * 0.18
        )
        let brightness = min(0.96, 0.58 + ((clampedEnergy + 1.0) / 2.0) * 0.28)
        return Color(hue: hue, saturation: saturation, brightness: brightness)
    }
}

struct TagChip: View {
    let tag: String

    private var label: String {
        EmotionTaxonomy.displayName(for: tag)
    }

    var body: some View {
        let color = EmotionPalette.color(forTag: tag)

        Text(label)
            .font(.caption.weight(.semibold))
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(
                LinearGradient(
                    colors: [
                        color.opacity(0.38),
                        color.opacity(0.16),
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                ),
                in: Capsule()
            )
            .overlay(
                Capsule()
                    .stroke(color.opacity(0.48), lineWidth: 1)
            )
    }
}
