import SwiftUI
import UIKit

// MARK: - Social

struct SocialView: View {
    @EnvironmentObject private var model: AppModel
    @State private var selectedDotID: String?
    @State private var dotFrames: [String: CGRect] = [:]

    private let columns = [GridItem(.adaptive(minimum: 66), spacing: 18)]
    private let bubbleWidth: CGFloat = 206

    private var selectedDot: APISocialDot? {
        guard let selectedDotID else { return nil }
        return model.socialDots.first(where: { $0.id == selectedDotID })
    }

    private struct SocialDotDayGroup: Identifiable {
        let key: String
        let title: String
        let dots: [APISocialDot]

        var id: String { key }
    }

    private var dayGroups: [SocialDotDayGroup] {
        guard !model.socialDots.isEmpty else { return [] }
        var grouped: [(String, [APISocialDot])] = []
        for dot in model.socialDots {
            let key = dot.localDate ?? "recent"
            if !grouped.isEmpty, grouped[grouped.count - 1].0 == key {
                grouped[grouped.count - 1].1.append(dot)
            } else {
                grouped.append((key, [dot]))
            }
        }

        return grouped.map { pair in
            SocialDotDayGroup(
                key: pair.0,
                title: dayTitle(for: pair.0),
                dots: pair.1
            )
        }
    }

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                ZStack(alignment: .topLeading) {
                    Color.black.opacity(0.02).ignoresSafeArea()

                    if model.socialDots.isEmpty {
                        Text("No friend dots yet.\nAdd friends in Settings.")
                            .font(.subheadline)
                            .multilineTextAlignment(.center)
                            .foregroundStyle(.secondary)
                            .padding()
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        ScrollView {
                            VStack(alignment: .leading, spacing: 14) {
                                Text("Newest dots first")
                                    .font(.footnote.weight(.semibold))
                                    .foregroundStyle(.secondary)

                                ForEach(dayGroups) { group in
                                    SocialDayDivider(title: group.title)

                                    LazyVGrid(columns: columns, spacing: 20) {
                                        ForEach(group.dots) { dot in
                                            SocialDotCell(dot: dot, isSelected: selectedDotID == dot.id) {
                                                withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                                                    if selectedDotID == dot.id {
                                                        selectedDotID = nil
                                                    } else {
                                                        selectedDotID = dot.id
                                                    }
                                                }
                                            }
                                            .background(
                                                GeometryReader { proxy in
                                                    Color.clear.preference(
                                                        key: SocialDotFramePreferenceKey.self,
                                                        value: [dot.id: proxy.frame(in: .named("social-grid"))]
                                                    )
                                                }
                                            )
                                            .zIndex(selectedDotID == dot.id ? 2 : 1)
                                        }
                                    }
                                }
                            }
                            .padding(.horizontal, 20)
                            .padding(.top, 18)
                            .padding(.bottom, 132)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .coordinateSpace(name: "social-grid")
                        .overlay(alignment: .topLeading) {
                            if let selectedDot,
                               let frame = dotFrames[selectedDot.id] {
                                SocialDotTagBubble(dot: selectedDot)
                                    .frame(width: bubbleWidth, alignment: .leading)
                                    .offset(
                                        x: bubbleOriginX(
                                            for: frame,
                                            containerWidth: geometry.size.width,
                                            bubbleWidth: bubbleWidth
                                        ),
                                        y: frame.maxY + 10
                                    )
                                    .transition(.opacity.combined(with: .scale(scale: 0.94)))
                                    .zIndex(3)
                            }
                        }
                    }
                }
            }
            .navigationTitle("Social")
            .task(id: model.sessionToken) {
                await model.refreshSocialDots()
            }
            .refreshable {
                await model.refreshSocialDots()
            }
            .onPreferenceChange(SocialDotFramePreferenceKey.self) { value in
                dotFrames = value
            }
            .onChange(of: model.socialDots) { _, _ in
                if let selectedDotID, !model.socialDots.contains(where: { $0.id == selectedDotID }) {
                    self.selectedDotID = nil
                }
                let knownIDs = Set(model.socialDots.map(\.id))
                dotFrames = dotFrames.filter { knownIDs.contains($0.key) }
            }
        }
    }

    private func bubbleOriginX(for frame: CGRect, containerWidth: CGFloat, bubbleWidth: CGFloat) -> CGFloat {
        let unclamped = frame.midX - (bubbleWidth / 2)
        let minimumX: CGFloat = 12
        let maximumX = max(minimumX, containerWidth - bubbleWidth - 12)
        return min(max(unclamped, minimumX), maximumX)
    }

    private func dayTitle(for key: String) -> String {
        guard key != "recent",
              let parsed = DateFormatter.localDate.date(from: key) else {
            return "Recent"
        }

        if Calendar.current.isDateInToday(parsed) {
            return "Today"
        }
        if Calendar.current.isDateInYesterday(parsed) {
            return "Yesterday"
        }
        return Self.dayHeaderFormatter.string(from: parsed)
    }

    private static let dayHeaderFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .none
        return formatter
    }()
}

private struct SocialDotFramePreferenceKey: PreferenceKey {
    static var defaultValue: [String: CGRect] = [:]

    static func reduce(value: inout [String: CGRect], nextValue: () -> [String: CGRect]) {
        value.merge(nextValue(), uniquingKeysWith: { _, new in new })
    }
}

private struct SocialDotCell: View {
    let dot: APISocialDot
    let isSelected: Bool
    let onTap: () -> Void

    var body: some View {
        VStack(spacing: 6) {
            Button(action: onTap) {
                EmotionMixedCircle(
                    tags: dot.dotTags ?? [],
                    fallbackHex: dot.dotColor,
                    diameter: 46,
                    borderOpacity: isSelected ? 0.62 : 0.38
                )
                .frame(width: 58, height: 58)
                .scaleEffect(isSelected ? 1.04 : 1.0)
            }
            .buttonStyle(.plain)

            Text(dot.label ?? "@\(dot.userId.prefix(6))")
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(1)
                .minimumScaleFactor(0.75)
        }
        .frame(width: 84)
    }
}

private struct SocialDayDivider: View {
    let title: String

    var body: some View {
        HStack(spacing: 10) {
            Rectangle()
                .fill(Color.secondary.opacity(0.22))
                .frame(height: 1)
            Text(title)
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)
                .lineLimit(1)
            Rectangle()
                .fill(Color.secondary.opacity(0.22))
                .frame(height: 1)
        }
        .padding(.top, 4)
    }
}

private struct SocialDotTagBubble: View {
    let dot: APISocialDot

    private var normalizedTags: [String] {
        var ordered: [String] = []
        for tag in dot.dotTags ?? [] {
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

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text(dot.label ?? "@\(dot.userId.prefix(6))")
                .font(.caption.weight(.semibold))
                .lineLimit(1)

            Text(dot.localDate ?? "Recent")
                .font(.caption2.weight(.semibold))
                .foregroundStyle(.secondary)

            if normalizedTags.isEmpty {
                Text("No tags shared")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                LazyVGrid(
                    columns: [GridItem(.adaptive(minimum: 70), spacing: 6)],
                    spacing: 6
                ) {
                    ForEach(normalizedTags, id: \.self) { tag in
                        TagChip(tag: tag)
                    }
                }
            }
        }
        .padding(10)
        .frame(width: 184, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .overlay(
            RoundedRectangle(cornerRadius: 12)
                .stroke(Color.white.opacity(0.18), lineWidth: 1)
        )
        .shadow(color: Color.black.opacity(0.16), radius: 10, x: 0, y: 4)
    }
}

struct ReminderScheduleEditor: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Days")
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(ReminderWeekday.ordered) { day in
                        let selected = model.isReminderWeekdaySelected(day)
                        Button(day.shortTitle) {
                            model.toggleReminderWeekday(day)
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 7)
                        .background(
                            selected
                            ? Color.teal.opacity(0.36)
                            : Color.secondary.opacity(0.16),
                            in: Capsule()
                        )
                        .overlay(
                            Capsule()
                                .stroke(
                                    selected ? Color.teal.opacity(0.75) : Color.white.opacity(0.14),
                                    lineWidth: 1
                                )
                        )
                    }
                }
            }

            ForEach(model.selectedReminderDays()) { day in
                HStack {
                    Text(day.title)
                        .font(.subheadline)
                    Spacer()
                    DatePicker(
                        "",
                        selection: Binding(
                            get: { model.reminderTime(for: day) },
                            set: { model.setReminderTime($0, for: day) }
                        ),
                        displayedComponents: .hourAndMinute
                    )
                    .labelsHidden()
                }
            }
        }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @EnvironmentObject private var model: AppModel
    @State private var acceptInviteToken: String = ""
    @State private var clipboardStatus: String?
    @State private var feedbackKind: FeedbackKind = .idea
    @State private var feedbackMessage: String = ""
    @State private var feedbackStatus: String?
    @State private var isSubmittingFeedback = false
    @State private var feedbackToastMessage: String?
    @FocusState private var focusedInput: FocusedInput?

    private enum FocusedInput: Hashable {
        case feedbackMessage
    }

    private enum FeedbackKind: String, CaseIterable, Identifiable {
        case idea
        case bug

        var id: String { rawValue }

        var title: String {
            switch self {
            case .idea:
                return "Idea"
            case .bug:
                return "Bug"
            }
        }
    }

    var body: some View {
        NavigationStack {
            Form {
                Section("Profile") {
                    TextField("Display name", text: $model.displayName)
                    Text("Anonymous handle: @\(model.anonymousHandle)")
                }

                Section("Check-In") {
                    DatePicker("Default time", selection: $model.dailyCheckin, displayedComponents: .hourAndMinute)
                    ReminderScheduleEditor()
                    Button("Save Reminder Schedule") {
                        Task {
                            await model.configureDailyReminder()
                        }
                    }
                    if let reminderStatus = model.reminderStatus {
                        Text(reminderStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("Integrations") {
                    HStack {
                        Label("Health", systemImage: healthIconName)
                        Spacer()
                        Text(model.healthAuthorizationState.displayLabel)
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }

                    if let snapshot = model.liveHealthSnapshot {
                        HStack {
                            Text("Latest score")
                            Spacer()
                            if let readinessScore = snapshot.readinessScore {
                                Text("\(readinessScore)")
                                    .font(.body.weight(.semibold))
                            } else {
                                Text("Unavailable")
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    if model.isFetchingLiveHealthSnapshot {
                        ProgressView("Refreshing health score...")
                            .font(.footnote)
                    }

                    HStack {
                        Button(healthConnectionActionLabel) {
                            Task {
                                await runHealthConnectionAction()
                            }
                        }
                        .disabled(!canRunHealthConnectionAction)

                        if model.healthAuthorizationState.isAuthorized {
                            Button("Refresh Now") {
                                Task {
                                    await model.refreshLiveHealthSnapshot()
                                }
                            }
                            .disabled(model.isFetchingLiveHealthSnapshot)
                        }
                    }

                    Text("Health data stays on this device in V1 and is not sent to social or backend.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("iCloud Sync") {
                    Toggle(
                        "Enable iCloud Sync",
                        isOn: Binding(
                            get: { model.iCloudSyncEnabled },
                            set: { model.setICloudSyncEnabled($0) }
                        )
                    )

                    HStack {
                        Text("Status")
                        Spacer()
                        Text(model.iCloudSyncStatusText)
                            .font(.footnote.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .minimumScaleFactor(0.75)
                    }

                    if let lastSynced = model.iCloudLastSyncAt {
                        Text("Last sync: \(lastSynced.formatted(date: .abbreviated, time: .shortened))")
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    Button("Sync Now") {
                        Task {
                            await model.syncNow()
                        }
                    }
                    .disabled(!model.iCloudSyncEnabled)

                    Text("Syncs local journal entries, transcripts/insights, audio, and drafts to your private iCloud account.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("On-Device AI") {
                    Button(model.isPreparingLiquidModel ? "Preparing Model..." : "Redownload Liquid Model") {
                        model.redownloadLiquidModel()
                    }
                    .disabled(model.isPreparingLiquidModel)

                    if model.isPreparingLiquidModel {
                        ProgressView(value: model.liquidModelPreparationProgress)
                    }

                    Text("Use this if local model files are corrupted or you want a fresh download.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Social") {
                    Button("Create Invite Link") {
                        Task {
                            await model.createInvite()
                        }
                    }

                    if let inviteToken = model.inviteToken {
                        HStack(alignment: .top) {
                            Text(inviteToken)
                                .font(.footnote.monospaced())
                                .textSelection(.enabled)
                            Spacer()
                            Button("Copy Token") {
                                copyToClipboard(inviteToken, label: "invite token")
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    if let clipboardStatus {
                        Text(clipboardStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }

                    TextField("Paste invite link or token", text: $acceptInviteToken)
                        .textInputAutocapitalization(.never)
                        .autocorrectionDisabled()

                    Button("Accept Invite") {
                        Task {
                            await model.acceptInvite(token: acceptInviteToken)
                            acceptInviteToken = ""
                        }
                    }
                }

                Section("Actions") {
                    Button("Save Settings") {
                        Task {
                            await model.saveProfile()
                        }
                    }
                    Button("Refresh") {
                        Task {
                            await model.refreshAll()
                        }
                    }
                    Button("Sign Out", role: .destructive) {
                        model.signOut()
                    }
                }

                Section("Send Idea / Bug Report") {
                    Picker("Type", selection: $feedbackKind) {
                        ForEach(FeedbackKind.allCases) { option in
                            Text(option.title).tag(option)
                        }
                    }
                    .pickerStyle(.segmented)

                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $feedbackMessage)
                            .frame(minHeight: 110)
                            .focused($focusedInput, equals: .feedbackMessage)
                        if feedbackMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text("Tell us what happened or what you'd like to see.")
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 8)
                                .allowsHitTesting(false)
                        }
                    }

                    Button(isSubmittingFeedback ? "Sending..." : "Send Report") {
                        Task {
                            await submitFeedback()
                        }
                    }
                    .disabled(
                        isSubmittingFeedback
                            || feedbackMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    )

                    if let feedbackStatus {
                        Text(feedbackStatus)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                }

                Section("About") {
                    Text("Made by Zubin @ lowercaseLabs")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .task {
                await model.refreshHealthAuthorizationState()
            }
            .scrollDismissesKeyboard(.interactively)
            .toolbar {
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") {
                        focusedInput = nil
                    }
                }
            }
            .overlay(alignment: .bottom) {
                if let feedbackToastMessage {
                    Text(feedbackToastMessage)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.white)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .background(
                            RoundedRectangle(cornerRadius: 12)
                                .fill(Color.black.opacity(0.82))
                        )
                        .padding(.bottom, 22)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
        }
    }

    private func copyToClipboard(_ value: String, label: String) {
        UIPasteboard.general.string = value
        clipboardStatus = "Copied \(label)."
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.8) {
            clipboardStatus = nil
        }
    }

    private func submitFeedback() async {
        guard !isSubmittingFeedback else { return }
        focusedInput = nil
        isSubmittingFeedback = true
        let succeeded = await model.submitFeedback(kind: feedbackKind.rawValue, message: feedbackMessage)
        if succeeded {
            feedbackMessage = ""
            feedbackStatus = "Submitted."
            showFeedbackToast("Report submitted")
        }
        isSubmittingFeedback = false
    }

    private func showFeedbackToast(_ message: String) {
        withAnimation(.easeInOut(duration: 0.2)) {
            feedbackToastMessage = message
        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 2.0) {
            withAnimation(.easeInOut(duration: 0.2)) {
                feedbackToastMessage = nil
            }
        }
    }

    private var healthIconName: String {
        switch model.healthAuthorizationState {
        case .authorizedAll:
            return "heart.fill"
        case .authorizedPartial:
            return "heart.circle.fill"
        case .notDetermined:
            return "heart"
        case .denied:
            return "heart.slash.fill"
        case .unavailable:
            return "heart.slash"
        }
    }

    private var healthConnectionActionLabel: String {
        switch model.healthAuthorizationState {
        case .authorizedAll:
            return "Manage in Health"
        case .authorizedPartial:
            return "Manage in Health"
        case .notDetermined:
            return "Connect Health"
        case .denied:
            return "Open Health"
        case .unavailable:
            return "Unavailable"
        }
    }

    private var canRunHealthConnectionAction: Bool {
        model.healthAuthorizationState != .unavailable
    }

    private func runHealthConnectionAction() async {
        switch model.healthAuthorizationState {
        case .notDetermined:
            await model.requestHealthAuthorization()
        case .denied:
            await model.requestHealthAuthorization()
            if model.healthAuthorizationState == .denied {
                await MainActor.run {
                    UIApplication.openHealthAccessManagement()
                }
            }
        case .unavailable:
            break
        case .authorizedAll, .authorizedPartial:
            await MainActor.run {
                UIApplication.openHealthAccessManagement()
            }
        }
    }
}
