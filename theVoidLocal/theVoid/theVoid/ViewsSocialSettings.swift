import SwiftUI
import UIKit

// MARK: - Social

struct SocialView: View {
    @EnvironmentObject private var model: AppModel
    @State private var selectedDotID: String?

    private let columns = [GridItem(.adaptive(minimum: 66), spacing: 18)]

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.opacity(0.02).ignoresSafeArea()

                if model.socialDots.isEmpty {
                    Text("No friend dots yet.\nAdd friends in Settings.")
                        .font(.subheadline)
                        .multilineTextAlignment(.center)
                        .foregroundStyle(.secondary)
                        .padding()
                } else {
                    ScrollView {
                        LazyVGrid(columns: columns, spacing: 20) {
                            ForEach(model.socialDots) { dot in
                                VStack(spacing: 6) {
                                    Button {
                                        withAnimation(.spring(response: 0.24, dampingFraction: 0.84)) {
                                            if selectedDotID == dot.id {
                                                selectedDotID = nil
                                            } else {
                                                selectedDotID = dot.id
                                            }
                                        }
                                    } label: {
                                        EmotionMixedCircle(
                                            tags: dot.dotTags ?? [],
                                            fallbackHex: dot.dotColor,
                                            diameter: 46,
                                            borderOpacity: 0.38
                                        )
                                        .frame(width: 58, height: 58)
                                    }
                                    .buttonStyle(.plain)

                                    Text(dot.label ?? "@\(dot.userId.prefix(6))")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                        .minimumScaleFactor(0.75)
                                }
                                .frame(width: 84)
                                .overlay(alignment: .bottom) {
                                    if selectedDotID == dot.id {
                                        SocialDotTagBubble(tags: dot.dotTags ?? [])
                                            .offset(y: 92)
                                            .transition(.opacity.combined(with: .scale(scale: 0.92)))
                                    }
                                }
                                .zIndex(selectedDotID == dot.id ? 2 : 0)
                            }
                        }
                        .padding(.horizontal, 26)
                        .padding(.top, 24)
                        .padding(.bottom, 132)
                        .frame(maxWidth: .infinity, alignment: .center)
                    }
                }
            }
            .navigationTitle("Social")
            .refreshable {
                await model.refreshSocialDots()
            }
            .onChange(of: model.socialDots) { _, _ in
                if let selectedDotID, !model.socialDots.contains(where: { $0.id == selectedDotID }) {
                    self.selectedDotID = nil
                }
            }
        }
    }
}

private struct SocialDotTagBubble: View {
    let tags: [String]

    private var normalizedTags: [String] {
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

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            Text("Dot makeup")
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
                    Text("HealthKit (V2 planned): sleep, HRV, and activity.")
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
}
