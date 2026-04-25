import SwiftUI

struct EditItemView: View {
    let original: Item
    let onSave: (Item) -> Void

    @State private var title: String
    @State private var content: String
    @State private var itemType: Item.ItemType
    @State private var hasDueDate: Bool
    @State private var dueDate: Date
    @State private var tags: [String]
    @State private var tagInput: String = ""
    @State private var isSaving = false
    @State private var alertMessage: String?
    @Environment(\.dismiss) private var dismiss

    init(item: Item, onSave: @escaping (Item) -> Void) {
        self.original = item
        self.onSave = onSave
        _title = State(initialValue: item.title)
        _content = State(initialValue: item.content)
        _itemType = State(initialValue: item.type)
        _hasDueDate = State(initialValue: item.dueDate != nil)
        _dueDate = State(initialValue: item.dueDateParsed ?? Calendar.current.date(byAdding: .day, value: 1, to: Date())!)
        _tags = State(initialValue: item.tags)
    }

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        typePicker
                        titleField
                        contentField
                        if itemType == .todo { dueDateField }
                        tagsField
                    }
                    .padding(16)
                    .padding(.bottom, 40)
                }
            }
            .navigationTitle("Edit")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") { dismiss() }
                        .tint(.white)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    if isSaving {
                        ProgressView().tint(.white)
                    } else {
                        Button("Save") { Task { await save() } }
                            .tint(.white)
                            .fontWeight(.semibold)
                            .disabled(title.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                }
            }
        }
        .preferredColorScheme(.dark)
        .alert("Error", isPresented: Binding(
            get: { alertMessage != nil },
            set: { if !$0 { alertMessage = nil } }
        )) {
            Button("OK") { alertMessage = nil }
        } message: {
            Text(alertMessage ?? "")
        }
    }

    // MARK: - Sections

    private var typePicker: some View {
        Picker("Type", selection: $itemType) {
            Label("Note", systemImage: "note.text").tag(Item.ItemType.note)
            Label("Todo", systemImage: "checkmark.circle").tag(Item.ItemType.todo)
            Label("Epic", systemImage: "square.stack.3d.up").tag(Item.ItemType.epic)
        }
        .pickerStyle(.segmented)
        .onChange(of: itemType) { _, newType in
            if newType != .todo { hasDueDate = false }
        }
    }

    private var titleField: some View {
        fieldCard(label: "Title") {
            TextField("Title", text: $title, axis: .vertical)
                .font(.system(.body, design: .rounded, weight: .semibold))
                .foregroundStyle(.white)
                .tint(.white)
                .lineLimit(1...4)
        }
    }

    private var contentField: some View {
        fieldCard(label: "Content") {
            TextEditor(text: $content)
                .font(.system(.body, design: .serif))
                .foregroundStyle(.white)
                .scrollContentBackground(.hidden)
                .frame(minHeight: 120)
                .tint(.white)
        }
    }

    private var dueDateField: some View {
        fieldCard(label: "Due Date") {
            Toggle("Set due date", isOn: $hasDueDate)
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white)
                .tint(.yellow)

            if hasDueDate {
                DatePicker(
                    "",
                    selection: $dueDate,
                    displayedComponents: [.date, .hourAndMinute]
                )
                .datePickerStyle(.compact)
                .colorScheme(.dark)
                .tint(.yellow)
                .labelsHidden()
                .padding(.top, 4)
            }
        }
    }

    private var tagsField: some View {
        fieldCard(label: "Tags") {
            if !tags.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 6) {
                        ForEach(tags, id: \.self) { tag in
                            HStack(spacing: 4) {
                                Text(tag)
                                    .font(.system(.caption2, design: .rounded, weight: .semibold))
                                    .foregroundStyle(.white.opacity(0.75))
                                Button {
                                    tags.removeAll { $0 == tag }
                                } label: {
                                    Image(systemName: "xmark")
                                        .font(.system(size: 9, weight: .bold))
                                        .foregroundStyle(.white.opacity(0.45))
                                }
                            }
                            .padding(.horizontal, 9)
                            .padding(.vertical, 5)
                            .background(Color.white.opacity(0.1))
                            .clipShape(Capsule())
                        }
                    }
                    .padding(.bottom, 4)
                }
            }

            HStack {
                TextField("Add tag…", text: $tagInput)
                    .font(.system(.subheadline, design: .rounded))
                    .foregroundStyle(.white)
                    .tint(.white)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .onSubmit { addTag() }
                if !tagInput.isEmpty {
                    Button("Add") { addTag() }
                        .font(.system(.caption, design: .rounded, weight: .semibold))
                        .foregroundStyle(.white.opacity(0.5))
                }
            }
        }
    }

    // MARK: - Helpers

    @ViewBuilder
    private func fieldCard<Content: View>(label: String, @ViewBuilder content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(label.uppercased())
                .font(.system(.caption2, design: .rounded, weight: .semibold))
                .foregroundStyle(.white.opacity(0.4))
            content()
        }
        .padding(14)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.white.opacity(0.08), lineWidth: 1))
    }

    private func addTag() {
        let trimmed = tagInput.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !trimmed.isEmpty, !tags.contains(trimmed) else { tagInput = ""; return }
        tags.append(trimmed)
        tagInput = ""
    }

    private func save() async {
        isSaving = true
        let dueDateISO: String? = hasDueDate ? ISO8601DateFormatter().string(from: dueDate) : nil
        do {
            let updated = try await APIClient.shared.editItem(
                id: original.id,
                title: title.trimmingCharacters(in: .whitespacesAndNewlines),
                content: content,
                type: itemType.rawValue,
                dueDate: dueDateISO,
                tags: tags
            )
            onSave(updated)
            dismiss()
        } catch {
            alertMessage = error.localizedDescription
            isSaving = false
        }
    }
}
