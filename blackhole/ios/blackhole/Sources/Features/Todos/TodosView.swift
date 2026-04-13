import SwiftUI

struct TodosView: View {
    @Bindable var model: TodosViewModel

    var body: some View {
        NavigationStack {
            ZStack {
                AppTheme.background
                    .ignoresSafeArea()

                VStack(spacing: 16) {
                    Picker("Filter", selection: $model.filter) {
                        ForEach(TodosViewModel.Filter.allCases) { filter in
                            Text(filter.rawValue).tag(filter)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal, 20)
                    .onChange(of: model.filter) { _, _ in
                        Task { await model.refresh() }
                    }

                    List {
                        ForEach(model.todos) { todo in
                            NavigationLink {
                                TodoDetailView(todo: todo) { updated in
                                    await model.save(todo: updated)
                                }
                            } label: {
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text(todo.title)
                                            .font(.headline)
                                        Spacer()
                                        Text(todo.status.rawValue.capitalized)
                                            .font(.caption.weight(.semibold))
                                            .foregroundStyle(todo.status == .completed ? .green : .secondary)
                                    }
                                    Text(todo.descriptionMarkdown)
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                    Text(todo.tags.joined(separator: " • "))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                .padding(.vertical, 4)
                            }
                            .listRowBackground(Color.white.opacity(0.74))
                        }
                    }
                    .scrollContentBackground(.hidden)
                    .background(Color.clear)

                    Button {
                        model.triggerVoiceAdd()
                    } label: {
                        Label("Voice Add Todo", systemImage: "mic.fill")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .foregroundStyle(.white)
                            .background(AppTheme.accentStrong, in: RoundedRectangle(cornerRadius: 18, style: .continuous))
                    }
                    .padding(.horizontal, 20)
                    .padding(.bottom, 12)
                }
            }
            .navigationTitle("Todos")
            .task {
                await model.refresh()
            }
        }
    }
}

struct TodoDetailView: View {
    @Environment(\.dismiss) private var dismiss
    @State var todo: TodoRecord
    let onSave: @MainActor (TodoRecord) async -> Void

    var body: some View {
        Form {
            TextField("Title", text: $todo.title)
            TextEditor(text: $todo.descriptionMarkdown)
                .frame(minHeight: 180)
            Picker("Priority", selection: $todo.priority) {
                ForEach(TodoPriority.allCases, id: \.self) { priority in
                    Text(priority.rawValue.capitalized).tag(priority)
                }
            }
            Picker("Status", selection: $todo.status) {
                ForEach(TodoStatus.allCases, id: \.self) { status in
                    Text(status.rawValue.capitalized).tag(status)
                }
            }
            TextField("Tags (comma separated)", text: Binding(
                get: { todo.tags.joined(separator: ", ") },
                set: { todo.tags = $0.split(separator: ",").map { $0.trimmingCharacters(in: .whitespaces) } }
            ))
        }
        .navigationTitle("Todo")
        .toolbar {
            Button("Save") {
                Task {
                    await onSave(todo)
                    dismiss()
                }
            }
        }
    }
}
