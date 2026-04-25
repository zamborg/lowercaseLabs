import SwiftUI

@MainActor
final class SearchViewModel: ObservableObject {
    @Published var query = ""
    @Published var results: [Item] = []
    @Published var isSearching = false
    @Published var hasSearched = false
    @Published var alertMessage: String?
    @Published private(set) var isListening = false
    @Published private(set) var isDictationBusy = false

    private let audioCapture = AudioCapturePipeline()
    private let engineCoordinator = DictationEngineCoordinator()
    private var currentSession: DictationSession?

    func search() async {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return }
        isSearching = true
        do { results = try await APIClient.shared.search(query: q); hasSearched = true }
        catch { alertMessage = error.localizedDescription }
        isSearching = false
    }

    func pressDidStart() {
        guard !isListening, !isDictationBusy else { return }
        Task { await startDictation() }
    }

    func pressDidEnd() {
        guard isListening else { return }
        Task { await stopDictation() }
    }

    private func startDictation() async {
        isDictationBusy = true
        let hasMic = await audioCapture.requestPermission()
        guard hasMic else {
            alertMessage = DictationError.microphonePermissionDenied.localizedDescription
            isDictationBusy = false
            return
        }
        do {
            let engine = try await engineCoordinator.engine(for: .apple)
            let session = try engine.makeSession(locale: .current)
            session.onUpdate = { [weak self] update in
                Task { @MainActor [weak self] in self?.query = update.text }
            }
            try audioCapture.start { [weak session] buffer, when in
                try? session?.append(buffer, at: when)
            }
            currentSession = session
            isListening = true
            isDictationBusy = false
        } catch {
            alertMessage = error.localizedDescription
            isDictationBusy = false
        }
    }

    private func stopDictation() async {
        guard isListening, let session = currentSession else { return }
        isDictationBusy = true
        isListening = false
        audioCapture.stop()
        currentSession = nil
        do {
            let text = try await session.finish()
            query = text.trimmingCharacters(in: .whitespacesAndNewlines)
            await search()
        } catch {
            alertMessage = error.localizedDescription
        }
        isDictationBusy = false
    }
}

struct SearchView: View {
    @StateObject private var viewModel = SearchViewModel()
    @FocusState private var isQueryFocused: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 0) {
                    searchBar.padding(16)
                    Divider().overlay(.white.opacity(0.1))
                    resultsArea
                }
            }
            .navigationTitle("Search")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.black, for: .navigationBar)
            .toolbarColorScheme(.dark, for: .navigationBar)
        }
        .preferredColorScheme(.dark)
        .alert("Error", isPresented: Binding(
            get: { viewModel.alertMessage != nil },
            set: { if !$0 { viewModel.alertMessage = nil } }
        )) {
            Button("OK") { viewModel.alertMessage = nil }
        } message: {
            Text(viewModel.alertMessage ?? "")
        }
    }

    private var searchBar: some View {
        HStack(spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.white.opacity(0.4))
                    .font(.system(size: 16))
                TextField("Search notes, epics, and to-dos…", text: $viewModel.query)
                    .font(.system(.body, design: .rounded))
                    .foregroundStyle(.white)
                    .tint(.white)
                    .focused($isQueryFocused)
                    .onSubmit { Task { await viewModel.search() } }
                if !viewModel.query.isEmpty {
                    Button { viewModel.query = "" } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.white.opacity(0.4))
                    }
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(Color.white.opacity(0.08))
            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))

            VoiceMicButton(
                isActive: viewModel.isListening,
                isDisabled: viewModel.isDictationBusy,
                onPressStart: { viewModel.pressDidStart() },
                onPressEnd: { viewModel.pressDidEnd() }
            )
        }
    }

    @ViewBuilder
    private var resultsArea: some View {
        if viewModel.isSearching {
            Spacer()
            ProgressView().tint(.white)
            Spacer()
        } else if viewModel.hasSearched && viewModel.results.isEmpty {
            Spacer()
            Text("No results.")
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white.opacity(0.4))
            Spacer()
        } else if !viewModel.results.isEmpty {
            List {
                ForEach(viewModel.results) { item in
                    SearchResultRow(item: item)
                        .listRowBackground(Color.clear)
                        .listRowSeparator(.hidden)
                        .listRowInsets(EdgeInsets(top: 4, leading: 16, bottom: 4, trailing: 16))
                }
            }
            .listStyle(.plain)
            .scrollContentBackground(.hidden)
        } else {
            Spacer()
            Text("Hold mic to dictate or type to search.")
                .font(.system(.subheadline, design: .rounded))
                .foregroundStyle(.white.opacity(0.3))
                .multilineTextAlignment(.center)
                .padding(.horizontal, 40)
            Spacer()
        }
    }
}

private struct VoiceMicButton: View {
    let isActive: Bool
    let isDisabled: Bool
    let onPressStart: () -> Void
    let onPressEnd: () -> Void

    var body: some View {
        ZStack {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(isActive ? Color(white: 0.22) : Color.white.opacity(0.1))
            Image(systemName: isActive ? "waveform" : "mic.fill")
                .font(.system(size: 17, weight: .semibold))
                .foregroundStyle(.white)
            PressAndHoldCaptureView(
                isDisabled: isDisabled,
                onPressStart: onPressStart,
                onPressEndInside: onPressEnd,
                onPressEndOutside: onPressEnd
            )
        }
        .frame(width: 44, height: 44)
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).stroke(.white.opacity(0.15), lineWidth: 1))
    }
}

private struct SearchResultRow: View {
    let item: Item

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Image(systemName: item.type.systemImage)
                    .font(.system(size: 13))
                    .foregroundStyle(item.type == .epic ? .cyan.opacity(0.65) : .white.opacity(0.4))
                Text(item.title)
                    .font(.system(.subheadline, design: .rounded, weight: .semibold))
                    .foregroundStyle(.white)
                Spacer()
                Text(item.timestampLabel)
                    .font(.system(.caption2, design: .rounded))
                    .foregroundStyle(.white.opacity(0.28))
            }
            Text(item.previewText)
                .font(.system(.caption, design: .serif))
                .foregroundStyle(.white.opacity(0.55))
                .lineLimit(3)
        }
        .padding(14)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 16, style: .continuous).stroke(.white.opacity(0.08), lineWidth: 1))
    }
}
