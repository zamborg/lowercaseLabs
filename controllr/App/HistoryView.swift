import SwiftUI
import SwiftData

struct HistoryView: View {
    @Query(sort: \RunRecord.date, order: .reverse) private var runs: [RunRecord]
    @Environment(\.modelContext) private var ctx

    var body: some View {
        NavigationStack {
            List(runs) { r in
                NavigationLink {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 16) {
                            Text(r.exportText).font(.footnote.monospaced()).textSelection(.enabled)
                            if let prompt = r.systemPrompt {
                                NavigationLink("System Prompt for This Run") {
                                    ScrollView {
                                        Text(prompt).font(.footnote.monospaced()).textSelection(.enabled).padding()
                                    }
                                    .navigationTitle("System Prompt")
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                    }
                        .toolbar { ToolbarItem { Button("Copy") { UIPasteboard.general.string = r.exportText } } }
                        .navigationTitle("Run")
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(r.transcript).lineLimit(1)
                        HStack {
                            Text(r.module).font(.caption.monospaced())
                            Text("\(r.latencyMs) ms").font(.caption)
                            if !r.error.isEmpty { Text("error").font(.caption).foregroundStyle(.red) }
                            Spacer()
                            Text(r.source).font(.caption).foregroundStyle(.secondary)
                        }.foregroundStyle(.secondary)
                    }
                }
            }
            .overlay { if runs.isEmpty { ContentUnavailableView("No runs yet", systemImage: "clock") } }
            .navigationTitle("History")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Copy all") { UIPasteboard.general.string = runs.prefix(30).map(\.exportText).joined(separator: "\n\n") }.disabled(runs.isEmpty)
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Clear", role: .destructive) { runs.forEach(ctx.delete); try? ctx.save() }.disabled(runs.isEmpty)
                }
            }
        }
    }
}
