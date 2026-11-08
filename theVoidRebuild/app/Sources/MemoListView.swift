import SwiftData
import SwiftUI

struct MemoListView: View {
    @Query(sort: \Memo.createdAt, order: .reverse) private var memos: [Memo]
    @Environment(\.modelContext) private var context

    var body: some View {
        NavigationStack {
            List {
                ForEach(memos) { memo in
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text(memo.title.isEmpty ? "untitled" : memo.title).font(.headline)
                            Spacer()
                            HStack(spacing: 4) {
                                ForEach(memo.tags, id: \.self) { Circle().fill($0.color).frame(width: 10, height: 10) }
                            }
                        }
                        Text(memo.transcript).font(.subheadline).foregroundStyle(.secondary).lineLimit(3)
                        Text(memo.createdAt.formatted(date: .abbreviated, time: .shortened))
                            .font(.caption2).foregroundStyle(.tertiary)
                    }
                    .padding(.vertical, 4)
                }
                .onDelete { offsets in
                    for i in offsets {
                        let id = memos[i].id
                        context.delete(memos[i])
                        Task { await DotStore.shared.unpublish(memoID: id) }
                    }
                }
            }
            .overlay { if memos.isEmpty { Text("no memos yet").foregroundStyle(.secondary) } }
            .navigationTitle("memos")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}
