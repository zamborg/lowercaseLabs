import SwiftUI

struct FriendsView: View {
    @State private var store = DotStore.shared
    @State private var invite: InviteItem?

    struct InviteItem: Identifiable { let id = UUID(); let share: CKShareBox }

    var body: some View {
        NavigationStack {
            List {
                if store.friendDots.isEmpty {
                    Text("no friend dots yet").foregroundStyle(.secondary)
                }
                ForEach(store.friendDots) { dot in
                    HStack {
                        Text("friend \(String(dot.ownerID.suffix(4)))").font(.subheadline.monospaced())
                        Spacer()
                        HStack(spacing: 4) {
                            ForEach(dot.tags, id: \.self) { Circle().fill($0.color).frame(width: 12, height: 12) }
                        }
                        Text(dot.createdAt.formatted(.relative(presentation: .named)))
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }
                if !store.status.isEmpty { Text(store.status).font(.caption).foregroundStyle(.orange) }
            }
            .navigationTitle("friends")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                Button("invite") {
                    Task {
                        do { invite = InviteItem(share: CKShareBox(try await store.shareForInvite())) }
                        catch { store.status = "invite failed: \(error.localizedDescription)" }
                    }
                }
            }
            .task { await store.refreshFriends() }
            .refreshable { await store.refreshFriends() }
            .sheet(item: $invite) { item in
                InviteSheet(share: item.share.share, container: store.cloudContainer)
            }
        }
    }
}

import CloudKit
struct CKShareBox: @unchecked Sendable { let share: CKShare; init(_ s: CKShare) { share = s } }
