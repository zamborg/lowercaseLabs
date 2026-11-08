import CloudKit
import SwiftUI

/// Tab 3: your friends' dots over the last week, one row per friend. Tags and time only, never content.
struct SocialView: View {
    @State private var store = DotStore.shared
    @State private var invite: InviteItem?
    @State private var selected: FriendDot?

    struct InviteItem: Identifiable { let id = UUID(); let share: CKShareBox }

    private let days = 7
    private var calendar: Calendar { .current }
    private var columns: [Date] {
        (0..<days).reversed().compactMap { calendar.date(byAdding: .day, value: -$0, to: calendar.startOfDay(for: .now)) }
    }

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 0) {
                    if store.friends.isEmpty {
                        empty
                    } else {
                        header
                        ForEach(store.friends) { friend in row(friend) }
                    }
                    if !store.status.isEmpty { Text(store.status).font(.caption).foregroundStyle(.orange).padding() }
                }
                .padding(.horizontal)
            }
            .background(Color.black)
            .navigationTitle("Friends")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                Button { startInvite() } label: { Image(systemName: "person.badge.plus") }
                    .accessibilityIdentifier("inviteButton")
            }
            .task { await store.refreshFriends() }
            .refreshable { await store.refreshFriends() }
            .sheet(item: $invite) { item in InviteSheet(share: item.share.share, container: store.cloudContainer) }
            .sheet(item: $selected) { dot in DotDetail(dot: dot).presentationDetents([.height(220)]) }
        }
    }

    private var empty: some View {
        VStack(spacing: 14) {
            Image(systemName: "person.2").font(.largeTitle).foregroundStyle(.secondary)
            Text("No friends yet").font(.headline)
            Text("Invite a friend to see each other's dots: just the colors of how you've been feeling, never what you said.")
                .font(.subheadline).foregroundStyle(.secondary).multilineTextAlignment(.center)
            Button("Invite a friend") { startInvite() }.buttonStyle(.borderedProminent).tint(.white).foregroundStyle(.black)
        }
        .padding(.top, 120)
    }

    private var header: some View {
        HStack(spacing: 0) {
            Color.clear.frame(width: 84)
            ForEach(columns, id: \.self) { day in
                Text(calendar.isDateInToday(day) ? "today" : day.formatted(.dateTime.weekday(.narrow)))
                    .font(.caption2).foregroundStyle(.secondary).frame(maxWidth: .infinity)
            }
        }
        .padding(.vertical, 10)
    }

    private func row(_ friend: Friend) -> some View {
        VStack(spacing: 4) {
            HStack(spacing: 0) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(friend.name).font(.subheadline.weight(.medium)).lineLimit(1)
                    if let mood = mood(of: friend) { Text(mood).font(.caption2).foregroundStyle(.secondary) }
                }
                .frame(width: 84, alignment: .leading)
                ForEach(columns, id: \.self) { day in
                    cell(friend.dots.filter { calendar.isDate($0.createdAt, inSameDayAs: day) })
                }
            }
            if !friend.sharesBack {
                Button("They can't see your dots yet. Send yours") { startInvite() }
                    .font(.caption2).frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .padding(.vertical, 12)
        .overlay(alignment: .bottom) { Divider().overlay(.white.opacity(0.1)) }
        .accessibilityIdentifier("friendRow")
    }

    private func cell(_ dots: [FriendDot]) -> some View {
        VStack(spacing: 3) {
            ForEach(dots.prefix(4)) { dot in
                Button { selected = dot } label: {
                    Circle().fill(dot.tags.first?.color ?? .gray).frame(width: 14, height: 14)
                        .overlay { if dot.tags.count > 1 { Circle().fill(dot.tags[1].color).frame(width: 6, height: 6) } }
                }
                .buttonStyle(.plain)
            }
            if dots.count > 4 { Text("+\(dots.count - 4)").font(.system(size: 9)).foregroundStyle(.secondary) }
        }
        .frame(maxWidth: .infinity, minHeight: 20, alignment: .top)
    }

    /// "mostly calm this week": the most common dominant tag over the visible days.
    private func mood(of friend: Friend) -> String? {
        let start = columns.first ?? .now
        let recent = friend.dots.filter { $0.createdAt >= start }.compactMap(\.tags.first)
        guard let top = Dictionary(grouping: recent, by: { $0 }).max(by: { $0.value.count < $1.value.count })?.key else { return nil }
        return "mostly \(top.shortLabel)"
    }

    private func startInvite() {
        Task {
            do { invite = InviteItem(share: CKShareBox(try await store.shareForInvite())) }
            catch { store.status = error.localizedDescription }
        }
    }
}

struct DotDetail: View {
    let dot: FriendDot

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(dot.createdAt.formatted(date: .complete, time: .shortened)).font(.headline)
            ForEach(dot.tags, id: \.self) { tag in
                HStack(spacing: 10) {
                    Circle().fill(tag.color).frame(width: 18, height: 18)
                    Text(tag.label)
                }
            }
            Spacer()
            Text("Friends only see how a message felt, never what was said.").font(.caption).foregroundStyle(.secondary)
        }
        .padding(24)
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

extension FriendDot: Hashable { static func == (a: FriendDot, b: FriendDot) -> Bool { a.id == b.id }; func hash(into h: inout Hasher) { h.combine(id) } }

struct CKShareBox: @unchecked Sendable { let share: CKShare; init(_ s: CKShare) { share = s } }
