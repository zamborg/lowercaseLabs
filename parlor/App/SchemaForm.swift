import SwiftUI
import ParlorCore

/// Renders a harness's config schema (the JSON-Schema subset: type, enum, default, minimum/maximum, format: multiline,
/// description) as a form over a dictionary. A blank field removes the key, so the host's or machine's default applies.
struct SchemaForm: View {
    let schema: JSON?
    @Binding var values: [String: JSON]

    private var props: [(String, [String: JSON])] {
        let p = schema?.objectValue?["properties"]?.objectValue ?? [:]
        return p.keys.sorted { a, b in
            let ma = p[a]?.objectValue?["format"]?.stringValue == "multiline", mb = p[b]?.objectValue?["format"]?.stringValue == "multiline"
            return ma != mb ? !ma : a < b
        }.map { ($0, p[$0]?.objectValue ?? [:]) }
    }

    private var unknownKeys: [String] {
        let known = Set(schema?.objectValue?["properties"]?.objectValue?.keys.map { $0 } ?? [])
        return values.keys.filter { !known.contains($0) }.sorted()
    }

    var body: some View {
        ForEach(props, id: \.0) { key, spec in field(key, spec) }
        if !unknownKeys.isEmpty {
            Text("Also in this agent's config, kept as is: " + unknownKeys.joined(separator: ", "))
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private func title(_ key: String) -> String { key.replacingOccurrences(of: "_", with: " ").capitalized }

    @ViewBuilder private func field(_ key: String, _ spec: [String: JSON]) -> some View {
        let desc = spec["description"]?.stringValue
        let def = spec["default"]
        VStack(alignment: .leading, spacing: 3) {
            if case .array(let options)? = spec["enum"] {
                Picker(title(key), selection: Binding(
                    get: { values[key]?.stringValue ?? "" },
                    set: { values[key] = $0.isEmpty ? nil : .string($0) })) {
                    Text(def?.stringValue.map { "Default (\($0))" } ?? "Default").tag("")
                    ForEach(options.compactMap(\.stringValue), id: \.self) { Text($0).tag($0) }
                }
                .accessibilityIdentifier("config-\(key)")
            } else {
                switch spec["type"]?.stringValue {
                case "boolean":
                    Toggle(title(key), isOn: Binding(
                        get: { if case .bool(let b)? = values[key] { return b }; if case .bool(let b)? = def { return b }; return false },
                        set: { values[key] = .bool($0) }))
                    .accessibilityIdentifier("config-\(key)")
                case "integer", "number":
                    LabeledContent(title(key)) {
                        HStack {
                            TextField("", text: Binding(get: { text(values[key]) }, set: { set(key, $0, spec) }),
                                      prompt: Text(def.map { text($0) } ?? "default"))
                                .multilineTextAlignment(.trailing).frame(width: 110)
                                .accessibilityIdentifier("config-\(key)")
                            if values[key] != nil {
                                Button("Reset") { values[key] = nil }.controlSize(.small).accessibilityIdentifier("reset-\(key)")
                            }
                        }
                    }
                default:
                    if spec["format"]?.stringValue == "multiline" {
                        Text(title(key)).font(.callout)
                        TextEditor(text: Binding(get: { values[key]?.stringValue ?? "" }, set: { values[key] = $0.isEmpty ? nil : .string($0) }))
                            .font(.body).frame(minHeight: 70).scrollContentBackground(.hidden)
                            .padding(6).background(.background.secondary, in: RoundedRectangle(cornerRadius: 8))
                            .accessibilityIdentifier("config-\(key)")
                    } else {
                        TextField(title(key), text: Binding(get: { values[key]?.stringValue ?? "" }, set: { values[key] = $0.isEmpty ? nil : .string($0) }),
                                  prompt: Text(def?.stringValue ?? ""))
                            .accessibilityIdentifier("config-\(key)")
                    }
                }
            }
            if let desc { Text(desc).font(.caption).foregroundStyle(.secondary) }
        }
    }

    private func text(_ j: JSON?) -> String {
        switch j {
        case .int(let i)?: return String(i)
        case .double(let d)?: return d == d.rounded() && abs(d) < 1e9 ? String(Int(d)) : String(d)
        default: return ""
        }
    }

    private func set(_ key: String, _ raw: String, _ spec: [String: JSON]) {
        let t = raw.trimmingCharacters(in: .whitespaces)
        if t.isEmpty { values[key] = nil; return }
        if spec["type"]?.stringValue == "integer" {
            if let i = Int(t) { values[key] = .int(i) }
        } else if let d = Double(t) {
            values[key] = .double(d)
        }
    }
}
