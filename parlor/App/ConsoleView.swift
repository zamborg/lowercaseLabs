import SwiftUI
import ParlorCore

/// Control console: every speed, memory and token knob, backed by `~/parlor/settings.json`.
struct ConsoleSection: View {
    @Environment(ParlorEngine.self) private var engine

    private func bind<T>(_ kp: WritableKeyPath<ParlorSettings, T>) -> Binding<T> {
        Binding(get: { engine.settings[keyPath: kp] }, set: { v in engine.updateSettings { $0[keyPath: kp] = v } })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Text("Performance").font(.headline)
                Spacer()
                Button("Reset to defaults") { engine.resetSettings() }.controlSize(.small)
            }
            liveStrip
            card("Language model", symbol: "text.bubble") {
                row("Prompt cache", "Keeps the model's attention state between turns so each reply only processes the new messages. The biggest speed-up on long chats.") {
                    Toggle("", isOn: bind(\.llm.promptCache)).labelsHidden().toggleStyle(.switch)
                }
                row("KV cache precision", "Shrinks memory on very long chats. Takes effect only when the prompt cache is off.") {
                    Picker("", selection: Binding(
                        get: { engine.settings.llm.kvBits ?? 0 },
                        set: { v in engine.updateSettings { $0.llm.kvBits = v == 0 ? nil : v } })) {
                        Text("Full").tag(0); Text("8-bit").tag(8); Text("4-bit").tag(4)
                    }.labelsHidden().pickerStyle(.segmented).frame(width: 180)
                }
                row("Attention window", "Rolling window in tokens; the oldest tokens fall out. Caps memory. Turns the prompt cache off.") {
                    optionalInt(bind(\.llm.maxKVSize), prompt: "off")
                }
                row("Prefill step", "Prompt tokens processed per step. Larger is faster but uses more peak memory.") {
                    intField(bind(\.llm.prefillStepSize), width: 70)
                }
                row("Reply limit", "Hard cap on tokens per reply, which also bounds runaway output. Blank for none.") {
                    optionalInt(bind(\.llm.maxReplyTokens), prompt: "none")
                }
                row("Top-p", "Sampling cutoff. 1.0 considers every token; lower is more focused.") {
                    doubleField(bind(\.llm.topP), width: 60)
                }
                row("Repetition penalty", "Values like 1.1 discourage loops, which small models fall into. Blank for off.") {
                    optionalDouble(bind(\.llm.repetitionPenalty), prompt: "off")
                }
                row("Memory budget", "Share of this Mac's memory the loaded models may use. Past it, the least recently used model is unloaded.") {
                    HStack(spacing: 4) {
                        doubleField(bind(\.memoryBudgetFraction), width: 56); Text("of RAM").foregroundStyle(.secondary)
                    }
                }
                row("Cache slots per model", "Prompt caches kept per loaded model. A request reuses the one sharing the longest start with its prompt, so alternating chats stay fast.") {
                    intField(bind(\.cacheSlotsPerModel), width: 56)
                }
                row("Unload when idle", "Minutes with no chat before the model is released from memory. 0 keeps it loaded.") {
                    HStack(spacing: 4) { intField(bind(\.llm.keepLoadedMinutes), width: 56); Text("min").foregroundStyle(.secondary) }
                }
            }
            card("Images", symbol: "photo") {
                row("Precision", "int8 is the balanced default; int4 is smaller and faster to load with some loss of detail; bf16 is full quality. Restarts the worker.") {
                    Picker("", selection: bind(\.image.precision)) {
                        Text("bf16").tag("bf16"); Text("int8").tag("int8"); Text("int4").tag("int4")
                    }.labelsHidden().pickerStyle(.segmented).frame(width: 180)
                }
                row("Fast size", "Square side in pixels for \"fast\" images. Render time scales with pixel count.") {
                    HStack(spacing: 4) { intField(bind(\.image.fastSide), width: 60); Text("px").foregroundStyle(.secondary) }
                }
                row("Detailed size", "Square side in pixels for \"detailed\" images.") {
                    HStack(spacing: 4) { intField(bind(\.image.detailedSide), width: 60); Text("px").foregroundStyle(.secondary) }
                }
                row("Fast steps", "Denoising steps. Blank uses the model's own default.") { optionalInt(bind(\.image.fastSteps), prompt: "default") }
                row("Detailed steps", "More steps cost time roughly linearly.") { optionalInt(bind(\.image.detailedSteps), prompt: "default") }
                row("Free worker after", "Seconds idle before the image worker exits and returns its memory. 0 keeps it running.") {
                    HStack(spacing: 4) { intField(bind(\.image.idleSeconds), width: 60); Text("sec").foregroundStyle(.secondary) }
                }
            }
            card("Agent loop and tools", symbol: "wrench.and.screwdriver") {
                row("Max tool steps", "Model-and-tool round trips allowed per message. The last one is forced to be a plain answer.") {
                    intField(bind(\.loop.maxSteps), width: 56)
                }
                row("Tool result limit", "Longer results are cut to this many characters (start and end kept) before the model sees them.") {
                    HStack(spacing: 4) { intField(bind(\.loop.toolResultChars), width: 70); Text("chars").foregroundStyle(.secondary) }
                }
                row("Tool timeout", "Seconds before a slow tool call is abandoned.") {
                    HStack(spacing: 4) { intField(bind(\.loop.toolTimeoutSeconds), width: 60); Text("sec").foregroundStyle(.secondary) }
                }
                row("Keep verbatim", "Recent messages kept word for word when older ones are summarised.") {
                    intField(bind(\.loop.keepRecent), width: 56)
                }
                row("Summary length", "Token budget for each rolling summary. Smaller summaries free more context.") {
                    HStack(spacing: 4) { intField(bind(\.loop.summaryTokens), width: 60); Text("tokens").foregroundStyle(.secondary) }
                }
            }
            Text("Saved to settings.json in your parlor folder as you type. Edit it by hand or let Claude Code change it; the app picks up changes live.")
                .font(.caption).foregroundStyle(.secondary)
        }
        .accessibilityIdentifier("performance-console")
    }

    // MARK: live numbers

    private var liveStrip: some View {
        let stats = engine.selection.flatMap { engine.room(for: $0).lastStats }
        return HStack(spacing: 18) {
            stat("Last reply", stats.map { String(format: "%.0f tok/s", $0.tokensPerSecond) } ?? "–")
            stat("Prompt", stats.map { $0.promptTokens.formatted() + " tok" } ?? "–")
            stat("From cache", stats.map { s in
                s.promptTokens > 0 ? "\(Int(Double(s.cachedTokens) / Double(s.promptTokens) * 100))%" : "–" } ?? "–")
            stat("Language model", engine.loadedModel.map(shortModelName) ?? "unloaded")
            stat("Queue", engine.queueDepth == 0 ? "empty" : "\(engine.queueDepth) waiting")
            stat("Image worker", engine.imageWorkerState)
            Spacer()
        }
        .padding(12).background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
        .task { while !Task.isCancelled { await engine.refreshLoaded(); try? await Task.sleep(nanoseconds: 2_000_000_000) } }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.system(.callout, design: .monospaced)).lineLimit(1)
        }
    }

    // MARK: pieces

    private func card<C: View>(_ title: String, symbol: String, @ViewBuilder content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Label(title, systemImage: symbol).font(.subheadline.weight(.semibold)).padding(.horizontal, 12).padding(.top, 10).padding(.bottom, 4)
            content()
        }
        .background(.background.secondary, in: RoundedRectangle(cornerRadius: 10))
    }

    private func row<C: View>(_ title: String, _ help: String, @ViewBuilder control: () -> C) -> some View {
        HStack(alignment: .center, spacing: 14) {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                Text(help).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Spacer(minLength: 12)
            control()
        }
        .padding(.horizontal, 12).padding(.vertical, 8)
    }

    private func intField(_ b: Binding<Int>, width: CGFloat) -> some View {
        TextField("", value: b, format: .number.grouping(.never)).textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.trailing).frame(width: width)
    }

    private func doubleField(_ b: Binding<Double>, width: CGFloat) -> some View {
        TextField("", value: b, format: .number.precision(.fractionLength(0...2))).textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.trailing).frame(width: width)
    }

    private func optionalInt(_ b: Binding<Int?>, prompt: String) -> some View {
        TextField(prompt, value: b, format: .number.grouping(.never)).textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.trailing).frame(width: 80)
    }

    private func optionalDouble(_ b: Binding<Double?>, prompt: String) -> some View {
        TextField(prompt, value: b, format: .number.precision(.fractionLength(0...2))).textFieldStyle(.roundedBorder)
            .multilineTextAlignment(.trailing).frame(width: 80)
    }
}
