import ActivityKit
import AlarmKit
import AppIntents
import SwiftUI
import WidgetKit

@main
struct TimerWidget: Widget {
    var body: some WidgetConfiguration {
        ActivityConfiguration(for: AlarmAttributes<TimerMetadata>.self) { context in
            HStack {
                Label("Controllr timer", systemImage: "timer")
                Spacer()
                countdown(context)
                stopButton(context)
            }
            .padding()
        } dynamicIsland: { context in
            DynamicIsland {
                DynamicIslandExpandedRegion(.leading) {
                    Label("Timer", systemImage: "timer")
                }
                DynamicIslandExpandedRegion(.center) {
                    countdown(context)
                }
                DynamicIslandExpandedRegion(.trailing) {
                    stopButton(context)
                }
            } compactLeading: {
                Image(systemName: "timer")
            } compactTrailing: {
                countdown(context)
            } minimal: {
                Image(systemName: "timer")
            }
        }
    }

    private func countdown(_ context: ActivityViewContext<AlarmAttributes<TimerMetadata>>) -> some View {
        Group {
            if case .countdown(let duration) = context.state.mode {
                Text(timerInterval: duration.startDate...duration.fireDate, countsDown: true)
                    .monospacedDigit()
            } else {
                Text("0:00")
            }
        }
    }

    private func stopButton(_ context: ActivityViewContext<AlarmAttributes<TimerMetadata>>) -> some View {
        Button(intent: StopTimerIntent(id: context.state.alarmID)) {
            Image(systemName: "stop.fill")
                .accessibilityLabel("Stop timer")
        }
    }
}
