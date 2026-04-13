import Foundation
import Testing
@testable import theVoid

struct HealthScoreEngineTests {
    @Test func fullDataProducesDeterministicScore() {
        let engine = HealthScoreEngine()
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let signals = HealthScoreSignals(readings: [
            .sleepHours: HealthMetricReading(rawValue: 7.9, sampledAt: timestamp.addingTimeInterval(-3600)),
            .hrvSdnnMs: HealthMetricReading(rawValue: 58, sampledAt: timestamp.addingTimeInterval(-1800)),
            .restingHeartRateBpm: HealthMetricReading(rawValue: 59, sampledAt: timestamp.addingTimeInterval(-1200)),
            .stepsToday: HealthMetricReading(rawValue: 5200, sampledAt: timestamp),
        ])
        let context = HealthScoreContext(timestamp: timestamp, elapsedDayFraction: 0.55)

        let first = engine.evaluate(signals: signals, context: context)
        let second = engine.evaluate(signals: signals, context: context)

        #expect(first.readinessScore != nil)
        #expect(abs(first.confidence - 1) < 0.000_01)
        #expect(first.components.count == 4)
        #expect(first.readinessScore == second.readinessScore)
        #expect(abs(first.confidence - second.confidence) < 0.000_01)
    }

    @Test func missingComponentsRenormalizeConfidence() {
        let engine = HealthScoreEngine()
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let signals = HealthScoreSignals(readings: [
            .sleepHours: HealthMetricReading(rawValue: 8.1, sampledAt: timestamp.addingTimeInterval(-1200)),
            .stepsToday: HealthMetricReading(rawValue: 2100, sampledAt: timestamp),
        ])
        let context = HealthScoreContext(timestamp: timestamp, elapsedDayFraction: 0.4)

        let result = engine.evaluate(signals: signals, context: context)

        #expect(result.readinessScore != nil)
        #expect(abs(result.confidence - 0.5) < 0.000_01)
        #expect(result.components.count == 2)
    }

    @Test func staleSignalsAreExcludedFromComposite() {
        let engine = HealthScoreEngine()
        let timestamp = Date(timeIntervalSince1970: 1_700_000_000)
        let staleHRVTime = timestamp.addingTimeInterval(-(40 * 3600))
        let signals = HealthScoreSignals(readings: [
            .sleepHours: HealthMetricReading(rawValue: 7.4, sampledAt: timestamp.addingTimeInterval(-1800)),
            .hrvSdnnMs: HealthMetricReading(rawValue: 55, sampledAt: staleHRVTime),
        ])

        let result = engine.evaluate(
            signals: signals,
            context: HealthScoreContext(timestamp: timestamp, elapsedDayFraction: 0.6)
        )

        let hrvComponent = result.components.first(where: { $0.type == .hrvSdnnMs })
        #expect(hrvComponent != nil)
        #expect(hrvComponent?.isStale == true)
        #expect(abs(result.confidence - 0.35) < 0.000_01)
    }
}
