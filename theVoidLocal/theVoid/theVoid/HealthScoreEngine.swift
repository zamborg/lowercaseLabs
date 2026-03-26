import Foundation

struct HealthMetricReading: Hashable {
    let rawValue: Double
    let sampledAt: Date?
}

struct HealthScoreSignals {
    let readings: [HealthMetricType: HealthMetricReading]

    func reading(for metricType: HealthMetricType) -> HealthMetricReading? {
        readings[metricType]
    }
}

struct HealthScoreContext {
    let timestamp: Date
    let elapsedDayFraction: Double

    var clampedElapsedDayFraction: Double {
        max(0, min(1, elapsedDayFraction))
    }
}

struct HealthScoreConfig {
    let weights: [HealthMetricType: Double]
    let staleThresholds: [HealthMetricType: TimeInterval]

    let sleepOptimalHours: Double
    let sleepMinimumHours: Double
    let sleepMaximumHours: Double

    let hrvRangeMs: ClosedRange<Double>
    let restingHeartRateRangeBpm: ClosedRange<Double>

    let stepsDailyGoal: Double
    let stepsDayFractionFloor: Double

    static let `default` = HealthScoreConfig(
        weights: [
            .sleepHours: 0.35,
            .hrvSdnnMs: 0.25,
            .restingHeartRateBpm: 0.25,
            .stepsToday: 0.15,
        ],
        staleThresholds: [
            .sleepHours: 36.0 * 60.0 * 60.0,
            .hrvSdnnMs: 36.0 * 60.0 * 60.0,
            .restingHeartRateBpm: 36.0 * 60.0 * 60.0,
            .stepsToday: 4.0 * 60.0 * 60.0,
        ],
        sleepOptimalHours: 8,
        sleepMinimumHours: 4,
        sleepMaximumHours: 11,
        hrvRangeMs: 20 ... 120,
        restingHeartRateRangeBpm: 45 ... 95,
        stepsDailyGoal: 9_000,
        stepsDayFractionFloor: 0.15
    )

    func weight(for metricType: HealthMetricType) -> Double {
        max(0, weights[metricType] ?? 0)
    }

    func staleThreshold(for metricType: HealthMetricType) -> TimeInterval {
        staleThresholds[metricType] ?? (36 * 60 * 60)
    }
}

protocol HealthComponentCalculator {
    var metricType: HealthMetricType { get }
    var unit: String { get }

    func componentScore(for rawValue: Double, context: HealthScoreContext, config: HealthScoreConfig) -> Double
}

extension HealthComponentCalculator {
    func makeSnapshot(
        reading: HealthMetricReading,
        context: HealthScoreContext,
        config: HealthScoreConfig,
        isStale: Bool
    ) -> HealthComponentSnapshot {
        HealthComponentSnapshot(
            type: metricType,
            rawValue: reading.rawValue,
            unit: unit,
            componentScore: clamp(componentScore(for: reading.rawValue, context: context, config: config), min: 0, max: 100),
            sampledAtISO8601: reading.sampledAt.map { ISO8601HealthFormatters.withFractional.string(from: $0) },
            isStale: isStale
        )
    }
}

struct HealthScoreResult {
    let readinessScore: Int?
    let confidence: Double
    let components: [HealthComponentSnapshot]
}

struct HealthScoreEngine {
    let config: HealthScoreConfig
    let calculators: [any HealthComponentCalculator]

    init(
        config: HealthScoreConfig = .default,
        calculators: [any HealthComponentCalculator] = HealthScoreEngine.defaultCalculators
    ) {
        self.config = config
        self.calculators = calculators
    }

    func evaluate(signals: HealthScoreSignals, context: HealthScoreContext) -> HealthScoreResult {
        let totalWeight = calculators.reduce(0) { partialResult, calculator in
            partialResult + config.weight(for: calculator.metricType)
        }

        var availableWeight = 0.0
        var weightedScore = 0.0
        var components: [HealthComponentSnapshot] = []

        for calculator in calculators {
            guard let reading = signals.reading(for: calculator.metricType) else {
                continue
            }

            let isStale: Bool
            if let sampledAt = reading.sampledAt {
                isStale = context.timestamp.timeIntervalSince(sampledAt) > config.staleThreshold(for: calculator.metricType)
            } else {
                isStale = false
            }

            let snapshot = calculator.makeSnapshot(
                reading: reading,
                context: context,
                config: config,
                isStale: isStale
            )
            components.append(snapshot)

            guard !snapshot.isStale else {
                continue
            }

            let weight = config.weight(for: calculator.metricType)
            availableWeight += weight
            weightedScore += (snapshot.componentScore * weight)
        }

        let readinessScore: Int?
        if availableWeight > 0 {
            readinessScore = Int((weightedScore / availableWeight).rounded())
        } else {
            readinessScore = nil
        }

        let confidence = totalWeight > 0 ? clamp(availableWeight / totalWeight, min: 0, max: 1) : 0

        return HealthScoreResult(
            readinessScore: readinessScore,
            confidence: confidence,
            components: components.sorted { lhs, rhs in
                lhs.type.displayOrder < rhs.type.displayOrder
            }
        )
    }

    private static var defaultCalculators: [any HealthComponentCalculator] {
        [
            SleepHoursCalculator(),
            HRVCalculator(),
            RestingHeartRateCalculator(),
            StepsCalculator(),
        ]
    }
}

private struct SleepHoursCalculator: HealthComponentCalculator {
    let metricType: HealthMetricType = .sleepHours
    let unit = "h"

    func componentScore(for rawValue: Double, context _: HealthScoreContext, config: HealthScoreConfig) -> Double {
        let clipped = clamp(rawValue, min: 0, max: 16)

        if clipped <= config.sleepMinimumHours {
            return (clipped / max(config.sleepMinimumHours, 0.1)) * 20
        }

        let optimal = config.sleepOptimalHours
        let distance = abs(clipped - optimal)
        var score = 100 - (pow(distance, 1.35) * 14)

        if clipped > config.sleepMaximumHours {
            score -= pow(clipped - config.sleepMaximumHours, 1.2) * 8
        }

        return clamp(score, min: 0, max: 100)
    }
}

private struct HRVCalculator: HealthComponentCalculator {
    let metricType: HealthMetricType = .hrvSdnnMs
    let unit = "ms"

    func componentScore(for rawValue: Double, context _: HealthScoreContext, config: HealthScoreConfig) -> Double {
        let normalized = normalize(rawValue, in: config.hrvRangeMs)
        return pow(normalized, 0.75) * 100
    }
}

private struct RestingHeartRateCalculator: HealthComponentCalculator {
    let metricType: HealthMetricType = .restingHeartRateBpm
    let unit = "bpm"

    func componentScore(for rawValue: Double, context _: HealthScoreContext, config: HealthScoreConfig) -> Double {
        let inverse = normalize(config.restingHeartRateRangeBpm.upperBound - rawValue, in: 0 ... (config.restingHeartRateRangeBpm.upperBound - config.restingHeartRateRangeBpm.lowerBound))
        return pow(inverse, 0.85) * 100
    }
}

private struct StepsCalculator: HealthComponentCalculator {
    let metricType: HealthMetricType = .stepsToday
    let unit = "count"

    func componentScore(for rawValue: Double, context: HealthScoreContext, config: HealthScoreConfig) -> Double {
        let fraction = max(context.clampedElapsedDayFraction, config.stepsDayFractionFloor)
        let expectedByNow = max(1, config.stepsDailyGoal * fraction)
        let ratio = max(0, rawValue) / expectedByNow

        if ratio >= 1 {
            let bonus = min(20, (ratio - 1) * 20)
            return min(100, 80 + bonus)
        }

        return pow(ratio, 0.75) * 80
    }
}

private func normalize(_ value: Double, in range: ClosedRange<Double>) -> Double {
    guard range.upperBound > range.lowerBound else {
        return 0
    }
    let normalized = (value - range.lowerBound) / (range.upperBound - range.lowerBound)
    return clamp(normalized, min: 0, max: 1)
}

private func clamp(_ value: Double, min minValue: Double, max maxValue: Double) -> Double {
    Swift.max(minValue, Swift.min(maxValue, value))
}

private enum ISO8601HealthFormatters {
    static let withFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
