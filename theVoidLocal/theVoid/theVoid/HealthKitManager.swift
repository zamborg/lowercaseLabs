import Foundation

#if canImport(HealthKit)
import HealthKit
#endif

enum HealthAuthorizationState: String, Codable, Hashable {
    case unavailable
    case notDetermined
    case denied
    case authorizedPartial
    case authorizedAll

    var isAuthorized: Bool {
        self == .authorizedPartial || self == .authorizedAll
    }

    var displayLabel: String {
        switch self {
        case .unavailable:
            return "Unavailable"
        case .notDetermined:
            return "Not Connected"
        case .denied:
            return "Denied"
        case .authorizedPartial:
            return "Partially Connected"
        case .authorizedAll:
            return "Connected"
        }
    }
}

final class HealthKitManager {
    private let scoreEngine: HealthScoreEngine
    private let calendar: Calendar

#if canImport(HealthKit)
    private let healthStore = HKHealthStore()
#endif

    init(scoreEngine: HealthScoreEngine = HealthScoreEngine(), calendar: Calendar = .current) {
        self.scoreEngine = scoreEngine
        self.calendar = calendar
    }

    func isHealthDataAvailable() -> Bool {
#if canImport(HealthKit)
        return HKHealthStore.isHealthDataAvailable()
#else
        return false
#endif
    }

    func authorizationStatus() async -> HealthAuthorizationState {
#if canImport(HealthKit)
        guard HKHealthStore.isHealthDataAvailable() else {
            return .unavailable
        }

        let probeStates = await probeReadAuthorizationStates(asOf: Date())
        let authorizedCount = probeStates.filter { $0 == .authorizedAll }.count
        if authorizedCount > 0 {
            return authorizedCount == probeStates.count ? .authorizedAll : .authorizedPartial
        }

        if probeStates.contains(.denied) {
            return .denied
        }

        return .notDetermined
#else
        return .unavailable
#endif
    }

    func requestReadAuthorization() async throws -> HealthAuthorizationState {
#if canImport(HealthKit)
        guard HKHealthStore.isHealthDataAvailable() else {
            return .unavailable
        }

        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            healthStore.requestAuthorization(toShare: nil, read: Set(requestedReadTypes)) { success, error in
                _ = success
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: ())
            }
        }
        return await authorizationStatus()
#else
        return .unavailable
#endif
    }

    func fetchSnapshot(at timestamp: Date) async throws -> EntryHealthSnapshot? {
#if canImport(HealthKit)
        guard HKHealthStore.isHealthDataAvailable() else {
            return nil
        }

        async let sleepReading = fetchSleepReading(asOf: timestamp)
        async let hrvReading = fetchLatestQuantityReading(
            identifier: .heartRateVariabilitySDNN,
            unit: HKUnit.secondUnit(with: .milli),
            lookbackHours: 36,
            asOf: timestamp
        )
        async let restingHeartRateReading = fetchLatestQuantityReading(
            identifier: .restingHeartRate,
            unit: HKUnit.count().unitDivided(by: .minute()),
            lookbackHours: 36,
            asOf: timestamp
        )
        async let stepsReading = fetchStepsReading(asOf: timestamp)

        var readings: [HealthMetricType: HealthMetricReading] = [:]

        if let sleepReading = try? await sleepReading {
            readings[.sleepHours] = sleepReading
        }
        if let hrvReading = try? await hrvReading {
            readings[.hrvSdnnMs] = hrvReading
        }
        if let restingHeartRateReading = try? await restingHeartRateReading {
            readings[.restingHeartRateBpm] = restingHeartRateReading
        }
        if let stepsReading = try? await stepsReading {
            readings[.stepsToday] = stepsReading
        }

        guard !readings.isEmpty else {
            return nil
        }

        let dayStart = calendar.startOfDay(for: timestamp)
        let nextDay = calendar.date(byAdding: .day, value: 1, to: dayStart) ?? dayStart.addingTimeInterval(86_400)
        let elapsedDayFraction = max(0, min(1, timestamp.timeIntervalSince(dayStart) / max(1, nextDay.timeIntervalSince(dayStart))))

        let scoreResult = scoreEngine.evaluate(
            signals: HealthScoreSignals(readings: readings),
            context: HealthScoreContext(timestamp: timestamp, elapsedDayFraction: elapsedDayFraction)
        )

        guard !scoreResult.components.isEmpty else {
            return nil
        }

        return EntryHealthSnapshot(
            capturedAtISO8601: ISO8601Formatters.withFractional.string(from: timestamp),
            readinessScore: scoreResult.readinessScore,
            confidence: scoreResult.confidence,
            components: scoreResult.components,
            version: 1
        )
#else
        return nil
#endif
    }
}

#if canImport(HealthKit)
private extension HealthKitManager {
    var requestedReadTypes: [HKObjectType] {
        [
            HKObjectType.categoryType(forIdentifier: .sleepAnalysis),
            HKObjectType.quantityType(forIdentifier: .heartRateVariabilitySDNN),
            HKObjectType.quantityType(forIdentifier: .restingHeartRate),
            HKObjectType.quantityType(forIdentifier: .stepCount),
        ].compactMap { $0 }
    }

    var requestedReadSampleTypes: [HKSampleType] {
        requestedReadTypes.compactMap { $0 as? HKSampleType }
    }

    func probeReadAuthorizationStates(asOf timestamp: Date) async -> [HealthAuthorizationState] {
        guard !requestedReadSampleTypes.isEmpty else {
            return [.notDetermined]
        }

        var states: [HealthAuthorizationState] = []
        states.reserveCapacity(requestedReadSampleTypes.count)
        for sampleType in requestedReadSampleTypes {
            let state = await probeReadAuthorization(for: sampleType, asOf: timestamp)
            states.append(state)
        }
        return states
    }

    func probeReadAuthorization(
        for sampleType: HKSampleType,
        asOf timestamp: Date
    ) async -> HealthAuthorizationState {
        let lookbackStart = timestamp.addingTimeInterval(-(12 * 3600))
        let predicate = HKQuery.predicateForSamples(withStart: lookbackStart, end: timestamp, options: [])

        do {
            _ = try await sampleQuery(
                sampleType: sampleType,
                predicate: predicate,
                limit: 1
            )
            return .authorizedAll
        } catch {
            if let authorizationErrorState = authorizationErrorState(for: error) {
                return authorizationErrorState
            }
            return .notDetermined
        }
    }

    func authorizationErrorState(for error: Error) -> HealthAuthorizationState? {
        let nsError = error as NSError
        guard nsError.domain == HKErrorDomain,
              let code = HKError.Code(rawValue: nsError.code) else {
            return nil
        }

        switch code {
        case .errorAuthorizationDenied:
            return .denied
        case .errorAuthorizationNotDetermined:
            return .notDetermined
        default:
            return nil
        }
    }

    func isReadAuthorizationError(_ error: Error) -> Bool {
        authorizationErrorState(for: error) != nil
    }

    func fetchSleepReading(asOf timestamp: Date) async throws -> HealthMetricReading? {
        guard let sleepType = HKObjectType.categoryType(forIdentifier: .sleepAnalysis) else {
            return nil
        }

        let dayStart = calendar.startOfDay(for: timestamp)
        let previousDay = calendar.date(byAdding: .day, value: -1, to: dayStart) ?? dayStart.addingTimeInterval(-86_400)
        let sleepWindowStart = calendar.date(bySettingHour: 15, minute: 0, second: 0, of: previousDay) ?? previousDay
        let predicate = HKQuery.predicateForSamples(withStart: sleepWindowStart, end: timestamp, options: [])

        let sortByStart = NSSortDescriptor(key: HKSampleSortIdentifierStartDate, ascending: true)
        let samples: [HKSample]
        do {
            samples = try await sampleQuery(
                sampleType: sleepType,
                predicate: predicate,
                limit: HKObjectQueryNoLimit,
                sortDescriptors: [sortByStart]
            )
        } catch {
            if isReadAuthorizationError(error) {
                return nil
            }
            throw error
        }

        var totalAsleepSeconds: TimeInterval = 0
        var latestSampleDate: Date?

        for sample in samples {
            guard let categorySample = sample as? HKCategorySample else {
                continue
            }
            guard isAsleepSleepAnalysisValue(categorySample.value) else {
                continue
            }

            let overlapStart = max(categorySample.startDate, sleepWindowStart)
            let overlapEnd = min(categorySample.endDate, timestamp)
            guard overlapEnd > overlapStart else {
                continue
            }

            totalAsleepSeconds += overlapEnd.timeIntervalSince(overlapStart)
            if latestSampleDate == nil || overlapEnd > (latestSampleDate ?? .distantPast) {
                latestSampleDate = overlapEnd
            }
        }

        guard totalAsleepSeconds > 0 else {
            return nil
        }

        return HealthMetricReading(
            rawValue: totalAsleepSeconds / 3600,
            sampledAt: latestSampleDate
        )
    }

    func fetchLatestQuantityReading(
        identifier: HKQuantityTypeIdentifier,
        unit: HKUnit,
        lookbackHours: Double,
        asOf timestamp: Date
    ) async throws -> HealthMetricReading? {
        guard let quantityType = HKObjectType.quantityType(forIdentifier: identifier) else {
            return nil
        }

        let start = timestamp.addingTimeInterval(-(lookbackHours * 3600))
        let predicate = HKQuery.predicateForSamples(withStart: start, end: timestamp, options: [.strictEndDate])
        let sortByEndDate = NSSortDescriptor(key: HKSampleSortIdentifierEndDate, ascending: false)

        let samples: [HKSample]
        do {
            samples = try await sampleQuery(
                sampleType: quantityType,
                predicate: predicate,
                limit: 1,
                sortDescriptors: [sortByEndDate]
            )
        } catch {
            if isReadAuthorizationError(error) {
                return nil
            }
            throw error
        }

        guard let quantitySample = samples.first as? HKQuantitySample else {
            return nil
        }

        return HealthMetricReading(
            rawValue: quantitySample.quantity.doubleValue(for: unit),
            sampledAt: quantitySample.endDate
        )
    }

    func fetchStepsReading(asOf timestamp: Date) async throws -> HealthMetricReading? {
        guard let stepsType = HKObjectType.quantityType(forIdentifier: .stepCount) else {
            return nil
        }

        let dayStart = calendar.startOfDay(for: timestamp)
        let predicate = HKQuery.predicateForSamples(withStart: dayStart, end: timestamp, options: [.strictStartDate])
        let stats: HKStatistics?
        do {
            stats = try await statisticsQuery(
                quantityType: stepsType,
                predicate: predicate,
                options: .cumulativeSum
            )
        } catch {
            if isReadAuthorizationError(error) {
                return nil
            }
            throw error
        }

        guard let sum = stats?.sumQuantity()?.doubleValue(for: .count()) else {
            return nil
        }

        return HealthMetricReading(rawValue: sum, sampledAt: timestamp)
    }

    func sampleQuery(
        sampleType: HKSampleType,
        predicate: NSPredicate?,
        limit: Int,
        sortDescriptors: [NSSortDescriptor]? = nil
    ) async throws -> [HKSample] {
        try await withCheckedThrowingContinuation { continuation in
            let query = HKSampleQuery(
                sampleType: sampleType,
                predicate: predicate,
                limit: limit,
                sortDescriptors: sortDescriptors
            ) { _, results, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: results ?? [])
            }
            healthStore.execute(query)
        }
    }

    func statisticsQuery(
        quantityType: HKQuantityType,
        predicate: NSPredicate?,
        options: HKStatisticsOptions
    ) async throws -> HKStatistics? {
        try await withCheckedThrowingContinuation { continuation in
            let query = HKStatisticsQuery(
                quantityType: quantityType,
                quantitySamplePredicate: predicate,
                options: options
            ) { _, result, error in
                if let error {
                    continuation.resume(throwing: error)
                    return
                }
                continuation.resume(returning: result)
            }
            healthStore.execute(query)
        }
    }

    func isAsleepSleepAnalysisValue(_ rawValue: Int) -> Bool {
        rawValue == HKCategoryValueSleepAnalysis.asleepUnspecified.rawValue
            || rawValue == HKCategoryValueSleepAnalysis.asleepCore.rawValue
            || rawValue == HKCategoryValueSleepAnalysis.asleepDeep.rawValue
            || rawValue == HKCategoryValueSleepAnalysis.asleepREM.rawValue
    }
}
#endif

private enum ISO8601Formatters {
    static let withFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()
}
