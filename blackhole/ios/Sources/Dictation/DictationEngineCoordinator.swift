import Foundation

@MainActor
final class DictationEngineCoordinator {
    private var loadedKind: DictationEngineKind?
    private var loadedEngine: DictationEngine?

    func engine(for kind: DictationEngineKind) async throws -> DictationEngine {
        if let loadedEngine, loadedKind == kind { return loadedEngine }
        await unloadCurrentEngine()
        let engine = makeEngine(for: kind)
        try await engine.prepare()
        loadedEngine = engine
        loadedKind = kind
        return engine
    }

    func unloadCurrentEngine() async {
        guard let loadedEngine else { loadedKind = nil; return }
        await loadedEngine.unload()
        self.loadedEngine = nil
        loadedKind = nil
    }

    private func makeEngine(for kind: DictationEngineKind) -> DictationEngine {
        switch kind {
        case .apple: return AppleSpeechDictationEngine()
        case .whisperKit: return WhisperKitDictationEngine()
        }
    }
}
