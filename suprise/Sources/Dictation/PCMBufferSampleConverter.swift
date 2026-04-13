import AVFoundation
import Foundation

final class PCMBufferSampleConverter {
    private let outputFormat = AVAudioFormat(
        commonFormat: .pcmFormatFloat32,
        sampleRate: 16_000,
        channels: 1,
        interleaved: false
    )!

    private var converter: AVAudioConverter?
    private var inputSignature: AudioFormatSignature?

    func samples(from buffer: AVAudioPCMBuffer) throws -> [Float] {
        if matchesOutputFormat(buffer.format) {
            return readFloatSamples(from: buffer)
        }

        let signature = AudioFormatSignature(format: buffer.format)
        if inputSignature != signature {
            converter = AVAudioConverter(from: buffer.format, to: outputFormat)
            inputSignature = signature
        }

        guard let converter else {
            throw DictationError.audioCaptureFailed("Unable to prepare audio conversion for dictation.")
        }

        let sampleRateRatio = outputFormat.sampleRate / buffer.format.sampleRate
        let expectedFrames = max(1, Int(ceil(Double(buffer.frameLength) * sampleRateRatio)))

        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: AVAudioFrameCount(expectedFrames)
        ) else {
            throw DictationError.audioCaptureFailed("Unable to allocate audio conversion buffer.")
        }

        var didProvideInput = false
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outputStatus in
            if didProvideInput {
                outputStatus.pointee = .noDataNow
                return nil
            }

            didProvideInput = true
            outputStatus.pointee = .haveData
            return buffer
        }

        if let conversionError {
            throw DictationError.audioCaptureFailed(conversionError.localizedDescription)
        }

        guard status != .error else {
            throw DictationError.audioCaptureFailed("Audio conversion failed while streaming dictation.")
        }

        return readFloatSamples(from: outputBuffer)
    }

    private func matchesOutputFormat(_ format: AVAudioFormat) -> Bool {
        format.sampleRate == outputFormat.sampleRate
            && format.channelCount == outputFormat.channelCount
            && format.commonFormat == outputFormat.commonFormat
            && format.isInterleaved == outputFormat.isInterleaved
    }

    private func readFloatSamples(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData?.pointee else {
            return []
        }

        let frameCount = Int(buffer.frameLength)
        return Array(UnsafeBufferPointer(start: channelData, count: frameCount))
    }
}

private struct AudioFormatSignature: Equatable {
    let sampleRate: Double
    let channelCount: AVAudioChannelCount
    let commonFormat: AVAudioCommonFormat
    let isInterleaved: Bool

    init(format: AVAudioFormat) {
        sampleRate = format.sampleRate
        channelCount = format.channelCount
        commonFormat = format.commonFormat
        isInterleaved = format.isInterleaved
    }
}
