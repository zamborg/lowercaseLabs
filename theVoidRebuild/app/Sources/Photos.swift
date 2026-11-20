import CoreML
import Foundation
import ImageIO
import UniformTypeIdentifiers
import Vision

/// The photo part's files and its text for enrichment. No UI, no camera: the app and the Mac harness share this.
enum PhotoFiles {
    /// Longest side kept. A journal photo doesn't need 12 megapixels.
    static let maxPixels = 2048

    /// Re-encodes as a plain JPEG: orientation baked into the pixels, and every other piece of metadata (location,
    /// device, time) dropped, because nothing is copied across. Returns the file name inside the photos folder.
    static func store(_ data: Data, as name: String) throws -> String {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                  kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true,   // bakes in orientation
                  kCGImageSourceThumbnailMaxPixelSize: maxPixels,
              ] as CFDictionary)
        else { throw PhotoError.unreadable }
        let url = Entry.photoDirectory().appendingPathComponent(name)
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil)
        else { throw PhotoError.unwritable }
        CGImageDestinationAddImage(destination, image, [kCGImageDestinationLossyCompressionQuality: 0.85] as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw PhotoError.unwritable }
        return name
    }

    static func remove(_ names: [String]) {
        for name in names where !name.isEmpty {
            try? FileManager.default.removeItem(at: Entry.photoDirectory().appendingPathComponent(name))
        }
    }

    /// Apple's on-device scene classifier, run on the CPU so it can never touch the GPU while the app is inactive
    /// (the lesson from the tagger). Labels from both images, most confident first, duplicates dropped.
    static func labels(for urls: [URL]) -> String {
        var best: [String: Float] = [:]
        for url in urls {
            let request = VNClassifyImageRequest()
            if let cpu = MLComputeDevice.allComputeDevices.first(where: { if case .cpu = $0 { true } else { false } }) {
                request.setComputeDevice(cpu, for: .main)
            }
            try? VNImageRequestHandler(url: url).perform([request])
            for o in request.results ?? [] where o.confidence >= 0.15 {
                best[o.identifier, default: 0] = max(best[o.identifier] ?? 0, o.confidence)
            }
        }
        let top = best.sorted { $0.value > $1.value }.prefix(6).map { $0.key.replacingOccurrences(of: "_", with: " ") }
        return top.isEmpty ? "nothing recognizable" : top.joined(separator: ", ")
    }
}

enum PhotoError: Error { case unreadable, unwritable }
