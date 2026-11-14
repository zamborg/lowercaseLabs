import CoreServices
import Foundation

/// FSEvents wrapper: calls `onChange` with changed paths, debounced.
final class FolderWatcher: @unchecked Sendable {
    private var stream: FSEventStreamRef?
    private let handler: ([String]) -> Void
    private let queue = DispatchQueue(label: "parlor.watcher")

    init(path: URL, onChange: @escaping ([String]) -> Void) {
        handler = onChange
        var ctx = FSEventStreamContext(version: 0, info: Unmanaged.passUnretained(self).toOpaque(), retain: nil, release: nil, copyDescription: nil)
        let cb: FSEventStreamCallback = { _, info, count, paths, _, _ in
            guard let info else { return }
            let me = Unmanaged<FolderWatcher>.fromOpaque(info).takeUnretainedValue()
            let arr = unsafeBitCast(paths, to: NSArray.self) as? [String] ?? []
            me.handler(Array(arr.prefix(count)))
        }
        stream = FSEventStreamCreate(
            nil, cb, &ctx, [path.path] as CFArray, FSEventStreamEventId(kFSEventStreamEventIdSinceNow),
            0.3, FSEventStreamCreateFlags(kFSEventStreamCreateFlagFileEvents | kFSEventStreamCreateFlagUseCFTypes | kFSEventStreamCreateFlagNoDefer))
        if let stream {
            FSEventStreamSetDispatchQueue(stream, queue)
            FSEventStreamStart(stream)
        }
    }

    func stop() {
        if let s = stream { FSEventStreamStop(s); FSEventStreamInvalidate(s); FSEventStreamRelease(s) }
        stream = nil
    }

    deinit { stop() }
}
