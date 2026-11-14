import Foundation
import MLX
import MLXToolKit
import MLXZImage

// parlor-imagegen <snapshot-dir> [bf16|int8|int4]
// stdin : one JSON request per line  {"id","prompt","width","height","seed","steps"}
// stdout: one JSON reply per line    {"id","ok":true,"path"} | {"id","ok":false,"error"}
// Anything the libraries print is diverted away from the protocol channel.


enum Loaded { case turbo(ZImageTurboT2IPackage), base(ZImageT2IPackage) }

@main
struct Worker {
    static func main() async {
        setvbuf(stdout, nil, _IOLBF, 0)
        let proto = FileHandle(fileDescriptor: dup(1), closeOnDealloc: false)
        let devnull = open("/dev/null", O_WRONLY)
        dup2(devnull, 1)

        func reply(_ obj: [String: Any]) {
            guard var d = try? JSONSerialization.data(withJSONObject: obj) else { return }
            d.append(0x0A)
            try? proto.write(contentsOf: d)
        }

        let args = CommandLine.arguments
        guard args.count >= 2 else {
            FileHandle.standardError.write("usage: parlor-imagegen <snapshot-dir> [bf16|int8|int4]\n".data(using: .utf8)!)
            exit(2)
        }
        let snapshot = args[1]
        let quant: Quant = {
            switch args.count > 2 ? args[2] : "int8" {
            case "bf16": return .bf16
            case "int4": return .int4
            default: return .int8
            }
        }()
        let isTurbo = snapshot.lowercased().contains("turbo")

        var loaded: Loaded?

        func ensureLoaded() async throws -> Loaded {
            if let l = loaded { return l }
            let l: Loaded
            if isTurbo {
                let p = ZImageTurboT2IPackage(configuration: .turbo(quant: quant, snapshotPath: snapshot))
                try await p.load(); l = .turbo(p)
            } else {
                let p = ZImageT2IPackage(configuration: .base(quant: quant, snapshotPath: snapshot))
                try await p.load(); l = .base(p)
            }
            loaded = l
            return l
        }

        while let line = readLine() {
            guard let data = line.data(using: .utf8),
                  let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let id = obj["id"] as? String,
                  let prompt = obj["prompt"] as? String else { continue }
            let width = obj["width"] as? Int ?? 768
            let height = obj["height"] as? Int ?? 768
            let seed = UInt64(obj["seed"] as? Int ?? 42)
            let steps = (obj["steps"] as? Int).flatMap { $0 > 0 ? $0 : nil }
            do {
                let l = try await ensureLoaded()
                let request = T2IRequest(prompt: prompt, width: width, height: height, steps: steps, seed: seed)
                let response: T2IResponse
                // Bind a progress sink the way the library's own CLI does; without one the run crashes at start.
                response = try await RunProgress.$sink.withValue({ _ in }) {
                    switch l {
                    case .turbo(let p): return try await p.run(request) as! T2IResponse
                    case .base(let p): return try await p.run(request) as! T2IResponse
                    }
                }
                let path = NSTemporaryDirectory() + "parlor-img-\(UUID().uuidString).png"
                try response.image.data.write(to: URL(fileURLWithPath: path))
                reply(["id": id, "ok": true, "path": path])
            } catch {
                reply(["id": id, "ok": false, "error": "\(error)"])
            }
        }
    }
}
