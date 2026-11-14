import Foundation
import MCP
import MLX
import MLXToolKit
import MLXZImage
import System

// parlor-imagegen <snapshot-dir> [bf16|int8|int4]
//
// An MCP server over stdio with one tool, `generate`. The app mounts it like any other server; an external
// client can too. Nothing the libraries print may reach stdout, so the protocol channel is a dup of fd 1 and
// fd 1 itself is pointed at /dev/null. Logs go to stderr (the app keeps them in logs/image.log).

enum Loaded { case turbo(ZImageTurboT2IPackage), base(ZImageT2IPackage) }

actor Generator {
    let snapshot: String
    let quant: Quant
    let isTurbo: Bool
    var loaded: Loaded?

    init(snapshot: String, quant: Quant) {
        self.snapshot = snapshot; self.quant = quant
        self.isTurbo = snapshot.lowercased().contains("turbo")
    }

    private func ensureLoaded() async throws -> Loaded {
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

    /// PNG bytes.
    func generate(prompt: String, width: Int, height: Int, seed: UInt64, steps: Int?) async throws -> Data {
        let l = try await ensureLoaded()
        let request = T2IRequest(prompt: prompt, width: width, height: height, steps: steps, seed: seed)
        // Bind a progress sink the way the library's own CLI does; without one the run crashes at start.
        let response: T2IResponse = try await RunProgress.$sink.withValue({ _ in }) {
            switch l {
            case .turbo(let p): return try await p.run(request) as! T2IResponse
            case .base(let p): return try await p.run(request) as! T2IResponse
            }
        }
        return response.image.data
    }
}

func log(_ s: String) { FileHandle.standardError.write(Data((s + "\n").utf8)) }

@main
struct Worker {
    static func main() async {
        let proto = dup(1)
        let devnull = open("/dev/null", O_WRONLY)
        dup2(devnull, 1)

        let args = CommandLine.arguments
        guard args.count >= 2 else {
            log("usage: parlor-imagegen <snapshot-dir> [bf16|int8|int4]")
            exit(2)
        }
        let quant: Quant = {
            switch args.count > 2 ? args[2] : "int8" {
            case "bf16": return .bf16
            case "int4": return .int4
            default: return .int8
            }
        }()
        let gen = Generator(snapshot: args[1], quant: quant)
        log("parlor-imagegen ready: \(args[1])")

        let schema: Value = .object([
            "type": .string("object"),
            "properties": .object([
                "prompt": .object(["type": .string("string"), "description": .string("What to draw.")]),
                "width": .object(["type": .string("integer"), "description": .string("Pixels, a multiple of 64. Default 768.")]),
                "height": .object(["type": .string("integer"), "description": .string("Pixels, a multiple of 64. Default 768.")]),
                "seed": .object(["type": .string("integer"), "description": .string("Optional seed for repeatable results.")]),
                "steps": .object(["type": .string("integer"), "description": .string("Denoising steps. Omit for the model's default.")]),
            ]),
            "required": .array([.string("prompt")]),
        ])
        let tool = Tool(name: "generate", description: "Generate an image from a text prompt with the local image model. Returns a PNG.", inputSchema: schema)

        let server = Server(name: "parlor-imagegen", version: "0.2.0", capabilities: .init(tools: .init(listChanged: false)))
        await server.withMethodHandler(ListTools.self) { _ in .init(tools: [tool]) }
        await server.withMethodHandler(CallTool.self) { params in
            guard params.name == "generate" else {
                return .init(content: [.text(text: "Unknown tool \(params.name)", annotations: nil, _meta: nil)], isError: true)
            }
            let a = params.arguments ?? [:]
            guard let prompt = a["prompt"]?.stringValue, !prompt.isEmpty else {
                return .init(content: [.text(text: "Missing prompt", annotations: nil, _meta: nil)], isError: true)
            }
            let width = a["width"]?.intValue ?? 768
            let height = a["height"]?.intValue ?? 768
            let seed = a["seed"]?.intValue ?? Int.random(in: 0..<100_000)
            let steps = a["steps"]?.intValue.flatMap { $0 > 0 ? $0 : nil }
            log("generate \(width)x\(height) seed \(seed) steps \(steps.map(String.init) ?? "default")")
            do {
                let png = try await gen.generate(prompt: prompt, width: width, height: height, seed: UInt64(seed), steps: steps)
                return .init(content: [
                    .image(data: png.base64EncodedString(), mimeType: "image/png", annotations: nil, _meta: nil),
                    .text(text: "Generated \(width)x\(height) image, seed \(seed).", annotations: nil, _meta: nil),
                ], isError: false)
            } catch {
                log("generate failed: \(error)")
                return .init(content: [.text(text: "Image generation failed: \(error)", annotations: nil, _meta: nil)], isError: true)
            }
        }
        do {
            try await server.start(transport: StdioTransport(input: FileDescriptor.standardInput, output: FileDescriptor(rawValue: proto)))
            await server.waitUntilCompleted()
        } catch {
            log("server error: \(error)")
            exit(1)
        }
    }
}
