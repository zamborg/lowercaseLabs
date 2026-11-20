import AVFoundation
import SwiftUI
import UIKit

/// The two images of a photo part, as the camera produced them.
struct Shot { let back: Data; let front: Data? }

/// The camera, back and front at once. Three rules, all from AGENTS.md priorities:
/// - It runs only while its page is showing and the app is active (the page calls `start`/`stop`, the scene phase calls `stop`).
/// - It has no audio input and never touches the audio session, so it can't break touch-to-record on Capture.
/// - It never saves anything itself: it hands back bytes; `Journal` stores them (metadata stripped) and saves the message.
///
/// `-fixtureCamera` is the test edge (D12): no hardware, fixed images, so flows run on the simulator and the Mac.
@MainActor @Observable
final class Camera: NSObject {
    enum State: Equatable { case off, starting, running, unavailable(String) }
    private(set) var state: State = .off
    private(set) var hasFront = false
    /// Measured, for the phone check and the Show timings setting.
    private(set) var arriveMs: Double?
    private(set) var snapGapMs: Double?

    let fixture = ProcessInfo.processInfo.arguments.contains("-fixtureCamera")

    let session: AVCaptureSession
    let backLayer: AVCaptureVideoPreviewLayer
    let frontLayer: AVCaptureVideoPreviewLayer
    private let backOutput = AVCapturePhotoOutput()
    private let frontOutput = AVCapturePhotoOutput()
    private let queue = DispatchQueue(label: "void.camera")
    private var configuredWithFront: Bool?
    private var grabs = Set<PhotoGrab>()

    override init() {
        session = AVCaptureMultiCamSession.isMultiCamSupported ? AVCaptureMultiCamSession() : AVCaptureSession()
        session.automaticallyConfiguresApplicationAudioSession = false   // no audio input, hands off the audio session
        backLayer = AVCaptureVideoPreviewLayer(sessionWithNoConnection: session)
        frontLayer = AVCaptureVideoPreviewLayer(sessionWithNoConnection: session)
        super.init()
        for layer in [backLayer, frontLayer] { layer.videoGravity = .resizeAspectFill }
    }

    // MARK: Lifecycle

    func start() async {
        guard state != .running, state != .starting else { return }
        let began = CFAbsoluteTimeGetCurrent()
        if fixture { hasFront = !Settings.cameraBackOnly; state = .running; arriveMs = 0; return }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .notDetermined: if !(await AVCaptureDevice.requestAccess(for: .video)) { fallthrough }
        case .denied, .restricted:
            state = .unavailable("Camera access is off. Allow it in Settings → The Void."); return
        default: break
        }
        state = .starting
        guard configure() else { state = .unavailable("This iPhone can't run the camera here."); return }
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            queue.async { [session] in session.startRunning(); done.resume() }
        }
        state = session.isRunning ? .running : .unavailable("The camera didn't start.")
        arriveMs = (CFAbsoluteTimeGetCurrent() - began) * 1000
    }

    func stop() {
        guard state != .off else { return }
        state = .off
        if fixture { return }
        queue.async { [session] in session.stopRunning() }
    }

    // MARK: Snap

    /// Both images at the moment of the call. Back-only when the setting says so or only one camera can run.
    func snap() async -> Shot? {
        if fixture { return Shot(back: Self.fixtureJPEG(.back), front: hasFront ? Self.fixtureJPEG(.front) : nil) }
        guard state == .running else { return nil }
        async let back = grab(backOutput)
        async let front: PhotoGrab.Result? = hasFront ? grab(frontOutput) : nil
        guard let b = await back else { return nil }
        let f = await front
        if let f { snapGapMs = abs(CMTimeGetSeconds(b.time) - CMTimeGetSeconds(f.time)) * 1000 }
        return Shot(back: b.data, front: f?.data)
    }

    private func grab(_ output: AVCapturePhotoOutput) async -> PhotoGrab.Result? {
        let settings = AVCapturePhotoSettings()
        settings.photoQualityPrioritization = .speed
        return await withCheckedContinuation { continuation in
            let grab = PhotoGrab { [weak self] result, grab in
                Task { @MainActor in self?.grabs.remove(grab) }
                continuation.resume(returning: result)
            }
            grabs.insert(grab)
            output.capturePhoto(with: settings, delegate: grab)
        }
    }

    // MARK: Session wiring

    private func configure() -> Bool {
        let wantFront = session is AVCaptureMultiCamSession && !Settings.cameraBackOnly
        if configuredWithFront == wantFront { return true }
        session.beginConfiguration()
        for input in session.inputs { session.removeInput(input) }
        for output in session.outputs { session.removeOutput(output) }
        for connection in session.connections { session.removeConnection(connection) }
        let ok = add(.back, to: backOutput, layer: backLayer) && (!wantFront || add(.front, to: frontOutput, layer: frontLayer))
        session.commitConfiguration()
        if let multi = session as? AVCaptureMultiCamSession, multi.hardwareCost > 1 || multi.systemPressureCost > 1 {
            hasFront = false; configuredWithFront = nil; return false
        }
        hasFront = wantFront && ok
        configuredWithFront = ok ? wantFront : nil
        return ok
    }

    private func add(_ position: AVCaptureDevice.Position, to output: AVCapturePhotoOutput, layer: AVCaptureVideoPreviewLayer) -> Bool {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position),
              let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else { return false }
        if session is AVCaptureMultiCamSession, let format = Self.multiCamFormat(device) {
            try? device.lockForConfiguration(); device.activeFormat = format; device.unlockForConfiguration()
        }
        session.addInputWithNoConnections(input)
        session.addOutputWithNoConnections(output)
        guard let port = input.ports(for: .video, sourceDeviceType: device.deviceType, sourceDevicePosition: position).first else { return false }
        let capture = AVCaptureConnection(inputPorts: [port], output: output)
        let preview = AVCaptureConnection(inputPort: port, videoPreviewLayer: layer)
        guard session.canAddConnection(capture), session.canAddConnection(preview) else { return false }
        session.addConnection(capture); session.addConnection(preview)
        for connection in [capture, preview] where connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }   // portrait
        if position == .front { preview.automaticallyAdjustsVideoMirroring = false; preview.isVideoMirrored = true }
        return true
    }

    /// The format nearest 1080p that can run two cameras at once; the full-resolution formats can't.
    private static func multiCamFormat(_ device: AVCaptureDevice) -> AVCaptureDevice.Format? {
        let area = { (f: AVCaptureDevice.Format) -> Int in let d = f.formatDescription.dimensions; return Int(d.width) * Int(d.height) }
        return device.formats.filter(\.isMultiCamSupported).min { abs(area($0) - 1920 * 1080) < abs(area($1) - 1920 * 1080) }
    }

    // MARK: Fixtures (-fixtureCamera)

    enum Side { case back, front }
    static func fixtureJPEG(_ side: Side) -> Data {
        let size = side == .back ? CGSize(width: 1500, height: 2000) : CGSize(width: 1200, height: 1600)
        return UIGraphicsImageRenderer(size: size).jpegData(withCompressionQuality: 0.8) { ctx in
            let colors = side == .back ? [UIColor(red: 0.35, green: 0.65, blue: 0.95, alpha: 1), UIColor(red: 0.95, green: 0.85, blue: 0.6, alpha: 1)]
                                       : [UIColor(red: 0.85, green: 0.5, blue: 0.5, alpha: 1), UIColor(red: 0.3, green: 0.2, blue: 0.4, alpha: 1)]
            let gradient = CGGradient(colorsSpace: nil, colors: colors.map(\.cgColor) as CFArray, locations: [0, 1])!
            ctx.cgContext.drawLinearGradient(gradient, start: .zero, end: CGPoint(x: 0, y: size.height), options: [])
            UIColor.white.setFill()
            ctx.cgContext.fillEllipse(in: CGRect(x: size.width * 0.3, y: size.height * 0.55, width: size.width * 0.4, height: size.width * 0.4))
        }
    }
}

/// One photo coming back from one output.
final class PhotoGrab: NSObject, AVCapturePhotoCaptureDelegate {
    struct Result { let data: Data; let time: CMTime }
    private let finish: (Result?, PhotoGrab) -> Void
    init(_ finish: @escaping (Result?, PhotoGrab) -> Void) { self.finish = finish }

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        finish(photo.fileDataRepresentation().map { Result(data: $0, time: photo.timestamp) }, self)
    }
}

/// The live view: back camera full-bleed, front camera inset in a corner (or flat colour stand-ins under -fixtureCamera).
struct CameraPreview: UIViewRepresentable {
    let camera: Camera
    func makeUIView(context: Context) -> PreviewHost { PreviewHost(camera) }
    func updateUIView(_ view: PreviewHost, context: Context) { view.sync() }
}

final class PreviewHost: UIView {
    private let camera: Camera
    private let back: CALayer, front: CALayer
    private var hasFront = false

    init(_ camera: Camera) {
        self.camera = camera
        back = camera.fixture ? Self.stand(UIColor(red: 0.2, green: 0.35, blue: 0.5, alpha: 1)) : camera.backLayer
        front = camera.fixture ? Self.stand(UIColor(red: 0.5, green: 0.3, blue: 0.35, alpha: 1)) : camera.frontLayer
        super.init(frame: .zero)
        backgroundColor = .black
        layer.addSublayer(back)
        front.cornerRadius = 14; front.masksToBounds = true; front.borderWidth = 2; front.borderColor = UIColor.white.withAlphaComponent(0.6).cgColor
        sync()
    }
    required init?(coder: NSCoder) { fatalError() }

    private static func stand(_ color: UIColor) -> CALayer { let l = CALayer(); l.backgroundColor = color.cgColor; return l }

    func sync() {
        if camera.hasFront != hasFront {
            hasFront = camera.hasFront
            if hasFront { layer.addSublayer(front) } else { front.removeFromSuperlayer() }
        }
        setNeedsLayout()
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        back.frame = bounds
        front.frame = CGRect(x: bounds.width - 118 - 16, y: safeAreaInsets.top + 12, width: 118, height: 158)
    }
}
