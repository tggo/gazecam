import AVFoundation
import CoreImage
import CoreMedia

/// Physical camera → scale to 1280x720 → effects → GazeCam sink.
final class CameraPipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    static let width = 1280, height = 720

    let effects: [Effect] = [GazeEffect(), HornsEffect(), MirrorEffect()]
    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "gazecam.capture")
    private let ci = CIContext(options: [.cacheIntermediates: false])
    private let sink = SinkWriter()
    private var pool: CVPixelBufferPool?
    private var format: CMFormatDescription?
    private var lastConnectTry = Date.distantPast

    var onStatus: ((String) -> Void)?

    static func cameras() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
                                         mediaType: .video, position: .unspecified)
            .devices.filter { !$0.localizedName.contains("GazeCam") }
    }

    override init() {
        super.init()
        let attrs: NSDictionary = [
            kCVPixelBufferWidthKey: Self.width, kCVPixelBufferHeightKey: Self.height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as NSDictionary,
        ]
        CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attrs, &pool)
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
    }

    func start(camera: AVCaptureDevice) {
        queue.async { [self] in
            session.beginConfiguration()
            session.inputs.forEach { session.removeInput($0) }
            if let input = try? AVCaptureDeviceInput(device: camera), session.canAddInput(input) { session.addInput(input) }
            if session.outputs.isEmpty, session.canAddOutput(output) { session.addOutput(output) }
            session.commitConfiguration()
            if !session.isRunning { session.startRunning() }
            onStatus?("Камера: \(camera.localizedName)")
        }
    }

    func stop() {
        queue.async { [self] in
            session.stopRunning()
            sink.disconnect()
            onStatus?("Зупинено")
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        if !sink.connected {
            guard Date().timeIntervalSince(lastConnectTry) > 2 else { return }
            lastConnectTry = Date()
            onStatus?(sink.connect() ? "Транслює у GazeCam" : "Віртуальна камера не знайдена — встанови розширення")
            if !sink.connected { return }
        }
        guard let src = CMSampleBufferGetImageBuffer(sampleBuffer), let pool else { return }
        var dst: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst) == kCVReturnSuccess, let dst else { return }

        // Aspect-fill into 1280x720.
        var img = CIImage(cvPixelBuffer: src)
        let s = max(CGFloat(Self.width) / img.extent.width, CGFloat(Self.height) / img.extent.height)
        img = img.transformed(by: CGAffineTransform(scaleX: s, y: s))
        img = img.transformed(by: CGAffineTransform(translationX: (CGFloat(Self.width) - img.extent.width) / 2 - img.extent.minX,
                                                    y: (CGFloat(Self.height) - img.extent.height) / 2 - img.extent.minY))
        if effects.contains(where: { $0.id == "mirror" && $0.enabled }) {
            img = img.transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -CGFloat(Self.width), y: 0))
        }
        ci.render(img, to: dst, bounds: CGRect(x: 0, y: 0, width: Self.width, height: Self.height), colorSpace: CGColorSpaceCreateDeviceRGB())

        for e in effects where e.enabled { e.apply(to: dst) }

        if format == nil {
            CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: dst, formatDescriptionOut: &format)
        }
        var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer), decodeTimeStamp: .invalid)
        var out: CMSampleBuffer?
        guard let format,
              CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: dst, formatDescription: format,
                                                       sampleTiming: &timing, sampleBufferOut: &out) == noErr, let out else { return }
        sink.enqueue(out)
    }
}
