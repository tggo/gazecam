import AVFoundation
import CoreImage
import CoreMedia
import QuartzCore

/// Physical camera → scale to 1280x720 → effects → GazeCam sink (+ latest frames for the preview window).
final class CameraPipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    static let width = 1280, height = 720

    let effects: [Effect] = [GazeEffect(), HornsEffect(), MirrorEffect()]
    var gaze: GazeEffect { effects.first { $0 is GazeEffect } as! GazeEffect }

    /// Snapshot for the preview window. Written on the capture queue, read on main.
    struct Stats {
        var fps: Double = 0
        var frameMs: Double = 0
        var effectMs: [String: Double] = [:]
        var sinkConnected = false
        var inputSize = CGSize.zero
        var dropped = 0
        var processed: CVPixelBuffer?
        var raw: CVPixelBuffer?
    }

    /// Keep an unprocessed copy of each frame for the before/after view (costs one extra render).
    var keepRaw = false
    let capture = DebugCapture()

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let queue = DispatchQueue(label: "gazecam.capture")
    private let ci = CIContext(options: [.cacheIntermediates: false])
    private let sink = SinkWriter()
    private var pool: CVPixelBufferPool?
    private var format: CMFormatDescription?
    private var lastConnectTry = Date.distantPast
    private var frameTimes: [CFTimeInterval] = []

    private let statsLock = NSLock()
    private var _stats = Stats()
    var stats: Stats { statsLock.lock(); defer { statsLock.unlock() }; return _stats }

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
            statsLock.lock(); _stats = Stats(); statsLock.unlock()
            onStatus?("Зупинено")
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        statsLock.lock(); _stats.dropped += 1; statsLock.unlock()
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        let t0 = CACurrentMediaTime()
        // Keep processing for the preview even when the virtual camera is not there yet.
        if !sink.connected, Date().timeIntervalSince(lastConnectTry) > 2 {
            lastConnectTry = Date()
            onStatus?(sink.connect() ? "Транслює у GazeCam" : "Віртуальна камера не знайдена — встанови розширення")
        }
        guard let src = CMSampleBufferGetImageBuffer(sampleBuffer), let pool else { return }
        var dst: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &dst) == kCVReturnSuccess, let dst else { return }

        // Aspect-fill into 1280x720.
        var img = CIImage(cvPixelBuffer: src)
        let inputSize = img.extent.size
        let s = max(CGFloat(Self.width) / img.extent.width, CGFloat(Self.height) / img.extent.height)
        img = img.transformed(by: CGAffineTransform(scaleX: s, y: s))
        img = img.transformed(by: CGAffineTransform(translationX: (CGFloat(Self.width) - img.extent.width) / 2 - img.extent.minX,
                                                    y: (CGFloat(Self.height) - img.extent.height) / 2 - img.extent.minY))
        if effects.contains(where: { $0.id == "mirror" && $0.enabled }) {
            img = img.transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -CGFloat(Self.width), y: 0))
        }
        let bounds = CGRect(x: 0, y: 0, width: Self.width, height: Self.height)
        ci.render(img, to: dst, bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())

        var raw: CVPixelBuffer?
        if keepRaw || capture.wantsFrames, CVPixelBufferPoolCreatePixelBuffer(nil, pool, &raw) == kCVReturnSuccess, let raw {
            ci.render(img, to: raw, bounds: bounds, colorSpace: CGColorSpaceCreateDeviceRGB())
        }

        var effectMs: [String: Double] = [:]
        for e in effects where e.enabled {
            let te = CACurrentMediaTime()
            e.apply(to: dst)
            effectMs[e.id] = (CACurrentMediaTime() - te) * 1000
        }

        if sink.connected {
            if format == nil {
                CMVideoFormatDescriptionCreateForImageBuffer(allocator: nil, imageBuffer: dst, formatDescriptionOut: &format)
            }
            var timing = CMSampleTimingInfo(duration: .invalid, presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(sampleBuffer), decodeTimeStamp: .invalid)
            var out: CMSampleBuffer?
            if let format,
               CMSampleBufferCreateReadyWithImageBuffer(allocator: nil, imageBuffer: dst, formatDescription: format,
                                                        sampleTiming: &timing, sampleBufferOut: &out) == noErr, let out {
                sink.enqueue(out)
            }
        }

        let now = CACurrentMediaTime()
        frameTimes.append(now)
        frameTimes.removeAll { now - $0 > 1 }
        if let raw, capture.wantsFrames {
            var st = stats; st.fps = Double(frameTimes.count); st.frameMs = (now - t0) * 1000; st.effectMs = effectMs
            capture.append(raw: raw, processed: dst, time: CMSampleBufferGetPresentationTimeStamp(sampleBuffer), debug: gaze.debug, stats: st)
        }
        statsLock.lock()
        _stats.fps = Double(frameTimes.count)
        _stats.frameMs = (now - t0) * 1000
        _stats.effectMs = effectMs
        _stats.sinkConnected = sink.connected
        _stats.inputSize = inputSize
        _stats.processed = dst
        _stats.raw = raw
        statsLock.unlock()
    }
}
