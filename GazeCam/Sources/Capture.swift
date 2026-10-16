import AppKit
import AVFoundation
import CoreImage

/// Debug captures: original | result side by side, with eye boxes, landmarks and stats burned in.
/// Tap Space = PNG snapshot, hold Space = MP4 while held. Files go to ~/Movies/GazeCam.
final class DebugCapture {
    static let dir: URL = {
        let d = FileManager.default.urls(for: .moviesDirectory, in: .userDomainMask)[0].appendingPathComponent("GazeCam")
        try? FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
        return d
    }()

    static let width = CameraPipeline.width * 2, height = CameraPipeline.height

    private let ci = CIContext(options: [.cacheIntermediates: false])
    private var pool: CVPixelBufferPool?
    private let lock = NSLock()

    private var writer: AVAssetWriter?
    private var input: AVAssetWriterInput?
    private var adaptor: AVAssetWriterInputPixelBufferAdaptor?
    private var startTime: CMTime?
    private var snapshotURL: URL?
    private var snapshotDone: ((URL?) -> Void)?

    init() {
        let attrs: NSDictionary = [
            kCVPixelBufferWidthKey: Self.width, kCVPixelBufferHeightKey: Self.height,
            kCVPixelBufferPixelFormatTypeKey: kCVPixelFormatType_32BGRA,
            kCVPixelBufferIOSurfacePropertiesKey: [:] as NSDictionary,
        ]
        CVPixelBufferPoolCreate(nil, nil, attrs, &pool)
    }

    var isRecording: Bool { lock.lock(); defer { lock.unlock() }; return writer != nil }
    var wantsFrames: Bool { lock.lock(); defer { lock.unlock() }; return writer != nil || snapshotURL != nil }

    private static func stamp() -> String {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd_HH-mm-ss-SSS"; return f.string(from: Date())
    }

    /// Next composed frame is written as PNG.
    func snapshot(done: @escaping (URL?) -> Void) {
        lock.lock(); defer { lock.unlock() }
        snapshotURL = Self.dir.appendingPathComponent("snap_\(Self.stamp()).png")
        snapshotDone = done
    }

    func startRecording() -> URL? {
        lock.lock(); defer { lock.unlock() }
        guard writer == nil else { return nil }
        let url = Self.dir.appendingPathComponent("rec_\(Self.stamp()).mp4")
        guard let w = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
        let inp = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: Self.width, AVVideoHeightKey: Self.height,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 16_000_000],
        ])
        inp.expectsMediaDataInRealTime = true
        w.add(inp)
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp, sourcePixelBufferAttributes: nil)
        w.startWriting()
        writer = w; input = inp; startTime = nil
        return url
    }

    /// Finishes the file; `discard` deletes it (a tap, not a hold).
    func stopRecording(discard: Bool, done: @escaping (URL?) -> Void) {
        lock.lock()
        let w = writer, inp = input, started = startTime != nil
        writer = nil; input = nil; adaptor = nil; startTime = nil
        lock.unlock()
        guard let w, let inp else { done(nil); return }
        if discard || !started {
            w.cancelWriting()
            try? FileManager.default.removeItem(at: w.outputURL)
            done(nil); return
        }
        inp.markAsFinished()
        w.finishWriting { done(w.status == .completed ? w.outputURL : nil) }
    }

    /// Called on the capture queue for every frame while something is pending.
    func append(raw: CVPixelBuffer, processed: CVPixelBuffer, time: CMTime, debug: GazeEffect.Debug, stats: CameraPipeline.Stats) {
        guard wantsFrames, let pool else { return }
        var out: CVPixelBuffer?
        guard CVPixelBufferPoolCreatePixelBuffer(nil, pool, &out) == kCVReturnSuccess, let out else { return }
        let cw = CGFloat(CameraPipeline.width)
        let img = CIImage(cvPixelBuffer: processed).transformed(by: CGAffineTransform(translationX: cw, y: 0))
            .composited(over: CIImage(cvPixelBuffer: raw))
        ci.render(img, to: out)
        Self.drawOverlay(on: out, debug: debug, stats: stats)

        lock.lock()
        let snapURL = snapshotURL, snapDone = snapshotDone
        snapshotURL = nil; snapshotDone = nil
        if let w = writer, let inp = input, let ad = adaptor {
            if startTime == nil { w.startSession(atSourceTime: time); startTime = time }
            if inp.isReadyForMoreMediaData { ad.append(out, withPresentationTime: time) }
        }
        lock.unlock()

        if let snapURL {
            let ok = (try? ci.writePNGRepresentation(of: CIImage(cvPixelBuffer: out), to: snapURL, format: .RGBA8,
                                                      colorSpace: CGColorSpaceCreateDeviceRGB())) != nil
            snapDone?(ok ? snapURL : nil)
        }
    }

    // MARK: Burned-in overlay

    static func drawOverlay(on pb: CVPixelBuffer, debug d: GazeEffect.Debug, stats s: CameraPipeline.Stats) {
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
        guard let cg = CGContext(data: CVPixelBufferGetBaseAddress(pb), width: w, height: h, bitsPerComponent: 8,
                                 bytesPerRow: CVPixelBufferGetBytesPerRow(pb), space: CGColorSpaceCreateDeviceRGB(),
                                 bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return }
        // Top-left origin, like the debug coordinates.
        cg.translateBy(x: 0, y: CGFloat(h)); cg.scaleBy(x: 1, y: -1)
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(cgContext: cg, flipped: true)
        defer { NSGraphicsContext.restoreGraphicsState() }

        let panelW = CGFloat(CameraPipeline.width)
        for dx in [0, panelW] {
            if let f = d.faceBox { stroke(cg, f.offsetBy(dx: dx, dy: 0), .systemYellow, 1) }
            for b in d.eyeBoxes { stroke(cg, b.offsetBy(dx: dx, dy: 0), .systemGreen, 2) }
            cg.setFillColor(NSColor.systemTeal.cgColor)
            for p in d.eyePoints { cg.fillEllipse(in: CGRect(x: p.x + dx - 2.5, y: p.y - 2.5, width: 5, height: 5)) }
            cg.setFillColor(NSColor.systemRed.cgColor)
            for p in d.pupils { cg.fillEllipse(in: CGRect(x: p.x + dx - 3, y: p.y - 3, width: 6, height: 6)) }
        }
        hud(lines(debug: d, stats: s)).draw(at: CGPoint(x: panelW + 14, y: 12))
        label("ОРИГІНАЛ").draw(at: CGPoint(x: 14, y: 12))
    }

    static func lines(debug d: GazeEffect.Debug, stats s: CameraPipeline.Stats) -> [String] {
        let deg = { (v: Float?) in v.map { String(format: "%.0f°", $0) } ?? "—" }
        return [
            String(format: "%.0f fps  кадр %.1f мс  (vision %.1f, модель %.1f)", s.fps, s.frameMs, d.visionMs, d.modelMs),
            "\(d.status)",
            "yaw \(deg(d.yawDeg))  pitch \(deg(d.pitchDeg))",
            String(format: "корекція верт %.1f°  гор %.1f°", d.angle.v, d.angle.h),
            String(format: "очі ~%.0f см  рух %.1f px%@", d.eyeDistanceCm, d.motionPx, d.smoothingReset ? "  [скид згладж.]" : ""),
            String(format: "зіниця %+.3f  нейтраль %+.3f  гасіння %+.1f°%@", d.pupilX ?? .nan, d.neutralX ?? .nan, d.followDeg, d.calibrating ? "  [калібрую]" : ""),
            "дроп \(s.dropped)  вірт.камера \(s.sinkConnected ? "так" : "ні")",
        ]
    }

    private static func stroke(_ cg: CGContext, _ r: CGRect, _ c: NSColor, _ w: CGFloat) {
        cg.setStrokeColor(c.cgColor); cg.setLineWidth(w); cg.stroke(r)
    }

    private static func hud(_ lines: [String]) -> NSAttributedString {
        let p = NSMutableParagraphStyle(); p.lineSpacing = 2
        return NSAttributedString(string: lines.joined(separator: "\n"), attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: 17, weight: .medium),
            .foregroundColor: NSColor.white, .backgroundColor: NSColor.black.withAlphaComponent(0.6), .paragraphStyle: p,
        ])
    }

    private static func label(_ s: String) -> NSAttributedString {
        NSAttributedString(string: " \(s) ", attributes: [
            .font: NSFont.monospacedSystemFont(ofSize: 17, weight: .bold),
            .foregroundColor: NSColor.white, .backgroundColor: NSColor.black.withAlphaComponent(0.6),
        ])
    }
}
