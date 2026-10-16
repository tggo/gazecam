import CoreML
import QuartzCore
import CoreVideo
import Foundation
import Vision

/// Eye-contact correction: per-eye warp network from "Look at Me! Correcting Eye Gaze in Live Video
/// Communication" (Hsu et al. 2019), weights via WangWilly/gaze-correction-cam, converted by tools/convert_gaze.py.
///
/// Per frame: Vision landmarks → 6 dlib-style points per eye → 48x64 crop + 12-ch anchor map + target angle →
/// Core ML predicts flow + light weights → bilinear warp here (mirrors `warp()` in convert_gaze.py) → paste back.
final class GazeEffect: Effect {
    let id = "gaze"
    let title = "Погляд у камеру"
    var enabled = true

    /// Camera position relative to the screen centre, cm (y negative = above). Mac + monitor webcam ≈ (0, -18, -1).
    var cameraOffset = SIMD3<Double>(0, -18, -1)
    /// Focal length in pixels for the 1280-wide frame (~65° horizontal FOV).
    var focalLength = 1000.0
    var ipd = 6.3
    /// Beyond this head yaw/pitch (radians) the warp looks wrong, so the effect steps aside.
    var maxHeadAngle: Float = 0.35
    /// Multiplies the computed redirection angle (1 = geometric estimate).
    var strength: Float = 1
    /// Edge blend width when pasting the corrected eye back, px.
    var feather: Float = 3
    /// Temporal smoothing of landmarks and angle, 0 = off, 0.9 = heavy. Fast moves reset it so the eyes do not lag.
    var smoothing: Float = 0.6

    /// How much of the measured horizontal eye movement to cancel (0 = fixed geometric angle only, 1 = all).
    /// Lets reading / teleprompter line sweeps disappear. Vertical stays geometric: Vision pupils carry no vertical signal.
    var followGaze: Float = 1
    /// Pupil shift (fraction of eye width) per degree of model redirection, measured: +15° → +0.05.
    var pupilPerDegree: Float = 0.0033
    /// Hard cap on the redirect, both axes; the network was trained on roughly ±20°.
    var maxRedirect: Float = 20
    /// "Looking at the screen" pupil position. Drifts slowly toward the running average; calibrate() pins it.
    private(set) var neutralX: Float?
    /// Set by corner calibration; stops the slow auto-drift of the neutral.
    var neutralLocked = false
    private var pupilX: Float?
    private var openRatioSmoothed: Float?

    /// Lid openness (relative to usual) when looking at the screen centre / top edge / bottom edge, and the
    /// angle from the eyes to those edges. Defaults measured on one user's corner snapshots; calibration replaces them.
    struct VerticalCal: Codable, Equatable { var rCenter: Float = 1.0, rTop: Float = 1.1, rBottom: Float = 0.62, halfDeg: Float = 15 }
    var vCal = VerticalCal()

    func setNeutral(_ x: Float) { neutralX = x; neutralLocked = true }

    /// Eyes count as closed (blink, looking far down) below this fraction of the user's usual openness:
    /// no correction — Vision invents pupils on a closed lid and the network paints an eye over it.
    /// The usual openness is learned (slow EMA over open frames), so it fits any face and lighting.
    var openRatio: Float = 0.4
    private(set) var usualOpenness: Float = 0.22

    /// Fixed [vertical, horizontal] redirection for experiments (test harness).
    var angleOverride: (Float, Float)?

    private var prevPoints: [CGPoint]?
    private var prevAngle: (Float, Float)?

    /// Last frame's state, for the preview window. Written on the capture queue, read on main.
    struct Debug {
        var status = "—"
        var yawDeg: Float?, pitchDeg: Float?
        var angle: (v: Float, h: Float) = (0, 0)
        var eyeDistanceCm: Double = 0
        var motionPx: Double = 0        // mean landmark displacement vs previous frame, before smoothing
        var smoothingReset = false
        var pupils: [CGPoint] = []
        var pupilX: Float?, neutralX: Float?
        var followDeg: Float = 0
        var openRatio: Float = 1
        var gazeUpDeg: Float = 0
        var followVDeg: Float = 0
        var openness: Float = 0
        var usualOpenness: Float = 0
        var eyeBoxes: [CGRect] = []     // pixel coords, top-left origin
        var eyePoints: [CGPoint] = []
        var faceBox: CGRect?
        var visionMs: Double = 0, modelMs: Double = 0
    }
    private let debugLock = NSLock()
    private var _debug = Debug()
    var debug: Debug { debugLock.lock(); defer { debugLock.unlock() }; return _debug }
    private func publish(_ d: Debug) { debugLock.lock(); _debug = d; debugLock.unlock() }

    private let H = 48, W = 64
    private let request = VNDetectFaceLandmarksRequest()
    private var models: [Side: MLModel] = [:]
    private let imgArr: MLMultiArray
    private let fpArr: MLMultiArray
    private let angArr: MLMultiArray

    enum Side: String { case L, R }

    init() {
        imgArr = try! MLMultiArray(shape: [1, 48, 64, 3], dataType: .float32)
        fpArr = try! MLMultiArray(shape: [1, 48, 64, 12], dataType: .float32)
        angArr = try! MLMultiArray(shape: [1, 2], dataType: .float32)
        let cfg = MLModelConfiguration()
        cfg.computeUnits = .cpuAndNeuralEngine
        for side in [Side.L, .R] {
            if let url = Bundle.main.url(forResource: "GazeWarp\(side.rawValue)", withExtension: "mlmodelc") {
                models[side] = try? MLModel(contentsOf: url, configuration: cfg)
            }
        }
        if models.count != 2 { NSLog("GazeCam: gaze models missing"); enabled = false }
    }

    func apply(to buffer: CVPixelBuffer) {
        var d = Debug()
        defer { publish(d) }
        guard models.count == 2 else { d.status = "моделі не завантажились"; return }
        let t0 = CACurrentMediaTime()
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
        let ok = (try? handler.perform([request])) != nil
        d.visionMs = (CACurrentMediaTime() - t0) * 1000
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        guard ok, let face = request.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }),
              let lm = face.landmarks, let le = lm.leftEye, let re = lm.rightEye
        else { d.status = "обличчя не знайдено"; prevPoints = nil; prevAngle = nil; return }
        let bb = face.boundingBox
        d.faceBox = CGRect(x: bb.minX * CGFloat(w), y: (1 - bb.maxY) * CGFloat(h), width: bb.width * CGFloat(w), height: bb.height * CGFloat(h))
        d.yawDeg = face.yaw.map { $0.floatValue * 180 / .pi }
        d.pitchDeg = face.pitch.map { $0.floatValue * 180 / .pi }
        if let yaw = face.yaw?.floatValue, abs(yaw) > maxHeadAngle { d.status = "голова повернута (yaw) — пропуск"; return }
        if let pitch = face.pitch?.floatValue, abs(pitch) > maxHeadAngle { d.status = "голова нахилена (pitch) — пропуск"; return }

        let size = CGSize(width: w, height: h)
        // Vision image points are bottom-left origin; flip to top-left like OpenCV.
        func pts(_ r: VNFaceLandmarkRegion2D) -> [CGPoint] { r.pointsInImage(imageSize: size).map { CGPoint(x: $0.x, y: CGFloat(h) - $0.y) } }
        let a = dlibEye(pts(le)), b = dlibEye(pts(re))
        // "R" model = dlib 36-41 = eye on the image-left side; "L" = dlib 42-47 = image-right side.
        let (rawLeft, rawRight) = a[0].x < b[0].x ? (a, b) : (b, a)

        // Temporal smoothing (EMA). A jump bigger than ~1/4 of an eye width is real motion: snap to it.
        var cur = rawLeft + rawRight
        let eyeW = max(abs(rawLeft[3].x - rawLeft[0].x), 1)
        var reset = true
        if let prev = prevPoints, prev.count == cur.count {
            let motion = zip(prev, cur).map { hypot($0.x - $1.x, $0.y - $1.y) }.reduce(0, +) / CGFloat(cur.count)
            d.motionPx = Double(motion)
            if motion < eyeW * 0.25 {
                reset = false
                let k = CGFloat(smoothing)
                cur = zip(prev, cur).map { CGPoint(x: $0.x * k + $1.x * (1 - k), y: $0.y * k + $1.y * (1 - k)) }
            }
        }
        d.smoothingReset = reset
        prevPoints = cur
        let imgLeftEye = Array(cur[0..<6]), imgRightEye = Array(cur[6..<12])
        d.eyePoints = imgLeftEye + imgRightEye

        // Eye aspect ratio on the 6-point layout: (|p1-p5| + |p2-p4|) / (2 |p0-p3|), averaged over both eyes.
        func ear(_ e: [CGPoint]) -> Float {
            let v = hypot(e[1].x - e[5].x, e[1].y - e[5].y) + hypot(e[2].x - e[4].x, e[2].y - e[4].y)
            return Float(v / max(2 * hypot(e[0].x - e[3].x, e[0].y - e[3].y), 1))
        }
        d.openness = (ear(imgLeftEye) + ear(imgRightEye)) / 2
        // Learn only from frames that are clearly open, so blinks do not drag the reference down.
        if d.openness > usualOpenness * 0.8 { usualOpenness = usualOpenness * 0.99 + d.openness * 0.01 }
        d.usualOpenness = usualOpenness
        if d.openness < usualOpenness * openRatio {
            d.status = "очі закриті — пропуск"
            prevAngle = nil
            return
        }

        let cR = mid(imgLeftEye[0], imgLeftEye[3]), cL = mid(imgRightEye[0], imgRightEye[3])
        var angle = angleOverride ?? gazeAngle(le: cL, re: cR, w: Double(w), h: Double(h))
        angle = (angle.0 * strength, angle.1 * strength)
        if !reset, let pa = prevAngle {
            angle = (pa.0 * smoothing + angle.0 * (1 - smoothing), pa.1 * smoothing + angle.1 * (1 - smoothing))
        }
        prevAngle = angle

        // Cancel eye movement relative to the neutral (screen centre) position.
        // Horizontal: pupil shift, undone in the network's own units (pupilPerDegree, measured).
        // Vertical: Vision pupils carry no vertical signal, but the lids follow gaze — openness / usual openness.
        let k = reset ? 0 : smoothing
        let ratio = d.openness / max(usualOpenness, 0.01)
        openRatioSmoothed = openRatioSmoothed.map { $0 * k + ratio * (1 - k) } ?? ratio
        let r = openRatioSmoothed ?? 1
        d.openRatio = r
        var followV: Float = 0, followH: Float = 0
        if angleOverride == nil {
            let up = r >= vCal.rCenter
                ? min((r - vCal.rCenter) / max(vCal.rTop - vCal.rCenter, 0.02), 1.3) * vCal.halfDeg
                : -min((vCal.rCenter - r) / max(vCal.rCenter - vCal.rBottom, 0.05), 1.5) * vCal.halfDeg
            d.gazeUpDeg = up
            followV = -up * followGaze
        }
        if angleOverride == nil, let lp = lm.leftPupil, let rp = lm.rightPupil,
           let x = pupilOffsetX(le, lp, re, rp) {
            let sx = pupilX.map { $0 * k + x * (1 - k) } ?? x
            pupilX = sx
            if !neutralLocked {
                // ~20 s time constant: follows posture changes, not individual glances.
                neutralX = neutralX.map { $0 * 0.998 + sx * 0.002 } ?? sx
            }
            if let n = neutralX { followH = -(sx - n) / pupilPerDegree * followGaze }
            d.pupilX = sx; d.neutralX = neutralX
            d.pupils = [lp, rp].compactMap { $0.pointsInImage(imageSize: size).first }.map { CGPoint(x: $0.x, y: CGFloat(h) - $0.y) }
        }
        // A look far beyond what the network can undo is deliberate (looking away): let it through,
        // fading the follow out between 1× and 1.6× the cap instead of a half-corrected rolled eye.
        let need = hypot(angle.0 + followV, angle.1 + followH)
        let fade = min(max((1.6 * maxRedirect - need) / (0.6 * maxRedirect), 0), 1)
        d.followDeg = followH * fade
        d.followVDeg = followV * fade
        angle = (angle.0 + followV * fade, angle.1 + followH * fade)
        d.angle = angle
        d.eyeDistanceCm = focalLength * ipd / max(hypot(Double(cL.x - cR.x), Double(cL.y - cR.y)), 1)

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else { return }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let frame = Frame(base: base, rowBytes: rowBytes, width: w, height: h)

        // Past ~20° total the network rolls the iris under the lid (white-eye look): cap the vector, not each axis.
        let mag = hypot(angle.0, angle.1)
        if mag > maxRedirect { angle = (angle.0 * maxRedirect / mag, angle.1 * maxRedirect / mag) }
        d.angle = angle

        let t1 = CACurrentMediaTime()
        let boxes = [correct(frame, points: imgLeftEye, center: cR, side: .R, seq: [0, 1, 2, 3, 4, 5], angle: angle),
                     correct(frame, points: imgRightEye, center: cL, side: .L, seq: [3, 2, 1, 0, 5, 4], angle: angle)]
        d.modelMs = (CACurrentMediaTime() - t1) * 1000
        d.eyeBoxes = boxes.compactMap { $0 }
        d.status = d.eyeBoxes.count == 2 ? "корекція активна" : "око біля краю кадру — пропуск"
    }

    // MARK: - Geometry

    /// Mean horizontal pupil offset from the eye-corner midpoint, in eye widths (same as PupilProbe).
    private func pupilOffsetX(_ le: VNFaceLandmarkRegion2D, _ lp: VNFaceLandmarkRegion2D,
                              _ re: VNFaceLandmarkRegion2D, _ rp: VNFaceLandmarkRegion2D) -> Float? {
        func off(_ e: VNFaceLandmarkRegion2D, _ p: VNFaceLandmarkRegion2D) -> Float? {
            let pts = e.normalizedPoints
            guard let l = pts.min(by: { $0.x < $1.x }), let r = pts.max(by: { $0.x < $1.x }), let c = p.normalizedPoints.first else { return nil }
            return Float((c.x - (l.x + r.x) / 2) / max(r.x - l.x, 0.001))
        }
        guard let a = off(le, lp), let b = off(re, rp) else { return nil }
        return (a + b) / 2
    }

    /// Six points in dlib order: [image-left corner, upper 1/3, upper 2/3, image-right corner, lower 2/3, lower 1/3].
    private func dlibEye(_ p: [CGPoint]) -> [CGPoint] {
        let l = p.min(by: { $0.x < $1.x })!, r = p.max(by: { $0.x < $1.x })!
        func lineY(_ x: CGFloat) -> CGFloat { l.y + (r.y - l.y) * (x - l.x) / max(r.x - l.x, 1) }
        let upper = p.filter { $0.y < lineY($0.x) }.sorted { $0.x < $1.x }
        let lower = p.filter { $0.y > lineY($0.x) }.sorted { $0.x < $1.x }
        func at(_ c: [CGPoint], _ t: CGFloat) -> CGPoint {
            let x = l.x + (r.x - l.x) * t
            let poly = [l] + c + [r]
            for i in 0..<(poly.count - 1) where poly[i].x <= x && x <= poly[i + 1].x {
                let s = (x - poly[i].x) / max(poly[i + 1].x - poly[i].x, 0.001)
                return CGPoint(x: x, y: poly[i].y + (poly[i + 1].y - poly[i].y) * s)
            }
            return CGPoint(x: x, y: lineY(x))
        }
        return [l, at(upper, 1.0 / 3), at(upper, 2.0 / 3), r, at(lower, 2.0 / 3), at(lower, 1.0 / 3)]
    }

    private func mid(_ a: CGPoint, _ b: CGPoint) -> CGPoint { CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2) }

    /// Port of GazeCorrector.estimate_gaze_angle: [vertical, horizontal] degrees, truncated to Int like the original.
    private func gazeAngle(le: CGPoint, re: CGPoint, w: Double, h: Double) -> (Float, Float) {
        let ipdPx = hypot(Double(le.x - re.x), Double(le.y - re.y))
        let z = -(focalLength * ipd) / ipdPx
        let o = cameraOffset
        let x = -abs(z) * (Double(le.x + re.x) - w) / (2 * focalLength) + o.x
        let y = abs(z) * (Double(le.y + re.y) - h) / (2 * focalLength) + o.y
        var av = atan((0 - y) / (0 - z)) * 180 / .pi
        var ah = atan((0 - x) / (0 - z)) * 180 / .pi
        av += atan((y - o.y) / (o.z - z)) * 180 / .pi
        ah += atan((x - o.x) / (o.z - z)) * 180 / .pi
        return (Float(Int(av)), Float(Int(ah)))
    }

    // MARK: - Per-eye correction

    struct Frame {
        let base: UnsafeMutablePointer<UInt8>
        let rowBytes, width, height: Int
        @inline(__always) func px(_ x: Int, _ y: Int) -> UnsafeMutablePointer<UInt8> { base + y * rowBytes + x * 4 }
    }

    @discardableResult
    private func correct(_ f: Frame, points: [CGPoint], center: CGPoint, side: Side, seq: [Int], angle: (Float, Float)) -> CGRect? {
        let eyeLen = abs(points[3].x - points[0].x)
        let halfW = eyeLen * 3 / 4, boxH = 1.5 * halfW
        let top = Int(center.y - boxH * 7 / 12), bottom = Int(center.y + boxH * 5 / 12)
        let left = Int(center.x - halfW), right = Int(center.x + halfW)
        guard top >= 0, left >= 0, bottom <= f.height, right <= f.width else { return nil }
        let oh = bottom - top, ow = right - left
        guard oh >= 12, ow >= 16 else { return nil }

        // Crop → 48x64 BGR [0,1] (cv2.resize INTER_LINEAR sampling).
        let img = imgArr.dataPointer.assumingMemoryBound(to: Float.self)
        var src = [Float](repeating: 0, count: H * W * 3)
        let sx = Float(ow) / Float(W), sy = Float(oh) / Float(H)
        for y in 0..<H {
            let fy = min(max((Float(y) + 0.5) * sy - 0.5, 0), Float(oh - 1))
            let y0 = Int(fy), y1 = min(y0 + 1, oh - 1), ty = fy - Float(y0)
            for x in 0..<W {
                let fx = min(max((Float(x) + 0.5) * sx - 0.5, 0), Float(ow - 1))
                let x0 = Int(fx), x1 = min(x0 + 1, ow - 1), tx = fx - Float(x0)
                let p00 = f.px(left + x0, top + y0), p01 = f.px(left + x1, top + y0)
                let p10 = f.px(left + x0, top + y1), p11 = f.px(left + x1, top + y1)
                for c in 0..<3 {
                    let v = (Float(p00[c]) * (1 - tx) + Float(p01[c]) * tx) * (1 - ty)
                          + (Float(p10[c]) * (1 - tx) + Float(p11[c]) * tx) * ty
                    src[(y * W + x) * 3 + c] = v / 255
                }
            }
        }
        src.withUnsafeBufferPointer { img.update(from: $0.baseAddress!, count: src.count) }

        // Anchor map: for each point, (col - px, row - py) in 48x64 space.
        let fp = fpArr.dataPointer.assumingMemoryBound(to: Float.self)
        let anchors = seq.map { i -> (Float, Float) in
            (Float(Int((points[i].x - CGFloat(left)) * CGFloat(W) / CGFloat(ow))),
             Float(Int((points[i].y - CGFloat(top)) * CGFloat(H) / CGFloat(oh))))
        }
        for y in 0..<H {
            for x in 0..<W {
                let o = (y * W + x) * 12
                for (k, a) in anchors.enumerated() {
                    fp[o + 2 * k] = Float(x) - a.0
                    fp[o + 2 * k + 1] = Float(y) - a.1
                }
            }
        }
        let ang = angArr.dataPointer.assumingMemoryBound(to: Float.self)
        ang[0] = angle.0; ang[1] = angle.1

        guard let input = try? MLDictionaryFeatureProvider(dictionary: ["img": imgArr, "fp": fpArr, "ang": angArr]),
              let out = try? models[side]!.prediction(from: input),
              let flowA = out.featureValue(for: "flow")?.multiArrayValue,
              let lcmA = out.featureValue(for: "lcm")?.multiArrayValue
        else { return nil }
        let fl = flowA.toFloats(), lc = lcmA.toFloats()

        // Backward warp + light correction (same as convert_gaze.py warp()).
        var dst = [Float](repeating: 0, count: H * W * 3)
        for y in 0..<H {
            let gy = -1 + 2 * Float(y) / Float(H - 1)
            for x in 0..<W {
                let i = y * W + x
                let gx = -1 + 2 * Float(x) / Float(W - 1)
                let pxf = 0.5 * (gx + fl[i * 2] + 1) * Float(W)
                let pyf = 0.5 * (gy + fl[i * 2 + 1] + 1) * Float(H)
                let fx0 = Int(floor(pxf)), fy0 = Int(floor(pyf))
                let x0 = min(max(fx0, 0), W - 1), x1 = min(max(fx0 + 1, 0), W - 1)
                let y0 = min(max(fy0, 0), H - 1), y1 = min(max(fy0 + 1, 0), H - 1)
                let wa = (Float(x1) - pxf) * (Float(y1) - pyf), wb = (Float(x1) - pxf) * (pyf - Float(y0))
                let wc = (pxf - Float(x0)) * (Float(y1) - pyf), wd = (pxf - Float(x0)) * (pyf - Float(y0))
                for c in 0..<3 {
                    let v = wa * src[(y0 * W + x0) * 3 + c] + wb * src[(y1 * W + x0) * 3 + c]
                          + wc * src[(y0 * W + x1) * 3 + c] + wd * src[(y1 * W + x1) * 3 + c]
                    dst[i * 3 + c] = v * lc[i * 2] + lc[i * 2 + 1]
                }
            }
        }

        // Paste back at original size; the original hard-cuts a (3,4) px border, we feather it instead.
        let cutY = 3, cutX = 4, feather = max(self.feather, 0.5)
        let rx = Float(W) / Float(ow), ry = Float(H) / Float(oh)
        for y in cutY..<(oh - cutY) {
            let fy = min(max((Float(y) + 0.5) * ry - 0.5, 0), Float(H - 1))
            let y0 = Int(fy), y1 = min(y0 + 1, H - 1), ty = fy - Float(y0)
            let ey = min(Float(y - cutY), Float(oh - cutY - 1 - y))
            for x in cutX..<(ow - cutX) {
                let ex = min(Float(x - cutX), Float(ow - cutX - 1 - x))
                let alpha = min(min(ex, ey) / feather, 1)
                let fx = min(max((Float(x) + 0.5) * rx - 0.5, 0), Float(W - 1))
                let x0 = Int(fx), x1 = min(x0 + 1, W - 1), tx = fx - Float(x0)
                let p = f.px(left + x, top + y)
                for c in 0..<3 {
                    let v = (dst[(y0 * W + x0) * 3 + c] * (1 - tx) + dst[(y0 * W + x1) * 3 + c] * tx) * (1 - ty)
                          + (dst[(y1 * W + x0) * 3 + c] * (1 - tx) + dst[(y1 * W + x1) * 3 + c] * tx) * ty
                    let nv = min(max(v, 0), 1) * 255
                    p[c] = UInt8(Float(p[c]) * (1 - alpha) + nv * alpha)
                }
            }
        }
        return CGRect(x: left, y: top, width: ow, height: oh)
    }
}

private extension MLMultiArray {
    /// Contiguous float copy regardless of whether Core ML handed back fp16 or fp32.
    func toFloats() -> [Float] {
        var out = [Float](repeating: 0, count: count)
        if dataType == .float16 {
            let p = dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { out[i] = Float(p[i]) }
        } else {
            let p = dataPointer.assumingMemoryBound(to: Float.self)
            for i in 0..<count { out[i] = p[i] }
        }
        return out
    }
}
