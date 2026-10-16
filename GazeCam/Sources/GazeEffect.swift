import CoreML
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
        guard models.count == 2 else { return }
        let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
        guard (try? handler.perform([request])) != nil,
              let face = request.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }),
              let lm = face.landmarks, let le = lm.leftEye, let re = lm.rightEye
        else { return }
        if let yaw = face.yaw?.floatValue, abs(yaw) > maxHeadAngle { return }
        if let pitch = face.pitch?.floatValue, abs(pitch) > maxHeadAngle { return }

        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        let size = CGSize(width: w, height: h)
        // Vision image points are bottom-left origin; flip to top-left like OpenCV.
        func pts(_ r: VNFaceLandmarkRegion2D) -> [CGPoint] { r.pointsInImage(imageSize: size).map { CGPoint(x: $0.x, y: CGFloat(h) - $0.y) } }
        let a = dlibEye(pts(le)), b = dlibEye(pts(re))
        // "R" model = dlib 36-41 = eye on the image-left side; "L" = dlib 42-47 = image-right side.
        let (imgLeftEye, imgRightEye) = a[0].x < b[0].x ? (a, b) : (b, a)

        let cR = mid(imgLeftEye[0], imgLeftEye[3]), cL = mid(imgRightEye[0], imgRightEye[3])
        let angle = gazeAngle(le: cL, re: cR, w: Double(w), h: Double(h))

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer)?.assumingMemoryBound(to: UInt8.self) else { return }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let frame = Frame(base: base, rowBytes: rowBytes, width: w, height: h)

        correct(frame, points: imgLeftEye, center: cR, side: .R, seq: [0, 1, 2, 3, 4, 5], angle: angle)
        correct(frame, points: imgRightEye, center: cL, side: .L, seq: [3, 2, 1, 0, 5, 4], angle: angle)
    }

    // MARK: - Geometry

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

    private func correct(_ f: Frame, points: [CGPoint], center: CGPoint, side: Side, seq: [Int], angle: (Float, Float)) {
        let eyeLen = abs(points[3].x - points[0].x)
        let halfW = eyeLen * 3 / 4, boxH = 1.5 * halfW
        let top = Int(center.y - boxH * 7 / 12), bottom = Int(center.y + boxH * 5 / 12)
        let left = Int(center.x - halfW), right = Int(center.x + halfW)
        guard top >= 0, left >= 0, bottom <= f.height, right <= f.width else { return }
        let oh = bottom - top, ow = right - left
        guard oh >= 12, ow >= 16 else { return }

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
        else { return }
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
        let cutY = 3, cutX = 4, feather: Float = 3
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
