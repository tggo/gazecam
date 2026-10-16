import CoreImage
import CoreGraphics
import CoreVideo
import Vision

/// One switchable processing step. Effects draw in place on a 1280x720 BGRA buffer.
protocol Effect: AnyObject {
    var id: String { get }
    var title: String { get }
    var enabled: Bool { get set }
    func apply(to buffer: CVPixelBuffer)
}

/// Two red horns above every detected face. Exists to prove the pipeline end to end.
final class HornsEffect: Effect {
    let id = "horns"
    let title = "Ріжки"
    var enabled = true

    private let request = VNDetectFaceRectanglesRequest()
    private var lastFaces: [CGRect] = []
    private var frame = 0

    func apply(to buffer: CVPixelBuffer) {
        frame += 1
        // Face detection every 3rd frame is plenty for a cosmetic overlay.
        if frame % 3 == 0 {
            let handler = VNImageRequestHandler(cvPixelBuffer: buffer, options: [:])
            if (try? handler.perform([request])) != nil {
                lastFaces = (request.results ?? []).map(\.boundingBox)
            }
        }
        guard !lastFaces.isEmpty else { return }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        guard let ctx = CGContext(data: CVPixelBufferGetBaseAddress(buffer), width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: CVPixelBufferGetBytesPerRow(buffer), space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return }

        ctx.setFillColor(CGColor(red: 0.85, green: 0.05, blue: 0.05, alpha: 1))
        ctx.setStrokeColor(CGColor(red: 0.3, green: 0, blue: 0, alpha: 1))
        ctx.setLineWidth(3)
        for n in lastFaces {
            // Vision boxes are normalized with origin bottom-left — same as CGContext.
            let f = CGRect(x: n.minX * CGFloat(w), y: n.minY * CGFloat(h), width: n.width * CGFloat(w), height: n.height * CGFloat(h))
            let base = f.maxY - f.height * 0.05
            let hornH = f.height * 0.45, hornW = f.width * 0.18
            for side in [-1.0, 1.0] {
                let cx = f.midX + CGFloat(side) * f.width * 0.28
                let p = CGMutablePath()
                p.move(to: CGPoint(x: cx - hornW / 2, y: base))
                p.addQuadCurve(to: CGPoint(x: cx + CGFloat(side) * hornW * 0.6, y: base + hornH),
                               control: CGPoint(x: cx - hornW / 2, y: base + hornH * 0.6))
                p.addQuadCurve(to: CGPoint(x: cx + hornW / 2, y: base),
                               control: CGPoint(x: cx + hornW / 2, y: base + hornH * 0.4))
                p.closeSubpath()
                ctx.addPath(p)
                ctx.drawPath(using: .fillStroke)
            }
        }
    }
}

/// Mirror flip — a second trivial effect to exercise the on/off plumbing.
final class MirrorEffect: Effect {
    let id = "mirror"
    let title = "Дзеркало"
    var enabled = false
    func apply(to buffer: CVPixelBuffer) {} // handled in the scale stage, see CameraPipeline
}
