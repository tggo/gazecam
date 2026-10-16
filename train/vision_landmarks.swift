// Dev helper: dump the app's 12 eye points (GazeEffect.swift: Vision leftEye/rightEye -> dlibEye -> sorted by x, top-left origin,
// no temporal smoothing) for every NNNN.jpg of a session that has no `lm` column (e.g. the old 2026-10-16_19-01-27 session).
//   swiftc -O vision_landmarks.swift -o /tmp/vision_landmarks && /tmp/vision_landmarks <session dir> <out.csv>
import Foundation
import Vision
import CoreGraphics
import ImageIO

func dlibEye(_ p: [CGPoint]) -> [CGPoint] {
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

let dir = CommandLine.arguments[1]
let files = try! FileManager.default.contentsOfDirectory(atPath: dir).filter { $0.hasSuffix(".jpg") }.sorted()
var out = "file,yaw,pitch,lm\n"
for f in files {
    let url = URL(fileURLWithPath: dir + "/" + f)
    guard let src = CGImageSourceCreateWithURL(url as CFURL, nil), let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { continue }
    let w = img.width, h = img.height
    let req = VNDetectFaceLandmarksRequest()
    guard (try? VNImageRequestHandler(cgImage: img, options: [:]).perform([req])) != nil,
          let face = req.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }),
          let lm = face.landmarks, let le = lm.leftEye, let re = lm.rightEye else { continue }
    let size = CGSize(width: w, height: h)
    func pts(_ r: VNFaceLandmarkRegion2D) -> [CGPoint] { r.pointsInImage(imageSize: size).map { CGPoint(x: $0.x, y: CGFloat(h) - $0.y) } }
    let a = dlibEye(pts(le)), b = dlibEye(pts(re))
    let all = a[0].x < b[0].x ? a + b : b + a
    let s = all.map { String(format: "%.2f %.2f", $0.x, $0.y) }.joined(separator: " ")
    out += "\(f),\(face.yaw.map { String(format: "%.1f", $0.floatValue * 180 / .pi) } ?? ""),\(face.pitch.map { String(format: "%.1f", $0.floatValue * 180 / .pi) } ?? ""),\(s)\n"
}
let outPath = CommandLine.arguments.count > 2 ? CommandLine.arguments[2] : "lm_vision.csv"
try! out.write(toFile: outPath, atomically: true, encoding: .utf8)
print("wrote", outPath)
