import AppKit
import CoreImage
import SwiftUI

/// Five-point gaze calibration: a dot over the whole screen at centre and the four corners, ~2 s each.
/// Records pupil position and lid openness per point → neutral pupil + vertical lid map in degrees
/// (screen size from the display, distance from the eye-distance estimate).
/// Every collected frame is also saved with its target as a per-user dataset:
/// ~/Movies/GazeCam/dataset/<session>/NNNN.jpg + labels.csv (target point, angles to it, pupil, lids, distance).
@MainActor
final class CornerCalibration: ObservableObject {
    struct Point { let name: String; let unit: CGPoint; var camera = false } // unit: -1…1, y up
    enum Mode { case calibrate, dataset }
    static let camera = Point(name: "камера", unit: .zero, camera: true)
    /// Dataset run: 3×3 grid with a look into the camera every few points, so every screen frame has a
    /// camera frame from the same head pose a few seconds away (training pairs for the warp network).
    static let datasetPoints: [Point] = {
        func g(_ x: CGFloat, _ y: CGFloat, _ n: String) -> Point { Point(name: n, unit: CGPoint(x: x, y: y)) }
        return [camera, g(0, 0, "центр"), g(-1, 1, "верх-ліво"), g(0, 1, "верх-центр"), camera,
                g(1, 1, "верх-право"), g(-1, 0, "ліво"), g(1, 0, "право"), camera,
                g(-1, -1, "низ-ліво"), g(0, -1, "низ-центр"), g(1, -1, "низ-право"), camera,
                g(-0.5, 0.5, "проміжна-1"), g(0.5, -0.5, "проміжна-2"), g(0.5, 0.5, "проміжна-3"), g(-0.5, -0.5, "проміжна-4"), camera]
    }()
    static let calibrationPoints = [
        Point(name: "центр", unit: .zero),
        Point(name: "верхній лівий кут", unit: CGPoint(x: -1, y: 1)),
        Point(name: "верхній правий кут", unit: CGPoint(x: 1, y: 1)),
        Point(name: "нижній правий кут", unit: CGPoint(x: 1, y: -1)),
        Point(name: "нижній лівий кут", unit: CGPoint(x: -1, y: -1)),
    ]
    static let inset: CGFloat = 60      // dot distance from the screen edge, pt
    static let settle = 0.8, collect = 1.6

    @Published var running = false
    @Published private(set) var mode = Mode.calibrate
    var points: [Point] { mode == .dataset ? Self.datasetPoints : Self.calibrationPoints }
    @Published var step = 0
    @Published var result: String?

    private var window: NSWindow?
    private var timer: Timer?
    private var stepStart = Date()
    private var samples: [[(x: Float?, r: Float, dist: Double)]] = []
    private weak var gaze: GazeEffect?
    private weak var pipeline: CameraPipeline?
    private var datasetDir: URL?
    private static let header = "file,point,unit_x,unit_y,target_h_deg,target_v_deg,pupil_x,openness,open_ratio,eye_dist_cm,kind,cam_h_deg,cam_v_deg,lm\n"
    private var labels = header
    private var frameNo = 0
    private let ci = CIContext()
    private var onDone: ((GazeEffect.VerticalCal, Float?) -> Void)?

    func start(pipeline: CameraPipeline, mode: Mode = .calibrate, onDone: @escaping (GazeEffect.VerticalCal, Float?) -> Void) {
        guard !running, let screen = NSScreen.main else { return }
        self.mode = mode
        labels = Self.header
        self.gaze = pipeline.gaze
        self.pipeline = pipeline
        self.onDone = onDone
        let f = DateFormatter(); f.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        datasetDir = DebugCapture.dir.appendingPathComponent("dataset/\(f.string(from: Date()))")
        try? FileManager.default.createDirectory(at: datasetDir!, withIntermediateDirectories: true)
        frameNo = 0
        samples = Array(repeating: [], count: points.count)
        step = 0; result = nil; running = true; stepStart = Date()

        let w = NSWindow(contentRect: screen.frame, styleMask: .borderless, backing: .buffered, defer: false)
        w.level = .screenSaver
        w.isOpaque = false
        w.backgroundColor = .clear
        w.ignoresMouseEvents = true
        w.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        w.contentView = NSHostingView(rootView: CalibrationOverlay(cal: self))
        w.setFrame(screen.frame, display: true)
        w.orderFrontRegardless()
        window = w

        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    func cancel() { finish(apply: false) }

    private func tick() {
        guard running, let g = gaze else { return }
        let t = Date().timeIntervalSince(stepStart)
        if t > Self.settle {
            let d = g.debug
            if d.status == "корекція активна" || d.status.hasPrefix("око") {
                samples[step].append((d.pupilX, d.openRatio, d.eyeDistanceCm))
            }
            saveFrame(d)
        }
        if t > Self.settle + Self.collect {
            step += 1; stepStart = Date()
            if step >= points.count { finish(apply: true) }
        }
    }

    /// Original (unprocessed) frame + label. Kept even for closed-eye frames: the label says so via openness.
    private func saveFrame(_ d: GazeEffect.Debug) {
        guard let dir = datasetDir, let raw = pipeline?.stats.raw else { return }
        let p = points[step]
        let dist = d.eyeDistanceCm > 0 ? d.eyeDistanceCm : 65
        // Camera position from the gaze settings (y negative = above the screen centre).
        let off = gaze?.cameraOffset ?? SIMD3(0, -18, -1)
        let camH = atan(off.x / dist) * 180 / .pi, camV = atan(-off.y / dist) * 180 / .pi
        let th = p.camera ? camH : atan(Double(p.unit.x) * tan(Self.halfAngle(horizontal: true, distanceCm: dist) * .pi / 180)) * 180 / .pi
        let tv = p.camera ? camV : atan(Double(p.unit.y) * tan(Self.halfAngle(horizontal: false, distanceCm: dist) * .pi / 180)) * 180 / .pi
        let lm = d.eyePoints.map { String(format: "%.1f %.1f", $0.x, $0.y) }.joined(separator: " ")
        let name = String(format: "%04d.jpg", frameNo); frameNo += 1
        let img = CIImage(cvPixelBuffer: raw)
        try? ci.writeJPEGRepresentation(of: img, to: dir.appendingPathComponent(name), colorSpace: CGColorSpaceCreateDeviceRGB(),
                                        options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9])
        labels += String(format: "%@,%@,%.2f,%.2f,%.2f,%.2f,%@,%.4f,%.4f,%.1f,%@,%.2f,%.2f,%@\n", name, p.name.replacingOccurrences(of: " ", with: "_"),
                         p.unit.x, p.unit.y, th, tv, d.pupilX.map { String(format: "%.4f", $0) } ?? "", d.openness, d.openRatio, d.eyeDistanceCm,
                         p.camera ? "camera" : "screen", camH, camV, lm)
    }

    private func finish(apply: Bool) {
        if let dir = datasetDir { try? labels.write(to: dir.appendingPathComponent("labels.csv"), atomically: true, encoding: .utf8) }
        timer?.invalidate(); timer = nil
        window?.orderOut(nil); window = nil
        running = false
        guard apply else { result = "скасовано"; return }
        if mode == .dataset {
            result = "Датасет: \(frameNo) кадрів → \(datasetDir?.lastPathComponent ?? "")"
            return
        }

        func median(_ v: [Float]) -> Float? { v.isEmpty ? nil : v.sorted()[v.count / 2] }
        let r = samples.map { median($0.map(\.r)) }
        let x = samples.map { median($0.compactMap(\.x)) }
        let dist = samples.flatMap { $0.map(\.dist) }.filter { $0 > 0 }.sorted()
        guard let rc = r[0], let rtl = r[1], let rtr = r[2], let rbr = r[3], let rbl = r[4], !dist.isEmpty else {
            result = "мало даних: обличчя не знайдено в деяких точках, спробуй ще раз"; return
        }
        let distCm = dist[dist.count / 2]
        var cal = GazeEffect.VerticalCal()
        cal.rCenter = rc
        cal.rTop = (rtl + rtr) / 2
        cal.rBottom = (rbr + rbl) / 2
        cal.halfDeg = Float(Self.halfAngle(horizontal: false, distanceCm: distCm))
        onDone?(cal, x[0])

        let f = { (v: Float?) in v.map { String(format: "%+.3f", $0) } ?? "—" }
        result = String(format: "Відкритість: центр %.2f, верх %.2f, низ %.2f; кут до верх/низу ±%.0f° на %.0f см.\nЗіниця: центр %@, ліві кути %@/%@, праві %@/%@",
                        rc, cal.rTop, cal.rBottom, cal.halfDeg, distCm, f(x[0]), f(x[1]), f(x[4]), f(x[2]), f(x[3]))
        if cal.rTop <= rc || cal.rBottom >= rc { result! += "\n⚠︎ повіки не змінились між верхом і низом — вертикаль буде неточна" }
    }

    /// Angle from the eyes (facing the screen centre) to the edge dots, from the display's physical size.
    static func halfAngle(horizontal: Bool, distanceCm: Double) -> Double {
        guard let screen = NSScreen.main,
              let id = screen.deviceDescription[NSDeviceDescriptionKey("NSScreenNumber")] as? CGDirectDisplayID else { return 15 }
        let mm = CGDisplayScreenSize(id)
        guard mm.height > 0 else { return 15 }
        let cmPerPt = Double(mm.height) / 10 / Double(screen.frame.height)
        let halfPt = horizontal ? screen.frame.width / 2 : screen.frame.height / 2
        let offsetCm = (Double(halfPt) - Double(inset)) * cmPerPt
        return atan(offsetCm / max(distanceCm, 20)) * 180 / .pi
    }
}

private struct CalibrationOverlay: View {
    @ObservedObject var cal: CornerCalibration

    var body: some View {
        GeometryReader { geo in
            ZStack {
                Color.black.opacity(0.55)
                if cal.step < cal.points.count {
                    let p = cal.points[cal.step]
                    let pos = CGPoint(x: geo.size.width / 2 + p.unit.x * (geo.size.width / 2 - CornerCalibration.inset),
                                      y: geo.size.height / 2 - p.unit.y * (geo.size.height / 2 - CornerCalibration.inset))
                    if p.camera {
                        Image(systemName: "arrow.up").font(.system(size: 90, weight: .bold)).foregroundStyle(.green)
                            .position(x: geo.size.width / 2, y: 90)
                    } else {
                        ZStack {
                            Circle().stroke(Color.white, lineWidth: 3).frame(width: 44, height: 44)
                            Circle().fill(Color.red).frame(width: 14, height: 14)
                        }
                        .position(pos)
                    }
                    VStack(spacing: 6) {
                        Text(p.camera ? "Дивись прямо в камеру" : "Дивись на точку: \(p.name)").font(.system(size: 28, weight: .semibold))
                        Text("\(cal.step + 1) / \(cal.points.count) · голову не повертай").font(.system(size: 18))
                    }
                    .foregroundStyle(.white)
                    .position(x: geo.size.width / 2, y: geo.size.height / 2 + 80)
                }
            }
        }
        .ignoresSafeArea()
    }
}
