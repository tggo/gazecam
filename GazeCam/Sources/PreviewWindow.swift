import AVKit
import CoreImage
import SwiftUI

/// Live preview + every setting + debug readout. Opened from the menu-bar popover.
struct PreviewWindow: View {
    @EnvironmentObject var model: AppModel
    @StateObject private var feed = PreviewFeed()

    var body: some View {
        HStack(spacing: 0) {
            VStack(spacing: 8) {
                if feed.showRaw, let raw = feed.raw {
                    HStack(spacing: 6) {
                        frameView(raw, label: "оригінал", overlay: false)
                        frameView(feed.processed, label: "результат", overlay: true)
                    }
                } else {
                    frameView(feed.processed, label: nil, overlay: true)
                }
                Spacer(minLength: 0)
            }
            .padding(12)
            .frame(minWidth: 640, maxWidth: .infinity, maxHeight: .infinity)
            .background(Color.black)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    controls
                    Divider()
                    gazeSettings
                    Divider()
                    capturesPanel
                    Divider()
                    debugPanel
                }
                .padding(14)
            }
            .frame(width: 320)
        }
        .frame(minHeight: 520)
        .onAppear { feed.attach(model.pipeline) }
        .onDisappear { feed.detach() }
        .sheet(item: $feed.reviewing) { CaptureReview(url: $0.url) }
    }

    // MARK: Frame + overlay

    @ViewBuilder
    private func frameView(_ img: CGImage?, label: String?, overlay: Bool) -> some View {
        ZStack(alignment: .topLeading) {
            if let img {
                Image(decorative: img, scale: 1).resizable().aspectRatio(16 / 9, contentMode: .fit)
                    .overlay {
                        if overlay && feed.showOverlay { DebugOverlay(debug: feed.gazeDebug) }
                    }
                    .overlay(alignment: .topLeading) {
                        if overlay && feed.showHUD {
                            Text(DebugCapture.lines(debug: feed.gazeDebug, stats: feed.stats).joined(separator: "\n"))
                                .font(.system(size: 12, weight: .medium, design: .monospaced))
                                .foregroundStyle(.white).padding(6)
                                .background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 4))
                                .padding(8)
                        }
                    }
                    .overlay(alignment: .topTrailing) {
                        if overlay && feed.recording {
                            Label("REC", systemImage: "record.circle.fill").font(.caption.bold())
                                .foregroundStyle(.white).padding(6).background(.red, in: Capsule()).padding(8)
                        }
                    }
            } else {
                Rectangle().fill(Color.gray.opacity(0.15)).aspectRatio(16 / 9, contentMode: .fit)
                    .overlay(Text(model.running ? "чекаю кадр…" : "натисни Старт").foregroundStyle(.secondary))
            }
            if let label {
                Text(label).font(.caption.bold()).padding(4).background(.black.opacity(0.6)).foregroundStyle(.white).padding(6)
            }
        }
    }

    // MARK: Panels

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Камера", selection: $model.cameraID) {
                ForEach(model.cameras, id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) }
            }
            .disabled(model.running)
            HStack {
                Button(model.running ? "Зупинити" : "Старт") { model.toggleRun() }.keyboardShortcut(.defaultAction)
                Text(model.status).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Text("Ефекти").font(.headline).padding(.top, 4)
            ForEach(model.pipeline.effects, id: \.id) { e in
                Toggle(e.title, isOn: Binding(get: { model.effectState[e.id] ?? false }, set: { model.setEffect(e.id, $0) }))
            }
            Text("Превʼю").font(.headline).padding(.top, 4)
            Toggle("Оверлей (очі, рамки)", isOn: $feed.showOverlay)
            Toggle("Статистика на відео", isOn: $feed.showHUD)
            Toggle("Поруч з оригіналом", isOn: Binding(get: { feed.showRaw }, set: { feed.setShowRaw($0) }))
        }
    }

    private var gazeSettings: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Погляд у камеру").font(.headline)
            slider("Камера над центром екрана", value: $model.gaze.cameraUp, range: 0...40, unit: "см")
            slider("Камера вбік від центру", value: $model.gaze.cameraRight, range: -30...30, unit: "см")
            slider("Камера перед екраном", value: $model.gaze.cameraDepth, range: -10...10, unit: "см")
            slider("Сила", value: $model.gaze.strength, range: 0...2, unit: "×", digits: 2)
            slider("Фокусна відстань", value: $model.gaze.focalLength, range: 500...2000, unit: "px")
            slider("Міжзіничкова відстань", value: $model.gaze.ipd, range: 5...7.5, unit: "см", digits: 1)
            slider("Макс. поворот голови", value: $model.gaze.maxHeadDeg, range: 5...45, unit: "°")
            slider("Розмиття краю вставки", value: $model.gaze.feather, range: 0...8, unit: "px", digits: 1)
            slider("Згладжування руху", value: $model.gaze.smoothing, range: 0...0.9, unit: "", digits: 2)
            slider("Гасити рухи очей (читання)", value: $model.gaze.followGaze, range: 0...1.5, unit: "×", digits: 2)
            slider("Макс. поворот погляду", value: $model.gaze.maxRedirect, range: 5...30, unit: "°")
            Button(feed.gazeDebug.calibrating ? "Дивись у центр екрана…" : "Калібрувати: дивлюсь у центр екрана") {
                model.pipeline.gaze.calibrate()
            }.font(.caption)
            Button("Скинути") { model.gaze = GazeSettings() }.font(.caption)
        }
    }

    private func slider(_ title: String, value: Binding<Double>, range: ClosedRange<Double>, unit: String, digits: Int = 0) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(title).font(.caption)
                Spacer()
                Text(String(format: "%.\(digits)f %@", value.wrappedValue, unit)).font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            }
            Slider(value: value, in: range)
        }
    }

    private var capturesPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Знімки й записи").font(.headline)
            Text("Пробіл: натиснути = знімок, тримати = запис. Зберігається оригінал і результат поруч зі статистикою.")
                .font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            Text(feed.recording ? "● ЗАПИС — відпусти пробіл" : "Тримай для запису")
                .font(.caption.bold()).frame(maxWidth: .infinity).padding(6)
                .background(feed.recording ? Color.red : Color.gray.opacity(0.2), in: RoundedRectangle(cornerRadius: 6))
                .onLongPressGesture(minimumDuration: 0, perform: {}, onPressingChanged: { $0 ? feed.press() : feed.release() })
            ForEach(feed.captures.prefix(8), id: \.self) { url in
                Button { feed.reviewing = .init(url: url) } label: {
                    Label(url.lastPathComponent, systemImage: url.pathExtension == "mp4" ? "film" : "photo").font(.caption)
                }.buttonStyle(.link)
            }
            Button("Відкрити папку") { NSWorkspace.shared.open(DebugCapture.dir) }.font(.caption)
        }
    }

    private var debugPanel: some View {
        let s = feed.stats, d = feed.gazeDebug
        return VStack(alignment: .leading, spacing: 3) {
            Text("Дебаг").font(.headline)
            row("FPS", String(format: "%.0f", s.fps))
            row("Кадр усього", String(format: "%.1f мс", s.frameMs))
            ForEach(s.effectMs.sorted(by: { $0.key < $1.key }), id: \.key) { k, v in row("  \(k)", String(format: "%.1f мс", v)) }
            row("  gaze: Vision", String(format: "%.1f мс", d.visionMs))
            row("  gaze: модель+warp", String(format: "%.1f мс", d.modelMs))
            row("Вхід камери", s.inputSize == .zero ? "—" : "\(Int(s.inputSize.width))×\(Int(s.inputSize.height))")
            row("Пропущено кадрів", "\(s.dropped)")
            row("Віртуальна камера", s.sinkConnected ? "підключена" : "ні")
            Divider().padding(.vertical, 4)
            row("Стан погляду", d.status)
            row("Yaw / pitch", [d.yawDeg, d.pitchDeg].map { $0.map { String(format: "%.0f°", $0) } ?? "—" }.joined(separator: " / "))
            row("Кут корекції (верт / гор)", String(format: "%.0f° / %.0f°", d.angle.v, d.angle.h))
            row("Відстань до очей", d.eyeDistanceCm > 0 ? String(format: "%.0f см", d.eyeDistanceCm) : "—")
            row("Зіниця / нейтраль", [d.pupilX, d.neutralX].map { $0.map { String(format: "%+.3f", $0) } ?? "—" }.joined(separator: " / "))
            row("Гасіння руху", String(format: "%+.1f°", d.followDeg))
            row("Рух очей", String(format: "%.1f px/кадр%@", d.motionPx, d.smoothingReset ? " (скид)" : ""))
        }
        .font(.caption)
    }

    private func row(_ k: String, _ v: String) -> some View {
        HStack(alignment: .top) {
            Text(k).foregroundStyle(.secondary)
            Spacer()
            Text(v).monospacedDigit().multilineTextAlignment(.trailing)
        }
    }
}

/// Face box, eye crops and the 12 landmark points, in 1280x720 frame coordinates.
struct DebugOverlay: View {
    let debug: GazeEffect.Debug

    var body: some View {
        Canvas { ctx, size in
            let sx = size.width / CGFloat(CameraPipeline.width), sy = size.height / CGFloat(CameraPipeline.height)
            func r(_ b: CGRect) -> CGRect { CGRect(x: b.minX * sx, y: b.minY * sy, width: b.width * sx, height: b.height * sy) }
            if let f = debug.faceBox { ctx.stroke(Path(r(f)), with: .color(.yellow.opacity(0.7)), lineWidth: 1) }
            for b in debug.eyeBoxes { ctx.stroke(Path(r(b)), with: .color(.green), lineWidth: 1.5) }
            for p in debug.pupils {
                let c = CGPoint(x: p.x * sx, y: p.y * sy)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 3, y: c.y - 3, width: 6, height: 6)), with: .color(.red))
            }
            for p in debug.eyePoints {
                let c = CGPoint(x: p.x * sx, y: p.y * sy)
                ctx.fill(Path(ellipseIn: CGRect(x: c.x - 2, y: c.y - 2, width: 4, height: 4)), with: .color(.cyan))
            }
        }
        .allowsHitTesting(false)
    }
}

/// Pulls the pipeline's latest frames ~30×/s while the window is open.
@MainActor
final class PreviewFeed: ObservableObject {
    @Published var processed: CGImage?
    @Published var raw: CGImage?
    @Published var stats = CameraPipeline.Stats()
    @Published var gazeDebug = GazeEffect.Debug()
    @Published var showOverlay = true
    @Published var showHUD = true
    @Published private(set) var showRaw = false
    @Published var recording = false
    @Published var captures: [URL] = []
    @Published var reviewing: ReviewItem?

    struct ReviewItem: Identifiable { let url: URL; var id: URL { url } }

    private var pressedAt: Date?
    private var pendingSnapshot: URL?
    private var keyMonitor: Any?

    private weak var pipeline: CameraPipeline?
    private var timer: Timer?
    private let ci = CIContext()

    func attach(_ p: CameraPipeline) {
        pipeline = p
        p.keepRaw = true
        loadCaptures()
        keyMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .keyUp]) { [weak self] e in
            guard e.keyCode == 49, e.window?.identifier?.rawValue.contains("preview") == true || e.window?.title == "GazeCam",
                  self?.reviewing == nil else { return e }
            if e.type == .keyDown { if !e.isARepeat { self?.press() } } else { self?.release() }
            return nil
        }
        timer?.invalidate()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0 / 30, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    func detach() {
        timer?.invalidate(); timer = nil
        pipeline?.keepRaw = false
        if let keyMonitor { NSEvent.removeMonitor(keyMonitor) }
        keyMonitor = nil
        if recording { release() }
    }

    func setShowRaw(_ on: Bool) { showRaw = on }

    /// Space down: snapshot this instant and start recording. Space up: a short tap keeps only the snapshot.
    func press() {
        guard let cap = pipeline?.capture, pressedAt == nil else { return }
        pressedAt = Date()
        cap.snapshot { [weak self] url in Task { @MainActor in self?.pendingSnapshot = url; self?.loadCaptures() } }
        if cap.startRecording() != nil { recording = true }
    }

    func release() {
        guard let cap = pipeline?.capture, let t = pressedAt else { return }
        pressedAt = nil
        let held = Date().timeIntervalSince(t) > 0.4
        recording = false
        cap.stopRecording(discard: !held) { [weak self] url in
            Task { @MainActor in
                guard let self else { return }
                if held, let url {
                    if let s = self.pendingSnapshot { try? FileManager.default.removeItem(at: s) }
                    self.reviewing = .init(url: url)
                }
                self.pendingSnapshot = nil
                self.loadCaptures()
            }
        }
    }

    func loadCaptures() {
        let fm = FileManager.default
        let urls = (try? fm.contentsOfDirectory(at: DebugCapture.dir, includingPropertiesForKeys: [.creationDateKey])) ?? []
        captures = urls.filter { ["png", "mp4"].contains($0.pathExtension) }
            .sorted { ($0.lastPathComponent) > ($1.lastPathComponent) }
    }

    private func tick() {
        guard let p = pipeline else { return }
        var s = p.stats
        processed = s.processed.flatMap(cgImage)
        raw = showRaw ? s.raw.flatMap(cgImage) : nil
        s.processed = nil; s.raw = nil
        stats = s
        gazeDebug = p.gaze.debug
    }

    private func cgImage(_ pb: CVPixelBuffer) -> CGImage? {
        ci.createCGImage(CIImage(cvPixelBuffer: pb), from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb)))
    }
}

/// Plays back a capture with frame stepping and slow motion, or shows a snapshot.
struct CaptureReview: View {
    let url: URL
    @Environment(\.dismiss) private var dismiss
    @State private var player: AVPlayer?

    var body: some View {
        VStack(spacing: 8) {
            if url.pathExtension == "mp4", let player {
                PlayerView(player: player).aspectRatio(32 / 9, contentMode: .fit)
                HStack {
                    Button("◀︎ кадр") { player.pause(); player.currentItem?.step(byCount: -1) }.keyboardShortcut(.leftArrow, modifiers: [])
                    Button("кадр ▶︎") { player.pause(); player.currentItem?.step(byCount: 1) }.keyboardShortcut(.rightArrow, modifiers: [])
                    Button("0.25×") { player.rate = 0.25 }
                    Button("1×") { player.rate = 1 }
                    Spacer()
                    Text(url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    Button("У Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    Button("Закрити") { dismiss() }.keyboardShortcut(.cancelAction)
                }
            } else if let img = NSImage(contentsOf: url) {
                Image(nsImage: img).resizable().aspectRatio(contentMode: .fit)
                HStack {
                    Text(url.lastPathComponent).font(.caption).foregroundStyle(.secondary)
                    Spacer()
                    Button("У Finder") { NSWorkspace.shared.activateFileViewerSelecting([url]) }
                    Button("Закрити") { dismiss() }.keyboardShortcut(.cancelAction)
                }
            }
        }
        .padding(12)
        .frame(minWidth: 1100, minHeight: 420)
        .onAppear { if url.pathExtension == "mp4" { player = AVPlayer(url: url) } }
    }
}

/// AppKit AVPlayerView. SwiftUI's VideoPlayer aborts on first use here (missing _AVKit_SwiftUI superclass metadata).
struct PlayerView: NSViewRepresentable {
    let player: AVPlayer
    func makeNSView(context: Context) -> AVPlayerView {
        let v = AVPlayerView()
        v.player = player
        v.controlsStyle = .inline
        v.showsFrameSteppingButtons = true
        return v
    }
    func updateNSView(_ v: AVPlayerView, context: Context) { v.player = player }
}
