import AVFoundation
import Combine
import SwiftUI
import SystemExtensions

@main
struct GazeCamApp: App {
    @StateObject private var model = AppModel()

    init() { TestHarness.runIfRequested(); PupilProbe.runIfRequested(); VideoHarness.runIfRequested() }

    var body: some Scene {
        MenuBarExtra("GazeCam", systemImage: model.running ? "eye.fill" : "eye") {
            MenuView().environmentObject(model)
        }
        .menuBarExtraStyle(.window)

        Window("GazeCam", id: "preview") {
            PreviewWindow().environmentObject(model)
        }
        .defaultSize(width: 1180, height: 640)
    }
}

/// Gaze tunables, persisted in UserDefaults and pushed into the effect on every change.
struct GazeSettings: Codable, Equatable {
    var cameraUp = 18.0
    var cameraRight = 0.0
    var cameraDepth = -1.0
    var strength = 1.0
    var focalLength = 1000.0
    var ipd = 6.3
    var maxHeadDeg = 20.0
    var feather = 3.0
    var smoothing = 0.6
    var followGaze = 1.0
    var maxRedirect = 20.0
    var maxRedirectH = 30.0
    var openRatio = 0.4
    var vCal: GazeEffect.VerticalCal?
    var schema = 2
    var userModel = false
    var pupilAtCamera: Double?
    var useEstimator = false

    init() {}
    // Tolerant decode: settings saved by an older build keep their values, new fields take defaults.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let d = GazeSettings()
        cameraUp = try c.decodeIfPresent(Double.self, forKey: .cameraUp) ?? d.cameraUp
        cameraRight = try c.decodeIfPresent(Double.self, forKey: .cameraRight) ?? d.cameraRight
        cameraDepth = try c.decodeIfPresent(Double.self, forKey: .cameraDepth) ?? d.cameraDepth
        strength = try c.decodeIfPresent(Double.self, forKey: .strength) ?? d.strength
        focalLength = try c.decodeIfPresent(Double.self, forKey: .focalLength) ?? d.focalLength
        ipd = try c.decodeIfPresent(Double.self, forKey: .ipd) ?? d.ipd
        maxHeadDeg = try c.decodeIfPresent(Double.self, forKey: .maxHeadDeg) ?? d.maxHeadDeg
        feather = try c.decodeIfPresent(Double.self, forKey: .feather) ?? d.feather
        smoothing = try c.decodeIfPresent(Double.self, forKey: .smoothing) ?? d.smoothing
        followGaze = try c.decodeIfPresent(Double.self, forKey: .followGaze) ?? d.followGaze
        maxRedirect = try c.decodeIfPresent(Double.self, forKey: .maxRedirect) ?? d.maxRedirect
        maxRedirectH = try c.decodeIfPresent(Double.self, forKey: .maxRedirectH) ?? d.maxRedirectH
        openRatio = try c.decodeIfPresent(Double.self, forKey: .openRatio) ?? d.openRatio
        vCal = try c.decodeIfPresent(GazeEffect.VerticalCal.self, forKey: .vCal)
        userModel = try c.decodeIfPresent(Bool.self, forKey: .userModel) ?? false
        pupilAtCamera = try c.decodeIfPresent(Double.self, forKey: .pupilAtCamera)
        useEstimator = try c.decodeIfPresent(Bool.self, forKey: .useEstimator) ?? false
        // Schema 1 → 2: strength >1 inflated the geometric base past the cap (28° → white eyes), and the closed-eye
        // ratio 0.6 caught normal downward looks. Reset both once.
        if (try c.decodeIfPresent(Int.self, forKey: .schema)) == nil { strength = 1; openRatio = d.openRatio }
    }
}

@MainActor
final class AppModel: NSObject, ObservableObject, OSSystemExtensionRequestDelegate {
    @Published var status = "Готово"
    @Published var extStatus = ""
    @Published var running = false
    @Published var cameras: [AVCaptureDevice] = CameraPipeline.cameras()
    @Published var cameraID: String = ""
    @Published var effectState: [String: Bool] = [:]
    @Published var gaze: GazeSettings {
        didSet { applyGaze(); if let d = try? JSONEncoder().encode(gaze) { UserDefaults.standard.set(d, forKey: "gaze") } }
    }

    let pipeline = CameraPipeline()
    private let extID = "dev.iamhero.GazeCam.Extension"

    override init() {
        gaze = (UserDefaults.standard.data(forKey: "gaze")).flatMap { try? JSONDecoder().decode(GazeSettings.self, from: $0) } ?? GazeSettings()
        super.init()
        applyGaze()
        cameraID = cameras.first?.uniqueID ?? ""
        for e in pipeline.effects { effectState[e.id] = e.enabled }
        pipeline.onStatus = { [weak self] s in Task { @MainActor in self?.status = s } }
        // Activation is idempotent; doing it on launch also picks up a rebuilt extension.
        installExtension()
        calibration.onCameraPupil = { [weak self] x in self?.gaze.pupilAtCamera = Double(x) }
        calSub = calibration.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
    }
    private var calSub: AnyCancellable?

    private func applyGaze() {
        let g = pipeline.gaze
        g.cameraOffset = SIMD3(gaze.cameraRight, -gaze.cameraUp, gaze.cameraDepth)
        g.strength = Float(gaze.strength)
        g.focalLength = gaze.focalLength
        g.ipd = gaze.ipd
        g.maxHeadAngle = Float(gaze.maxHeadDeg * .pi / 180)
        g.feather = Float(gaze.feather)
        g.smoothing = Float(gaze.smoothing)
        g.followGaze = Float(gaze.followGaze)
        g.maxRedirect = Float(gaze.maxRedirect)
        g.maxRedirectH = Float(gaze.maxRedirectH)
        g.openRatio = Float(gaze.openRatio)
        g.vCal = gaze.vCal ?? GazeEffect.VerticalCal()
        if g.useUserModel != gaze.userModel { g.useUserModel = gaze.userModel }
        g.useEstimator = gaze.useEstimator
        if let p = gaze.pupilAtCamera, g.neutralX != Float(p) { g.setNeutral(Float(p)) }
    }

    let calibration = CornerCalibration()

    func startDatasetRun() {
        calibration.start(pipeline: pipeline, mode: .dataset) { _, _ in }
        objectWillChange.send()
    }

    func startCalibration() {
        calibration.start(pipeline: pipeline) { [weak self] cal, neutral in
            self?.gaze.vCal = cal
            if let neutral { self?.gaze.pupilAtCamera = Double(neutral) }
            self?.objectWillChange.send()
        }
        objectWillChange.send()
    }

    func setEffect(_ id: String, _ on: Bool) {
        pipeline.effects.first { $0.id == id }?.enabled = on
        effectState[id] = on
    }

    func toggleRun() {
        if running { pipeline.stop(); running = false; return }
        guard let cam = cameras.first(where: { $0.uniqueID == cameraID }) ?? cameras.first else { status = "Немає камери"; return }
        AVCaptureDevice.requestAccess(for: .video) { ok in
            Task { @MainActor in
                guard ok else { self.status = "Нема доступу до камери"; return }
                self.pipeline.start(camera: cam)
                self.running = true
            }
        }
    }

    func installExtension() {
        let r = OSSystemExtensionRequest.activationRequest(forExtensionWithIdentifier: extID, queue: .main)
        r.delegate = self
        OSSystemExtensionManager.shared.submitRequest(r)
        extStatus = "Встановлюю…"
    }

    func uninstallExtension() {
        let r = OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: extID, queue: .main)
        r.delegate = self
        OSSystemExtensionManager.shared.submitRequest(r)
        extStatus = "Видаляю…"
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, actionForReplacingExtension existing: OSSystemExtensionProperties,
                             withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction { .replace }

    nonisolated func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        Task { @MainActor in self.extStatus = "Дозволь у System Settings → General → Login Items & Extensions → Camera Extensions" }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
        Task { @MainActor in self.extStatus = result == .completed ? "Розширення OK" : "Потрібен перезапуск" }
    }

    nonisolated func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        Task { @MainActor in self.extStatus = "Помилка: \(error.localizedDescription)" }
    }
}

struct MenuView: View {
    @EnvironmentObject var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("GazeCam").font(.headline)
            Picker("Камера", selection: $model.cameraID) {
                ForEach(model.cameras, id: \.uniqueID) { Text($0.localizedName).tag($0.uniqueID) }
            }
            .disabled(model.running)

            Button(model.running ? "Зупинити" : "Старт") { model.toggleRun() }
                .keyboardShortcut(.defaultAction)
            Text(model.status).font(.caption).foregroundStyle(.secondary)

            Divider()
            Text("Ефекти").font(.subheadline)
            ForEach(model.pipeline.effects, id: \.id) { e in
                Toggle(e.title, isOn: Binding(get: { model.effectState[e.id] ?? false }, set: { model.setEffect(e.id, $0) }))
            }
            Button("Превʼю і налаштування…") {
                openWindow(id: "preview")
                NSApp.activate(ignoringOtherApps: true)
            }

            Divider()
            HStack {
                Button("Встановити камеру") { model.installExtension() }
                Button("Видалити") { model.uninstallExtension() }
            }
            if !model.extStatus.isEmpty { Text(model.extStatus).font(.caption).foregroundStyle(.secondary) }

            Divider()
            Button("Вийти") { NSApp.terminate(nil) }
        }
        .padding(14)
        .frame(width: 300)
    }
}
