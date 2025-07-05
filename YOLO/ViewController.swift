//  Ultralytics YOLO 🚀 - AGPL-3.0 License
//
//  Main View Controller for Ultralytics YOLO App
//  This file is part of the Ultralytics YOLO app, enabling real-time object detection using YOLO11 models on iOS devices.
//  Licensed under AGPL-3.0. For commercial use, refer to Ultralytics licensing: https://ultralytics.com/license
//  Access the source code: https://github.com/ultralytics/yolo-ios-app
//
//  This ViewController manages the app's main screen, handling video capture, model selection, detection visualization,
//  and user interactions. It sets up and controls the video preview layer, handles model switching via a segmented control,
//  manages UI elements like sliders for confidence and IoU thresholds, and displays detection results on the video feed.
//  It leverages CoreML, Vision, and AVFoundation frameworks to perform real-time object detection and to interface with
//  the device's camera.

import AVFoundation
import CoreML
import CoreMedia
import UIKit
import Vision

// Change to only use YOLO11n model
var mlModel = try! yolo11n(configuration: mlmodelConfig).model
var mlmodelConfig: MLModelConfiguration = {
  let config = MLModelConfiguration()

  if #available(iOS 17.0, *) {
    config.setValue(1, forKey: "experimentalMLE5EngineUsage")
  }

  return config
}()

var numberDetectModel = try! best(configuration: mlmodelConfig).model
var numberDetector = try! VNCoreMLModel(for: numberDetectModel)

class ViewController: UIViewController {
  @IBOutlet var videoPreview: UIView!
  @IBOutlet var View0: UIView!
  @IBOutlet var segmentedControl: UISegmentedControl!
  @IBOutlet var playButtonOutlet: UIBarButtonItem!
  @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
  @IBOutlet var slider: UISlider!
  @IBOutlet var sliderConf: UISlider!
  @IBOutlet weak var sliderConfLandScape: UISlider!
  @IBOutlet var sliderIoU: UISlider!
  @IBOutlet weak var sliderIoULandScape: UISlider!
  @IBOutlet weak var labelName: UILabel!
  @IBOutlet weak var labelFPS: UILabel!
  @IBOutlet weak var labelZoom: UILabel!
  @IBOutlet weak var labelVersion: UILabel!
  @IBOutlet weak var labelSlider: UILabel!
  @IBOutlet weak var labelSliderConf: UILabel!
  @IBOutlet weak var labelSliderConfLandScape: UILabel!
  @IBOutlet weak var labelSliderIoU: UILabel!
  @IBOutlet weak var labelSliderIoULandScape: UILabel!
  @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
  @IBOutlet weak var focus: UIImageView!
  @IBOutlet weak var toolBar: UIToolbar!

  let selection = UISelectionFeedbackGenerator()
  var detector = try! VNCoreMLModel(for: mlModel)
  var session: AVCaptureSession!
  var videoCapture: VideoCapture!
  var currentBuffer: CVPixelBuffer?
  var framesDone = 0
  var t0 = 0.0  // inference start
  var t1 = 0.0  // inference dt
  var t2 = 0.0  // inference dt smoothed
  var t3 = CACurrentMediaTime()  // FPS start
  var t4 = 0.0  // FPS dt smoothed
  // var cameraOutput: AVCapturePhotoOutput!
  var longSide: CGFloat = 3
  var shortSide: CGFloat = 4
  var frameSizeCaptured = false

  // Developer mode
  let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")  // developer mode selected in settings
  let save_detections = false  // write every detection to detections.txt
  let save_frames = false  // write every frame to frames.txt

  lazy var visionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: {
        [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    // NOTE: BoundingBoxView object scaling depends on request.imageCropAndScaleOption https://developer.apple.com/documentation/vision/vnimagecropandscaleoption
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    return request
  }()

  lazy var numberDetectionRequest: VNCoreMLRequest = {
    let request = VNCoreMLRequest(
      model: numberDetector,
      completionHandler: { [weak self] request, error in
        self?.processNumberDetections(for: request, error: error, originalBusRect: self?.currentBusRect)
      })
    request.imageCropAndScaleOption = .scaleFill
    return request
  }()

  // Add this to store the current bus rect for reference
  private var currentBusRect: CGRect?

  // Add these properties to the ViewController class
  private var detectedBusNumbers: [String] = []
  private var busNumberTimer: Timer?
  private var speechSynthesizer = AVSpeechSynthesizer()
  private var lastAnnouncementTime: TimeInterval = 0
  private var lastAnnouncedNumber: String?

  override func viewDidLoad() {
    super.viewDidLoad()
    slider.value = 30
    setLabels()
    setUpBoundingBoxViews()
    setUpOrientationChangeNotification()
    startVideo()
    
    // Hide the segmented control since we're only using YOLO11n
    segmentedControl.isHidden = true
    
    // Hide confidence and IoU sliders since we're using fixed values of 0.5
    sliderConf.isHidden = true
    labelSliderConf.isHidden = true
    sliderIoU.isHidden = true
    labelSliderIoU.isHidden = true
    
    // Also hide landscape versions
    sliderConfLandScape.isHidden = true
    labelSliderConfLandScape.isHidden = true
    sliderIoULandScape.isHidden = true
    labelSliderIoULandScape.isHidden = true
    
    // Hide the max items slider and its label
    slider.isHidden = true
    labelSlider.isHidden = true
    
    // Hide the Ultralytics logo
    hideUltralyticsLogo()
    
    // Set fixed threshold values
    detector.featureProvider = ThresholdProvider(iouThreshold: 0.5, confidenceThreshold: 0.5)
    
    // Set up timer for bus number announcements
    busNumberTimer = Timer.scheduledTimer(timeInterval: 5.0, target: self, selector: #selector(checkAndAnnounceBusNumber), userInfo: nil, repeats: true)
    
    // Set speech synthesizer delegate
    speechSynthesizer.delegate = self
  }

  override func viewWillTransition(
    to size: CGSize, with coordinator: any UIViewControllerTransitionCoordinator
  ) {
    super.viewWillTransition(to: size, with: coordinator)

    // No need to toggle sliders visibility based on orientation anymore
    if size.width > size.height {
      toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)
    } else {
      toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
      toolBar.setShadowImage(nil, forToolbarPosition: .any)
    }
    
    self.videoCapture.previewLayer?.frame = CGRect(
      x: 0, y: 0, width: size.width, height: size.height)
  }

  private func setUpOrientationChangeNotification() {
    NotificationCenter.default.addObserver(
      self, selector: #selector(orientationDidChange),
      name: UIDevice.orientationDidChangeNotification, object: nil)
  }

  @objc func orientationDidChange() {
    videoCapture.updateVideoOrientation()
    //      frameSizeCaptured = false
  }

  @IBAction func vibrate(_ sender: Any) {
    selection.selectionChanged()
  }

  @IBAction func indexChanged(_ sender: Any) {
    selection.selectionChanged()
    activityIndicator.startAnimating()

    mlModel = try! yolo11n(configuration: .init()).model
    
    setModel()
    setUpBoundingBoxViews()
    activityIndicator.stopAnimating()
  }

  func setModel() {

    /// VNCoreMLModel
    detector = try! VNCoreMLModel(for: mlModel)
    detector.featureProvider = ThresholdProvider()

    /// VNCoreMLRequest
    let request = VNCoreMLRequest(
      model: detector,
      completionHandler: { [weak self] request, error in
        self?.processObservations(for: request, error: error)
      })
    request.imageCropAndScaleOption = .scaleFill  // .scaleFit, .scaleFill, .centerCrop
    visionRequest = request
    t2 = 0.0  // inference dt smoothed
    t3 = CACurrentMediaTime()  // FPS start
    t4 = 0.0  // FPS dt smoothed
  }

  @IBAction func takePhoto(_ sender: Any?) {
    let t0 = DispatchTime.now().uptimeNanoseconds

    // 1. captureSession and cameraOutput
    // session = videoCapture.captureSession  // session = AVCaptureSession()
    // session.sessionPreset = AVCaptureSession.Preset.photo
    // cameraOutput = AVCapturePhotoOutput()
    // cameraOutput.isHighResolutionCaptureEnabled = true
    // cameraOutput.isDualCameraDualPhotoDeliveryEnabled = true
    // print("1 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)

    // 2. Settings
    let settings = AVCapturePhotoSettings()
    // settings.flashMode = .off
    // settings.isHighResolutionPhotoEnabled = cameraOutput.isHighResolutionCaptureEnabled
    // settings.isDualCameraDualPhotoDeliveryEnabled = self.videoCapture.cameraOutput.isDualCameraDualPhotoDeliveryEnabled

    // 3. Capture Photo
    usleep(20_000)  // short 10 ms delay to allow camera to focus
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    print("3 Done: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
  }

  @IBAction func logoButton(_ sender: Any) {
    selection.selectionChanged()
    if let link = URL(string: "https://www.ultralytics.com") {
      UIApplication.shared.open(link)
    }
  }

  func setLabels() {
    self.labelName.text = "Oculis"
    self.labelVersion.text = "Version " + UserDefaults.standard.string(forKey: "app_version")!
  }

  @IBAction func playButton(_ sender: Any) {
    selection.selectionChanged()
    self.videoCapture.start()
    playButtonOutlet.isEnabled = false
    pauseButtonOutlet.isEnabled = true
  }

  @IBAction func pauseButton(_ sender: Any?) {
    selection.selectionChanged()
    self.videoCapture.stop()
    playButtonOutlet.isEnabled = true
    pauseButtonOutlet.isEnabled = false
  }

  @IBAction func switchCameraTapped(_ sender: Any) {
    self.videoCapture.captureSession.beginConfiguration()
    let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
    self.videoCapture.captureSession.removeInput(currentInput!)
    guard let currentPosition = currentInput?.device.position else { return }

    let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back

    let newCameraDevice = bestCaptureDevice(for: nextCameraPosition)

    guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else {
      return
    }

    self.videoCapture.captureSession.addInput(videoInput1)
    self.videoCapture.updateVideoOrientation()

    self.videoCapture.captureSession.commitConfiguration()

  }

  // share image
  @IBAction func shareButton(_ sender: Any) {
    selection.selectionChanged()
    let settings = AVCapturePhotoSettings()
    self.videoCapture.cameraOutput.capturePhoto(
      with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
  }

  // share screenshot
  @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
    // let layer = UIApplication.shared.keyWindow!.layer
    // let scale = UIScreen.main.scale
    // UIGraphicsBeginImageContextWithOptions(layer.frame.size, false, scale);
    // layer.render(in: UIGraphicsGetCurrentContext()!)
    // let screenshot = UIGraphicsGetImageFromCurrentImageContext()
    // UIGraphicsEndImageContext()

    // let screenshot = UIApplication.shared.screenShot
    // UIImageWriteToSavedPhotosAlbum(screenshot!, nil, nil, nil)
  }

  let maxBoundingBoxViews = 100
  var boundingBoxViews = [BoundingBoxView]()
  var colors: [String: UIColor] = [:]
  let ultralyticsColorsolors: [UIColor] = [
    UIColor(red: 4 / 255, green: 42 / 255, blue: 255 / 255, alpha: 0.6),  // #042AFF
    UIColor(red: 11 / 255, green: 219 / 255, blue: 235 / 255, alpha: 0.6),  // #0BDBEB
    UIColor(red: 243 / 255, green: 243 / 255, blue: 243 / 255, alpha: 0.6),  // #F3F3F3
    UIColor(red: 0 / 255, green: 223 / 255, blue: 183 / 255, alpha: 0.6),  // #00DFB7
    UIColor(red: 17 / 255, green: 31 / 255, blue: 104 / 255, alpha: 0.6),  // #111F68
    UIColor(red: 255 / 255, green: 111 / 255, blue: 221 / 255, alpha: 0.6),  // #FF6FDD
    UIColor(red: 255 / 255, green: 68 / 255, blue: 79 / 255, alpha: 0.6),  // #FF444F
    UIColor(red: 204 / 255, green: 237 / 255, blue: 0 / 255, alpha: 0.6),  // #CCED00
    UIColor(red: 0 / 255, green: 243 / 255, blue: 68 / 255, alpha: 0.6),  // #00F344
    UIColor(red: 189 / 255, green: 0 / 255, blue: 255 / 255, alpha: 0.6),  // #BD00FF
    UIColor(red: 0 / 255, green: 180 / 255, blue: 255 / 255, alpha: 0.6),  // #00B4FF
    UIColor(red: 221 / 255, green: 0 / 255, blue: 186 / 255, alpha: 0.6),  // #DD00BA
    UIColor(red: 0 / 255, green: 255 / 255, blue: 255 / 255, alpha: 0.6),  // #00FFFF
    UIColor(red: 38 / 255, green: 192 / 255, blue: 0 / 255, alpha: 0.6),  // #26C000
    UIColor(red: 1 / 255, green: 255 / 255, blue: 179 / 255, alpha: 0.6),  // #01FFB3
    UIColor(red: 125 / 255, green: 36 / 255, blue: 255 / 255, alpha: 0.6),  // #7D24FF
    UIColor(red: 123 / 255, green: 0 / 255, blue: 104 / 255, alpha: 0.6),  // #7B0068
    UIColor(red: 255 / 255, green: 27 / 255, blue: 108 / 255, alpha: 0.6),  // #FF1B6C
    UIColor(red: 252 / 255, green: 109 / 255, blue: 47 / 255, alpha: 0.6),  // #FC6D2F
    UIColor(red: 162 / 255, green: 255 / 255, blue: 11 / 255, alpha: 0.6),  // #A2FF0B
  ]

  func setUpBoundingBoxViews() {
    // Ensure all bounding box views are initialized up to the maximum allowed.
    while boundingBoxViews.count < maxBoundingBoxViews {
      boundingBoxViews.append(BoundingBoxView())
    }

    // Retrieve class labels directly from the CoreML model's class labels, if available.
    guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
      fatalError("Class labels are missing from the model description")
    }

    // Assign random colors to the classes.
    var count = 0
    for label in classLabels {
      let color = ultralyticsColorsolors[count]
      count += 1
      if count > 19 {
        count = 0
      }
      colors[label] = color

    }
  }

  func startVideo() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    videoCapture.setUp(sessionPreset: .photo) { success in
      // .hd4K3840x2160 or .photo (4032x3024)  Warning: 4k may not work on all devices i.e. 2019 iPod
      if success {
        // Add the video preview into the UI.
        if let previewLayer = self.videoCapture.previewLayer {
          self.videoPreview.layer.addSublayer(previewLayer)
          self.videoCapture.previewLayer?.frame = self.videoPreview.bounds  // resize preview layer
        }

        // Add the bounding box layers to the UI, on top of the video preview.
        for box in self.boundingBoxViews {
          box.addToLayer(self.videoPreview.layer)
        }

        // Once everything is set up, we can start capturing live video.
        self.videoCapture.start()
      }
    }
  }

  func processObservations(for request: VNRequest, error: Error?) {
    DispatchQueue.main.async {
      if let results = request.results as? [VNRecognizedObjectObservation] {
        self.show(predictions: results)
      } else {
        self.show(predictions: [])
      }

      // Measure FPS
      if self.t1 < 10.0 {  // valid dt
        self.t2 = self.t1 * 0.05 + self.t2 * 0.95  // smoothed inference time
      }
      self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95  // smoothed delivered FPS
      self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)  // t2 seconds to ms
      self.t3 = CACurrentMediaTime()
    }
  }

  // Save text file
  func saveText(text: String, file: String = "saved.txt") {
    if let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first {
      let fileURL = dir.appendingPathComponent(file)

      // Writing
      do {  // Append to file if it exists
        let fileHandle = try FileHandle(forWritingTo: fileURL)
        fileHandle.seekToEndOfFile()
        fileHandle.write(text.data(using: .utf8)!)
        fileHandle.closeFile()
      } catch {  // Create new file and write
        do {
          try text.write(to: fileURL, atomically: false, encoding: .utf8)
        } catch {
          print("no file written")
        }
      }

      // Reading
      // do {let text2 = try String(contentsOf: fileURL, encoding: .utf8)} catch {/* error handling here */}
    }
  }

  // Save image file
  func saveImage() {
    let dir = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
    let fileURL = dir!.appendingPathComponent("saved.jpg")
    let image = UIImage(named: "lighthouse.png")
    FileManager.default.createFile(
      atPath: fileURL.path, contents: image!.jpegData(compressionQuality: 0.5), attributes: nil)
  }

  // Return hard drive space (GB)
  func freeSpace() -> Double {
    let fileURL = URL(fileURLWithPath: NSHomeDirectory() as String)
    do {
      let values = try fileURL.resourceValues(forKeys: [
        .volumeAvailableCapacityForImportantUsageKey
      ])
      return Double(values.volumeAvailableCapacityForImportantUsage!) / 1E9  // Bytes to GB
    } catch {
      print("Error retrieving storage capacity: \(error.localizedDescription)")
    }
    return 0
  }

  // Return RAM usage (GB)
  func memoryUsage() -> Double {
    var taskInfo = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
    let kerr: kern_return_t = withUnsafeMutablePointer(to: &taskInfo) {
      $0.withMemoryRebound(to: integer_t.self, capacity: 1) {
        task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
      }
    }
    if kerr == KERN_SUCCESS {
      return Double(taskInfo.resident_size) / 1E9  // Bytes to GB
    } else {
      return 0
    }
  }

  func show(predictions: [VNRecognizedObjectObservation]) {
    // Clear previous bus bounding boxes
    
    // Track if we find a bus in the current frame
    var foundBusInCurrentFrame = false

    var str = ""
    // date
    let date = Date()
    let calendar = Calendar.current
    let hour = calendar.component(.hour, from: date)
    let minutes = calendar.component(.minute, from: date)
    let seconds = calendar.component(.second, from: date)
    let nanoseconds = calendar.component(.nanosecond, from: date)
    let sec_day =
      Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9  // seconds in the day

    self.labelSlider.text =
      String(predictions.count) + " items (max " + String(Int(slider.value)) + ")"
    let width = videoPreview.bounds.width  // 375 pix
    let height = videoPreview.bounds.height  // 812 pix

    if UIDevice.current.orientation == .portrait {

      // ratio = videoPreview AR divided by sessionPreset AR
      var ratio: CGFloat = 1.0
      if videoCapture.captureSession.sessionPreset == .photo {
        ratio = (height / width) / (4.0 / 3.0)  // .photo
      } else {
        ratio = (height / width) / (16.0 / 9.0)  // .hd4K3840x2160, .hd1920x1080, .hd1280x720 etc.
      }

      for i in 0..<boundingBoxViews.count {
        if i < predictions.count && i < Int(slider.value) {
          let prediction = predictions[i]
          
          // Get prediction details and check class ID
          // Note: This assumes the model outputs class indexes directly
          // You might need to adjust based on how your model represents classes
          let bestClass = prediction.labels[0].identifier
          let classIndex = getClassIndex(for: bestClass) // You'll need to implement this function
          
          // Skip if not a bus (index 5)
          if classIndex != 5 {
            boundingBoxViews[i].hide()
            continue
          }
          
          // If we get here, it's a bus detection - proceed with the normal display code
          let confidence = prediction.labels[0].confidence
          
          if confidence > 0.2 {
            var rect = prediction.boundingBox  // normalized xywh, origin lower left
            switch UIDevice.current.orientation {
            case .portraitUpsideDown:
              rect = CGRect(
                x: 1.0 - rect.origin.x - rect.width,
                y: 1.0 - rect.origin.y - rect.height,
                width: rect.width,
                height: rect.height)
            case .landscapeLeft:
              rect = CGRect(
                x: rect.origin.x,
                y: rect.origin.y,
                width: rect.width,
                height: rect.height)
            case .landscapeRight:
              rect = CGRect(
                x: rect.origin.x,
                y: rect.origin.y,
                width: rect.width,
                height: rect.height)
            case .unknown:
              print("The device orientation is unknown, the predictions may be affected")
              fallthrough
            default: break
            }

            if ratio >= 1 {  // iPhone ratio = 1.218
              let offset = (1 - ratio) * (0.5 - rect.minX)
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
              rect = rect.applying(transform)
              rect.size.width *= ratio
            } else {  // iPad ratio = 0.75
              let offset = (ratio - 1) * (0.5 - rect.maxY)
              let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
              rect = rect.applying(transform)
              ratio = (height / width) / (3.0 / 4.0)
              rect.size.height /= ratio
            }

            // Scale normalized to pixels [375, 812] [width, height]
            rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))

            // Show the bounding box.
            boundingBoxViews[i].show(
              frame: rect,
              label: String(format: "%@ %.1f", bestClass, confidence * 100),
              color: colors[bestClass] ?? UIColor.white,
              alpha: CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9))

            if developerMode {
              // Write
              if save_detections {
                str += String(
                  format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                  sec_day, freeSpace(), UIDevice.current.batteryLevel, bestClass, confidence,
                  rect.origin.x, rect.origin.y, rect.size.width, rect.size.height)
              }
            }

            // Store the normalized bus rect for number detection
            let busRect = prediction.boundingBox
            
            // If we have a current pixel buffer, process it for number detection
            if let currentPixelBuffer = self.currentBuffer {
                self.processBusDetection(busRect: busRect, originalBuffer: currentPixelBuffer)
            }

            // Mark that we found a bus
            foundBusInCurrentFrame = true
          } else {
            boundingBoxViews[i].hide()
          }
        } else {
          boundingBoxViews[i].hide()
        }
      }
    } else {
      let frameAspectRatio = longSide / shortSide
      let viewAspectRatio = width / height
      var scaleX: CGFloat = 1.0
      var scaleY: CGFloat = 1.0
      var offsetX: CGFloat = 0.0
      var offsetY: CGFloat = 0.0

      if frameAspectRatio > viewAspectRatio {
        scaleY = height / shortSide
        scaleX = scaleY
        offsetX = (longSide * scaleX - width) / 2
      } else {
        scaleX = width / longSide
        scaleY = scaleX
        offsetY = (shortSide * scaleY - height) / 2
      }

      for i in 0..<boundingBoxViews.count {
        if i < predictions.count {
          let prediction = predictions[i]
          
          // Get prediction details and check class ID
          // Note: This assumes the model outputs class indexes directly
          // You might need to adjust based on how your model represents classes
          let bestClass = prediction.labels[0].identifier
          let classIndex = getClassIndex(for: bestClass) // You'll need to implement this function
          
          // Skip if not a bus (index 5)
          if classIndex != 5 {
            boundingBoxViews[i].hide()
            continue
          }
          
          // If we get here, it's a bus detection - proceed with the normal display code
          let confidence = prediction.labels[0].confidence
          
          if confidence > 0.2 {
            var rect = prediction.boundingBox

            rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
            rect.origin.y =
              height
              - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY)
            rect.size.width *= longSide * scaleX
            rect.size.height *= shortSide * scaleY

            let label = String(format: "%@ %.1f", bestClass, confidence * 100)
            let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
            // Show the bounding box.
            boundingBoxViews[i].show(
              frame: rect,
              label: label,
              color: colors[bestClass] ?? UIColor.white,
              alpha: alpha)

            // Store the normalized bus rect for number detection
            let busRect = prediction.boundingBox
            
            // If we have a current pixel buffer, process it for number detection
            if let currentPixelBuffer = self.currentBuffer {
                self.processBusDetection(busRect: busRect, originalBuffer: currentPixelBuffer)
            }

            // Mark that we found a bus
            foundBusInCurrentFrame = true
          } else {
            boundingBoxViews[i].hide()
          }
        } else {
          boundingBoxViews[i].hide()
        }
      }
    }
    // Write
    if developerMode {
      if save_detections {
        saveText(text: str, file: "detections.txt")  // Write stats for each detection
      }
      if save_frames {
        str = String(
          format: "%.3f %.3f %.3f %.3f %.1f %.1f %.1f\n",
          sec_day, freeSpace(), memoryUsage(), UIDevice.current.batteryLevel,
          self.t1 * 1000, self.t2 * 1000, 1 / self.t4)
        saveText(text: str, file: "frames.txt")  // Write stats for each image
      }
    }

    // Debug
    // print(str)
    // print(UIDevice.current.identifierForVendor!)
    // saveImage()
  }

  // Helper function to get class index from class name
  func getClassIndex(for className: String) -> Int {
    // This would depend on your model's class list
    // A simple implementation might be:
    let classList = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ] // complete COCO class list
    return classList.firstIndex(of: className) ?? -1
  }

  // Pinch to Zoom Start ---------------------------------------------------------------------------------------------
  let minimumZoom: CGFloat = 1.0
  let maximumZoom: CGFloat = 10.0
  var lastZoomFactor: CGFloat = 1.0

  @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
    let device = videoCapture.captureDevice

    // Return zoom value between the minimum and maximum zoom values
    func minMaxZoom(_ factor: CGFloat) -> CGFloat {
      return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
    }

    func update(scale factor: CGFloat) {
      do {
        try device.lockForConfiguration()
        defer {
          device.unlockForConfiguration()
        }
        device.videoZoomFactor = factor
      } catch {
        print("\(error.localizedDescription)")
      }
    }

    let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
    switch pinch.state {
    case .began, .changed:
      update(scale: newScaleFactor)
      self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
    case .ended:
      lastZoomFactor = minMaxZoom(newScaleFactor)
      update(scale: lastZoomFactor)
      self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
    default: break
    }
  }  // Pinch to Zoom End --------------------------------------------------------------------------------------------

  // Function to find and hide the Ultralytics logo
  private func hideUltralyticsLogo() {
    // Remove the logo by accessing it directly in the storyboard by tag if it has one
    if let logoView = self.view.viewWithTag(123) { // Replace 123 with the actual tag if it exists
      logoView.isHidden = true
    }
    
    // Or, alternatively, find it by its position and dimensions from the storyboard
    for subview in self.view.subviews {
      if let imageView = subview as? UIImageView,
         imageView.frame.origin.x > 200 && imageView.frame.origin.y > 500 && 
         imageView.frame.width > 150 && imageView.frame.height > 60 {
        imageView.isHidden = true
      }
    }
  }

  func predict(sampleBuffer: CMSampleBuffer) {
    if let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
        currentBuffer = pixelBuffer
        if !frameSizeCaptured {
            let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            longSide = max(frameWidth, frameHeight)
            shortSide = min(frameWidth, frameHeight)
            frameSizeCaptured = true
        }
        
        // Always add empty string to gradually clear the array - regardless of orientation
        detectedBusNumbers.append("")
        
        // Keep array at a reasonable size
        if detectedBusNumbers.count > 10 {
            detectedBusNumbers.removeFirst()
        }
        
        let imageOrientation: CGImagePropertyOrientation
        switch UIDevice.current.orientation {
        case .portrait:
            imageOrientation = .up
        case .portraitUpsideDown:
            imageOrientation = .down
        case .landscapeLeft:
            imageOrientation = .up
        case .landscapeRight:
            imageOrientation = .up
        case .unknown:
            imageOrientation = .up
        default:
            imageOrientation = .up
        }

        // Invoke a VNRequestHandler with that image
        let handler = VNImageRequestHandler(
            cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
        if UIDevice.current.orientation != .faceUp {
            t0 = CACurrentMediaTime()
            do {
                try handler.perform([visionRequest])
            } catch {
                print(error)
            }
            t1 = CACurrentMediaTime() - t0
        }

        // Don't clear currentBuffer immediately as we need it for number detection
        // Instead, clear it after processing the detection results
    }
  }

  // This function will be called when a bus is detected
  private func processBusDetection(busRect: CGRect, originalBuffer: CVPixelBuffer) {
    // Store the current bus rect for reference when processing results
    currentBusRect = busRect
    
    // Crop the buffer to the bus region
    guard let croppedBuffer = cropPixelBuffer(originalBuffer, to: busRect) else {
        print("Failed to crop buffer")
        return
    }
    
    // Create a handler for the cropped buffer
    let handler = VNImageRequestHandler(cvPixelBuffer: croppedBuffer, orientation: .up, options: [:])
    
    // Perform the number detection on the cropped region
    do {
        try handler.perform([numberDetectionRequest])
    } catch {
        print("Error performing number detection: \(error)")
    }
  }

  private func processNumberDetections(for request: VNRequest, error: Error?, originalBusRect: CGRect?) {
    guard let results = request.results as? [VNRecognizedObjectObservation],
          let busRect = originalBusRect else {
        return
    }
    
    // Filter for digits and relevant letters
    let numberDetections = results.filter { observation in
        guard let topLabel = observation.labels.first else { return false }
        // Include digits 0-9 and letters A, E, G, M, W with confidence > 0.5
        return topLabel.confidence > 0.5
    }
    
    DispatchQueue.main.async { [weak self] in
        guard let self = self else { return }
        self.showNumberDetections(detections: numberDetections, inBusRect: busRect)
    }
  }

  private func showNumberDetections(detections: [VNRecognizedObjectObservation], inBusRect busRect: CGRect) {
    // Start with index after existing boxes that might be used
    let startIndex = boundingBoxViews.count / 2
    
    // Hide all previous digit boxes
    for i in startIndex..<boundingBoxViews.count {
        boundingBoxViews[i].hide()
    }
    
    // The preview view dimensions
    let width = videoPreview.bounds.width
    let height = videoPreview.bounds.height
    
    // Collection of detected numbers with their screen positions for ordering
    var detectedNumbersWithPositions: [(number: String, xPosition: CGFloat)] = []
    
    for (i, detection) in detections.enumerated() {
        let boxIndex = startIndex + i
        if boxIndex >= boundingBoxViews.count {
            break // Don't exceed available boxes
        }
        
        let label = detection.labels[0].identifier
        let confidence = detection.labels[0].confidence
        
        // Map normalized coordinates from cropped space to original image space
        let normalizedBox = CGRect(
            x: busRect.origin.x + (detection.boundingBox.origin.x * busRect.width),
            y: busRect.origin.y + (detection.boundingBox.origin.y * busRect.height), 
            width: detection.boundingBox.width * busRect.width,
            height: detection.boundingBox.height * busRect.height
        )
        
        // Convert to screen coordinates
        let screenBox = VNImageRectForNormalizedRect(normalizedBox, Int(width), Int(height))
        
        // Adjust for the case where image aspect ratio differs from screen
        var adjustedScreenBox = screenBox
        
        // Handle portrait mode transformations similar to how the bus boxes are done
        if UIDevice.current.orientation == .portrait {
            var ratio: CGFloat = 1.0
            if videoCapture.captureSession.sessionPreset == .photo {
                ratio = (height / width) / (4.0 / 3.0)
            } else {
                ratio = (height / width) / (16.0 / 9.0)
            }
            
            if ratio >= 1 { // iPhone ratio adjustment
                adjustedScreenBox.size.width *= ratio
            }
        }
        
        // Store the number with its X position for ordering
        detectedNumbersWithPositions.append((number: label, xPosition: adjustedScreenBox.midX))
    }
    
    // Sort numbers by X position (left to right)
    let sortedNumbers = detectedNumbersWithPositions.sorted { $0.xPosition < $1.xPosition }
    
    // Create a string by joining all detected numbers
    let numberString = sortedNumbers.map { $0.number }.joined()
    
    // Always add to detectedBusNumbers array - either the detected number or empty string
    // Only add if the number is not empty
    if !numberString.isEmpty {
        detectedBusNumbers.append(numberString)
    }
    
    // Keep array at a reasonable size
    if detectedBusNumbers.count > 10 {
        detectedBusNumbers.removeFirst()
    }
    
    // Display the number string at the bottom of the screen only if not empty
    if !numberString.isEmpty {
        displayBusNumberAtBottom(numberString)
    }
  }

  // Update displayBusNumberAtBottom to not add to the array since we do it above
  private func displayBusNumberAtBottom(_ numberString: String) {
    // Remove any existing number string label
    for subview in self.view.subviews {
        if subview.tag == 3001 {
            subview.removeFromSuperview()
        }
    }
    
    // Create a new label
    let label = UILabel()
    label.text = "Bus #: " + numberString
    label.textAlignment = .center
    label.font = UIFont.boldSystemFont(ofSize: 32)
    label.textColor = UIColor.white
    label.backgroundColor = UIColor.black.withAlphaComponent(0.7)
    label.layer.cornerRadius = 10
    label.layer.masksToBounds = true
    label.tag = 3001
    
    // Size and position the label
    label.sizeToFit()
    label.frame.size.width += 40 // Add padding
    label.frame.size.height += 20
    label.center.x = self.view.center.x
    label.frame.origin.y = self.view.bounds.height - label.frame.height - 50 // Position from bottom
    
    // Add to view
    self.view.addSubview(label)
    
    // Animate in
    label.alpha = 0
    UIView.animate(withDuration: 0.3) {
        label.alpha = 1.0
    }
    
    // Auto-remove after 5 seconds
    DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
        UIView.animate(withDuration: 0.3, animations: {
            label.alpha = 0
        }, completion: { _ in
            label.removeFromSuperview()
        })
    }
  }

  // New method to find majority bus number and announce it
  @objc private func checkAndAnnounceBusNumber() {
    print("Checking and announcing bus number")
    print("detectedBusNumbers: \(detectedBusNumbers)")
    guard !detectedBusNumbers.isEmpty else { return }
    
    // Find the frequency of each bus number
    let counts = Dictionary(grouping: detectedBusNumbers, by: { $0 })
                .mapValues { $0.count }
    
    // Find the most frequent number
    if let (majorityNumber, majorityCount) = counts.max(by: { $0.value < $1.value }) {
        let currentTime = CACurrentMediaTime()
        var numberToAnnounce: String? = nil
        
        // If the majority is not empty, announce it
        if !majorityNumber.isEmpty {
            numberToAnnounce = majorityNumber
        } 
        // If the majority is empty, look for next most frequent that appears at least 3 times
        else {
            let nonEmptyNumbers = counts.filter { !$0.key.isEmpty && $0.value >= 3 }
            if let bestNonEmpty = nonEmptyNumbers.max(by: { $0.value < $1.value }) {
                numberToAnnounce = bestNonEmpty.key
            }
        }
        
        // Announce if we have a valid number
        if let number = numberToAnnounce, 
           (number != lastAnnouncedNumber || currentTime - lastAnnouncementTime > 15.0) {
            print("announcing bus number: \(number)")
            announceBusNumber(number)
            lastAnnouncementTime = currentTime
            lastAnnouncedNumber = number
        }
    }
  }

  // Method to speak the bus number
  private func announceBusNumber(_ number: String) {
    // Stop any ongoing speech
    if speechSynthesizer.isSpeaking {
        speechSynthesizer.stopSpeaking(at: .immediate)
    }
    
    // Configure audio session
    let audioSession = AVAudioSession.sharedInstance()
    do {
        try audioSession.setCategory(.playback, mode: .default)
        try audioSession.setActive(true)
        print("Audio session category: \(audioSession.category.rawValue)")
        print("Audio session volume: \(audioSession.outputVolume)")
    } catch {
        print("Failed to configure audio session: \(error.localizedDescription)")
    }
    
    // Create and configure the utterance
    let utterance = AVSpeechUtterance(string: "Bus \(number)")
    utterance.rate = 0.5
    utterance.pitchMultiplier = 1.0
    utterance.volume = 1.0
    
    // Use a voice that matches the user's language
    utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
    
    // Speak the utterance
    speechSynthesizer.speak(utterance)
    
    print("🔊 Announcing Bus \(number)")
  }

  func cropPixelBuffer(_ pixelBuffer: CVPixelBuffer, to rect: CGRect) -> CVPixelBuffer? {
    let inputWidth = CVPixelBufferGetWidth(pixelBuffer)
    let inputHeight = CVPixelBufferGetHeight(pixelBuffer)
    
    // Convert normalized rect to pixel coordinates
    let cropX = Int(rect.origin.x * CGFloat(inputWidth))
    let cropY = Int(rect.origin.y * CGFloat(inputHeight))
    let cropWidth = Int(rect.width * CGFloat(inputWidth))
    let cropHeight = Int(rect.height * CGFloat(inputHeight))
    
    // Ensure crop dimensions are valid
    guard cropX >= 0, cropY >= 0, cropWidth > 0, cropHeight > 0,
          cropX + cropWidth <= inputWidth, cropY + cropHeight <= inputHeight else {
        print("Invalid crop dimensions")
        return nil
    }
    
    // Create output buffer
    var croppedBuffer: CVPixelBuffer?
    let options = [
        kCVPixelBufferCGImageCompatibilityKey: true,
        kCVPixelBufferCGBitmapContextCompatibilityKey: true
    ] as CFDictionary
    
    let status = CVPixelBufferCreate(
        kCFAllocatorDefault,
        cropWidth, cropHeight,
        CVPixelBufferGetPixelFormatType(pixelBuffer),
        options,
        &croppedBuffer
    )
    
    guard status == kCVReturnSuccess, let croppedBuffer = croppedBuffer else {
        print("Failed to create pixel buffer")
        return nil
    }
    
    // Lock base buffer for reading
    CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
    CVPixelBufferLockBaseAddress(croppedBuffer, [])
    
    // Copy pixel data
    let srcBaseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)!
    let dstBaseAddress = CVPixelBufferGetBaseAddress(croppedBuffer)!
    
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
    let dstBytesPerRow = CVPixelBufferGetBytesPerRow(croppedBuffer)
    
    let bytesPerPixel = 4 // BGRA
    
    for row in 0..<cropHeight {
        let srcRowStart = srcBaseAddress.advanced(by: (cropY + row) * srcBytesPerRow + cropX * bytesPerPixel)
        let dstRowStart = dstBaseAddress.advanced(by: row * dstBytesPerRow)
        
        memcpy(dstRowStart, srcRowStart, cropWidth * bytesPerPixel)
    }
    
    // Unlock buffers
    CVPixelBufferUnlockBaseAddress(croppedBuffer, [])
    CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
    
    return croppedBuffer
  }

  private func createNumberLabel(forRect rect: CGRect, withText text: String) {
    // Create a label with larger text
    let label = UILabel(frame: CGRect(x: rect.midX - 50, y: rect.minY - 40, width: 100, height: 30))
    label.text = text
    label.textAlignment = .center
    label.font = UIFont.boldSystemFont(ofSize: 24)
    label.textColor = UIColor.black
    label.backgroundColor = UIColor.yellow
    label.layer.cornerRadius = 8
    label.layer.masksToBounds = true
    label.tag = 2002 // Use a specific tag for number labels
    
    // Add it to the view
    self.view.addSubview(label)
    
    // Remove it after a few seconds
    DispatchQueue.main.asyncAfter(deadline: .now() + 3.0) {
        label.removeFromSuperview()
    }
  }
}  // ViewController class End

extension ViewController: VideoCaptureDelegate {
  func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
    predict(sampleBuffer: sampleBuffer)
  }
}

// Programmatically save image
extension ViewController: AVCapturePhotoCaptureDelegate {
  func photoOutput(
    _ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?
  ) {
    if let error = error {
      print("error occurred : \(error.localizedDescription)")
    }
    if let dataImage = photo.fileDataRepresentation() {
      let dataProvider = CGDataProvider(data: dataImage as CFData)
      let cgImageRef: CGImage! = CGImage(
        jpegDataProviderSource: dataProvider!, decode: nil, shouldInterpolate: true,
        intent: .defaultIntent)
      var isCameraFront = false
      if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
        currentInput.device.position == .front
      {
        isCameraFront = true
      }
      var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
      switch UIDevice.current.orientation {
      case .landscapeLeft:
        orientation = isCameraFront ? .downMirrored : .up
      case .landscapeRight:
        orientation = isCameraFront ? .upMirrored : .down
      default:
        break
      }
      var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
      if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
        let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent)
      {
        image = UIImage(cgImage: cgImage)
      }
      let imageView = UIImageView(image: image)
      imageView.contentMode = .scaleAspectFill
      imageView.frame = videoPreview.frame
      let imageLayer = imageView.layer
      videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)

      let bounds = UIScreen.main.bounds
      UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
      self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
      let img = UIGraphicsGetImageFromCurrentImageContext()
      UIGraphicsEndImageContext()
      imageLayer.removeFromSuperlayer()
      let activityViewController = UIActivityViewController(
        activityItems: [img!], applicationActivities: nil)
      activityViewController.popoverPresentationController?.sourceView = self.View0
      self.present(activityViewController, animated: true, completion: nil)
      //
      //            // Save to camera roll
      //            UIImageWriteToSavedPhotosAlbum(img!, nil, nil, nil);
    } else {
      print("AVCapturePhotoCaptureDelegate Error")
    }
  }
}

// Add this AFTER the closing brace of ViewController class
extension ViewController: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didStart utterance: AVSpeechUtterance) {
        print("Speech started: \(utterance.speechString)")
    }
    
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        print("Speech finished: \(utterance.speechString)")
    }
}
