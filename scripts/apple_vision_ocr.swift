import Foundation
import ImageIO
import Vision

struct OCRObservation: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

struct OCRFrame: Codable {
    let frame: String
    let frameIndex: Int
    let timestamp: Double
    let observations: [OCRObservation]
    let error: String?
}

func argument(_ name: String) -> String? {
    guard let index = CommandLine.arguments.firstIndex(of: name), index + 1 < CommandLine.arguments.count else {
        return nil
    }
    return CommandLine.arguments[index + 1]
}

guard
    let framesValue = argument("--frames"),
    let outputValue = argument("--output"),
    let fpsValue = argument("--fps"),
    let fps = Double(fpsValue),
    fps > 0
else {
    fputs("Usage: apple_vision_ocr --frames DIR --output FILE.jsonl --fps FPS\n", stderr)
    exit(2)
}

let framesDirectory = URL(fileURLWithPath: framesValue, isDirectory: true)
let outputURL = URL(fileURLWithPath: outputValue)
let fileManager = FileManager.default
let timestampMap: [String: Double]
if let timestampsValue = argument("--timestamps"),
   let data = fileManager.contents(atPath: timestampsValue),
   let decoded = try? JSONDecoder().decode([String: Double].self, from: data) {
    timestampMap = decoded
} else {
    timestampMap = [:]
}
let supportedExtensions = Set(["jpg", "jpeg", "png"])
let frameURLs = try fileManager.contentsOfDirectory(
    at: framesDirectory,
    includingPropertiesForKeys: nil,
    options: [.skipsHiddenFiles]
).filter { supportedExtensions.contains($0.pathExtension.lowercased()) }
 .sorted { $0.lastPathComponent < $1.lastPathComponent }

fileManager.createFile(atPath: outputURL.path, contents: nil)
guard let output = try? FileHandle(forWritingTo: outputURL) else {
    fputs("Unable to open output: \(outputURL.path)\n", stderr)
    exit(2)
}
defer { try? output.close() }

let encoder = JSONEncoder()
encoder.outputFormatting = [.withoutEscapingSlashes]
let languageProbe = VNRecognizeTextRequest()
languageProbe.recognitionLevel = .accurate
let supportedLanguages = (try? languageProbe.supportedRecognitionLanguages()) ?? []
let preferredLanguages = ["zh-Hans", "zh-Hant", "en-US"]
let recognitionLanguages = preferredLanguages.filter { supportedLanguages.contains($0) }
fputs(
    "Apple Vision languages: \(recognitionLanguages.joined(separator: ",")) "
        + "(revision \(languageProbe.revision))\n",
    stderr
)

for (position, frameURL) in frameURLs.enumerated() {
    autoreleasepool {
        let frameIndex = position + 1
        let timestamp = timestampMap[frameURL.lastPathComponent] ?? Double(position) / fps
        var result: OCRFrame
        if
            let source = CGImageSourceCreateWithURL(frameURL as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        {
            let request = VNRecognizeTextRequest()
            request.recognitionLevel = .accurate
            request.recognitionLanguages = recognitionLanguages
            request.usesLanguageCorrection = false
            request.minimumTextHeight = 0.018
            do {
                try VNImageRequestHandler(cgImage: image, options: [:]).perform([request])
                let observations = (request.results ?? []).compactMap { observation -> OCRObservation? in
                    guard let candidate = observation.topCandidates(1).first else { return nil }
                    let box = observation.boundingBox
                    return OCRObservation(
                        text: candidate.string,
                        confidence: candidate.confidence,
                        x: box.origin.x,
                        y: box.origin.y,
                        width: box.width,
                        height: box.height
                    )
                }
                result = OCRFrame(
                    frame: frameURL.lastPathComponent,
                    frameIndex: frameIndex,
                    timestamp: timestamp,
                    observations: observations,
                    error: nil
                )
            } catch {
                let nsError = error as NSError
                result = OCRFrame(
                    frame: frameURL.lastPathComponent,
                    frameIndex: frameIndex,
                    timestamp: timestamp,
                    observations: [],
                    error: "\(nsError.domain) code=\(nsError.code) "
                        + "\(nsError.localizedDescription) userInfo=\(nsError.userInfo)"
                )
            }
        } else {
            result = OCRFrame(
                frame: frameURL.lastPathComponent,
                frameIndex: frameIndex,
                timestamp: timestamp,
                observations: [],
                error: "Unable to decode image"
            )
        }

        if let data = try? encoder.encode(result) {
            output.write(data)
            output.write(Data([0x0A]))
        }
    }
    if position == 0 || (position + 1) % 50 == 0 || position + 1 == frameURLs.count {
        fputs("Apple Vision OCR: \(position + 1)/\(frameURLs.count)\n", stderr)
    }
}
