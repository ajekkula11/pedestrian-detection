#include "PedestrianDetector.h"

#include <iostream>
#include <fstream>
#include <stdexcept>

// ──────────────────────────────────────────────────────────────
// Constructor / Destructor
// ──────────────────────────────────────────────────────────────

PedestrianDetector::PedestrianDetector() {}

PedestrianDetector::~PedestrianDetector() {
    if (context) { delete context; context = nullptr; }
    if (engine)  { delete engine;  engine  = nullptr; }
    if (runtime) { delete runtime; runtime = nullptr; }
}

// ──────────────────────────────────────────────────────────────
// readEngineFile()
// Reads the .engine binary from disk into a byte buffer
// ──────────────────────────────────────────────────────────────

std::vector<char> PedestrianDetector::readEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open engine file: " + path);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size))
        throw std::runtime_error("Failed to read engine file: " + path);

    std::cout << "[PedestrianDetector] Engine file read: "
              << size / (1024 * 1024) << " MB" << std::endl;
    return buffer;
}

// ──────────────────────────────────────────────────────────────
// loadEngine()
// Deserializes .engine file into TRT runtime/engine/context
// ──────────────────────────────────────────────────────────────

bool PedestrianDetector::loadEngine(const std::string& enginePath) {
    try {
        // 1. Read raw engine bytes
        std::vector<char> engineData = readEngineFile(enginePath);

        // 2. Create TensorRT runtime
        runtime = nvinfer1::createInferRuntime(logger);
        if (!runtime) {
            std::cerr << "[PedestrianDetector] Failed to create TRT runtime" << std::endl;
            return false;
        }

        // 3. Deserialize engine
        engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
        if (!engine) {
            std::cerr << "[PedestrianDetector] Failed to deserialize engine" << std::endl;
            return false;
        }

        // 4. Create execution context
        context = engine->createExecutionContext();
        if (!context) {
            std::cerr << "[PedestrianDetector] Failed to create execution context" << std::endl;
            return false;
        }

        // 5. Fetch tensor names (TensorRT 10 API)
        inputTensorName  = engine->getIOTensorName(0);
        outputTensorName = engine->getIOTensorName(1);

        std::cout << "[PedestrianDetector] Engine loaded successfully" << std::endl;
        std::cout << "[PedestrianDetector] Input tensor:  " << inputTensorName  << std::endl;
        std::cout << "[PedestrianDetector] Output tensor: " << outputTensorName << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "[PedestrianDetector] loadEngine error: " << e.what() << std::endl;
        return false;
    }
}

// ──────────────────────────────────────────────────────────────
// Stubs — implemented in Phase 2 and beyond
// ──────────────────────────────────────────────────────────────

std::vector<Detection> PedestrianDetector::runInference(const cv::Mat& frame) {
    // Phase 2 — MIN
    return {};
}

std::vector<TrackedPedestrian> PedestrianDetector::detect(const cv::Mat& frame) {
    // Phase 2 + 3
    return {};
}

void PedestrianDetector::visualize(cv::Mat& frame, const std::vector<TrackedPedestrian>& peds) {
    // Phase 2 + 3
}

void PedestrianDetector::updateTracks(const std::vector<Detection>& detections) {
    // Phase 3 — TARGET
}

float PedestrianDetector::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    // Phase 3 — TARGET
    return 0.0f;
}

void PedestrianDetector::initROI(int frame_width, int frame_height) {
    // Phase 3 — TARGET
}

bool PedestrianDetector::isInsideROI(const cv::Rect& bbox) {
    // Phase 3 — TARGET
    return false;
}

RiskLevel PedestrianDetector::computeRisk(const Track& t, bool inside_roi) {
    // Phase 4 — OPTIMAL
    return RiskLevel::LOW;
}
