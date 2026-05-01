#ifndef PEDESTRIAN_DETECTOR_H
#define PEDESTRIAN_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <NvInfer.h>
#include <iostream>

// ─────────────────────────────────────────────────────────────
// TensorRT Logger
// ─────────────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

// ─────────────────────────────────────────────────────────────
// Detection (raw model output after NMS)
// ─────────────────────────────────────────────────────────────
struct Detection {
    cv::Rect    bbox;
    float       confidence;
    int         class_id;
    std::string label;
};

// ─────────────────────────────────────────────────────────────
// Risk levels
// ─────────────────────────────────────────────────────────────
enum class RiskLevel { LOW, MEDIUM, HIGH };

// ─────────────────────────────────────────────────────────────
// Tracked pedestrian
// ─────────────────────────────────────────────────────────────
struct TrackedPedestrian {
    Detection   detection;
    int         track_id;
    RiskLevel   risk;
    cv::Point2f centroid;
    cv::Point2f motion;
    int         frames_tracked;
    bool        inside_roi;
};

// ─────────────────────────────────────────────────────────────
// Main Detector Class
// ─────────────────────────────────────────────────────────────
class PedestrianDetector {
public:
    PedestrianDetector();
    ~PedestrianDetector();

    // Engine
    bool loadEngine(const std::string& enginePath);

    // Main pipeline
    std::vector<TrackedPedestrian> detect(const cv::Mat& frame);
    void visualize(cv::Mat& frame, const std::vector<TrackedPedestrian>& peds);

    
    std::vector<Detection> getRawDetections(const cv::Mat& frame) {
        return runInference(frame);
    }

private:
    // ── Engine helpers ───────────────────────────────────────
    std::vector<char> readEngineFile(const std::string& path);

    Logger                       logger;
    nvinfer1::IRuntime*          runtime{nullptr};
    nvinfer1::ICudaEngine*       engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    std::string inputTensorName;
    std::string outputTensorName;

    // ── Inference ────────────────────────────────────────────
    std::vector<Detection> runInference(const cv::Mat& frame);

    // ── Tracking ─────────────────────────────────────────────
    struct Track {
        int          id;
        cv::Rect     bbox;
        float        confidence;   
        cv::Point2f  centroid;
        cv::Point2f  prev_centroid;
        int          frames_tracked;
        int          frames_missing;
    };

    std::vector<Track> active_tracks;
    int                next_track_id{0};

    void  updateTracks(const std::vector<Detection>& detections);
    float computeIoU(const cv::Rect& a, const cv::Rect& b);

    // ── ROI ──────────────────────────────────────────────────
    std::vector<cv::Point> roi_polygon;
    void  initROI(int frame_width, int frame_height);
    bool  isInsideROI(const cv::Rect& bbox);

    // ── Risk ─────────────────────────────────────────────────
    int        frame_height_{0};
    RiskLevel  computeRisk(const Track& t, bool inside_roi);
};

#endif // PEDESTRIAN_DETECTOR_H
