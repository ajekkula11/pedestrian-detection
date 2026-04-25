#ifndef PEDESTRIAN_DETECTOR_H
#define PEDESTRIAN_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <NvInfer.h>

// ── Logger ────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::cout << "[TRT] " << msg << std::endl;
    }
};

// ── Detection (raw inference output) ─────────────────────────
struct Detection {
    cv::Rect bbox;
    float    confidence;
    int      class_id;
    std::string label;
};

// ── Risk levels for OPTIMAL tier ─────────────────────────────
enum class RiskLevel { LOW, MEDIUM, HIGH };

// ── Tracked pedestrian (tracker output) ──────────────────────
struct TrackedPedestrian {
    Detection   detection;
    int         track_id;
    RiskLevel   risk;
    cv::Point2f centroid;
    cv::Point2f motion;
    int         frames_tracked;
    bool        inside_roi;
};

// ── Main detector class ───────────────────────────────────────
class PedestrianDetector {
public:
    PedestrianDetector();
    ~PedestrianDetector();

    bool loadEngine(const std::string& enginePath);
    std::vector<TrackedPedestrian> detect(const cv::Mat& frame);
    void visualize(cv::Mat& frame, const std::vector<TrackedPedestrian>& peds);

private:
    // Engine loading
    std::vector<char>      readEngineFile(const std::string& path);

    // Inference
    std::vector<Detection> runInference(const cv::Mat& frame);

    // TensorRT handles
    Logger                       logger;
    nvinfer1::IRuntime*          runtime{nullptr};
    nvinfer1::ICudaEngine*       engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    std::string inputTensorName;
    std::string outputTensorName;

    // IoU Tracker
    struct Track {
        int          id;
        cv::Rect     bbox;
        cv::Point2f  centroid;
        cv::Point2f  prev_centroid;
        int          frames_tracked;
        int          frames_missing;
    };

    std::vector<Track> active_tracks;
    int                next_track_id{0};

    void  updateTracks(const std::vector<Detection>& detections);
    float computeIoU(const cv::Rect& a, const cv::Rect& b);

    // ROI
    std::vector<cv::Point> roi_polygon;
    void      initROI(int frame_width, int frame_height);
    bool      isInsideROI(const cv::Rect& bbox);

    // Risk scoring
    RiskLevel computeRisk(const Track& t, bool inside_roi);
};

#endif // PEDESTRIAN_DETECTOR_H
