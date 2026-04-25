#ifndef PEDESTRIAN_DETECTOR_H
#define PEDESTRIAN_DETECTOR_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include <NvInfer.h>
#include "stopSignDetector.h"   // reuse Detection struct + Logger

enum class RiskLevel { LOW, MEDIUM, HIGH };

struct TrackedPedestrian {
    Detection    detection;       // bbox, confidence, class_id, label
    int          track_id;        // persistent ID from IoU tracker
    RiskLevel    risk;            // scoring output
    cv::Point2f  centroid;        // current centroid
    cv::Point2f  motion;          // centroid displacement from last frame
    int          frames_tracked;  // persistence counter
    bool         inside_roi;      // inside trapezoidal ROI?
};

class PedestrianDetector {
public:
    PedestrianDetector();
    ~PedestrianDetector();

    bool loadEngine(const std::string& enginePath);
    std::vector<TrackedPedestrian> detect(const cv::Mat& frame);
    void visualize(cv::Mat& frame, const std::vector<TrackedPedestrian>& peds);

private:
    std::vector<char> readEngineFile(const std::string& path);
    std::vector<Detection> runInference(const cv::Mat& frame);

    Logger                       logger;
    nvinfer1::IRuntime*          runtime{nullptr};
    nvinfer1::ICudaEngine*       engine{nullptr};
    nvinfer1::IExecutionContext* context{nullptr};

    std::string inputTensorName;
    std::string outputTensorName;

    struct Track {
        int          id;
        cv::Rect     bbox;
        cv::Point2f  centroid;
        cv::Point2f  prev_centroid;
        int          frames_tracked;
        int          frames_missing;
    };

    std::vector<Track>  active_tracks;
    int                 next_track_id{0};

    void  updateTracks(const std::vector<Detection>& detections);
    float computeIoU(const cv::Rect& a, const cv::Rect& b);

    std::vector<cv::Point> roi_polygon;
    void      initROI(int frame_width, int frame_height);
    bool      isInsideROI(const cv::Rect& bbox);
    RiskLevel computeRisk(const Track& t, bool inside_roi);
};

#endif // PEDESTRIAN_DETECTOR_H
