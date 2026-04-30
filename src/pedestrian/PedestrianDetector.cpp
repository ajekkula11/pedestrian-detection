#include "PedestrianDetector.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cmath>

// ─────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────
static constexpr int   INPUT_W       = 640;
static constexpr int   INPUT_H       = 640;
static constexpr int   NUM_CLASSES   = 80;
static constexpr int   NUM_BOXES     = 8400;
static constexpr float CONF_THRESH   = 0.22f;
static constexpr float NMS_THRESH    = 0.35f;
static constexpr int   PERSON_CLS_ID = 0;

static constexpr float IOU_THRESH  = 0.15f;
static constexpr int   MAX_MISSING = 10;
static constexpr float EMA_ALPHA   = 0.4f;

// ─────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────
PedestrianDetector::PedestrianDetector() {}

PedestrianDetector::~PedestrianDetector() {
    if (context) delete context;
    if (engine)  delete engine;
    if (runtime) delete runtime;
}

// ─────────────────────────────────────────────────────────────
// loadEngine
// ─────────────────────────────────────────────────────────────
std::vector<char> PedestrianDetector::readEngineFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("Failed to open engine file");

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    file.read(buffer.data(), size);
    return buffer;
}

bool PedestrianDetector::loadEngine(const std::string& enginePath) {
    auto engineData = readEngineFile(enginePath);

    runtime = nvinfer1::createInferRuntime(logger);
    engine  = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    context = engine->createExecutionContext();

    inputTensorName  = engine->getIOTensorName(0);
    outputTensorName = engine->getIOTensorName(1);

    return true;
}

// ─────────────────────────────────────────────────────────────
// runInference (YOLOv8)
// IMPORTANT: scores are already sigmoid-activated in the ONNX export.
// Do NOT apply sigmoid again -- it pushes all scores to ~1.0 and
// floods the detector with false positives.
// ─────────────────────────────────────────────────────────────
std::vector<Detection> PedestrianDetector::runInference(const cv::Mat& frame) {
    std::vector<Detection> detections;

    int orig_w = frame.cols;
    int orig_h = frame.rows;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INPUT_W, INPUT_H));

    cv::Mat blob;
    resized.convertTo(blob, CV_32F, 1.0f / 255.0f);

    std::vector<cv::Mat> channels(3);
    cv::split(blob, channels);

    std::vector<float> inputData(3 * INPUT_H * INPUT_W);
    for (int c = 0; c < 3; ++c)
        std::memcpy(inputData.data() + c * INPUT_H * INPUT_W,
                    channels[c].data,
                    INPUT_H * INPUT_W * sizeof(float));

    size_t inputSize  = inputData.size() * sizeof(float);
    size_t outputSize = NUM_BOXES * (4 + NUM_CLASSES) * sizeof(float);

    void* d_input  = nullptr;
    void* d_output = nullptr;
    cudaMalloc(&d_input,  inputSize);
    cudaMalloc(&d_output, outputSize);

    cudaMemcpy(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice);

    context->setTensorAddress(inputTensorName.c_str(),  d_input);
    context->setTensorAddress(outputTensorName.c_str(), d_output);
    context->enqueueV3(0);

    std::vector<float> output((4 + NUM_CLASSES) * NUM_BOXES);
    cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);

    float scale_x = static_cast<float>(orig_w) / INPUT_W;
    float scale_y = static_cast<float>(orig_h) / INPUT_H;

    std::vector<cv::Rect> boxes;
    std::vector<float>    scores;
    std::vector<int>      class_ids;

    for (int i = 0; i < NUM_BOXES; ++i) {
        float best_score = 0.0f;
        int   best_cls   = -1;

        for (int c = 0; c < NUM_CLASSES; ++c) {
            float score = output[(4 + c) * NUM_BOXES + i];  // already a probability
            if (score > best_score) {
                best_score = score;
                best_cls   = c;
            }
        }

        if (best_cls != PERSON_CLS_ID || best_score < CONF_THRESH)
            continue;

        float cx = output[0 * NUM_BOXES + i];
        float cy = output[1 * NUM_BOXES + i];
        float bw = output[2 * NUM_BOXES + i];
        float bh = output[3 * NUM_BOXES + i];

        int x = static_cast<int>((cx - bw / 2.0f) * scale_x);
        int y = static_cast<int>((cy - bh / 2.0f) * scale_y);
        int w = static_cast<int>(bw * scale_x);
        int h = static_cast<int>(bh * scale_y);

        if (w < 23 || h < 42) continue;

        x = std::max(0, x);
        y = std::max(0, y);
        w = std::min(w, orig_w - x);
        h = std::min(h, orig_h - y);

        boxes.push_back(cv::Rect(x, y, w, h));
        scores.push_back(best_score);
        class_ids.push_back(best_cls);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, indices);

    for (int idx : indices) {
        detections.push_back({boxes[idx], scores[idx], class_ids[idx], "person"});
    }

    return detections;
}

// ─────────────────────────────────────────────────────────────
// computeIoU
// ─────────────────────────────────────────────────────────────
float PedestrianDetector::computeIoU(const cv::Rect& a, const cv::Rect& b) {
    cv::Rect inter = a & b;
    if (inter.area() <= 0) return 0.0f;

    float intersection = static_cast<float>(inter.area());
    float uni          = static_cast<float>(a.area() + b.area()) - intersection;
    return (uni > 0.0f) ? (intersection / uni) : 0.0f;
}

// ─────────────────────────────────────────────────────────────
// initROI
// Trapezoid for 1920x1080 dashcam:
//   top-left (30%,45%)  top-right (70%,45%)
//   bottom-right (95%,95%)  bottom-left (5%,95%)
// ─────────────────────────────────────────────────────────────
void PedestrianDetector::initROI(int frame_width, int frame_height) {
    frame_height_ = frame_height;
    roi_polygon.clear();
    roi_polygon.emplace_back(static_cast<int>(0.30f * frame_width),
                             static_cast<int>(0.45f * frame_height));
    roi_polygon.emplace_back(static_cast<int>(0.70f * frame_width),
                             static_cast<int>(0.45f * frame_height));
    roi_polygon.emplace_back(static_cast<int>(0.95f * frame_width),
                             static_cast<int>(0.95f * frame_height));
    roi_polygon.emplace_back(static_cast<int>(0.05f * frame_width),
                             static_cast<int>(0.95f * frame_height));
}

// ─────────────────────────────────────────────────────────────
// isInsideROI
// ─────────────────────────────────────────────────────────────
bool PedestrianDetector::isInsideROI(const cv::Rect& bbox) {
    if (roi_polygon.empty()) return false;

    cv::Point2f centroid(bbox.x + bbox.width  * 0.5f,
                         bbox.y + bbox.height * 0.5f);

    double result = cv::pointPolygonTest(roi_polygon, centroid, /*measureDist=*/false);
    return result >= 0.0;
}

// ─────────────────────────────────────────────────────────────
// updateTracks -- hybrid IoU + centroid-distance matching
// ─────────────────────────────────────────────────────────────
void PedestrianDetector::updateTracks(const std::vector<Detection>& detections) {
    int T = static_cast<int>(active_tracks.size());
    int D = static_cast<int>(detections.size());

    std::vector<bool> trk_used(T, false);
    std::vector<bool> det_used(D, false);

    struct Triple { float score; int t, d; };
    std::vector<Triple> pairs;

    for (int t = 0; t < T; ++t) {
        for (int d = 0; d < D; ++d) {
            float iou = computeIoU(active_tracks[t].bbox, detections[d].bbox);

            cv::Point2f dc(detections[d].bbox.x + detections[d].bbox.width  * 0.5f,
                           detections[d].bbox.y + detections[d].bbox.height * 0.5f);

            float dist     = static_cast<float>(cv::norm(active_tracks[t].centroid - dc));
            float avg_size = (active_tracks[t].bbox.width + active_tracks[t].bbox.height) * 0.5f;

            bool iou_match  = (iou >= IOU_THRESH);
            bool dist_match = (dist < avg_size * 0.5f && iou > 0.0f);

            if (iou_match || dist_match) {
                float score = iou + (1.0f / (1.0f + dist));
                pairs.push_back({score, t, d});
            }
        }
    }

    std::sort(pairs.begin(), pairs.end(),
              [](const Triple& a, const Triple& b){ return a.score > b.score; });

    for (const auto& p : pairs) {
        if (trk_used[p.t] || det_used[p.d]) continue;

        Track& trk      = active_tracks[p.t];
        const auto& det = detections[p.d];

        cv::Point2f new_centroid(det.bbox.x + det.bbox.width  * 0.5f,
                                 det.bbox.y + det.bbox.height * 0.5f);

        trk.prev_centroid = trk.centroid;
        trk.centroid.x    = EMA_ALPHA * new_centroid.x + (1.0f - EMA_ALPHA) * trk.centroid.x;
        trk.centroid.y    = EMA_ALPHA * new_centroid.y + (1.0f - EMA_ALPHA) * trk.centroid.y;

        trk.bbox           = det.bbox;
        trk.confidence     = det.confidence;
	    trk.frames_tracked++;
        trk.frames_missing = 0;

        trk_used[p.t] = true;
        det_used[p.d] = true;
    }

    // Age unmatched tracks
    for (int t = 0; t < T; ++t)
        if (!trk_used[t])
            active_tracks[t].frames_missing++;

    // Drop stale tracks
    active_tracks.erase(
        std::remove_if(active_tracks.begin(), active_tracks.end(),
                       [](const Track& t){ return t.frames_missing > MAX_MISSING; }),
        active_tracks.end());

    // Spawn new tracks for unmatched detections
    for (int d = 0; d < D; ++d) {
        if (det_used[d]) continue;

        Track trk;
        trk.id             = next_track_id++;
        trk.bbox           = detections[d].bbox;
        trk.confidence     = detections[d].confidence;
        trk.centroid       = cv::Point2f(detections[d].bbox.x + detections[d].bbox.width  * 0.5f,
                                         detections[d].bbox.y + detections[d].bbox.height * 0.5f);
        trk.prev_centroid  = trk.centroid;
        trk.frames_tracked = 1;
        trk.frames_missing = 0;
        active_tracks.push_back(trk);
    }
}

// ─────────────────────────────────────────────────────────────
// computeRisk (Phase 4) — Zone-based
// Trapezoid split at 70% frame height:
//   Outside ROI                  -> LOW   (green)
//   Inside ROI, above 70% line   -> MEDIUM (yellow)
//   Inside ROI, below 70% line   -> HIGH  (red)
// ─────────────────────────────────────────────────────────────
RiskLevel PedestrianDetector::computeRisk(const Track& trk, bool inside_roi) {
    if (!inside_roi) return RiskLevel::LOW;

    // Centroid y position as fraction of frame height
    float cy_norm = static_cast<float>(trk.bbox.y + trk.bbox.height) / static_cast<float>(frame_height_);

    // Below the danger line (70% of frame height) = HIGH risk
    if (cy_norm >= 0.70f) return RiskLevel::HIGH;

    // Inside ROI but above danger line = MEDIUM risk
    return RiskLevel::MEDIUM;
}

// ─────────────────────────────────────────────────────────────
// detect
// ─────────────────────────────────────────────────────────────
std::vector<TrackedPedestrian> PedestrianDetector::detect(const cv::Mat& frame) {
    if (roi_polygon.empty())
        initROI(frame.cols, frame.rows);

    std::vector<Detection> raw = runInference(frame);
    updateTracks(raw);

    std::vector<TrackedPedestrian> results;
    results.reserve(active_tracks.size());

    for (const Track& trk : active_tracks) {
        TrackedPedestrian tp;
        tp.detection      = Detection{trk.bbox, trk.confidence, PERSON_CLS_ID, "person"};
        tp.track_id       = trk.id;
        tp.centroid       = trk.centroid;
        tp.motion         = trk.centroid - trk.prev_centroid;
        tp.frames_tracked = trk.frames_tracked;
        tp.inside_roi     = isInsideROI(trk.bbox);
        tp.risk           = computeRisk(trk, tp.inside_roi);
        results.push_back(tp);
    }

    // ── Post-tracking NMS: centroid-distance suppression ────────────────
    // If two tracks have centroids within 80px, keep the older one
    std::vector<TrackedPedestrian> filtered;
    for (int i = 0; i < (int)results.size(); i++) {
        bool suppressed = false;
        for (int j = 0; j < (int)results.size(); j++) {
            if (i == j) continue;
            float dx   = results[i].centroid.x - results[j].centroid.x;
            float dy   = results[i].centroid.y - results[j].centroid.y;
            float dist = std::sqrt(dx*dx + dy*dy);
            if (dist < 150.0f &&
                results[j].frames_tracked > results[i].frames_tracked) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) filtered.push_back(results[i]);
    }
    return filtered;
}

// ─────────────────────────────────────────────────────────────
// visualize
// Green box  = inside ROI
// Blue box   = outside ROI
// Yellow     = ROI trapezoid outline
// Label: "ID:N f:N"
// ─────────────────────────────────────────────────────────────
void PedestrianDetector::visualize(cv::Mat& frame,
                                   const std::vector<TrackedPedestrian>& peds) {
    int W = frame.cols, H = frame.rows;

    // ── Draw ROI trapezoid outline (cyan) ────────────────────────────────
    if (!roi_polygon.empty()) {
        std::vector<std::vector<cv::Point>> polys = {roi_polygon};
        cv::polylines(frame, polys, true, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
    }

    // ── Draw danger zone divider (dotted white line at 70% height) ───────
    // Interpolate left and right edges of trapezoid at 70% frame height
    // Trapezoid: TL(30%,45%) TR(70%,45%) BR(95%,95%) BL(5%,95%)
    float t = (0.70f - 0.45f) / (0.95f - 0.45f);  // lerp factor along height
    int div_lx = static_cast<int>((0.30f + t * (0.05f - 0.30f)) * W);
    int div_rx = static_cast<int>((0.70f + t * (0.95f - 0.70f)) * W);
    int div_y  = static_cast<int>(0.70f * H);

    // Draw dotted line
    int dash = 12, gap = 8;
    for (int x = div_lx; x < div_rx; x += dash + gap) {
        int x2 = std::min(x + dash, div_rx);
        cv::line(frame, cv::Point(x, div_y), cv::Point(x2, div_y),
                 cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
    }

    // Zone labels
    cv::putText(frame, "DANGER",
                cv::Point(div_lx + 8, div_y + 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
    cv::putText(frame, "CAUTION",
                cv::Point(div_lx + 8, div_y - 8),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 255), 1, cv::LINE_AA);

    // ── Draw each tracked pedestrian ─────────────────────────────────────
    for (const auto& tp : peds) {
        cv::Rect box = tp.detection.bbox;

        // Color by risk zone
        cv::Scalar box_color;
        std::string risk_str;
        if (tp.risk == RiskLevel::HIGH) {
            box_color = cv::Scalar(0, 0, 255);    // red
            risk_str  = "HIGH";
        } else if (tp.risk == RiskLevel::MEDIUM) {
            box_color = cv::Scalar(0, 200, 255);  // yellow
            risk_str  = "MED";
        } else {
            box_color = cv::Scalar(0, 255, 0);    // green
            risk_str  = "LOW";
        }

        // Smooth box using EMA centroid
        int cx = static_cast<int>(tp.centroid.x);
        int cy = static_cast<int>(tp.centroid.y);
        int hw = box.width  / 2;
        int hh = box.height / 2;
        cv::Rect smooth_box(cx - hw, cy - hh, box.width, box.height);
        smooth_box &= cv::Rect(0, 0, W, H);  // clamp to frame

        cv::rectangle(frame, smooth_box, box_color, 2, cv::LINE_AA);

        // Label background + text
        std::string label = "ID:" + std::to_string(tp.track_id)
                          + " " + risk_str
                          + " f:" + std::to_string(tp.frames_tracked);
        int baseline = 0;
        cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.42, 1, &baseline);
        int lx = smooth_box.x;
        int ly = std::max(smooth_box.y - ts.height - 6, 0);
        cv::rectangle(frame,
                      cv::Point(lx, ly),
                      cv::Point(lx + ts.width, ly + ts.height + 4),
                      box_color, cv::FILLED);
        cv::putText(frame, label,
                    cv::Point(lx, ly + ts.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.42,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }
}
