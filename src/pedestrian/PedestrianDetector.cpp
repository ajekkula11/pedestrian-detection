#include "PedestrianDetector.h"

#include <algorithm>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cmath>

// Constants
static constexpr int INPUT_W = 640;
static constexpr int INPUT_H = 640;
static constexpr int NUM_CLASSES = 80;
static constexpr int NUM_BOXES = 8400;
static constexpr float CONF_THRESH = 0.30f;   // 🔴 raised threshold
static constexpr float NMS_THRESH = 0.35f;
static constexpr int PERSON_CLS_ID = 0;

PedestrianDetector::PedestrianDetector() {}

PedestrianDetector::~PedestrianDetector() {
  if (context) delete context;
  if (engine) delete engine;
  if (runtime) delete runtime;
}

// Read engine
std::vector<char> PedestrianDetector::readEngineFile(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Failed to open engine file");

  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  return buffer;
}

// Load engine
bool PedestrianDetector::loadEngine(const std::string &enginePath) {
  auto engineData = readEngineFile(enginePath);

  runtime = nvinfer1::createInferRuntime(logger);
  engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
  context = engine->createExecutionContext();

  inputTensorName = engine->getIOTensorName(0);
  outputTensorName = engine->getIOTensorName(1);

  return true;
}

// Inference
std::vector<Detection> PedestrianDetector::runInference(const cv::Mat &frame) {
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
    std::memcpy(inputData.data() + c * INPUT_H * INPUT_W, channels[c].data,
                INPUT_H * INPUT_W * sizeof(float));

  size_t inputSize = inputData.size() * sizeof(float);
  size_t outputSize = NUM_BOXES * (4 + NUM_CLASSES) * sizeof(float);

  void *d_input = nullptr;
  void *d_output = nullptr;
  cudaMalloc(&d_input, inputSize);
  cudaMalloc(&d_output, outputSize);

  cudaMemcpy(d_input, inputData.data(), inputSize, cudaMemcpyHostToDevice);

  context->setTensorAddress(inputTensorName.c_str(), d_input);
  context->setTensorAddress(outputTensorName.c_str(), d_output);
  context->enqueueV3(0);

  std::vector<float> output((4 + NUM_CLASSES) * NUM_BOXES);
  cudaMemcpy(output.data(), d_output, outputSize, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  float scale_x = static_cast<float>(orig_w) / INPUT_W;
  float scale_y = static_cast<float>(orig_h) / INPUT_H;

  // ✅ YOLOv8 parsing (NO sigmoid, NO objectness)
  for (int i = 0; i < NUM_BOXES; ++i) {

    float best_score = 0.0f;
    int best_cls = -1;

    for (int c = 0; c < NUM_CLASSES; ++c) {
      float score = output[(4 + c) * NUM_BOXES + i];

      if (score > best_score) {
        best_score = score;
        best_cls = c;
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

    x = std::max(0, x);
    y = std::max(0, y);
    w = std::min(w, orig_w - x);
    h = std::min(h, orig_h - y);

    boxes.push_back(cv::Rect(x, y, w, h));
    scores.push_back(best_score);   // 🔴 use best_score directly
    class_ids.push_back(best_cls);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, CONF_THRESH, NMS_THRESH, indices);

  for (int idx : indices) {
    Detection det;
    det.bbox = boxes[idx];
    det.confidence = scores[idx];
    det.class_id = class_ids[idx];
    det.label = "person";
    detections.push_back(det);
  }

  return detections;
}

// detect wrapper
std::vector<TrackedPedestrian>
PedestrianDetector::detect(const cv::Mat &frame) {
  std::vector<Detection> raw = runInference(frame);
  std::vector<TrackedPedestrian> results;

  for (auto &det : raw) {
    TrackedPedestrian tp;
    tp.detection = det;
    tp.track_id = -1;
    tp.risk = RiskLevel::LOW;
    tp.centroid = cv::Point2f(det.bbox.x + det.bbox.width / 2.0f,
                              det.bbox.y + det.bbox.height / 2.0f);
    tp.motion = cv::Point2f(0, 0);
    tp.frames_tracked = 1;
    tp.inside_roi = false;
    results.push_back(tp);
  }

  return results;
}

// visualize
void PedestrianDetector::visualize(
    cv::Mat &frame,
    const std::vector<TrackedPedestrian> &peds) {

  for (const auto &p : peds) {
    const cv::Rect &box = p.detection.bbox;

    cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);

    std::string label =
        "person " +
        std::to_string(static_cast<int>(p.detection.confidence * 100)) + "%";

    cv::putText(frame, label,
                cv::Point(box.x, box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
  }
}
