#include "../src/pedestrian/PedestrianDetector.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/stat.h>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: ./pedestrian_test <engine_path> <video_path>"
              << std::endl;
    return 1;
  }

  std::string enginePath = argv[1];
  std::string videoPath = argv[2];

  // Load engine
  PedestrianDetector detector;
  if (!detector.loadEngine(enginePath)) {
    std::cerr << "Failed to load engine." << std::endl;
    return 1;
  }

  // Open video
  cv::VideoCapture cap(videoPath);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video: " << videoPath << std::endl;
    return 1;
  }

  // Make sure predictions dir exists
  mkdir("/home/acv5/pedestrian-detection/dataset/predictions", 0755);

  cv::Mat frame;
  int frame_count = 0; // counts only sampled frames (= output file index)
  int frame_index = 0; // counts every video frame
  double total_ms = 0.0;
  int frame_offset = 14;

  double fps = cap.get(cv::CAP_PROP_FPS);
  int frame_interval = static_cast<int>(std::round(fps)); // round, not truncate

  std::cout << "Video FPS: " << fps << " | Sampling every " << frame_interval
            << " frames" << std::endl;

  while (cap.read(frame)) {

    // Sample at 1 FPS to match dataset images
    if (frame_index < frame_offset ||
        (frame_index - frame_offset) % frame_interval != 0) {
      frame_index++;
      continue;
    }

    frame_count++; // 1-based, matches frame_0001.jpg naming from LabelImg
    if (frame_count > 120) {
      break;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    // ── EVALUATION MODE ──────────────────────────────────────────────────
    // Use raw detections (not tracked) so coasting ghost tracks don't
    // create false positives against ground truth labels.
    // The tracker is still used for the live display below.

    auto raw_dets =
        detector.getRawDetections(frame); // raw NMS output (for eval)
    // ─────────────────────────────────────────────────────────────────────

    // Write predictions in YOLO format (raw detections, not tracked)
    std::ostringstream filename;
    filename << "/home/acv5/pedestrian-detection/dataset/predictions/frame_"
             << std::setw(4) << std::setfill('0') << frame_count << ".txt";

    std::ofstream pred_file(filename.str());
    for (const auto &det : raw_dets) {
      float x_center = (det.bbox.x + det.bbox.width * 0.5f) / frame.cols;
      float y_center = (det.bbox.y + det.bbox.height * 0.5f) / frame.rows;
      float width = static_cast<float>(det.bbox.width) / frame.cols;
      float height = static_cast<float>(det.bbox.height) / frame.rows;

      pred_file << "0 " << x_center << " " << y_center << " " << width << " "
                << height << " " << det.confidence << "\n";
    }
    pred_file.close();

    // Live display uses tracked output

    auto t2 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    total_ms += ms;

    // FPS overlay
    std::string fps_str =
        "FPS: " + std::to_string(static_cast<int>(1000.0 / ms));
    cv::putText(frame, fps_str, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(0, 255, 255), 2);

    // Stats every 10 sampled frames
    if (frame_count % 10 == 0) {
      double avg_fps = 1000.0 / (total_ms / frame_count);
      std::cout << "Sampled frame " << frame_count << " (video frame "
                << frame_index << ")"
                << " | Avg FPS: " << std::fixed << std::setprecision(1)
                << avg_fps << " | Raw dets: " << raw_dets.size() << std::endl;
    }

    frame_index++;
  }

  std::cout << "\n=== Summary ===" << std::endl;
  std::cout << "Sampled frames : " << frame_count << std::endl;
  std::cout << "Avg FPS        : " << (1000.0 / (total_ms / frame_count))
            << std::endl;
  std::cout << "Predictions written to ../dataset/predictions/" << std::endl;

  return 0;
}
