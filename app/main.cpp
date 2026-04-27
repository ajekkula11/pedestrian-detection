#include <iostream>
#include <chrono>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include "../src/pedestrian/PedestrianDetector.h"
#include <opencv2/opencv.hpp>
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./pedestrian_test <engine_path> <video_path>" << std::endl;
        return 1;
    }

    std::string enginePath = argv[1];
    std::string videoPath  = argv[2];

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

    cv::Mat frame;
    int frame_count = 0;
    double total_ms = 0.0;

    while (cap.read(frame)) {
        auto t1 = std::chrono::high_resolution_clock::now();

        auto peds = detector.detect(frame);
        detector.visualize(frame, peds);

        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
        frame_count++;

        // FPS overlay
        std::string fps_str = "FPS: " + std::to_string(static_cast<int>(1000.0 / ms));
        cv::putText(frame, fps_str, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        std::string det_str = "Pedestrians: " + std::to_string(peds.size());
        cv::putText(frame, det_str, cv::Point(10, 65),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);

        cv::imshow("Pedestrian Detection", frame);
        if (cv::waitKey(1) == 'q') break;

        // Print stats every 30 frames
        if (frame_count % 30 == 0) {
            double avg_fps = 1000.0 / (total_ms / frame_count);
            std::cout << "Frame " << frame_count
                      << " | Avg FPS: " << avg_fps
                      << " | Detections: " << peds.size() << std::endl;
        }
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Total frames: " << frame_count << std::endl;
    std::cout << "Avg FPS: " << (1000.0 / (total_ms / frame_count)) << std::endl;

    return 0;
}
