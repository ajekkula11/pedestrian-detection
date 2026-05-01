#include "../src/pedestrian/PedestrianDetector.h"

#include <chrono>
#include <iomanip>
#include <iostream>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./pedestrian_test <engine_path> <video_path>" << std::endl;
        return 1;
    }

    PedestrianDetector detector;
    if (!detector.loadEngine(argv[1])) {
        std::cerr << "Failed to load engine." << std::endl;
        return 1;
    }

    cv::VideoCapture cap(argv[2]);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << argv[2] << std::endl;
        return 1;
    }

    cv::Mat frame;
    int    frame_count = 0;
    double total_ms    = 0.0;

    std::cout << "Video FPS: " << cap.get(cv::CAP_PROP_FPS) << std::endl;

    while (cap.read(frame)) {
        auto t1 = std::chrono::high_resolution_clock::now();

        auto peds = detector.detect(frame);
        detector.visualize(frame, peds);

        auto t2 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        total_ms += ms;
        frame_count++;

        int n_high = 0, n_med = 0, n_low = 0;
        for (const auto& p : peds) {
            if      (p.risk == RiskLevel::HIGH)   n_high++;
            else if (p.risk == RiskLevel::MEDIUM)  n_med++;
            else                                   n_low++;
        }

        cv::putText(frame,
            "FPS: " + std::to_string(static_cast<int>(1000.0 / ms)),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        cv::putText(frame,
            "Peds: " + std::to_string(peds.size()) +
            "  H:" + std::to_string(n_high) +
            " M:" + std::to_string(n_med) +
            " L:" + std::to_string(n_low),
            cv::Point(10, 65), cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        if (frame_count % 30 == 0) {
            std::cout << "Frame " << frame_count
                      << " | Avg FPS: " << std::fixed << std::setprecision(1)
                      << (1000.0 / (total_ms / frame_count))
                      << " | H:" << n_high << " M:" << n_med << " L:" << n_low
                      << std::endl;
        }

        cv::imshow("Pedestrian Detection — Phase 4", frame);
        if (cv::waitKey(1) == 'q') break;
    }

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Frames : " << frame_count << std::endl;
    std::cout << "Avg FPS: " << (1000.0 / (total_ms / frame_count)) << std::endl;
    return 0;
}
