// ════════════════════════════════════════════════════════════════════
// image_classical_detector.cpp
// Universal Classical Blob Detector (DoG / Hough / Hybrid) for Images & Videos
// ════════════════════════════════════════════════════════════════════

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <optional>
#include <cmath>
#include <limits>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

enum class DetMode
{
    DoG,
    HoughCircles,
    Hybrid
};

enum class PriorMode
{
    Size,
    Contrast,
    Center
};

struct TrackerConfig
{
    float CONF_THRESH = 0.5f;
    float HYBRID_LOG_WEIGHT = 0.5f;
};

struct BBox
{
    int x, y, w, h;
    float score = 0.0f;
    string type = "detection";

    int cx() const { return x + w / 2; }
    int cy() const { return y + h / 2; }
    Rect rect() const { return Rect(x, y, w, h); }
};

class BlobDetector
{
public:
    BlobDetector(const TrackerConfig &cfg, int roi_w, int roi_h, PriorMode prior = PriorMode::Center)
        : cfg_(cfg), roi_w_(roi_w), roi_h_(roi_h), prior_(prior) {}

    Mat blob_mask(const Mat& r32, float thresh = 18.f)
    {
        Mat pos_clip, neg_clip;
        Mat pos_mask, neg_mask;

        max(r32, 0, pos_clip);
        pos_clip.convertTo(pos_clip, CV_8U);
        threshold(pos_clip, pos_mask, thresh, 255, THRESH_BINARY);

        Mat neg_r32 = -r32;
        max(neg_r32, 0, neg_clip);
        neg_clip.convertTo(neg_clip, CV_8U);
        threshold(neg_clip, neg_mask, thresh, 255, THRESH_BINARY);

        Mat out;
        bitwise_or(pos_mask, neg_mask, out);
        return out;
    }

    std::vector<BBox> select_all(const Mat& gray,
                                 const Mat& fused_mask,
                                 const Mat& dog_conf,
                                 const Mat& log_conf,
                                 float hybrid_w) const
    {
        std::vector<std::vector<Point>> contours;
        findContours(fused_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        const double roi_area = static_cast<double>(roi_w_) * roi_h_;

        struct Candidate
        {
            BBox box;
            double score;
        };

        std::vector<Candidate> candidates;

        for (auto& cnt : contours)
        {
            double area = contourArea(cnt);
            if (area <= 4)
                continue;

            Rect r = boundingRect(cnt);
            if (r.width <= 0 || r.height <= 0)
                continue;

            if (r.x < 0 || r.y < 0 || r.x + r.width > gray.cols || r.y + r.height > gray.rows)
                continue;

            // Horizon Filter: Ignore ground/buildings in the bottom portion
            if (r.y > gray.rows * 0.85)
                continue;

            double aspect = static_cast<double>(r.width) / r.height;
            double extent = area / (r.width * r.height);
            double rel = (area / roi_area) * 100.0;

            if (!(0.10 < aspect && aspect < 10.0 && extent > 0.05 && rel < 40.0))
            {
                continue;
            }

            double dog_mean = 0.0;
            double log_mean = 0.0;

            if (!dog_conf.empty())
                dog_mean = mean(dog_conf(r))[0];

            if (!log_conf.empty())
                log_mean = mean(log_conf(r))[0];

            double dog_n = std::min(dog_mean / 50.0, 1.0);
            double log_n = std::min(log_mean / 50.0, 1.0);

            double hyb_conf = (1.0 - hybrid_w) * dog_n + hybrid_w * log_n;

            double score = area * (0.5 + hyb_conf);

            if (prior_ == PriorMode::Contrast)
            {
                Scalar mu, sigma;
                meanStdDev(gray(r), mu, sigma);
                score = sigma[0] * (0.5 + hyb_conf);
            }
            else if (prior_ == PriorMode::Center)
            {
                const double cx = r.x + r.width * 0.5;
                const double cy = r.y + r.height * 0.5;
                score = -std::hypot(cx - roi_w_ * 0.5, cy - roi_h_ * 0.5) +
                        hyb_conf * std::max(roi_w_, roi_h_);
            }

            candidates.push_back({BBox{r.x, r.y, r.width, r.height, (float)hyb_conf, "dog_detected"}, score});
        }

        sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
            return a.score > b.score;
        });

        std::vector<BBox> boxes;
        boxes.reserve(candidates.size());
        for (const auto& c : candidates)
            boxes.push_back(c.box);

        return boxes;
    }

    vector<BBox> detect_dog_all(const Mat &roi)
    {
        Mat gray;
        if (roi.channels() == 3)
            cvtColor(roi, gray, COLOR_BGR2GRAY);
        else
            gray = roi.clone();

        Mat g1a, g2a;
        GaussianBlur(gray, g1a, {3, 3}, 0.8);
        GaussianBlur(gray, g2a, {7, 7}, 2.0);

        Mat dog_a;
        subtract(g1a, g2a, dog_a, noArray(), CV_16S);

        Mat g1b, g2b;
        GaussianBlur(gray, g1b, {5, 5}, 1.5);
        GaussianBlur(gray, g2b, {11, 11}, 3.5);

        Mat dog_b;
        subtract(g1b, g2b, dog_b, noArray(), CV_16S);

        Mat dog_a32, dog_b32;
        dog_a.convertTo(dog_a32, CV_32F);
        dog_b.convertTo(dog_b32, CV_32F);

        Mat abs_a, abs_b, dog_conf;
        absdiff(dog_a32, Scalar(0), abs_a);
        absdiff(dog_b32, Scalar(0), abs_b);
        max(abs_a, abs_b, dog_conf);

        Mat m1 = blob_mask(dog_a32, 18.f);
        Mat m2 = blob_mask(dog_b32, 18.f);

        Mat combined;
        bitwise_or(m1, m2, combined);

        return select_all(gray, combined, dog_conf, Mat{}, 0.0f);
    }

    vector<BBox> detect_hough_circles(const Mat &img)
    {
        Mat gray;
        if (img.channels() == 3)
            cvtColor(img, gray, COLOR_BGR2GRAY);
        else
            gray = img.clone();

        GaussianBlur(gray, gray, Size(3, 3), 1.0);

        vector<Vec3f> circles;
        int min_radius = 2;
        int max_radius = 40;

        HoughCircles(gray, circles, HOUGH_GRADIENT, 1,
                     gray.rows / 20,
                     60,
                     10,
                     min_radius, max_radius);

        vector<BBox> boxes;
        for (const auto &c : circles)
        {
            int cx = cvRound(c[0]);
            int cy = cvRound(c[1]);
            int r  = cvRound(c[2]);

            int x = max(0, cx - r);
            int y = max(0, cy - r);
            int w = min(img.cols - x, 2 * r);
            int h = min(img.rows - y, 2 * r);

            if ((y + h) < (img.rows * 0.88))
            {
                boxes.push_back({x, y, w, h, 0.95f, "hough_circle"});
            }
        }
        return boxes;
    }

    vector<BBox> detect_hybrid_all(const Mat &img)
    {
        auto dog_boxes = detect_dog_all(img);
        auto hough_boxes = detect_hough_circles(img);

        vector<BBox> combined = dog_boxes;
        for (const auto &hb : hough_boxes)
        {
            bool matched = false;
            for (auto &db : dog_boxes)
            {
                if ((db.rect() & hb.rect()).area() > 0)
                {
                    matched = true;
                    break;
                }
            }
            if (!matched)
            {
                combined.push_back(hb);
            }
        }
        for (auto &b : combined) {
            b.type = "hybrid";
        }
        return combined;
    }

private:
    TrackerConfig cfg_;
    int roi_w_;
    int roi_h_;
    PriorMode prior_;
};

DetMode get_detection_mode()
{
    cout << "\n--- Detection Engine ---\n"
         << " [1] Filtered DoG (Exact detectors.cpp Scale-Space Logic)\n"
         << " [2] Hough Circle Transform (Recommended for Balloons)\n"
         << " [3] Hybrid (DoG + Hough)\n";

    string sel;
    cout << " Select (default 2): ";
    getline(cin, sel);

    if (sel == "1") return DetMode::DoG;
    if (sel == "3") return DetMode::Hybrid;
    return DetMode::HoughCircles;
}

string create_run_folder()
{
    auto t = chrono::system_clock::now();
    auto tt = chrono::system_clock::to_time_t(t);
    stringstream ss;
    ss << "Res/run_" << put_time(localtime(&tt), "%Y%m%d_%H%M%S");
    string dir = ss.str();
    fs::create_directories(dir);
    return dir;
}

void process_frame_detections(Mat &frame, BlobDetector &detector, DetMode det_mode, 
                              long long &elapsed_ms, int &det_count, vector<BBox> &out_detections)
{
    auto start_time = chrono::high_resolution_clock::now();

    switch (det_mode)
    {
    case DetMode::DoG:
        out_detections = detector.detect_dog_all(frame);
        break;
    case DetMode::HoughCircles:
        out_detections = detector.detect_hough_circles(frame);
        break;
    case DetMode::Hybrid:
        out_detections = detector.detect_hybrid_all(frame);
        break;
    }

    auto end_time = chrono::high_resolution_clock::now();
    elapsed_ms = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
    det_count = out_detections.size();

    // Define Frame Center
    Point frame_center(frame.cols / 2, frame.rows / 2);

    // Draw center crosshair on frame
    drawMarker(frame, frame_center, Scalar(0, 0, 255), MARKER_CROSS, 15, 2);

    for (size_t i = 0; i < out_detections.size(); ++i)
    {
        const auto &b = out_detections[i];
        
        // Draw bounding box
        rectangle(frame, Rect(b.x, b.y, b.w, b.h), Scalar(0, 255, 0), 2);
        
        // Target Center coordinates
        Point target_center(b.cx(), b.cy());

        // Draw line from frame center to target center
        line(frame, frame_center, target_center, Scalar(255, 0, 0), 1, LINE_AA);

        // Label with Target ID and Exact Coordinates (X, Y)
        string label = "T" + to_string(i + 1) + " (X:" + to_string(b.x) + " Y:" + to_string(b.y) + ")";
        putText(frame, label, Point(b.x, max(0, b.y - 5)),
                FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0, 255, 255), 1);
    }
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        cerr << "[ERROR] Usage: ./Exe-image_classical_detector <path_to_image_or_video>\n";
        return -1;
    }

    string input_path = argv[1];
    Mat sample_frame = imread(input_path);
    if (sample_frame.empty())
    {
        cerr << "[ERROR] Failed to load image from: " << input_path << "\n";
        return -1;
    }

    cout << "[INFO] Loaded image size: " << sample_frame.cols << "x" << sample_frame.rows << "\n";

    TrackerConfig cfg;
    DetMode det_mode = get_detection_mode();
    BlobDetector detector(cfg, sample_frame.cols, sample_frame.rows, PriorMode::Center);

    string run_dir = create_run_folder();
    string mode_str = (det_mode == DetMode::DoG) ? "Dog" : (det_mode == DetMode::HoughCircles) ? "Hough" : "Hybrid";

    string ext = fs::path(input_path).extension().string();
    transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    bool is_video = (ext == ".mp4" || ext == ".avi" || ext == ".mov" || ext == ".mkv" || ext == ".m4v");

    if (!is_video)
    {
        long long processing_ms = 0;
        int det_count = 0;
        vector<BBox> detections;

        process_frame_detections(sample_frame, detector, det_mode, processing_ms, det_count, detections);

        string txt_path = run_dir + "/detection_report.txt";
        ofstream report(txt_path);
        report << "Mode: " << mode_str << "\n";
        report << "Processing Time: " << processing_ms << " ms\n";
        report << "Number of Detections: " << det_count << "\n\n";
        report << "Detections:\n";
        report << "ID\tX\tY\tW\tH\tConfidence\tType\tDistance(px)\n";
        report << "----------------------------------------------------\n";

        for (size_t i = 0; i < detections.size(); ++i)
        {
            const auto &b = detections[i];
            double center_dist = sqrt(pow(b.cx() - sample_frame.cols / 2.0, 2) + pow(b.cy() - sample_frame.rows / 2.0, 2));
            report << (i + 1) << "\t" << b.x << "\t" << b.y << "\t" << b.w << "\t" << b.h 
                   << "\t" << b.score << "\t" << b.type << "\t" << center_dist << "\n";
        }
        report.close();

        string parent_txt_path = "Res/latest_detection_report.txt";
        fs::copy_file(txt_path, parent_txt_path, fs::copy_options::overwrite_existing);

        string det_name_str = (det_mode == DetMode::DoG) ? "DoG" : (det_mode == DetMode::HoughCircles) ? "Hough" : "Hybrid";
        string out_path = run_dir + "/Output_" + det_name_str + "_Result.jpg";
        imwrite(out_path, sample_frame);

        string parent_out_path = "Res/latest_Output_Result.jpg";
        fs::copy_file(out_path, parent_out_path, fs::copy_options::overwrite_existing);

        cout << "\n[DONE] Processing complete. Output saved in subfolder: " << run_dir << "\n";

        namedWindow("Detector Result", WINDOW_NORMAL);
        imshow("Detector Result", sample_frame);
        waitKey(0);
    }
    else
    {
        VideoCapture cap(input_path);
        if (!cap.isOpened())
        {
            cerr << "[ERROR] Failed to open video stream: " << input_path << "\n";
            return -1;
        }

        double fps = cap.get(CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;
        int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
        int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));

        string det_name_str = (det_mode == DetMode::DoG) ? "DoG" : (det_mode == DetMode::HoughCircles) ? "Hough" : "Hybrid";
        string out_video_path = run_dir + "/Output_" + det_name_str + "_Video.mp4";
        VideoWriter writer(out_video_path, VideoWriter::fourcc('a', 'v', 'c', '1'), fps, Size(frame_width, frame_height));

        string txt_path = run_dir + "/video_detection_report.txt";
        ofstream report(txt_path);
        report << "Mode: " << mode_str << "\n";
        report << "Video Stream: " << input_path << "\n\n";
        report << "Frame\tID\tX\tY\tW\tH\tConfidence\tType\tDistance(px)\n";
        report << "--------------------------------------------------------\n";

        Mat frame;
        int frame_idx = 0;
        namedWindow("Video Detector Result", WINDOW_NORMAL);

        while (cap.read(frame))
        {
            if (frame.empty()) break;
            frame_idx++;

            long long processing_ms = 0;
            int det_count = 0;
            vector<BBox> detections;

            process_frame_detections(frame, detector, det_mode, processing_ms, det_count, detections);

            for (size_t i = 0; i < detections.size(); ++i)
            {
                const auto &b = detections[i];
                double center_dist = sqrt(pow(b.cx() - frame.cols / 2.0, 2) + pow(b.cy() - frame.rows / 2.0, 2));
                report << frame_idx << "\t" << (i + 1) << "\t" << b.x << "\t" << b.y << "\t" << b.w << "\t" << b.h 
                       << "\t" << b.score << "\t" << b.type << "\t" << center_dist << "\n";
            }

            if (writer.isOpened()) writer.write(frame);

            imshow("Video Detector Result", frame);
            if ((waitKey(1) & 0xFF) == 27) break;
        }

        cap.release();
        if (writer.isOpened()) writer.release();
        report.close();

        string parent_txt_path = "Res/latest_video_report.txt";
        fs::copy_file(txt_path, parent_txt_path, fs::copy_options::overwrite_existing);

        cout << "\n[DONE] Video processing complete. Report and output saved in: " << run_dir << "\n";
    }

    return 0;
}
