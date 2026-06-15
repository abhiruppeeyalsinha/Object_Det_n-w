// ════════════════════════════════════════════════════════════════════
// detectors.cpp
// ════════════════════════════════════════════════════════════════════

#include "detectors.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

using namespace std;
using namespace cv;

// ════════════════════════════════════════════════════════════════════
// Constructor
// ════════════════════════════════════════════════════════════════════

BlobDetector::BlobDetector(const Config& cfg,
    int roi_w,
    int roi_h,
    PriorMode prior) : cfg_(cfg), roi_w_(roi_w), roi_h_(roi_h), prior_(prior)
{
}

// ════════════════════════════════════════════════════════════════════
// blob_mask
// ════════════════════════════════════════════════════════════════════

Mat BlobDetector::blob_mask(const Mat& r32, float thresh)
{
    Mat pos_clip, neg_clip;
    Mat pos_mask, neg_mask;

    // Positive
    max(r32, 0, pos_clip);
    pos_clip.convertTo(pos_clip, CV_8U);

    threshold(pos_clip, pos_mask, thresh, 255, THRESH_BINARY);

    // Negative
    Mat neg_r32 = -r32;

    max(neg_r32, 0, neg_clip);
    neg_clip.convertTo(neg_clip, CV_8U);

    threshold(neg_clip,
        neg_mask,
        thresh,
        255,
        THRESH_BINARY);

    Mat out;
    bitwise_or(pos_mask, neg_mask, out);

    return out;
}

// ════════════════════════════════════════════════════════════════════
// select_best
// ════════════════════════════════════════════════════════════════════

std::optional<BBox>
BlobDetector::select_best(const Mat& gray,
    const Mat& fused_mask,
    const Mat& dog_conf,
    const Mat& log_conf,
    float hybrid_w) const
{
    std::vector<std::vector<Point>> contours;

    findContours(fused_mask,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);

    const double roi_area =
        static_cast<double>(roi_w_) * roi_h_;

    std::optional<BBox> best;

    double best_score = 0.0;
    double min_dist = std::numeric_limits<double>::max();

    for (auto& cnt : contours)
    {
        double area = contourArea(cnt);

        if (area <= 4)
            continue;

        Rect r = boundingRect(cnt);

        if (r.width <= 0 || r.height <= 0)
            continue;

        double aspect =
            static_cast<double>(r.width) / r.height;

        double extent =
            area / (r.width * r.height);

        double rel =
            (area / roi_area) * 100.0;

        // Same geometric gates
        if (!(0.25 < aspect &&
            aspect < 4.0 &&
            extent > 0.10 &&
            0.002 < rel &&
            rel < 28.0))
        {
            continue;
        }

        const double cx =
            r.x + r.width * 0.5;

        const double cy =
            r.y + r.height * 0.5;

        // ─────────────────────────────────────────────
        // Hybrid confidence
        // ─────────────────────────────────────────────

        double dog_mean = 0.0;
        double log_mean = 0.0;

        if (!dog_conf.empty())
        {
            Mat patch = dog_conf(r);
            dog_mean = mean(patch)[0];
        }

        if (!log_conf.empty())
        {
            Mat patch = log_conf(r);
            log_mean = mean(patch)[0];
        }

        double dog_n =
            std::min(dog_mean / 50.0, 1.0);

        double log_n =
            std::min(log_mean / 50.0, 1.0);

        double hyb_conf =
            (1.0 - hybrid_w) * dog_n +
            hybrid_w * log_n;

        BBox candidate{
            r.x,
            r.y,
            r.width,
            r.height };

        // ─────────────────────────────────────────────
        // Priority modes
        // ─────────────────────────────────────────────

        if (prior_ == PriorMode::Size)
        {
            double score =
                area * (0.5 + hyb_conf);

            if (!best || score > best_score)
            {
                best_score = score;
                best = candidate;
            }
        }
        else if (prior_ == PriorMode::Contrast)
        {
            Mat patch = gray(r);

            Scalar mu, sigma;

            meanStdDev(patch, mu, sigma);

            double score =
                sigma[0] * (0.5 + hyb_conf);

            if (!best || score > best_score)
            {
                best_score = score;
                best = candidate;
            }
        }
        else
        {
            double dist =
                std::hypot(cx - roi_w_ * 0.5,
                    cy - roi_h_ * 0.5);

            double adj_dist =
                dist / (0.5 + hyb_conf);

            if (adj_dist < min_dist)
            {
                min_dist = adj_dist;
                best = candidate;
            }
        }
    }

    return best;
}

std::vector<BBox>
BlobDetector::select_all(const Mat& gray,
    const Mat& fused_mask,
    const Mat& dog_conf,
    const Mat& log_conf,
    float hybrid_w) const
{
    std::vector<std::vector<Point>> contours;
    findContours(fused_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    const double roi_area =
        static_cast<double>(roi_w_) * roi_h_;

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

        double aspect =
            static_cast<double>(r.width) / r.height;

        double extent =
            area / (r.width * r.height);

        double rel =
            (area / roi_area) * 100.0;

        if (!(0.25 < aspect &&
            aspect < 4.0 &&
            extent > 0.10 &&
            0.002 < rel &&
            rel < 28.0))
        {
            continue;
        }

        double dog_mean = 0.0;
        double log_mean = 0.0;

        if (!dog_conf.empty())
            dog_mean = mean(dog_conf(r))[0];

        if (!log_conf.empty())
            log_mean = mean(log_conf(r))[0];

        double dog_n =
            std::min(dog_mean / 50.0, 1.0);

        double log_n =
            std::min(log_mean / 50.0, 1.0);

        double hyb_conf =
            (1.0 - hybrid_w) * dog_n +
            hybrid_w * log_n;

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
            score = -std::hypot(cx - roi_w_ * 0.5,
                                cy - roi_h_ * 0.5) +
                    hyb_conf * std::max(roi_w_, roi_h_);
        }

        candidates.push_back({BBox{r.x, r.y, r.width, r.height}, score});
    }

    sort(candidates.begin(),
         candidates.end(),
         [](const Candidate& a,
            const Candidate& b)
         {
             return a.score > b.score;
         });

    std::vector<BBox> boxes;
    boxes.reserve(candidates.size());

    for (const auto& c : candidates)
        boxes.push_back(c.box);

    return boxes;
}

// ════════════════════════════════════════════════════════════════════
// Detect_dog
// ════════════════════════════════════════════════════════════════════

std::optional<BBox>
BlobDetector::detect_dog(const Mat& roi)
{
    Mat gray;

    cvtColor(roi,
        gray,
        COLOR_BGR2GRAY);

    // ─────────────────────────────────────────────
    // Small objects
    // ─────────────────────────────────────────────

    Mat g1a, g2a;

    GaussianBlur(gray, g1a, { 3, 3 }, 0.8);

    GaussianBlur(gray, g2a, { 7, 7 }, 2.0);

    Mat dog_a;

    subtract(g1a, g2a, dog_a, noArray(), CV_16S);

    // ─────────────────────────────────────────────
    // Medium objects
    // ─────────────────────────────────────────────

    Mat g1b, g2b;

    GaussianBlur(gray, g1b, { 5, 5 }, 1.5);

    GaussianBlur(gray, g2b, { 11, 11 }, 3.5);

    Mat dog_b;

    subtract(g1b, g2b, dog_b, noArray(), CV_16S);

    // Convert → float32
    Mat dog_a32, dog_b32;

    dog_a.convertTo(dog_a32, CV_32F);
    dog_b.convertTo(dog_b32, CV_32F);

    // ─────────────────────────────────────────────
    // Confidence maps
    // ─────────────────────────────────────────────

    Mat abs_a, abs_b, dog_conf;

    absdiff(dog_a32,
        Scalar(0),
        abs_a);

    absdiff(dog_b32,
        Scalar(0),
        abs_b);

    max(abs_a,
        abs_b,
        dog_conf);

    // ─────────────────────────────────────────────
    // Combined mask
    // ─────────────────────────────────────────────

    Mat m1 =
        blob_mask(dog_a32, 18.f);

    Mat m2 =
        blob_mask(dog_b32, 18.f);

    Mat combined;

    bitwise_or(m1,
        m2,
        combined);

    return select_best(gray,
        combined,
        dog_conf,
        Mat{},
        0.0f);
}

std::vector<BBox>
BlobDetector::detect_dog_all(const Mat& roi)
{
    Mat gray;

    cvtColor(roi, gray, COLOR_BGR2GRAY);

    Mat g1a, g2a;
    GaussianBlur(gray, g1a, { 3, 3 }, 0.8);
    GaussianBlur(gray, g2a, { 7, 7 }, 2.0);

    Mat dog_a;
    subtract(g1a, g2a, dog_a, noArray(), CV_16S);

    Mat g1b, g2b;
    GaussianBlur(gray, g1b, { 5, 5 }, 1.5);
    GaussianBlur(gray, g2b, { 11, 11 }, 3.5);

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

// ════════════════════════════════════════════════════════════════════
// detect_log
// ════════════════════════════════════════════════════════════════════

std::optional<BBox>
BlobDetector::detect_log(const Mat& roi)
{
    Mat gray;

    cvtColor(roi,
        gray,
        COLOR_BGR2GRAY);

    Mat gray32;

    gray.convertTo(gray32, CV_32F);

    const float sigma_medium = 1.5f;

    int k_size_gauss =
        static_cast<int>(6 * sigma_medium + 1);

    if (k_size_gauss % 2 == 0)
        k_size_gauss++;

    // Morph kernel
    Mat kernel_morph =
        getStructuringElement(
            MORPH_RECT,
            { 5, 5 });

    // ─────────────────────────────────────────────
    // LoG
    // ─────────────────────────────────────────────

    Mat blur_a;

    GaussianBlur(gray32,
        blur_a,
        { k_size_gauss, k_size_gauss },
        sigma_medium);

    Mat log_raw;

    Laplacian(blur_a,
        log_raw,
        CV_32F,
        3);

    Mat log_abs;

    absdiff(log_raw,
        Scalar(0),
        log_abs);

    Mat log_conf =
        log_abs * (sigma_medium * sigma_medium);

    // ─────────────────────────────────────────────
    // Threshold
    // ─────────────────────────────────────────────

    Mat log_8u;

    convertScaleAbs(log_abs, log_8u);

    Mat thresh_a;

    threshold(log_8u, thresh_a, 30, 255, THRESH_BINARY);

    Mat fused;
    morphologyEx(thresh_a, fused, MORPH_CLOSE, kernel_morph);

    return select_best(gray, fused, Mat{}, log_conf, 1.0f);
}

std::vector<BBox>
BlobDetector::detect_log_all(const Mat& roi)
{
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);

    Mat gray32;
    gray.convertTo(gray32, CV_32F);

    const float sigma_medium = 1.5f;

    int k_size_gauss =
        static_cast<int>(6 * sigma_medium + 1);

    if (k_size_gauss % 2 == 0)
        k_size_gauss++;

    Mat kernel_morph =
        getStructuringElement(MORPH_RECT, { 5, 5 });

    Mat blur_a;
    GaussianBlur(gray32,
        blur_a,
        { k_size_gauss, k_size_gauss },
        sigma_medium);

    Mat log_raw;
    Laplacian(blur_a, log_raw, CV_32F, 3);

    Mat log_abs;
    absdiff(log_raw, Scalar(0), log_abs);

    Mat log_conf =
        log_abs * (sigma_medium * sigma_medium);

    Mat log_8u;
    convertScaleAbs(log_abs, log_8u);

    Mat thresh_a;
    threshold(log_8u, thresh_a, 30, 255, THRESH_BINARY);

    Mat fused;
    morphologyEx(thresh_a, fused, MORPH_CLOSE, kernel_morph);

    return select_all(gray, fused, Mat{}, log_conf, 1.0f);
}

// ════════════════════════════════════════════════════════════════════
// detect_hybrid
//
// ════════════════════════════════════════════════════════════════════

std::optional<BBox>
BlobDetector::detect_hybrid(const Mat& roi)
{
    Mat gray;

    cvtColor(roi,
        gray,
        COLOR_BGR2GRAY);

    Mat gray32;

    gray.convertTo(gray32,
        CV_32F);

    const float sigma_medium = 1.5f;
    const int kernel_size_medium = 7;
    const int thresh_val = 15;

    Mat kernel_morph =
        getStructuringElement(
            MORPH_RECT,
            { kernel_size_medium,
             kernel_size_medium });

    // ═══════════════════════════════════════════
    // DoG branch
    // ═══════════════════════════════════════════

    Mat g1a, g2a, g1b, g2b;

    GaussianBlur(gray, g1a, { 3, 3 }, 0.8);
    GaussianBlur(gray, g2a, { 7, 7 }, 2.0);

    GaussianBlur(gray, g1b, { 5, 5 }, 1.5);
    GaussianBlur(gray, g2b, { 11, 11 }, 3.5);

    Mat dog_a, dog_b;

    subtract(g1a, g2a, dog_a, noArray(), CV_32F);

    subtract(g1b, g2b, dog_b, noArray(), CV_32F);

    // ═══════════════════════════════════════════
    // LoG branch
    // ═══════════════════════════════════════════

    int k_size_gauss =
        static_cast<int>(6 * sigma_medium + 1);

    if (k_size_gauss % 2 == 0)
        k_size_gauss++;

    Mat blur_a, blur_b;

    GaussianBlur(gray32, blur_a, { k_size_gauss, k_size_gauss }, sigma_medium);

    GaussianBlur(gray32, blur_b, { 9, 9 }, 1.5);

    Mat log_a, log_b;

    Laplacian(blur_a, log_a, CV_32F, 3);

    Laplacian(blur_b, log_b, CV_32F, 3);

    log_a *= (0.8f * 0.8f);
    log_b *= (1.5f * 1.5f);

    // ═══════════════════════════════════════════
    // Confidence maps
    // ═══════════════════════════════════════════

    Mat abs_dog_a, abs_dog_b;
    Mat abs_log_a, abs_log_b;

    absdiff(dog_a,
        Scalar(0),
        abs_dog_a);

    absdiff(dog_b,
        Scalar(0),
        abs_dog_b);

    absdiff(log_a,
        Scalar(0),
        abs_log_a);

    absdiff(log_b,
        Scalar(0),
        abs_log_b);

    Mat dog_conf, log_conf;

    max(abs_dog_a,
        abs_dog_b,
        dog_conf);

    max(abs_log_a,
        abs_log_b,
        log_conf);

    // ═══════════════════════════════════════════
    // Binary masks
    // ═══════════════════════════════════════════

    Mat m1 =
        blob_mask(dog_a);

    Mat m2 =
        blob_mask(dog_b);

    Mat m3 =
        blob_mask(abs_log_a,
            thresh_val);

    Mat m4 =
        blob_mask(abs_log_b,
            thresh_val);

    Mat fused_raw, temp1, temp2;

    bitwise_or(m1, m2, temp1);
    bitwise_or(m3, m4, temp2);
    bitwise_or(temp1, temp2, fused_raw);

    // ═══════════════════════════════════════════
    // Morph close
    // ═══════════════════════════════════════════

    Mat fused_closed;

    morphologyEx(fused_raw,
        fused_closed,
        MORPH_CLOSE,
        kernel_morph);

    // ═══════════════════════════════════════════
    // Keep largest contours
    // ═══════════════════════════════════════════

    Mat cleaned =
        Mat::zeros(fused_closed.size(),
            CV_8U);

    std::vector<std::vector<Point>> contours;

    findContours(fused_closed,
        contours,
        RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);

    if (!contours.empty())
    {
        std::sort(contours.begin(),
            contours.end(),
            [](const auto& a,
                const auto& b)
            {
                return contourArea(a) >
                    contourArea(b);
            });

        int max_cnt =
            std::min((int)contours.size(), 5);

        for (int i = 0; i < max_cnt; ++i)
        {
            double area =
                contourArea(contours[i]);

            if (area > 4)
            {
                drawContours(cleaned,
                    contours,
                    i,
                    Scalar(255),
                    -1);
            }
        }
    }

    return select_best(gray,
        cleaned,
        dog_conf,
        log_conf,
        cfg_.HYBRID_LOG_WEIGHT);
}

std::vector<BBox>
BlobDetector::detect_hybrid_all(const Mat& roi)
{
    Mat gray;
    cvtColor(roi, gray, COLOR_BGR2GRAY);

    Mat gray32;
    gray.convertTo(gray32, CV_32F);

    const float sigma_medium = 1.5f;
    const int kernel_size_medium = 7;
    const int thresh_val = 15;

    Mat kernel_morph =
        getStructuringElement(MORPH_RECT,
            { kernel_size_medium, kernel_size_medium });

    Mat g1a, g2a, g1b, g2b;
    GaussianBlur(gray, g1a, { 3, 3 }, 0.8);
    GaussianBlur(gray, g2a, { 7, 7 }, 2.0);
    GaussianBlur(gray, g1b, { 5, 5 }, 1.5);
    GaussianBlur(gray, g2b, { 11, 11 }, 3.5);

    Mat dog_a, dog_b;
    subtract(g1a, g2a, dog_a, noArray(), CV_32F);
    subtract(g1b, g2b, dog_b, noArray(), CV_32F);

    int k_size_gauss =
        static_cast<int>(6 * sigma_medium + 1);

    if (k_size_gauss % 2 == 0)
        k_size_gauss++;

    Mat blur_a, blur_b;
    GaussianBlur(gray32, blur_a, { k_size_gauss, k_size_gauss }, sigma_medium);
    GaussianBlur(gray32, blur_b, { 9, 9 }, 1.5);

    Mat log_a, log_b;
    Laplacian(blur_a, log_a, CV_32F, 3);
    Laplacian(blur_b, log_b, CV_32F, 3);

    log_a *= (0.8f * 0.8f);
    log_b *= (1.5f * 1.5f);

    Mat abs_dog_a, abs_dog_b;
    Mat abs_log_a, abs_log_b;

    absdiff(dog_a, Scalar(0), abs_dog_a);
    absdiff(dog_b, Scalar(0), abs_dog_b);
    absdiff(log_a, Scalar(0), abs_log_a);
    absdiff(log_b, Scalar(0), abs_log_b);

    Mat dog_conf, log_conf;
    max(abs_dog_a, abs_dog_b, dog_conf);
    max(abs_log_a, abs_log_b, log_conf);

    Mat m1 = blob_mask(dog_a);
    Mat m2 = blob_mask(dog_b);
    Mat m3 = blob_mask(abs_log_a, thresh_val);
    Mat m4 = blob_mask(abs_log_b, thresh_val);

    Mat fused_raw, temp1, temp2;
    bitwise_or(m1, m2, temp1);
    bitwise_or(m3, m4, temp2);
    bitwise_or(temp1, temp2, fused_raw);

    Mat fused_closed;
    morphologyEx(fused_raw, fused_closed, MORPH_CLOSE, kernel_morph);

    Mat cleaned =
        Mat::zeros(fused_closed.size(), CV_8U);

    std::vector<std::vector<Point>> contours;
    findContours(fused_closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    sort(contours.begin(),
         contours.end(),
         [](const auto& a,
            const auto& b)
         {
             return contourArea(a) > contourArea(b);
         });

    int max_cnt =
        std::min(static_cast<int>(contours.size()), 50);

    for (int i = 0; i < max_cnt; ++i)
    {
        if (contourArea(contours[i]) > 4)
            drawContours(cleaned, contours, i, Scalar(255), -1);
    }

    return select_all(gray,
        cleaned,
        dog_conf,
        log_conf,
        cfg_.HYBRID_LOG_WEIGHT);
}
