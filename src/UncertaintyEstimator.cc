#include "DenseFeatureTensor.h"
#include "MapPoint.h"
#include "Converter.h"
#include <opencv2/opencv.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iostream>
#include "UncertaintyEstimator.h"

namespace ORB_SLAM3
{
    namespace
    {
        bool GetDenseFeatureKeyPoint(const Frame &frame, int idx, cv::Point2f &pt)
        {
            if (idx < 0)
                return false;

            // Dense features are currently extracted from the left/raw image.
            if (frame.Nleft == -1)
            {
                if (idx >= static_cast<int>(frame.mvKeys.size()))
                    return false;

                pt = frame.mvKeys[idx].pt;
                return true;
            }

            if (idx >= frame.Nleft || idx >= static_cast<int>(frame.mvKeys.size()))
                return false;

            pt = frame.mvKeys[idx].pt;
            return true;
        }

        float ComputeDenseFeatureScale(int featSize, int imgSize)
        {
            if (featSize <= 0 || imgSize <= 0)
                return 1.0f;

            if (featSize == 1 || imgSize == 1)
                return 0.0f;

            return static_cast<float>(featSize - 1) / static_cast<float>(imgSize - 1);
        }

        bool BuildDenseFeatureTensor(const cv::Mat &image, DenseFeatureTensor &feat)
        {
            if (image.empty())
                return false;

            cv::Mat gray;
            if (image.channels() == 1)
            {
                gray = image.clone();
            }
            else if (image.channels() == 3)
            {
                cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
            }
            else if (image.channels() == 4)
            {
                cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
            }
            else
            {
                return false;
            }

            cv::Mat grayF;
            gray.convertTo(grayF, CV_32F, 1.0 / 255.0);

            cv::Mat gradX, gradY, lap;
            cv::Sobel(grayF, gradX, CV_32F, 1, 0, 3);
            cv::Sobel(grayF, gradY, CV_32F, 0, 1, 3);
            cv::Laplacian(grayF, lap, CV_32F, 3);

            cv::Mat blur3, blur5, blur9, sqr;
            cv::blur(grayF, blur3, cv::Size(3, 3));
            cv::blur(grayF, blur5, cv::Size(5, 5));
            cv::blur(grayF, blur9, cv::Size(9, 9));
            sqr = grayF.mul(grayF);

            const int H = grayF.rows;
            const int W = grayF.cols;
            const int C = 8;

            feat.reset(H, W, C);

            for (int y = 0; y < H; ++y)
            {
                for (int x = 0; x < W; ++x)
                {
                    feat.at(y, x, 0) = grayF.at<float>(y, x);
                    feat.at(y, x, 1) = gradX.at<float>(y, x);
                    feat.at(y, x, 2) = gradY.at<float>(y, x);
                    feat.at(y, x, 3) = lap.at<float>(y, x);
                    feat.at(y, x, 4) = blur3.at<float>(y, x);
                    feat.at(y, x, 5) = blur5.at<float>(y, x);
                    feat.at(y, x, 6) = blur9.at<float>(y, x);
                    feat.at(y, x, 7) = sqr.at<float>(y, x);
                }
            }

            return true;
        }
    }

    UncertaintyEstimator::UncertaintyEstimator()
    {
        mUncertaintyEps = 0.05f;
        mHardRejectTh = 0.6f;
        mbEnableHardReject = false;

        mMinDynWeight = 0.2f;
        mMaxDynWeight = 3.0f;
        mDefaultUncertainty = 0.1f;
        mpDenseFeatureExtractor = new DenseFeatureExtractor("./dinov2_vits14_dense_224.onnx", 224, 224, true);
    }
    UncertaintyEstimator::~UncertaintyEstimator() 
    {
        if (mpDenseFeatureExtractor){
            delete mpDenseFeatureExtractor;
            mpDenseFeatureExtractor = nullptr;
        }
    }

    bool UncertaintyEstimator::BilinearSampleFeature(const DenseFeatureTensor &feat, float x, float y, std::vector<float> &outFeat) const
    {
        outFeat.clear();

        if (feat.empty())
            return false;

        if (x < 0.0f || y < 0.0f || x > feat.W - 1 || y > feat.H - 1)
            return false;

        int x0 = static_cast<int>(floor(x));
        int y0 = static_cast<int>(floor(y));
        int x1 = std::min(x0 + 1, feat.W - 1);
        int y1 = std::min(y0 + 1, feat.H - 1);

        float dx = x - x0;
        float dy = y - y0;

        outFeat.assign(feat.C, 0.0f);

        for (int c = 0; c < feat.C; ++c)
        {
            float v00 = feat.at(y0, x0, c);
            float v01 = feat.at(y0, x1, c);
            float v10 = feat.at(y1, x0, c);
            float v11 = feat.at(y1, x1, c);

            float v0 = (1.0f - dx) * v00 + dx * v01;
            float v1 = (1.0f - dx) * v10 + dx * v11;
            float v = (1.0f - dy) * v0 + dy * v1;

            outFeat[c] = v;
        }

        return true;
    }

    void UncertaintyEstimator::ExtractDenseFeature(Frame &frame)
    {
        if (frame.mbDenseFeatureReady)
            return;

        const cv::Mat &img = frame.mImGray.empty() ? frame.imgLeft : frame.mImGray;

        if (img.empty()) {
            std::cerr << "[UnceratintyEstimator] Empty frame image. " << std::endl;
        }

        if (!mpDenseFeatureExtractor) {
            std::cerr << "[UncertaintyEstimator] DenseFeatureExtractor is null ." << std::endl;
        }

        if (!mpDenseFeatureExtractor->Extract(img, frame.mDenseFeat)) {
            std::cerr << "[UncertaintyEstimator] Dense feature exrtact failed. " << std::endl;
            return;
        }

        frame.mfFeatScaleX = ComputeDenseFeatureScale(frame.mDenseFeat.W, img.cols);
        frame.mfFeatScaleY = ComputeDenseFeatureScale(frame.mDenseFeat.H, img.rows);
        frame.mbDenseFeatureReady = true;
    }

    void UncertaintyEstimator::ExtractDenseFeature(KeyFrame *pKF)
    {
        if (!pKF || pKF->mbDenseFeatureReady)
            return;

        const cv::Mat &img = pKF->mImGray;

        if (img.empty()) {
            std::cerr << "[UnceratintyEstimator] Empty frame image. " << std::endl;
        }

        if (!mpDenseFeatureExtractor) {
            std::cerr << "[UncertaintyEstimator] DenseFeatureExtractor is null ." << std::endl;
        }

        if (!mpDenseFeatureExtractor->Extract(img, pKF->mDenseFeat)) {
            std::cerr << "[UncertaintyEstimator] Dense feature exrtact failed. " << std::endl;
            return;
        }
        pKF->mfFeatScaleX = ComputeDenseFeatureScale(pKF->mDenseFeat.W, img.cols);
        pKF->mfFeatScaleY = ComputeDenseFeatureScale(pKF->mDenseFeat.H, img.rows);

        pKF->mbDenseFeatureReady = true;
    }

    void UncertaintyEstimator::ComputeFrameUncertainty(Frame &currentFrame, KeyFrame *pRefKF)
    {
        if (!pRefKF)
            return;

        if (!currentFrame.mbDenseFeatureReady || currentFrame.mDenseFeat.empty())
        {
            std::cerr << "[UncertaintyEstimator] Current frame dense feature is not ready." << std::endl;
            return;
        }

        if (!pRefKF->mbDenseFeatureReady || pRefKF->mDenseFeat.empty())
        {
            std::cerr << "[UncertaintyEstimator] Reference keyframe dense feature is not ready." << std::endl;
            return;
        }

        const int N = currentFrame.N;
        currentFrame.mvUncertainty.assign(N, mDefaultUncertainty);
        currentFrame.mvDynWeight.assign(N, UncertaintyToWeight(mDefaultUncertainty));

        std::vector<float> featCur, featRef;

        for (int i = 0; i < N; ++i)
        {
            MapPoint *pMP = currentFrame.mvpMapPoints[i];

            if (!pMP || pMP->isBad())
                continue;

            cv::Point2f ptCur;
            if (!GetDenseFeatureKeyPoint(currentFrame, i, ptCur))
                continue;

            // 当前帧某一个关键点采样特征
            float xCurFeat = ptCur.x * currentFrame.mfFeatScaleX;
            float yCurFeat = ptCur.y * currentFrame.mfFeatScaleY;
            if (!BilinearSampleFeature(currentFrame.mDenseFeat, xCurFeat, yCurFeat, featCur))
                continue;

            // 地图点投影到参考关键帧
            float uRef, vRef;
            if (!ProjectMapPointToFrame(pMP, pRefKF, uRef, vRef))
                continue;

            // 参考帧该关键点采样特征
            float xRefFeat = uRef * pRefKF->mfFeatScaleX;
            float yRefFeat = vRef * pRefKF->mfFeatScaleY;
            if (!BilinearSampleFeature(pRefKF->mDenseFeat, xRefFeat, yRefFeat, featRef))
                continue;

            float sim = CosineSimilarity(featCur, featRef);
            sim = std::max(-1.0f, std::min(1.0f, sim));

            float u = 1.0f - sim;

            u = std::max(0.0f, std::min(1.0f, u));

            currentFrame.mvUncertainty[i] = u;
            currentFrame.mvDynWeight[i] = UncertaintyToWeight(u);
        }
        currentFrame.mbUncertaintyReady = true;
    }

    float UncertaintyEstimator::CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) const
    {
        if (a.empty() || b.empty() || a.size() != b.size())
            return 0.0f;

        double dot = 0.0f;
        double na = 0.0f;
        double nb = 0.0f;

        for (size_t i = 0; i < a.size(); ++i)
        {
            dot += static_cast<double>(a[i]) * static_cast<double>(b[i]);
            na += static_cast<double>(a[i]) * static_cast<double>(a[i]);
            nb += static_cast<double>(b[i]) * static_cast<double>(b[i]);
        }

        if (na < 1e-12 || nb < 1e-12)
            return 0.0f;

        return static_cast<float>(dot / (std::sqrt(na) * std::sqrt(nb)));
    }

    float UncertaintyEstimator::UncertaintyToWeight(float u) const
    {
        float w = 1.0f / (u + mUncertaintyEps);
        w = std::max(mMinDynWeight, std::min(w, mMaxDynWeight));
        return w;
    }

    bool UncertaintyEstimator::ProjectMapPointToFrame(MapPoint *pMP, KeyFrame *pKF, float &u, float &v) const
    {
        if (!pMP || !pKF || pMP->isBad())
            return false;

        const cv::Mat &img = pKF->mImGray;
        if (img.empty() || !pKF->mpCamera)
            return false;

        Eigen::Vector3f Xw = pMP->GetWorldPos();
        Sophus::SE3f Tcw = pKF->GetPose();
        Eigen::Vector3f Xc = Tcw * Xw;

        if (Xc[2] <= 0.0f)
            return false;

        const Eigen::Vector2f uv = pKF->mpCamera->project(Xc);
        u = uv(0);
        v = uv(1);

        if (pKF->mpCamera->GetType() == GeometricCamera::CAM_PINHOLE && pKF->mDistCoef.total() >= 4)
        {
            const float x = (u - pKF->cx) * pKF->invfx;
            const float y = (v - pKF->cy) * pKF->invfy;
            const float r2 = x * x + y * y;

            const float k1 = pKF->mDistCoef.at<float>(0);
            const float k2 = pKF->mDistCoef.at<float>(1);
            const float p1 = pKF->mDistCoef.at<float>(2);
            const float p2 = pKF->mDistCoef.at<float>(3);
            const float k3 = (pKF->mDistCoef.total() >= 5) ? pKF->mDistCoef.at<float>(4) : 0.0f;

            float xDistort = x * (1.0f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);
            float yDistort = y * (1.0f + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2);

            xDistort += 2.0f * p1 * x * y + p2 * (r2 + 2.0f * x * x);
            yDistort += p1 * (r2 + 2.0f * y * y) + 2.0f * p2 * x * y;

            u = xDistort * pKF->fx + pKF->cx;
            v = yDistort * pKF->fy + pKF->cy;
        }

        if (u < 0.0f || u > static_cast<float>(img.cols - 1) ||
            v < 0.0f || v > static_cast<float>(img.rows - 1))
            return false;

        return true;
    }

    void UncertaintyEstimator::PrintUncertaintyStats(const Frame &frame) const
    {
        if (frame.mvUncertainty.empty())
            return;

        float minU = 1e9f, maxU = -1e9f, sumU = 0.0f;
        int cnt = 0;

        for (float u : frame.mvUncertainty)
        {
            minU = std::min(u, minU);
            maxU = std::max(u, maxU);
            sumU += u;
            cnt++;
        }

        std::cout << "[Uncertainty] min = " << minU << " max = " << maxU << " mean = " << (cnt > 0 ? sumU / cnt : 0.0f) << std::endl;
    }

    cv::Mat UncertaintyEstimator::VistualizeFrameUncertainty(const Frame &frame) const 
    {
        cv::Mat vis;
        const cv::Mat &img = frame.mImGray.empty() ? frame.imgLeft : frame.mImGray;
        if (img.empty())
            return vis;

        if (img.channels() == 1)
            cv::cvtColor(img, vis, cv::COLOR_GRAY2BGR);
        else if (img.channels() == 3)
            vis = img.clone();
        else
            cv::cvtColor(img, vis, cv::COLOR_BGRA2BGR);

        const size_t nVis = std::min(frame.mvUncertainty.size(),
                                     (frame.Nleft == -1) ? frame.mvKeys.size()
                                                         : static_cast<size_t>(frame.Nleft));

        for (size_t i = 0; i < nVis; ++i) {
            float u = (i < frame.mvUncertainty.size() ? frame.mvUncertainty[i] : mDefaultUncertainty);
            u = std::max(0.0f, std::min(1.0f, u));

            cv::Scalar color(255.0f * (1.0f - u), 0.0f, 255.0f * u);

            cv::circle(vis, frame.mvKeys[i].pt, 2, color, -1);
        }
        return vis;
    }
}
