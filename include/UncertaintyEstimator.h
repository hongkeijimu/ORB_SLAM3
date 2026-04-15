#ifndef UNCERTAINTY_ESTIMATOR_H
#define UNCERTAINTY_ESTIMATOR_H

#include "Frame.h"
#include "KeyFrame.h"
#include <vector>

namespace ORB_SLAM3
{
    class UncertaintyEstimator
    {
    public:
        UncertaintyEstimator();

        void ExtractDenseFeature(Frame &frame);
        void ExtractDenseFeature(KeyFrame *pKF);

        void ComputeFrameUncertainty(Frame &currentFrame, KeyFrame *pRefKF);

        bool BilinearSampleFeature(const DenseFeatureTensor &feat, float x, float y, std::vector<float> &outFeat) const;

        float CosineSimilarity(const std::vector<float> &a, const std::vector<float> &b) const;

        float UncertaintyToWeight(float u) const;

        void PrintUncertaintyStats(const Frame &frame) const;

        cv::Mat VistualizeFrameUncertainty(const Frame &frame) const;

    public:
        float mUncertaintyEps;
        float mHardRejectTh;
        bool mbEnableHardReject;

        float mMinDynWeight;
        float mMaxDynWeight;
        float mDefaultUncertainty;

    private:
        bool ProjectMapPointToFrame(MapPoint* pMP, KeyFrame* pKF, float &u, float &v) const;
    };
}

#endif