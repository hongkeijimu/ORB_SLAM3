#include "SemanticProcessor.h"

#include <algorithm>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace ORB_SLAM3
{

SemanticProcessor::SemanticProcessor()
    : mbInitialized(false)
{
}

SemanticProcessor::~SemanticProcessor()
{
}

bool SemanticProcessor::Initialize(const std::string &modelPath, const std::string &configPath)
{
    mModelPath = modelPath;
    mConfigPath = configPath;
    mbInitialized = true;

    if (mDynamicClasses.empty())
    {
        // Label 1 is used by the dummy inference path below.
        mDynamicClasses.insert(1);
    }

    std::cout << "[SemanticProcessor] Initialize success" << std::endl;
    return true;
}

bool SemanticProcessor::Infer(const cv::Mat &im, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap)
{
    if (!IsInitialized())
    {
        std::cerr << "[SemanticProcessor] Infer failed: processor not initialized." << std::endl;
        return false;
    }

    if (im.empty())
    {
        std::cerr << "[SemanticProcessor] Infer failed: input image is empty." << std::endl;
        return false;
    }

    return InferDummy(im, dynamicMask, semanticLabelMap);
}

void SemanticProcessor::SetDynamicClasses(const std::set<int> &dynamicClasses)
{
    mDynamicClasses = dynamicClasses;
}

void SemanticProcessor::SetTestBoxes(const std::vector<cv::Rect> &testBoxes)
{
    mvTestBoxes = testBoxes;
}

bool SemanticProcessor::IsDynamicClass(int cls) const
{
    return mDynamicClasses.find(cls) != mDynamicClasses.end();
}

bool SemanticProcessor::IsInitialized() const
{
    return mbInitialized;
}

bool SemanticProcessor::InferDummy(const cv::Mat &im, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap)
{
    semanticLabelMap = cv::Mat::zeros(im.rows, im.cols, CV_8UC1);
    dynamicMask = cv::Mat::zeros(im.rows, im.cols, CV_8UC1);

    std::vector<cv::Rect> testBoxes = mvTestBoxes;
    if (testBoxes.empty())
    {
        // Default manual test box. Adjust this placeholder before wiring a detector.
        const int boxWidth = std::max(1, im.cols / 5);
        const int boxHeight = std::max(1, im.rows / 3);
        const int boxX = std::max(0, (im.cols - boxWidth) / 2);
        const int boxY = std::max(0, (im.rows - boxHeight) / 2);
        testBoxes.push_back(cv::Rect(boxX, boxY, boxWidth, boxHeight));
    }

    const cv::Rect imageBounds(0, 0, im.cols, im.rows);

    // cv::Mat vis;
    // if (im.channels() == 1) {
    //     cv::cvtColor(im, vis, cv::COLOR_GRAY2RGB);
    // } else {
    //     vis = im.clone();
    // }

    mvDebugBoxes.clear();

    for (size_t i = 0; i < testBoxes.size(); ++i)
    {
        const cv::Rect roi = testBoxes[i] & imageBounds;
        if (roi.empty())
        {
            continue;
        }

        semanticLabelMap(roi).setTo(cv::Scalar(1));
        mvDebugBoxes.push_back(roi);
        // cv::rectangle(vis, roi, cv::Scalar(0, 255, 0), 2);
    }

    BuildDynamicMaskFromLabels(semanticLabelMap, dynamicMask);

    // cv::imshow("debug", vis);
    // cv::waitKey(1);

    return true;
}

void SemanticProcessor::BuildDynamicMaskFromLabels(const cv::Mat &semanticLabelMap, cv::Mat &dynamicMask) const
{
    if (semanticLabelMap.empty())
    {
        dynamicMask.release();
        return;
    }

    dynamicMask = cv::Mat::zeros(semanticLabelMap.rows, semanticLabelMap.cols, CV_8UC1);

    for (int y = 0; y < semanticLabelMap.rows; ++y)
    {
        const uchar *pLabel = semanticLabelMap.ptr<uchar>(y);
        uchar *pMask = dynamicMask.ptr<uchar>(y);
        for (int x = 0; x < semanticLabelMap.cols; ++x)
        {
            if (IsDynamicClass(static_cast<int>(pLabel[x])))
            {
                pMask[x] = 255;
            }
        }
    }
}

} // namespace ORB_SLAM3
