#ifndef ORB_SLAM3_SEMANTICPROCESSOR_H
#define ORB_SLAM3_SEMANTICPROCESSOR_H

#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace ORB_SLAM3 {
class SemanticProcessor {
public:
    SemanticProcessor();
    ~SemanticProcessor();

    bool Initialize(const std::string &modelPath = "", const std::string &configPath = "");

    bool Infer(const cv::Mat &im, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap);

    void SetDynamicClasses(const std::set<int> &dynamicClasses);
    void SetTestBoxes(const std::vector<cv::Rect> &testBoxes);

    bool IsDynamicClass(int cls) const;
    bool IsInitialized() const;
    const std::vector<cv::Rect>& GetDebugBoxes() const {return mvDebugBoxes;}

private:
    bool InferDummy(const cv::Mat &im, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap);
    void BuildDynamicMaskFromLabels(const cv::Mat &semanticLabelMap, cv::Mat &dynamicMask) const;

    bool mbInitialized;
    std::string mModelPath;
    std::string mConfigPath;
    std::set<int> mDynamicClasses;
    std::vector<cv::Rect> mvTestBoxes;
    std::vector<cv::Rect> mvDebugBoxes;
};
}

#endif
