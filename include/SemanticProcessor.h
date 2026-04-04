#ifndef ORB_SLAM3_SEMANTICPROCESSOR_H
#define ORB_SLAM3_SEMANTICPROCESSOR_H

#include <set>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include <onnxruntime_cxx_api.h>

namespace ORB_SLAM3 {

struct SemanticDetection {
    int class_id;
    float score;
    cv::Rect box;
    bool is_dynamic;
    std::string class_name;

    SemanticDetection() 
        : class_id(0), score(0.0f), box(), is_dynamic(false), class_name("")
    {

    }

    SemanticDetection(int cls, float s, const cv::Rect& b, bool dyn, const std::string& name = "")
        : class_id(cls), score(s), box(b), is_dynamic(dyn), class_name(name)
    {

    }
};
class SemanticProcessor {
public:
    SemanticProcessor();
    ~SemanticProcessor();

    bool Initialize(const std::string &modelPath = "", const std::string &configPath = "");

    bool Infer(const cv::Mat &im, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap);

    void SetDynamicClasses(const std::set<int> &dynamicClasses);
    void SetTestBoxes(const std::vector<cv::Rect> &testBoxes);
    void SetUseDummy(bool useDummy);

    bool IsDynamicClass(int cls) const;
    bool IsInitialized() const;

    const std::vector<cv::Rect>& GetDebugBoxes() const {return mvDebugBoxes;}
    const std::vector<SemanticDetection>& GetLastDetections() const {return mvLastDetections;}

private:
    bool InferDummyDetections(const cv::Mat& im, std::vector<SemanticDetection>& detections);
    bool InferRealDetections(const cv::Mat& im, std::vector<SemanticDetection>& detections);

    // bool InferDummy(const cv::Mat &im, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap);
    void BuildSemanticMapsFromDetections(const cv::Size& imageSize, const std::vector<SemanticDetection>& detections, cv::Mat& dynamicMask, cv::Mat& semanticLabelMap);
    void BuildDynamicMaskFromLabels(const cv::Mat &semanticLabelMap, cv::Mat &dynamicMask) const;

    static float IoU(const cv::Rect& a, const cv::Rect& b);
    void NMS(const std::vector<SemanticDetection>& input, std::vector<SemanticDetection>& output, float iouThreshold) const;

private:
    bool mbInitialized;
    bool mbUseDummy;

    std::string mModelPath;
    std::string mConfigPath;
    
    std::set<int> mDynamicClasses;
    
    std::vector<cv::Rect> mvTestBoxes;
    std::vector<cv::Rect> mvDebugBoxes;

    std::vector<SemanticDetection> mvLastDetections;

    std::unique_ptr<Ort::Env> mpOrtEnv;
    std::unique_ptr<Ort::Session> mpOrtSession;
    Ort::SessionOptions mSessionOptions;

    std::string mInputName;
    std::string mOutputName;

    int mInputWidth = 640;
    int mInputHeight = 640;
    float mConfThreshold = 0.25f;
    float mNmsThreshold = 0.45f;

};
}

#endif
