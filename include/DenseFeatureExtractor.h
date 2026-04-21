#ifndef DENSE_FEATURE_EXTRACTOR
#define DENSE_FEATURE_EXTRACTOR

#include "DenseFeatureTensor.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>

namespace ORB_SLAM3
{
    class DenseFeatureExtractor
    {
    public:
        DenseFeatureExtractor(const std::string &modelPath, int intputW = 224, int intputH = 224, bool useCUDA = true);

        ~DenseFeatureExtractor() = default;

        bool isReady() const { return mbReady; }

        bool Extract(const cv::Mat &image, DenseFeatureTensor &feat);

        int GetInputW() const { return mInputW; }
        int GetInputH() const { return mInputH; }

    private:
        bool InitializeSession();
        bool Preprocess(const cv::Mat &image, std::vector<float> &inputTensorValues) const;
        bool RunInference(const std::vector<float> &inputTensorValues, DenseFeatureTensor &feat);
        bool ParseOutputToTensor(const Ort::Value &outputValue, DenseFeatureTensor &feat) const;

    private:
        bool mbReady;
        std::string mModelPath;
        int mInputW;
        int mInputH;
        bool mbUseCUDA;

        Ort::Env mEnv;
        Ort::SessionOptions mSessionOptions;
        std::unique_ptr<Ort::Session> mpSession;

        std::vector<std::string> mInputNameStrings;
        std::vector<std::string> mOutputNameStrings;
        std::vector<const char*> mInputNames;
        std::vector<const char*> mOutputNames;
    };
}

#endif