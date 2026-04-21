#include "DenseFeatureExtractor.h"
#include <iostream>
#include <iostream>
#include <stdexcept>
#include <cstring>

namespace ORB_SLAM3
{
    DenseFeatureExtractor::DenseFeatureExtractor(const std::string &modelPath, int intputW, int intputH, bool useCUDA)
        : mModelPath(modelPath), mInputW(intputW), mInputH(intputH), mbUseCUDA(useCUDA), mbReady(false), mEnv(ORT_LOGGING_LEVEL_WARNING, "DenseFeatureExtractor")
    {

        mbReady = InitializeSession();
    }

    bool DenseFeatureExtractor::InitializeSession()
    {
        try
        {
            mSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            mSessionOptions.SetIntraOpNumThreads(1);

#ifdef USE_CUDA
            if (mbUseCUDA)
            {
                OrtCUDAProviderOptions cuda_options{};
                mSessionOptions.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[DenseFeatureExtractor] Using ONNX Runtime CUDA EP. " std::endl;
            }
#endif
            mpSession = std::make_unique<Ort::Session>(
                mEnv,
                mModelPath.c_str(),
                mSessionOptions);

            Ort::AllocatorWithDefaultOptions allocator;

            const size_t numInputs = mpSession->GetInputCount();
            const size_t numOutputs = mpSession->GetOutputCount();

            mInputNameStrings.clear();
            mOutputNameStrings.clear();
            mInputNames.clear();
            mOutputNames.clear();

            for (size_t i = 0; i < numInputs; ++i)
            {
                Ort::AllocatedStringPtr name = mpSession->GetInputNameAllocated(i, allocator);
                mInputNameStrings.emplace_back(name.get());
            }

            for (size_t i = 0; i < numOutputs; ++i)
            {
                Ort::AllocatedStringPtr name = mpSession->GetOutputNameAllocated(i, allocator);
                mOutputNameStrings.emplace_back(name.get());
            }

            for (auto &x : mInputNameStrings)
            {
                mInputNames.push_back(x.c_str());
            }

            for (auto &x : mOutputNameStrings)
            {
                mOutputNames.push_back(x.c_str());
            }

            std::cout << "[DenseFeatureExtractor] Model path: " << mModelPath << std::endl;
            std::cout << "[DenseFeatureExtractor] Inputs: ";
            for (auto &x : mInputNameStrings)
            {
                std::cout << x << " ";
            }
            std::cout << std::endl;

            std::cout << "[DenseFeatureExtractor] Output: ";
            for (auto &x : mOutputNameStrings)
            {
                std::cout << x << " ";
            }
            std::cout << std::endl;

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[DenseFeatureExtractor] Failed to initialize ONNX session: " << e.what() << std::endl;
            return false;
        }
    }

    bool DenseFeatureExtractor::Preprocess(const cv::Mat &image, std::vector<float> &inputTensorValues) const
    {
        if (image.empty())
            return false;

        cv::Mat img;
        if (image.channels() == 1)
        {
            cv::cvtColor(image, img, cv::COLOR_GRAY2BGR);
        }
        else if (image.channels() == 3)
        {
            img = image.clone();
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, img, cv::COLOR_BGRA2BGR);
        }
        else
        {
            return false;
        }

        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(mInputW, mInputH), 0, 0, cv::INTER_LINEAR);

        cv::Mat floatImg;
        resized.convertTo(floatImg, CV_32F, 1.0 / 255.0);

        const std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        const std::vector<float> std = {0.229f, 0.224f, 0.225f};

        inputTensorValues.assign(1 * 3 * mInputH * mInputW, 0.0f);

        std::vector<cv::Mat> channels(3);
        cv::split(floatImg, channels);

        for (int c = 0; c < 3; c++)
        {
            channels[c] = (channels[c] - mean[c]) / std[c];
            std::memcpy(
                inputTensorValues.data() + c * mInputH * mInputW,
                channels[c].ptr<float>(),
                sizeof(float) * mInputH * mInputW);
        }
        return true;
    }

    bool DenseFeatureExtractor::ParseOutputToTensor(const Ort::Value &outputValue, DenseFeatureTensor &feat) const
    {
        if (!outputValue.IsTensor()) 
        {
            std::cerr << "[DenseFeatureExtractor] Output is not a tensor. " << std::endl; 
            return false;
        }
            

        auto tensorInfo = outputValue.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> shape = tensorInfo.GetShape();
        const float *outPtr = outputValue.GetTensorData<float>();

        if (shape.size() != 4 || shape[0] != 1)
        {
            std::cerr << "[DenseFeatureExtractor] Unsupported output shape. " << std::endl;
            return false;
        }

        bool assumeNCHW = true;

        if (assumeNCHW)
        {
            int C = static_cast<int>(shape[1]);
            int H = static_cast<int>(shape[2]);
            int W = static_cast<int>(shape[3]);

            feat.reset(H, W, C);

            for (int c = 0; c < C; ++c)
            {
                for (int y = 0; y < H; ++y)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        const size_t srcIdx = static_cast<int>(c) * H * W + static_cast<int>(y) * W + x;
                        feat.at(y, x, c) = outPtr[srcIdx];
                    }
                }
            }
        }
        else
        {
            int H = static_cast<int>(shape[1]);
            int W = static_cast<int>(shape[2]);
            int C = static_cast<int>(shape[3]);

            feat.reset(H, W, C);
            for (int c = 0; c < C; ++c)
            {
                for (int y = 0; y < H; ++y)
                {
                    for (int x = 0; x < W; ++x)
                    {
                        const size_t srcIdx = static_cast<int>(c) * H * W + static_cast<int>(y) * W + x;
                        feat.at(y, x, c) = outPtr[srcIdx];
                    }
                }
            }
        }
        return true;
    }

    bool DenseFeatureExtractor::RunInference(const std::vector<float> &inputTensorValues, DenseFeatureTensor &feat)
    {
        if (!mpSession)
            return false;

        try
        {
            std::array<int64_t, 4> inputShape = {1, 3, mInputH, mInputW};

            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
                OrtAllocatorType::OrtArenaAllocator,
                OrtMemTypeDefault);

            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                const_cast<float *>(inputTensorValues.data()),
                inputTensorValues.size(),
                inputShape.data(),
                inputShape.size());

            auto outputTensors = mpSession->Run(
                Ort::RunOptions{nullptr},
                mInputNames.data(),
                &inputTensor,
                1,
                mOutputNames.data(),
                mOutputNames.size());

            if (outputTensors.empty())
            {
                std::cerr << "[DenseFeatureExtractor] Empty output tensor. " << std::endl;
                return false;
            }
            return ParseOutputToTensor(outputTensors[0], feat);
        }
        catch (const std::exception &e)
        {
            std::cerr << "[DenseFeatureExtractor] Inference failed: " << e.what() << std::endl;
            return false;
        }
    }

    bool DenseFeatureExtractor::Extract(const cv::Mat &image, DenseFeatureTensor &feat)
    {
        feat = DenseFeatureTensor();

        if (!mbReady)
            return false;

        std::vector<float> inputTensorValues;
        if (!Preprocess(image, inputTensorValues))
        {
            std::cerr << "[DenseFeatureExtractor] Preprocess failed. " << std::endl;
            return false;
        }

        if (!RunInference(inputTensorValues, feat))
        {
            std::cerr << "[DenseFeatureExtractor] RunInference failed. " << std::endl;
            return false;
        }
        return !feat.empty();
    }

}