#include "SemanticProcessor.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace ORB_SLAM3
{

    SemanticProcessor::SemanticProcessor()
        : mbInitialized(false), mbUseDummy(true)
    {
    }

    SemanticProcessor::~SemanticProcessor()
    {
    }

    bool SemanticProcessor::Initialize(const std::string &modelPath, const std::string &configPath)
    {
        mModelPath = modelPath;
        mConfigPath = configPath;

        if (mDynamicClasses.empty())
        {
            mDynamicClasses.insert(0);
            mDynamicClasses.insert(1);
            mDynamicClasses.insert(2);
            mDynamicClasses.insert(3);
            mDynamicClasses.insert(5);
            mDynamicClasses.insert(7);
        }

        mpOrtSession.reset();
        mpOrtEnv.reset();
        mInputName.clear();
        mOutputName.clear();
        mbInitialized = false;

        if (mModelPath.empty())
        {
            mbUseDummy = true;
            mbInitialized = true;
            std::cout << "[SemanticProcessor] Initialize success, mode = dummy" << std::endl;
            return true;
        }

        try
        {
            mbUseDummy = false;

            mpOrtEnv.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "SemanticProcessor"));
            mSessionOptions = Ort::SessionOptions();
            mSessionOptions.SetIntraOpNumThreads(1);
            mSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

            mpOrtSession.reset(new Ort::Session(*mpOrtEnv, mModelPath.c_str(), mSessionOptions));

            Ort::AllocatorWithDefaultOptions allocator;

            {
                Ort::AllocatedStringPtr inputName = mpOrtSession->GetInputNameAllocated(0, allocator);
                mInputName = inputName.get();
            }

            {
                Ort::AllocatedStringPtr outputName = mpOrtSession->GetOutputNameAllocated(0, allocator);
                mOutputName = outputName.get();
            }

            auto inputTypeInfo = mpOrtSession->GetInputTypeInfo(0);
            auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
            const std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

            if (inputDims.size() == 4)
            {
                if (inputDims[2] > 0)
                    mInputHeight = static_cast<int>(inputDims[2]);
                if (inputDims[3] > 0)
                    mInputWidth = static_cast<int>(inputDims[3]);
            }

            mbInitialized = true;

            std::cout << "[SemanticProcessor] Initialize success, mode = real, input = "
                      << mInputWidth << "x" << mInputHeight << std::endl;
            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[SemanticProcessor] Initialize failed: " << e.what() << std::endl;
            mpOrtSession.reset();
            mpOrtEnv.reset();
            mbInitialized = false;
            return false;
        }
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

        std::vector<SemanticDetection> detections;
        bool ok = false;

        if (mbUseDummy)
        {
            ok = InferDummyDetections(im, detections);
        }
        else
        {
            ok = InferRealDetections(im, detections);
        }

        if (!ok)
        {
            std::cerr << "[SemanticProcessor] Infer failed: detection stage failed. " << std::endl;
            dynamicMask.release();
            semanticLabelMap.release();
            mvDebugBoxes.clear();
            mvLastDetections.clear();
            return false;
        }

        mvLastDetections = detections;
        BuildSemanticMapsFromDetections(im.size(), detections, dynamicMask, semanticLabelMap);

        return true;
    }

    void SemanticProcessor::SetDynamicClasses(const std::set<int> &dynamicClasses)
    {
        mDynamicClasses = dynamicClasses;
    }

    void SemanticProcessor::SetTestBoxes(const std::vector<cv::Rect> &testBoxes)
    {
        mvTestBoxes = testBoxes;
    }

    void SemanticProcessor::SetUseDummy(bool useDummy)
    {
        mbUseDummy = useDummy;
    }

    bool SemanticProcessor::IsDynamicClass(int cls) const
    {
        return mDynamicClasses.find(cls) != mDynamicClasses.end();
    }

    bool SemanticProcessor::IsInitialized() const
    {
        return mbInitialized;
    }

    bool SemanticProcessor::InferDummyDetections(const cv::Mat &im, std::vector<SemanticDetection> &detections)
    {
        detections.clear();

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

        for (size_t i = 0; i < testBoxes.size(); ++i)
        {
            const cv::Rect roi = testBoxes[i] & imageBounds;
            if (roi.empty())
            {
                continue;
            }

            int cls = 0;
            bool isDyn = IsDynamicClass(cls);

            detections.push_back(SemanticDetection(cls, 1.0f, roi, isDyn, "dummy_person"));
        }

        return true;
    }

    bool SemanticProcessor::InferRealDetections(const cv::Mat &im, std::vector<SemanticDetection> &detections)
    {
        detections.clear();

        if (!mpOrtSession)
        {
            std::cerr << "[SemanticProcessor] InferRealDetections failed: ONNX session is null." << std::endl;
            return false;
        }

        try
        {
            cv::Mat rgb;
            if (im.channels() == 1)
            {
                cv::cvtColor(im, rgb, cv::COLOR_GRAY2RGB);
            }
            else
            {
                cv::cvtColor(im, rgb, cv::COLOR_BGR2RGB);
            }

            cv::Mat resized;
            cv::resize(rgb, resized, cv::Size(mInputWidth, mInputHeight));

            resized.convertTo(resized, CV_32F, 1.0f / 255.0f);

            std::vector<float> inputTensorValues(1 * 3 * mInputHeight * mInputWidth);
            const int channelSize = mInputHeight * mInputWidth;

            for (int y = 0; y < mInputHeight; ++y)
            {
                const cv::Vec3f *rowPtr = resized.ptr<cv::Vec3f>(y);
                for (int x = 0; x < mInputWidth; ++x)
                {
                    inputTensorValues[0 * channelSize + y * mInputWidth + x] = rowPtr[x][0];
                    inputTensorValues[1 * channelSize + y * mInputWidth + x] = rowPtr[x][1];
                    inputTensorValues[2 * channelSize + y * mInputWidth + x] = rowPtr[x][2];
                }
            }

            std::vector<int64_t> inputShape = {1, 3, mInputHeight, mInputWidth};
            Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputTensorValues.data(),
                inputTensorValues.size(),
                inputShape.data(),
                inputShape.size());

            const char *inputNames[] = {mInputName.c_str()};
            const char *outputNames[] = {mOutputName.c_str()};

            auto outputTensors = mpOrtSession->Run(
                Ort::RunOptions{nullptr},
                inputNames,
                &inputTensor,
                1,
                outputNames,
                1);

            if (outputTensors.empty() || !outputTensors[0].IsTensor())
            {
                std::cerr << "[SemanticProcessor] Invalid output tensor." << std::endl;
                return false;
            }

            // 4) 解析输出
            Ort::Value &outputTensor = outputTensors[0];
            auto outputInfo = outputTensor.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> outputShape = outputInfo.GetShape();

            // YOLOv5 常见输出: [1, 25200, 85]
            if (outputShape.size() != 3)
            {
                std::cerr << "[SemanticProcessor] Unexpected output dims: " << outputShape.size() << std::endl;
                return false;
            }

            const int64_t numPreds = outputShape[1];
            const int64_t elemPerPred = outputShape[2]; // 85 = x,y,w,h,obj,80cls
            if (elemPerPred < 6)
            {
                std::cerr << "[SemanticProcessor] Invalid YOLO output format." << std::endl;
                return false;
            }

            const float *outputData = outputTensor.GetTensorData<float>();
            const int numClasses = static_cast<int>(elemPerPred - 5);

            const float scaleX = static_cast<float>(im.cols) / static_cast<float>(mInputWidth);
            const float scaleY = static_cast<float>(im.rows) / static_cast<float>(mInputHeight);

            std::vector<SemanticDetection> rawDetections;

            for (int64_t i = 0; i < numPreds; ++i)
            {
                const float *pred = outputData + i * elemPerPred;

                const float cx = pred[0];
                const float cy = pred[1];
                const float w = pred[2];
                const float h = pred[3];
                const float obj = pred[4];

                if (obj < 1e-6f)
                    continue;

                int bestClass = -1;
                float bestClassScore = 0.f;

                for (int c = 0; c < numClasses; ++c)
                {
                    float clsScore = pred[5 + c];
                    if (clsScore > bestClassScore)
                    {
                        bestClassScore = clsScore;
                        bestClass = c;
                    }
                }

                if (bestClass < 0)
                    continue;

                const float conf = obj * bestClassScore;
                if (conf < mConfThreshold)
                    continue;

                float x1 = (cx - 0.5f * w) * scaleX;
                float y1 = (cy - 0.5f * h) * scaleY;
                float x2 = (cx + 0.5f * w) * scaleX;
                float y2 = (cy + 0.5f * h) * scaleY;

                int left = std::max(0, std::min(im.cols - 1, static_cast<int>(std::round(x1))));
                int top = std::max(0, std::min(im.rows - 1, static_cast<int>(std::round(y1))));
                int right = std::max(0, std::min(im.cols - 1, static_cast<int>(std::round(x2))));
                int bottom = std::max(0, std::min(im.rows - 1, static_cast<int>(std::round(y2))));

                int boxW = right - left;
                int boxH = bottom - top;
                if (boxW <= 1 || boxH <= 1)
                    continue;

                cv::Rect box(left, top, boxW, boxH);
                bool isDyn = IsDynamicClass(bestClass);

                rawDetections.push_back(
                    SemanticDetection(bestClass, conf, box, isDyn, ""));
            }

            // 5) NMS
            NMS(rawDetections, detections, mNmsThreshold);

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "[SemanticProcessor] InferRealDetections exception: " << e.what() << std::endl;
            return false;
        }
    }

    float SemanticProcessor::IoU(const cv::Rect &a, const cv::Rect &b)
    {
        int interX1 = std::max(a.x, b.x);
        int interY1 = std::max(a.y, b.y);
        int interX2 = std::min(a.x + a.width, b.x + b.width);
        int interY2 = std::min(a.y + a.height, b.y + b.height);

        int interW = std::max(0, interX2 - interX1);
        int interH = std::max(0, interY2 - interY1);

        float interArea = static_cast<float>(interW * interH);
        float unionArea = static_cast<float>(a.area() + b.area() - interArea);

        if (unionArea <= 1e-6f)
            return 0.f;

        return interArea / unionArea;
    }

    void SemanticProcessor::NMS(const std::vector<SemanticDetection> &input, std::vector<SemanticDetection> &output, float iouThreshold) const
    {
        output.clear();
        if (input.empty())
            return;

        std::vector<SemanticDetection> sorted = input;
        std::sort(sorted.begin(), sorted.end(), [](const SemanticDetection &a, const SemanticDetection &b)
                  { return a.score > b.score; });
        
        std::vector<bool> removed(sorted.size(), false);

        for (size_t i = 0; i < sorted.size(); ++i) {
            if (removed[i]) {
                continue;
            }

            output.push_back(sorted[i]);
            for (size_t j = i + 1; j < sorted.size(); ++j) {
                if (removed[j]) {
                    continue;
                }
                if (sorted[i].class_id != sorted[j].class_id) {
                    continue;
                }
                if (IoU(sorted[i].box, sorted[j].box) > iouThreshold) {
                    removed[j] = true;
                }
            }
        }
    }

    void SemanticProcessor::BuildSemanticMapsFromDetections(const cv::Size &imageSize, const std::vector<SemanticDetection> &detections, cv::Mat &dynamicMask, cv::Mat &semanticLabelMap)
    {
        semanticLabelMap = cv::Mat::zeros(imageSize.height, imageSize.width, CV_8UC1);
        dynamicMask = cv::Mat::zeros(imageSize.height, imageSize.width, CV_8UC1);
        mvDebugBoxes.clear();

        const cv::Rect imageBounds(0, 0, imageSize.width, imageSize.height);

        for (size_t i = 0; i < detections.size(); ++i)
        {
            const SemanticDetection &det = detections[i];
            cv::Rect roi = det.box & imageBounds;
            if (roi.empty())
            {
                continue;
            }

            semanticLabelMap(roi).setTo(cv::Scalar(det.class_id));

            mvDebugBoxes.push_back(roi);
            if (det.is_dynamic)
            {
                dynamicMask(roi).setTo(cv::Scalar(255));
            }
        }
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
