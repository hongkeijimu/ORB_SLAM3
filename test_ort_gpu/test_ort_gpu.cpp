#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <numeric>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

static int64_t SafeDim(int64_t d, int64_t fallback) {
    return d > 0 ? d : fallback;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./test_ort_gpu <model.onnx>\n";
        return 1;
    }

    const std::string model_path = argv[1];

    try {
        // 1. 创建 ORT 环境
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test_ort_gpu");

        // 2. SessionOptions
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        // 3. 启用 CUDA Execution Provider
        OrtCUDAProviderOptions cuda_options{};
        cuda_options.device_id = 0;
        cuda_options.arena_extend_strategy = 0;
        cuda_options.gpu_mem_limit = SIZE_MAX;
        cuda_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchExhaustive;
        cuda_options.do_copy_in_default_stream = 1;

        session_options.AppendExecutionProvider_CUDA(cuda_options);

        std::cout << "[INFO] CUDA Execution Provider appended.\n";

        // 4. 创建 Session
        Ort::Session session(env, model_path.c_str(), session_options);
        std::cout << "[INFO] Model loaded: " << model_path << "\n";

        Ort::AllocatorWithDefaultOptions allocator;

        // 5. 读取输入信息
        size_t num_inputs = session.GetInputCount();
        size_t num_outputs = session.GetOutputCount();

        std::cout << "[INFO] Num inputs : " << num_inputs << "\n";
        std::cout << "[INFO] Num outputs: " << num_outputs << "\n";

        if (num_inputs < 1) {
            throw std::runtime_error("Model has no input.");
        }

        auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
        std::string input_name = input_name_alloc.get();

        auto input_type_info = session.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_shape = input_tensor_info.GetShape();

        std::cout << "[INFO] Input name: " << input_name << "\n";
        std::cout << "[INFO] Input shape from model: [";
        for (size_t i = 0; i < input_shape.size(); ++i) {
            std::cout << input_shape[i] << (i + 1 < input_shape.size() ? ", " : "");
        }
        std::cout << "]\n";

        // 6. 给动态维度一个默认值
        // YOLOv5 通常输入为 [1, 3, 640, 640]
        if (input_shape.size() != 4) {
            throw std::runtime_error("Expected 4D input tensor.");
        }

        input_shape[0] = SafeDim(input_shape[0], 1);
        input_shape[1] = SafeDim(input_shape[1], 3);
        input_shape[2] = SafeDim(input_shape[2], 640);
        input_shape[3] = SafeDim(input_shape[3], 640);

        int64_t batch = input_shape[0];
        int64_t channels = input_shape[1];
        int64_t height = input_shape[2];
        int64_t width = input_shape[3];

        size_t input_tensor_size = static_cast<size_t>(batch * channels * height * width);

        std::cout << "[INFO] Using input shape: ["
                  << batch << ", " << channels << ", "
                  << height << ", " << width << "]\n";

        // 7. 构造一个全 0 的 dummy 输入
        std::vector<float> input_tensor_values(input_tensor_size, 0.0f);

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensor_values.data(),
            input_tensor_values.size(),
            input_shape.data(),
            input_shape.size());

        // 8. 获取输出名
        std::vector<std::string> output_name_strs;
        std::vector<const char*> output_names;

        for (size_t i = 0; i < num_outputs; ++i) {
            auto out_name_alloc = session.GetOutputNameAllocated(i, allocator);
            output_name_strs.emplace_back(out_name_alloc.get());
        }
        for (auto& s : output_name_strs) {
            output_names.push_back(s.c_str());
        }

        const char* input_names[] = {input_name.c_str()};

        // 9. 跑一次推理
        std::cout << "[INFO] Running inference...\n";

        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,            // 输入 tensor 数组首元素地址
            1,
            output_names.data(),
            output_names.size());

        std::cout << "[INFO] Inference done.\n";
        std::cout << "[INFO] Output tensors: " << output_tensors.size() << "\n";

        // 10. 打印输出形状
        for (size_t i = 0; i < output_tensors.size(); ++i) {
            auto type_info = output_tensors[i].GetTensorTypeAndShapeInfo();
            auto out_shape = type_info.GetShape();

            std::cout << "[INFO] Output " << i << " shape: [";
            for (size_t j = 0; j < out_shape.size(); ++j) {
                std::cout << out_shape[j] << (j + 1 < out_shape.size() ? ", " : "");
            }
            std::cout << "]\n";
        }

        std::cout << "[SUCCESS] ONNX Runtime GPU test passed.\n";
        return 0;
    }
    catch (const Ort::Exception& e) {
        std::cerr << "[ORT ERROR] " << e.what() << "\n";
        return 2;
    }
    catch (const std::exception& e) {
        std::cerr << "[STD ERROR] " << e.what() << "\n";
        return 3;
    }
}
