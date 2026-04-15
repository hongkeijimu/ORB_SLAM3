#ifndef DENSE_FEATURE_TENSOR_H
#define DENSE_FEATURE_TENSOR_H
#include <vector>

namespace ORB_SLAM3 {
    struct DenseFeatureTensor
    {
        int H = 0;
        int W = 0;
        int C = 0;
        std::vector<float> data;

        bool empty() const
        {
            return data.empty() || H <= 0 || W <= 0 || C <= 0;
        }
        
        inline float at(int y, int x, int c) const 
        {
            return data[(y * W + x) * C + c];
        }
        
        inline float& at(int y, int x, int c)
        {
            return data[(y * W + x) * C + c];
        }

        void reset(int h, int w, int c)
        {
            H = h;
            W = w;
            C = c;
            data.assign(H * W * C, 0.0f);
        }

    };
    
    
}
#endif