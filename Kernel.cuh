#ifndef KERNEL_CUH
#define KERNEL_CUH

// #include <thrust/extrema.h>
#include <vector>
#include <utility>
#include <thread>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "sm_20_atomic_functions.h"

#include <opencv2/opencv.hpp>
//#include <opencv2/cudaimgproc.hpp>

//#include "Frame.h"
#include "Voxel.h"

//#define MULTI_SAMPLE_AA_

#define CHECK(call)                                                             \
{                                                                               \
    const cudaError_t error = call;                                             \
    if (error != cudaSuccess){                                                  \
        printf("Error: %s:%d, ", __FILE__, __LINE__);                           \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));      \
        exit(1);                                                                \
    }                                                                           \
}

namespace Jiamera {
    class Frame;

    // 体素块属性数组
    __device__ float* d_color_b;
    __device__ float* d_color_g;
    __device__ float* d_color_r;

    __device__ int* d_instance;
    __device__ int* d_label;
    __device__ float* d_tsdf;

    __device__ float* d_bgr_weight;
    __device__ float* d_label_weight;
    __device__ float* d_instance_weight;

    // 点云数组
    __device__ float* d_gl_rgb;
    __device__ float* d_gl_label;
    __device__ float* d_gl_instance;
    __device__ unsigned int* d_gl_point_num;


    // 帧图片数组
    __constant__ int d_rgb_height;
    __constant__ int d_rgb_width;    
    __constant__ float d_rgb_intrinsics[9];

    __constant__ int d_depth_height;
    __constant__ int d_depth_width;
    __constant__ float d_depth_intrinsics[9];

    __constant__ int d_panoptic_height;
    __constant__ int d_panoptic_width;
    __constant__ float d_panoptic_intrinsics[9];

    __device__ float* d_pose;
    __device__ float* d_rgb_image;
    __device__ float* d_depth_image;
    __device__ float* d_panoptic_image;
    #ifdef GPU_PROCESS_IMAGE_
    __device__ uchar* d_rgb_uchar;
    __device__ uchar* d_depth_uchar;
    __device__ uchar* d_panoptic_uchar;
    #endif

    #ifdef MULTI_SAMPLE_AA_
    // 保证第一个元素为中心点
    //__constant__ int X[27] = { 0, -1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1 };
    //__constant__ int Y[27] = { 0, -1, -1, -1,  0,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  1,  1,  1, -1, -1, -1,  0,  0,  0,  1,  1,  1 };
    //__constant__ int Z[27] = { 0, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1, -1,  0,  1 };
    __constant__ int X[9] = { 0, -1, -1, -1, -1,  1,  1,  1,  1};
    __constant__ int Y[9] = { 0, -1, -1,  1,  1, -1, -1,  1,  1};
    __constant__ int Z[9] = { 0, -1,  1, -1,  1, -1,  1, -1,  1};
    __constant__ const int d_sub_point_num = 9;
    #endif // MULTI_SAMPLE_AA_

    // 空间属性数组
    __constant__ int d_grid_num_x;
    __constant__ int d_grid_num_y;
    __constant__ int d_grid_num_z;
    __constant__ float d_origin_x;
    __constant__ float d_origin_y;
    __constant__ float d_origin_z;
    __constant__ float d_grid_size;
    __constant__ float d_trunc_margin;

    extern "C" __host__ void CudaCheck(const int lineNumber, cudaError_t status);

    extern "C" __host__ void CudaPreWork(Voxel*, Frame*);

    extern "C" __host__ void CudaInWork(Voxel*, Frame*, std::vector<Frame*>);

    extern "C" __host__ void CudaPostWork(Voxel*, Frame*, std::vector<Frame*>);

    __global__ void DeviceMain(const int index, float* color_b, float* color_g, float* color_r, int* instance, int* label, float* tsdf, float* bgr_weight, float* label_weight, float* instance_weight,
        float* pose, float* rgb_image, float* depth_image, float* panoptic_image,
        float* gl_rgb, float* gl_label, float* gl_instance, unsigned int* gl_point_num);

    // 1: rgb   2: depth    3: panoptic
    __global__ void ProcessImage(const uchar* src, float* des, uint type, int step);

    // x - 1, i > 0
    __device__ int GetLeft(int x, int X);

    // x + 1, i < d_grid_num_x - 1
    __device__ int GetRight(int x, int X);

    // y + 1, threadIdx.x < d_grid_num_y - 1
    __device__ int GetUp(int x, int X);

    // y - 1, threadIdx.x > 0
    __device__ int GetDown(int x, int X);

    // z + 1, blockIdx.x < d_grid_num_z - 1
    __device__ int GetFront(int x, int X);

    // z - 1, blockIdx.x > 0
    __device__ int GetBehind(int x, int X);

    __device__ float GetMajorityLabel(bool valid_point[], float list[], int len);
    __device__ float GetAverageDiffByLabel(bool valid_point[], float depth_list[], float camera_z_list[], float label_list[], int len, float label);
    __device__ float GetMajorityLabelWeightByLabel(bool valid_point[], float label_weight_list[], float label_list[], int len, float label);
    __device__ float GetMajorityLabelWeightByLabel(bool valid_point[], float label_weight_list[], float label_list[], int len, float label);
    __device__ float GetAverageLabelWeightByLabel(bool valid_point[], float label_weight_list[], float label_list[], int len, float label);

    __global__ void test(float* data);
}

#endif