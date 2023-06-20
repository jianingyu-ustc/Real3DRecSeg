#ifndef JIAMERA_VOXEL_H_
#define JIAMERA_VOXEL_H_

#include <math.h>
#include <map>
#include <map>
#include <fstream>
#include <algorithm>
#include <atomic>
#include <shared_mutex>
#include <iostream>
#include <algorithm>

// #include "cuda_runtime.h"
// #include "device_launch_parameters.h"
#include "Object.h"

#include "Matrix.h"
#include "VoxelHash.h"
#include "Frame.h"

namespace Jiamera{

    class Voxel : public Object/*, public VoxelHashInterface*/ {
        friend VoxelBlock::VoxelBlock(int, int, int, int);
    public:

        Voxel();
        explicit Voxel(float, float, float, float);

        Voxel(const Voxel&) = delete;
        Voxel& operator=(const Voxel&) = delete;

        //virtual void Display(GLFWwindow* const window, const unsigned int window_idx);

        void data_init();

        // 初始化 OpenGL 点云数组
        // 预留空间要保证最多数量时 VAO 不会崩溃
        void gl_data_init() {
            gl_rgb_ = (float*)malloc(grid_num_ * 6 * sizeof(float));
            std::cout<<grid_num_ * 6 * sizeof(float)<<"\n";
            gl_label_ = (float*)malloc(grid_num_ * 6 * sizeof(float));
            gl_instance_ = (float*)malloc(grid_num_ * 6 * sizeof(float));
        }

        // 向 OpenGL 渲染数组传输点云
        void gl_data_update() {
            //int thread_num = this->belonged_gl_->thread_list_.size();
            thread_atom.store(1); // 信号量 += 线程数量
        }

        unsigned int get_grid_num() {
            return this->grid_num_;
        }

        const float k_origin_x_ = -1.5f; // xx 坐标系下的原点坐标
        const float k_origin_y_ = -1.5f;
        const float k_origin_z_ = -1.5f;

        float truncation_margin_;  // 截断边界

        const float k_space_x = 3.0f;
        const float k_space_y = 3.0f;
        const float k_space_z = 3.0f;

        const float k_grid_size_ = 0.006f;    // 每个网格 xyz 轴上的边长 (单位：米)，体素网格为立方体

        unsigned int grid_num_x_;     // x 轴上网格个数
        unsigned int grid_num_y_;
        unsigned int grid_num_z_;
        unsigned int grid_num_;       // xyz 轴上网格个数 (体积)

        //float* data_;  // 每个网格的：times, x, y, z, r, g, b, a, tsdf, weight, instance, semantic

        float* color_b_;
        float* color_g_;
        float* color_r_;
        int* instance_;
        int* label_;
        float* tsdf_;
        float* bgr_weight_;
        float* label_weight_;
        float* instance_weight_;

        // 传递给 OpenGL 显示的点云数组，格式均为 x y z r g b
        float* gl_rgb_;
        float* gl_label_;
        float* gl_instance_;
        unsigned int gl_point_num_ = 0;    // gl_data_ 中的点的数量


        void SaveCloud(const std::string &scene_id, float* pose);

        void SaveParamaters();

        void UpdateDataList() {

        }

        // =i时由i窗口更新点云输组
        std::atomic<int> thread_atom = 0;

        // Window对象互斥读取数据
        mutable std::shared_mutex gl_data_mtx_;


    protected:
    private:

    };

}

#endif // !VOXEL_H