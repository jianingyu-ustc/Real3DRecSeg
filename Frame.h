#ifndef JIAMERA_FRAME_H_
#define JIAMERA_FRAME_H_

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <atomic> 

#include "Matrix.h"
#include "Viewer.h"

namespace Jiamera {
#define FRAME_NUM_ 5
#define GPU_PROCESS_IMAGE_

	static std::atomic<int> frame_idx_atomic = 0;

	class Frame {
	public:
		Frame(int rgb_width, int rgb_height, int depth_width, int depth_height, int panoptic_width, int panoptic_height,
			  int base_frame_idx, int first_frame_idx, int num_frames, 
			  const std::string &frame,
			  const std::string &rgb_intrinsic,
			  const std::string &depth_intrinsic,
			  const std::string &panoptic_intrinsic,
			  const std::string &base_pose);

		Frame(const Frame* f) {
			this->rgb_mat_ = f->rgb_mat_.clone();
			this->depth_mat_ = f->depth_mat_.clone();
			this->panoptic_mat_ = f->panoptic_mat_.clone();
			this->rgb_viewer_->current_pose_ = f->rgb_viewer_->clone_pose();
		}

		// 更新图像帧(rgbdp+pose)
		void Update(int frame_idx);

		void update_image(float* image, const int height, const int width, const std::string& file, int type) {
			// type(rgb|depth|panoptic|)
			ReadImage(image, height, width, file, type);
		}

		void update_viewer(Viewer* viewer, const std::string &pose) {
			viewer->Update(pose);
		}

		std::string dataset_path;		// 每一帧包含信息的目录

		int base_frame_index_ = 0;		// 初始位姿矩阵对应的帧
		int first_frame_index_ = 0;
		int last_frame_index_;
		int current_frame_index_ = -1;
		unsigned int frame_num_;	// 总帧数

		unsigned int rgb_width_;
		unsigned int rgb_height_;

		unsigned int depth_width_;
		unsigned int depth_height_;

		unsigned int panoptic_width_;
		unsigned int panoptic_height_;

		// float图像数组，由于图像预处理花费较长时间，
		// 可选择使用GPU预处理图像，无需使用
		float* rgb_image_;
		float* depth_image_;
		float* panoptic_image_;

		cv::Mat rgb_mat_;
		cv::Mat depth_mat_;
		cv::Mat panoptic_mat_;

		Viewer* rgb_viewer_;
		Viewer* depth_viewer_;
		Viewer* panoptic_viewer_;

		void ReadImage(float* image, int height, int width,
			           const std::string& path, int type);
					   
	};
	//std::vector<Frame*> frame_buffer(FRAME_NUM_);
}



































#endif
