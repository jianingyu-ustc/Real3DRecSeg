#ifndef VIEWER
#define VIEWER

#include <iostream>
#include <vector>
#include "Matrix.h"

namespace Jiamera {
	extern float gaze[12];
	class Viewer {

	public:

		Viewer(int width, int height, const std::string &intrinsics, 
			   const std::string &base_pose);

		void Update(const std::string &pose) {
			update_pose_file(pose);
		}

		void update_pose_file(const std::string& pose);

		float* get_pose() {
			return current_pose_->grid_;
		}
		Matrix<float>* clone_pose() {
			Matrix<float>* new_pose = new Matrix<float>(4, 4);
			memcpy(new_pose->grid_, current_pose_->grid_, sizeof(float) * 4 * 4);
			return new_pose;
		}

		int current_frame_index_;	// 当前帧序号
		Matrix<float>* intrinsics_;	// 相机内参, 构造时赋值

		// 第一帧世界系下的相机位姿, 构造时赋值, 存储在 frame-000150.pose.txt
		Matrix<float>* base_pose_;

		Matrix<float>* base_pose_inverse_;	// base_pose 的逆
		Matrix<float>* delta_pose_;		// 当前帧与第一帧变换的位姿

		// 当前帧的位姿，= base_pose_inverse * delta_pose
		Matrix<float>* current_pose_;

		float position_[3];	// 当前相机中心坐标
		float front_[3];	// 当前正相机前方 1m 的坐标
		float gaze_[3];


	};

}

#endif //