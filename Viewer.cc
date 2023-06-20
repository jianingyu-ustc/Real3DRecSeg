#include "Viewer.h"

namespace Jiamera {
	Viewer::Viewer(int width, int height, const std::string& intrinsics, 
		           const std::string& file) {
		//std::cout << "Creating Viewer Class -----------------------------------------\n";

		intrinsics_ = new Matrix<float>(3, 3);

		intrinsics_->read_txt(intrinsics, 4, 4);
		//intrinsics_->view_dim2();

		base_pose_ = new Matrix<float>(4, 4);
		base_pose_->read_txt(file, 4, 4);

		base_pose_inverse_ = base_pose_->get_inverse_4x4();

		delta_pose_ = new Matrix<float>(4, 4);
		current_pose_ = new Matrix<float>(4, 4);

	}

	void Viewer::update_pose_file(const std::string& pose) {
		delta_pose_->read_txt(pose, 4, 4);
		current_pose_ = matrix_multiply_2d(base_pose_inverse_, delta_pose_);

		position_[0] = current_pose_->grid_[3];
		position_[1] = current_pose_->grid_[7];
		position_[2] = current_pose_->grid_[11];

		float p[16];

		// 根据相机系中的(0,0,0)(0,0,1)向量反推出其在世界系中的向量，用于在动画窗口中绘画相机坐标及方向
		for(size_t i = 0; i < 15; ++i) p[i] = current_pose_->grid_[i];
		front_[0] = (p[0]*p[10]*p[3]*p[5] - p[0]*p[3]*p[6]*p[9] - p[1]*p[10]*p[3]*p[4] + 
							  p[1]*p[3]*p[6]*p[8] + p[2]*p[3]*p[4]*p[9] - p[2]*p[3]*p[5]*p[8] + p[4]*p[9] - p[5]*p[8]) / 
							 (p[0]*p[10]*p[5] - p[0]*p[6]*p[9] - p[1]*p[10]*p[4] + 
							  p[1]*p[6]*p[8] + p[2]*p[4]*p[9] - p[2]*p[5]*p[8]);

		front_[1] = (p[0]*p[10]*p[5]*p[7] - p[0]*p[6]*p[7]*p[9] - p[0]*p[9] - 
							  p[1]*p[10]*p[4]*p[7] + p[1]*p[6]*p[7]*p[8] + p[1]*p[8] + 
							  p[2]*p[4]*p[7]*p[9] - p[2]*p[5]*p[7]*p[8]) / 
							 (p[0]*p[10]*p[5] - p[0]*p[6]*p[9] - p[1]*p[10]*p[4] + 
							  p[1]*p[6]*p[8] + p[2]*p[4]*p[9] - p[2]*p[5]*p[8]);

		front_[2] = (p[0]*p[10]*p[11]*p[5] - p[0]*p[11]*p[6]*p[9] + p[0]*p[5] - 
							  p[1]*p[10]*p[11]*p[4] + p[1]*p[11]*p[6]*p[8] - p[1]*p[4] + 
							  p[11]*p[2]*p[4]*p[9] - p[11]*p[2]*p[5]*p[8]) /
							 (p[0]*p[10]*p[5] - p[0]*p[6]*p[9] - p[1]*p[10]*p[4] + 
							  p[1]*p[6]*p[8] + p[2]*p[4]*p[9] - p[2]*p[5]*p[8]);

		gaze_[0] = front_[0] - position_[0];
		gaze_[1] = front_[1] - position_[1];
		gaze_[2] = front_[2] - position_[2];

		gaze[0]=position_[0];
		gaze[1]=position_[1];
		gaze[2]=position_[2];

		gaze[6]=front_[0];
		gaze[7]=front_[1];
		gaze[8]=front_[2];
	}
}
