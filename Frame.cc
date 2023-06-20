#include "Frame.h"

namespace Jiamera {
	Frame::Frame(int rgb_width, int rgb_height, int depth_width, int depth_height, int panoptic_width, int panoptic_height,
		         int base_frame_idx, int first_frame_idx, int num_frames, 
		         const std::string& frame,
		         const std::string& rgb_intrinsic,
				 const std::string& depth_intrinsic,
				 const std::string& panoptic_intrinsic,
		         const std::string& base_pose) 
		: rgb_width_(rgb_width), rgb_height_(rgb_height), 
		  depth_width_(depth_width), depth_height_(depth_height),
		  panoptic_width_(panoptic_width), panoptic_height_(panoptic_height),
		  dataset_path(frame),
		  base_frame_index_(base_frame_idx), first_frame_index_(first_frame_idx), 
		  frame_num_(num_frames) {

		//std::cout << "Creating Frame Class -----------------------------------------\n";
		last_frame_index_ = first_frame_index_ + frame_num_ - 1;

#ifndef GPU_PROCESS_IMAGE_
		rgb_image_ = (float*)malloc(rgb_height * rgb_width * sizeof(float) * 3);
		depth_image_ = (float*)malloc(depth_height * depth_width * sizeof(float));
		panoptic_image_ = (float*)malloc(panoptic_height * panoptic_width * sizeof(float) * 3);
#endif

		rgb_viewer_ = new Viewer(rgb_width, rgb_height, rgb_intrinsic, base_pose);
		depth_viewer_ = new Viewer(depth_width, depth_height, depth_intrinsic, base_pose);
		panoptic_viewer_ = new Viewer(panoptic_width, panoptic_height, panoptic_intrinsic, base_pose);

	}

	void Frame::Update(int frame_idx) {
		std::string index = std::to_string(frame_idx);
		std::string pose_file = dataset_path + "pose/" + index + ".txt";

		if (frame_idx > last_frame_index_) {
			std::cout << "already the last frame" << "\n";
			return;
		}

#pragma omp parallel for
		for (int i = 0; i < 4; ++i) {
			if (i == 0) {
				std::string rgb_file = dataset_path + "color/" + index + ".jpg";
				update_image(rgb_image_, rgb_height_, rgb_width_, rgb_file, cv::IMREAD_COLOR);
			}
			else if (i == 1) {
				std::string depth_file = dataset_path + "depth/" + index + ".png";
				update_image(depth_image_, depth_height_, depth_width_, depth_file, cv::IMREAD_UNCHANGED);
			}
			else if (i == 2){
				std::string panoptic_file = dataset_path + "panoptic/" + index + ".png";
				update_image(panoptic_image_, panoptic_height_, panoptic_width_, panoptic_file, 3);
			}
			else{
				std::string rgb_file = dataset_path + "color/" + index + ".jpg";
				update_viewer(rgb_viewer_, pose_file);
			}
		}
		//while (frame_idx_atomic != frame_idx) continue;
		//if (frame_idx_atom.compare_exchange_strong(frame_idx, frame_idx + 1)) printf("%d\n", frame_idx);
		//frame_idx_atomic.fetch_add(1);
		//frame_buffer[frame_idx % FRAME_NUM_] = new Frame(this);
	}

	void Frame::ReadImage(float* image, int height, int width, const std::string& path, int type) {
		cv::Mat img = cv::imread(path, type);	// 30 ms -> 6ms
		if (img.rows != height || img.cols != width) {
			std::cout << "Error: image size does not match.\n";
			return;
		}

		if (img.empty()) {
			std::cout << "Error: can't read a matrix from png file!" <<
				std::endl;
			cv::waitKey(0);
		}

		// rgbdp图像预处理，可选择是否使用GPU
		if (type == cv::IMREAD_UNCHANGED) {		// 5 ms
			depth_mat_ = img;
#ifndef GPU_PROCESS_IMAGE_
			for (int r = 0; r < height; ++r) {
				for (int c = 0; c < width; ++c) {
					image[r * width + c] = (float)(img.at<ushort>(r, c)) / 1000.0f;
					// Only consider depth < 6m
					if (image[r * width + c] > 6.0f) {
						image[r * width + c] = 0;
					}
				}
			}
#endif
		}
			
		else if (type == cv::IMREAD_COLOR) {	// 70 ms
			rgb_mat_ = img;
#ifndef GPU_PROCESS_IMAGE_
			for (int r = 0; r < height; ++r) {
				for (int c = 0; c < width; ++c) {
					// bgr 顺序
					image[(r * width + c) * 3 + 0] =
						int(img.at<cv::Vec3b>(r, c)[0]) / 255.0f;
					image[(r * width + c) * 3 + 1] =
						int(img.at<cv::Vec3b>(r, c)[1]) / 255.0f;
					image[(r * width + c) * 3 + 2] =
						int(img.at<cv::Vec3b>(r, c)[2]) / 255.0f;
					//std::cout << img.step << std::endl;
					//std::cout << int(img.data[r * img.step + c * img.channels() + 0]) << " " << int(img.at<cv::Vec3b>(r, c)[0]) << "\n";	// 正确
				}
			}
#endif // GPU_PROCESS_IMAGE_
		}

		else if(type == 3) {	// 60 ms
			panoptic_mat_ = img;
#ifndef GPU_PROCESS_IMAGE_
			for (int r = 0; r < height; ++r) {
				for (int c = 0; c < width; ++c) {
					image[(r * width + c) * 3 + 0] = int(img.at<cv::Vec3b>(r, c)[0]);	// instance
					image[(r * width + c) * 3 + 1] = int(img.at<cv::Vec3b>(r, c)[1]);	// label
					image[(r * width + c) * 3 + 2] = int(img.at<cv::Vec3b>(r, c)[2]) / 255.0f;	// confidence
				}
			}
#endif
		}

		img.release();
		return;
	}





}
