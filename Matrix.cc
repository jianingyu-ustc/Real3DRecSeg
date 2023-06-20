#include "Matrix.h"
#include "Voxel.h"

template<class T>
Matrix<T>::Matrix(int r, T default_value) : dim_(1) {
	size_[0] = r;
	len_ = r;
	grid_initialize(default_value);
}
template Matrix<float>::Matrix(int, float);

template<class T>
Matrix<T>::Matrix(int r, int c, T default_value) : dim_(2) {
	size_[0] = r;
	size_[1] = c;
	len_ = r * c;
	grid_initialize(default_value);
}
template Matrix<int>::Matrix(int, int, int);
template Matrix<float>::Matrix(int, int, float);


template<class T>
Matrix<T>::Matrix(int r, int c, int w, T default_value) : dim_(3) {
	size_[0] = r;
	size_[1] = c;
	size_[2] = w;
	len_ = r * c * w;
	grid_initialize(default_value);
}
template Matrix<float>::Matrix(int, int, int, float);
template Matrix<VoxelBlock*>::Matrix(int, int, int, VoxelBlock*);


template<class T>
void Matrix<T>::view_dim2() {
	for (int i = 0; i < size_[0]; i++) {
		for (int j = 0; j < size_[1]; j++) {
			std::cout << grid_[size_[1] * i + j] << " ";
		}
		std::cout << "\n";
	}
}
template void Matrix<int>::view_dim2();
template void Matrix<float>::view_dim2();


template<class T>
Matrix<T>* Matrix<T>::get_inverse_4x4() {

	Matrix* ans = new Matrix(4, 4, 0.0f);
	float* inv = (float*)malloc(16 * sizeof(float));

	float det;
	float* m = grid_;

	inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] +
		     m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

	inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] -
		     m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

	inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + 
		     m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

	inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] -
		     m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

	inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - 
		     m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

	inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + 
		     m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

	inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - 
		     m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

	inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + 
		     m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

	inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + 
		     m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

	inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - 
		     m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

	inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + 
		     m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

	inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - 
		     m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

	inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - 
		     m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

	inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + 
		     m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

	inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - 
		     m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

	inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + 
		     m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

	det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
	if (det == 0) {
		std::cout << "inverse fail\n";
		return nullptr;
	}
	det = 1.0 / det;
	for (int i = 0; i < 16; i++) ans->grid_[i] = inv[i] * det;

	return ans;
}
template Matrix<float>* Matrix<float>::get_inverse_4x4();


template<class T>
void Matrix<T>::read_txt(const std::string &file_name,const int row, const int col) {
	std::vector<float> matrix;
	FILE* fp = fopen(file_name.c_str(), "r");

	for (int r = 0; r < size_[0]; ++r) {
		int c = 0;
		float temp_num;
		for (; c < size_[1]; ++c) {
			fscanf(fp, "%f", &grid_[r * size_[1] + c]);
		}
		for (; c < col; ++c) {
			fscanf(fp, "%f", &temp_num);
		}
	}

	fclose(fp);
	return;
}
template void Matrix<float>::read_txt(const std::string &, const int, const int);

//template<class T>
//void Matrix<T>::read_png(const std::string file_name, int type) {
//	cv::Mat image = cv::imread(file_name, type);
//	std::cout << image.rows << " " << image.cols << " " << image.channels() << "%%%%%%%%%%%%%%%%%%%%%%\n";
//
//	if (image.empty()) {
//		std::cout << "Error: can't read a matrix from png file!" << std::endl;
//		cv::waitKey(0);
//	}
//
//	if (type == cv::IMREAD_UNCHANGED) {
//
//		for (int r = 0; r < size_[0]; ++r) {
//			for (int c = 0; c < size_[1]; ++c) {
//				grid_[r * size_[1] + c] = (float)(image.at<unsigned short>(r, c)) / 1000.0f;
//				if (grid_[r * size_[1] + c] > 6.0f) // Only consider depth < 6m
//					grid_[r * size_[1] + c] = 0;
//			}
//		}
//	}
//	
//	else if (type == cv::IMREAD_COLOR) {
//
//		//std::cout << "(" << int(image.at<cv::Vec3b>(r, c)[0]) << ", " << int(image.at<cv::Vec3b>(r, c)[1]) << ", " << int(image.at<cv::Vec3b>(r, c)[2]) << ")  ";
//
//		//std::cout << image.at<cv::Vec3b>(100, 100)[0] << " " << image.at<cv::Vec3b>(100, 100)[1] << " " << image.at<cv::Vec3b>(100, 100)[2] << "$$$$$$$$$$$$$$$$$$$$\n";
//
//		std::cout << "cv::IMREAD_COLOR needs to be rewrote\n";
//	}
//
//
//
//	return;
//}
//template void Matrix<float>::read_png(std::string, int);

