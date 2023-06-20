#ifndef JIAMERA_MATRIX_H_
#define JIAMERA_MATRIX_H_


#include <iostream>
#include <opencv2/opencv.hpp>

template<class T>
class Matrix
{
	// template<class U> friend Matrix* matrix_multiply_2d(Matrix<U>* m1, Matrix<U>* m2);
public:
	
	explicit Matrix(int r, T default_value = 0);
	explicit Matrix(int r, int c, T default_value = 0);
	explicit Matrix(int r, int c, int w, T default_value = 0);

	void view_dim2();

	//void read_png(std::string file_name, int type);

	void read_txt(const std::string &file_name, const int row, const int col);

	Matrix* get_inverse_4x4();

	T* grid_;
	int size_[10];	// 每个维度的长度，最高10维

private:
	int dim_;	// 维度
	int len_;	// size_[0] * size_[1] * ... * size_[dim_ - 1]

	void grid_initialize(T default_value) {
		grid_ = (T*)malloc(len_ * sizeof(T));
		for (int i = 0; i < len_; ++i) grid_[i] = default_value;
		//for (int i = 0; i < len_; ++i) grid_[i] = (T)(i + 1);
	}

};

template<class T>
Matrix<T>* matrix_multiply_2d(const Matrix<T>* m1, const Matrix<T>* m2) {
	int m1_row = m1->size_[0];
	int m1_col = m1->size_[1];
	int m2_row = m2->size_[0];
	int m2_col = m2->size_[1];

	if (m1_col != m2_row) {
		std::cout << "can't multiply these two matrix \n";
		return nullptr;
	}

	Matrix<T>* ans = new Matrix<T>(m1_row, m2_col);
	for (int r = 0; r < m1_row; ++r) {
		for (int c = 0; c < m2_col; ++c) {
			double temp_val = 0;
			for (int i = 0; i < m1_col; ++i) 
				temp_val += m1->grid_[r * m1_col + i] * 
				            m2->grid_[i * m2_col + c];

			ans->grid_[r * m2_col + c] = temp_val;
		}
	}

	return ans;
}


#endif