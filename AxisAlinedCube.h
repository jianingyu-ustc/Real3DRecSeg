#ifndef JIAMERA_AXIS_ALINED_CUBE_H_
#define JIAMERA_AXIS_ALINED_CUBE_H_

#include "Cube.h"

const int X[8] = { -1,1,1,-1,-1,1,1,-1 };
const int Y[8] = { -1,-1,-1,-1,1,1,1,1 };
const int Z[8] = { 1,1,-1,-1,1,1,-1,-1 };

namespace Jiamera {

	// 轴对齐立方体，用重心+边长的方法表示
	class AxisAlinedCube :public Cube {
	public:
		AxisAlinedCube(Point* center, float length) :Cube(center, length) {
			float half_size_length = length / 2;

			for (int i = 0; i < 8; ++i) {
				Point* vertex = new Point(center->x_ + X[i] * half_size_length,
										  center->y_ + Y[i] * half_size_length,
										  center->z_ + Z[i] * half_size_length);
				vertex_list_.push_back(vertex);
			}

		}
	};
}


#endif
