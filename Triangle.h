#ifndef JIAMERA_TRIANGLE_H_
#define JIAMERA_TRIANGLE_H_

#include <vector>

#include "Window.h"
#include "Object.h"


namespace Jiamera {
	class Triangle :public Object {
	public:
		Triangle() {}
		virtual void Display(GLFWwindow* const window, 
							 const unsigned int window_idx) override {}
	};
}

#endif