#ifndef JIAMERA_OBJECT_H_
#define JIAMERA_OBJECT_H_

#include <string>
#include <iostream>

#include <glad/glad.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GLFW/glfw3.h>
#include <GL/glu.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// #include <cufft.h>

//#include "Gui.h"

namespace Jiamera {
	class Gui;

	class Object {
	public:
		Object() {
			//std::cout << "Creating Object Class -----------------------------------------\n";

		}


		Object(const Object&) = delete;
		Object& operator=(const Object&) = delete;


		virtual ~Object() {}



		virtual void Display(GLFWwindow* const window, 
			                 const unsigned int window_idx) = 0;

		void bind_belonged_gl_(Gui* gui) {
			belonged_gl_ = gui;
		}

	protected:

		Gui* belonged_gl_;

	};


}


#endif