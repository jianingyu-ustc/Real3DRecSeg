#ifndef JIAMERA_WINDOW_H_
#define JIAMERA_WINDOW_H_

#include <iostream>
#include <vector>
#include <map>
#include <math.h>

// #pragma comment (lib,"glew32.lib")
// #pragma comment (lib,"glu32.lib")
// #pragma comment (lib,"openGL32.lib")
// #pragma comment (lib,"glfw3.lib")



#include "Object.h"
#include "Shader.h"
#include "StbImage.h"
#include "Camera.h"

namespace Jiamera{

	GLFWwindow* rgb_window;	// 作为主窗口，控制回调函数的参数

	class Window {

	public:
		Window() {
			//std::cout << "Creating Window Class -----------------------------------------\n";

			//if(!universal_camera) 
			// universal_camera = new Camera(glm::vec3(0.0f, 0.0f, 3.0f));
		};
		Window(const std::string &title, unsigned int x, unsigned int y, 
			   unsigned int w, unsigned int h) 
			: title_(title), screen_position_x_(x), screen_position_y_(y), 
			  screen_width_(w), screen_height_(h) {
			//std::cout << "Creating Window Class -----------------------------------------\n";

			lastX = screen_width_ / 2.0f;
			lastY = screen_height_ / 2.0f;

			//if (!universal_camera) 
			// universal_camera = new Camera(glm::vec3(0.0f, 0.0f, 3.0f));
		}

		~Window() {};

		void show_object() {
			for (int i = 0; i < object_.size(); ++i)
				object_[i]->Display(this->window_, this->index_);
		}

		void add_object(Object* const object) {
			object_.push_back(object);
		}

		void operator()() {
			glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
			glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
			glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
			window_ = glfwCreateWindow(this->screen_width_, this->screen_height_, this->title_.c_str(), NULL, NULL);
			if (window_ == NULL) {
					std::cout << "Failed to create GLFW window" << std::endl;
					glfwTerminate();
					return;
				}

			if (title_ == "rgb") {
				rgb_window = window_;
			}

			if (window_ == nullptr) {
				std::cout << "Fail to create GLFW window.\n";
				return;
			}

			// 将窗口的上下文设置为当前线程的上下文
			glfwMakeContextCurrent(window_);

			glfwSetFramebufferSizeCallback(window_, framebuffer_size_callback);
			glfwSetCursorPosCallback(window_, mouse_move_callback);
			glfwSetScrollCallback(window_, scroll_callback);
			//glfwSetMouseButtonCallback(window_, mouse_button_callback);

			// 加载所有 OpenGL 函数指针
			if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
				std::cout << "Fail to initialize GLAD\n";
				return;
			}

			glEnable(GL_DEPTH_TEST);

			this->show_object();
		}


		GLFWwindow* window_;
		std::string title_ = "window";
		unsigned int index_;	// 本对象在 Gui 中的序列，以 1 为起点

		unsigned int screen_position_x_, screen_position_y_;
		unsigned int screen_width_ = 800, screen_height_ = 600;

		std::vector<Object*> object_;

	};


}



#endif