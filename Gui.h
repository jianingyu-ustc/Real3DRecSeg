#ifndef JIAMERA_GUI_H_
#define JIAMERA_GUI_H_

#include <vector>
#include <thread>
#include <pthread.h>
#include "Object.h"
// #include "Window.h"

namespace Jiamera {
//#define GPU_UPDATE_GL_
	class Gui {
	public:
		Gui() {
			std::cout << "Creating Gui Class -----------------------------------------\n";
			glfwInit();
		}

		Gui(const Gui&) = delete;
		Gui& operator=(const Gui&) = delete;

		~Gui() {
			glfwTerminate();
		}

		void AddWindow(Window* window) {
			if (window) {
				window->index_ = window_list_.size() + 1;
				window_list_.push_back(window);
			}
		}

		void Display() {
			for (size_t i = 0; i < window_list_.size(); ++i) {
				Window* temp_window = window_list_[i];
				thread_list_.push_back(std::thread(*temp_window));
			}
		}

		void Synchronize() {
			for (size_t i = 0; i < thread_list_.size(); ++i)
				thread_list_[i].join();
		}

		static std::vector<Window*> window_list_;
		std::vector<std::thread> thread_list_;
	};

	std::vector<Window*> Gui::window_list_;

}

#endif