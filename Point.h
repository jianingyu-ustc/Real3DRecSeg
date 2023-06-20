#ifndef JIAMERA_POINT_H_
#define JIAMERA_POINT_H_

#include "Window.h"
#include "Object.h"
#include "Color.h"
#include "Voxel.h"



float vertices[18] = {
	// 位置             // 颜色 
	 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, // 右下 
	-0.5f, -0.5f, 0.0f, 0.0f, 1.0f, 0.0f, // 左下 
	 0.0f,  0.5f, 0.0f, 0.0f, 0.0f, 1.0f  // 上 
};

namespace Jiamera {
	class Point : public Object {
	public:
		Point() : x_(0), y_(0), z_(0), color_(Color(0, 0, 0)) {}
		Point(float x, float y) : x_(x), y_(y), z_(0), 
			  color_(Color(0, 0, 0)) {}

		Point(float x, float y, const Color& color) : x_(x), y_(y), z_(0),
			  color_(color) {}

		Point(float x, float y, float z) : x_(x), y_(y), z_(z),
		      color_(Color(0, 0, 0)) {}

		Point(float x, float y, float z, const Color& color) 
			: x_(x), y_(y), z_(z), color_(color) {}
		~Point() {};

		void Display(GLFWwindow* const window, 
			         const unsigned int window_idx) override {
			Shader ourShader("Shader.vs", "Shader.fs");

			//创建VBO和VAO对象，并赋予ID
			unsigned int VBO, VAO;
			glGenVertexArrays(1, &VAO);
			glGenBuffers(1, &VBO);
			//绑定VBO和VAO对象
			glBindVertexArray(VAO);
			glBindBuffer(GL_ARRAY_BUFFER, VBO);
			//为当前绑定到target的缓冲区对象创建一个新的数据存储。
			//如果data不是NULL，则使用来自此指针的数据初始化数据存储
			glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, 
				         GL_STATIC_DRAW);

			//告知Shader如何解析缓冲里的属性值
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
				                  (void*)0);
			//开启VAO管理的第一个属性值
			glEnableVertexAttribArray(0);

			//告知Shader如何解析缓冲里的属性值
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float),
				                  (void*)(3 * sizeof(float)));
			//开启VAO管理的第一个属性值
			glEnableVertexAttribArray(1);

			glBindBuffer(GL_ARRAY_BUFFER, 0);
			glBindVertexArray(0);

			// 渲染循环
			while (!glfwWindowShouldClose(window)) {
				//processInput(window);
				glClearColor(0.2f, 0.3f, 0.3f, 1.0f); //状态设置
				glClear(GL_COLOR_BUFFER_BIT); //状态使用

				ourShader.use();

				// 设置uniform值
				float timeValue = glfwGetTime();
				float greenValue = sin(timeValue) / 2.0f + 0.5f;
				ourShader.setVec4("ourColor", 0.0f, greenValue, 0.0f, 1.0f);
				ourShader.setFloat("offsetX", 0.5);
				glBindVertexArray(VAO);
				// glfw: 交换缓冲区和轮询IO事件（按键按下/释放、鼠标移动等）
				glDrawArrays(GL_TRIANGLES, 0, 3);
				glfwSwapBuffers(window);
				glfwPollEvents();
			}
			// glfw: 回收前面分配的GLFW先关资源. 
			glDeleteVertexArrays(1, &VAO);
			glDeleteBuffers(1, &VBO);
			glDeleteProgram(ourShader.ID);
		}

		float x_;
		float y_;
		float z_;
		Color color_;
	};



}








#endif