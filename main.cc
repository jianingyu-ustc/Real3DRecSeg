#include <thread>
#include <stdio.h>
#include <dirent.h>
#include <cstdlib>
#include <vector>

#include "Matrix.h"
#include "Frame.h"
#include "Voxel.h"

#include "Window.h"
#include "Color.h"
#include "Point.h"
#include "Particles.h"
#include "Kernel.cuh"
#include "Triangle.h"
#include "AxisAlinedCube.h"
#include "Gui.h"
#define DISPLAY_

std::string GetCmdResult(const std::string &strCmd){
	char buf[10240] = {0};
	FILE *pf = NULL;
	
	if( (pf = popen(strCmd.c_str(), "r")) == NULL )
	{
		return "";
	}
 
	std::string strResult;
	while(fgets(buf, sizeof buf, pf))
	{
		strResult += buf;
	}
	
	pclose(pf);
 
	unsigned int iSize =  strResult.size();
	if(iSize > 0 && strResult[iSize - 1] == '\n')  // linux
	{
		strResult = strResult.substr(0, iSize - 1);
	}
 
	return strResult;
}

int main(int argc, char* argv[]) {
	#ifdef DISPLAY_		// 展示动画
		// Gui类包含3个Window类成员
		Jiamera::Gui* gl_system = new Jiamera::Gui();
		// title: 1 rgb, 2 label, 3 instance
		Jiamera::Window* rgb_window = new Jiamera::Window("rgb", 100, 100, 640, 480);	
		Jiamera::Window* label_window = new Jiamera::Window(std::string("label"), 1000, 100, 640, 480);
		Jiamera::Window* instance_window = new Jiamera::Window(std::string("instance"), 500, 100, 640, 480);
	#endif

	// 数据集目录
	std::string scene_dir = std::string(argv[1]);
	std::string rgb_intrinsic_file = scene_dir + "intrinsic/intrinsic_color.txt";
	std::string depth_intrinsic_file = scene_dir + "intrinsic/intrinsic_depth.txt";
	std::string panooptic_intrinsic_file = scene_dir + "intrinsic/intrinsic_color.txt";
	std::string base_pose_file = scene_dir + "pose/0.txt";
	int frame_num = stoi(GetCmdResult("ls " + scene_dir + "color/ | wc -l"));

	// 存储数据及rgbd及panoptic图像参数
	Jiamera::Frame* chief_frame = new Jiamera::Frame(1296, 968, 640, 480, 1296, 968, 0, 0, frame_num, 
													scene_dir, rgb_intrinsic_file, depth_intrinsic_file, 
													panooptic_intrinsic_file, base_pose_file);
	// 多线程读取图像帧
	std::vector<Jiamera::Frame*> frame_workers;
	for (size_t i = 0; i < FRAME_NUM_; ++i) {
		frame_workers.push_back(new Jiamera::Frame(1296, 968, 640, 480, 1296, 968, 0, 0, frame_num, 
												scene_dir, rgb_intrinsic_file, depth_intrinsic_file, 
												panooptic_intrinsic_file, base_pose_file));
	}

	/*
		when gpu block size = 2m, grid size must >= 
		when gpu block size = 4m, grid size must >= 0.8cm
		when gpu block size = 8m, grid size must >= 
		when gpu block size = 16m, grid size must >= 3cm
	*/


	// Voxel类保存体素块数据，Particles类将体素块以点云形式显示在OpenGL窗口中
	Jiamera::Particles* voxel = new Jiamera::Particles(12.0f, 10.0f, 10.0f, 0.02);


	// 将体素空间和图像参数拷贝进设备端
	CudaPreWork(voxel, chief_frame);

	#ifdef DISPLAY_
		// 为Voxel的父类Object绑定对应的Gui对象
		voxel->bind_belonged_gl_(gl_system);

		// 为三个Window对象添加要现实的Object对象
		rgb_window->add_object(voxel);
		label_window->add_object(voxel);
		instance_window->add_object(voxel);
		// 将三个Window对象添加到Gui对象中，每个窗口共用同一个camera，位姿始终相同
		// Windows OS可以同时显示三个窗口，Linux 遇到了问题，
		// 暂时在 Particles.h line 154 修改 rgb_window 显示的属性
		gl_system->AddWindow(rgb_window);
		// gl_system->AddWindow(label_window);
		//gl_system->AddWindow(instance_window);

		// 展示Gui对象
		gl_system->Display();
		// std::this_thread::sleep_for(std::chrono::milliseconds(10000));
	#endif

	// 调用核函数进行重建
	CudaInWork(voxel, chief_frame, frame_workers);
	// 清理设备端数据
	CudaPostWork(voxel, chief_frame, frame_workers);

	// 将voxel体素块数据以点云形式存储到ply文件中
	voxel->SaveCloud(scene_dir.substr(11, 12), chief_frame->rgb_viewer_->base_pose_->grid_);
	//voxel->SaveParamaters();

	#ifdef DISPLAY_
	// 待手动关闭窗口后，程序退出
	gl_system->Synchronize();
	#endif

	return 0;
}
