# 环境配置
- Ubuntu 22.04.1 LTS
- GCC 11.3.0
- CUDA Version: 11.7
- OpenCV 2.4
- 安装OpenGL并确保glad、GLFW、GL、glm文件夹已放置在/usr/include/目录中
- Python 2.7
- Numpy 1.21.5

## 目录结构
将数据集、项目文件、结果文件放在同一级目录中：
`/home/jia/Jiamera`

```
/home/jia/scannet/
|-- <scanId/>
|-- ...

<scanId/>
|-- <scanId>_vh_clean.ply
|-- <scanId>_vh_clean_2.ply
|-- <scanId>.aggregation.json, <scanId>_vh_clean.aggregation.json
|-- <scanId>_vh_clean_2.labels.ply
|-- <scanId>_2d-label.zip
|-- <color/>
|-- <depth/>
|-- <panoptic/>
```

# 参数设定

**main.cc**
- scene_id：选择重建场景
- Jiamera::Particles(12.0f, 10.0f, 10.0f, 0.02)：设置重建空间的xyz边长和体素块边长，单位：米
- `#define DISPLAY_`：展示实时重建动画

**Kernel.cuh**
- `#define MULTI_SAMPLE_AA_`：对体素块多采样

**Particles.h, line 179**
- `glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * current_point_num_, this->gl_rgb_, GL_STATIC_DRAW);`：展示rgb模型
- `glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * current_point_num_, this->gl_label_, GL_STATIC_DRAW);`：展示label模型

# 重建

创建目录：`mkdir /home/jia/pred`

进入项目目录：`cd /home/jia/Jiamera`

**重建单个场景：**
```sh
sh construct.sh ../scannet/scenexxxx_xx/
```

**重建所有场景：**
```sh
sh construct_all.sh
```

重建动画可以进行720°浏览，按WSADQE键进行平移，按住鼠标左键拖动改变视角，滑动滚轮进行缩放。运行结束后按Esc键退出动画窗口，生成的ply点云文件存储在/Jiamera/result目录下。

# 测试

**eval.py，line 51：** 选择pred模型所在文件夹

进入项目目录：`cd /home/jia/Jiamera`

**测试单个场景：**
```sh
python2 eval.py scenexxxx_xx iou
```

**测试所有场景：**
```sh
for file in ../scannet/*; do echo ${file: -12}; python2 eval.py ${file: -12} iou; done
```

以下场景图像尺寸异常，重建失败：
- scene0088_03
- scene0144_00
- scene0144_01
- scene0354_00
- scene0474_03
- scene0689_00
- scene0704_00
- scene0704_01