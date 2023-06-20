#ifndef JIAMERA_CUBE_H_
#define JIAMERA_CUBE_H_


#include <vector>

#include "Window.h"
#include "Object.h"
#include "Color.h"
#include "Point.h"

namespace Jiamera {
    class Cube :public Object {
    public:
        Cube(Point* center, float length) : center_(center), 
                                            side_length_(length) {}

        virtual void Display(GLFWwindow* const window, 
                             const unsigned int window_idx) override {

            // 每行由 4 个点组成，代表一个面
            GLfloat vertices[] = {
                // bottom
                vertex_list_[0]->x_, vertex_list_[0]->y_, vertex_list_[0]->z_, 
                vertex_list_[1]->x_, vertex_list_[1]->y_, vertex_list_[1]->z_,
                vertex_list_[2]->x_, vertex_list_[2]->y_, vertex_list_[2]->z_, 
                vertex_list_[3]->x_, vertex_list_[3]->y_, vertex_list_[3]->z_,

                //// top
                vertex_list_[4]->x_, vertex_list_[4]->y_, vertex_list_[4]->z_, 
                vertex_list_[5]->x_, vertex_list_[5]->y_, vertex_list_[5]->z_,
                vertex_list_[6]->x_, vertex_list_[6]->y_, vertex_list_[6]->z_, 
                vertex_list_[7]->x_, vertex_list_[7]->y_, vertex_list_[7]->z_,

                // left
                vertex_list_[0]->x_, vertex_list_[0]->y_, vertex_list_[0]->z_, 
                vertex_list_[3]->x_, vertex_list_[3]->y_, vertex_list_[3]->z_,
                vertex_list_[7]->x_, vertex_list_[7]->y_, vertex_list_[7]->z_, 
                vertex_list_[4]->x_, vertex_list_[4]->y_, vertex_list_[4]->z_,

                // right
                vertex_list_[1]->x_, vertex_list_[1]->y_, vertex_list_[1]->z_, 
                vertex_list_[2]->x_, vertex_list_[2]->y_, vertex_list_[2]->z_,
                vertex_list_[6]->x_, vertex_list_[6]->y_, vertex_list_[6]->z_, 
                vertex_list_[5]->x_, vertex_list_[5]->y_, vertex_list_[5]->z_,

                // front
                vertex_list_[0]->x_, vertex_list_[0]->y_, vertex_list_[0]->z_, 
                vertex_list_[1]->x_, vertex_list_[1]->y_, vertex_list_[1]->z_,
                vertex_list_[5]->x_, vertex_list_[5]->y_, vertex_list_[5]->z_, 
                vertex_list_[4]->x_, vertex_list_[4]->y_, vertex_list_[4]->z_,

                // behind
                vertex_list_[3]->x_, vertex_list_[3]->y_, vertex_list_[3]->z_, 
                vertex_list_[2]->x_, vertex_list_[2]->y_, vertex_list_[2]->z_,
                vertex_list_[6]->x_, vertex_list_[6]->y_, vertex_list_[6]->z_, 
                vertex_list_[7]->x_, vertex_list_[7]->y_, vertex_list_[7]->z_,

            };

            GLfloat colors[] = {
                0, 0, 0,   0, 0, 1,   0, 1, 1,   0, 1, 0,
                1, 0, 0,   1, 0, 1,   1, 1, 1,   1, 1, 0,
                0, 0, 0,   0, 0, 1,   1, 0, 1,   1, 0, 0,
                0, 1, 0,   0, 1, 1,   1, 1, 1,   1, 1, 0,
                0, 0, 0,   0, 1, 0,   1, 1, 0,   1, 0, 0,
                0, 0, 1,   0, 1, 1,   1, 1, 1,   1, 0, 1
            };



            //glEnableClientState(GL_VERTEX_ARRAY);
            //glEnableClientState(GL_COLOR_ARRAY);

            //glVertexPointer(3, GL_FLOAT, 0, vertices);
            //glColorPointer(3, GL_FLOAT, 0, colors);

            //glDrawArrays(GL_QUADS, 0, 24);

            //glDisableClientState(GL_COLOR_ARRAY);
            //glDisableClientState(GL_VERTEX_ARRAY);
        }

    protected:
        Point* center_;
        float side_length_;
        std::vector<Point*> vertex_list_;



    };


}

#endif // !CUBE_H
