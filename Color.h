#ifndef JIAMERA_COLOR_H_
#define JIAMERA_COLOR_H_

struct Color {
	double R, G, B;
	Color(int r, int g, int b) {
		this->R = (double)r / 255;
		this->G = (double)g / 255;
		this->B = (double)b / 255;
	}
};

#endif