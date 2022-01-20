#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <vector>       // std::vector



typedef struct {
	int x,y;
} Point;

typedef struct {
	float r,g,b,a;
    Point start;
} item;

void plot(std::vector<item> M;
          int step;) {

    static const std::string filename   = "../../jump_flood/output.png";
}

