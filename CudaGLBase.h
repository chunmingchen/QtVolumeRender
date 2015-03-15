/////////////////////////////////////////////////////
/// Cuda volume renderer base class
/// Handles cuda setups, cuda-gl interop and geometry translation
///  Chun-Ming Chen
/////////////////////////////////////////////////////

#ifndef _CudaGLBase_h_
#define _CudaGLBase_h_
// This .H file needs VRParam defined in advance

#include <exception>
#include <stdio.h>

#include <GL/glew.h>
#include <GL/gl.h>

#include "HostCommon.h"


class CudaGLBase{

protected:
	bool resized;

	GLuint pbo;                 // OpenGL pixel buffer object
	GLuint pbo_texid;
	float4 bgColor;

	inline int getCoalescedBufferSize(size_t original_width) {
		int base = BLOCK_SIZE_W;
		return ((original_width + base - 1) / base ) * base;
	}

	void setViewParam();

	void initPixelBuffer(int width, int height);

	void executeGpu();


	void executeRender();

public:
    VRParam vrParam;
	cudaDeviceProp device_prop;

	CudaGLBase() : pbo(0), resized(true) {
		bgColor = make_float4(0,0,0,0);
	}

	~CudaGLBase() ;

	void executeGpuRender() ;

	bool initDevice() ;

	void resize(int width, int height);

	inline bool isResized() { return resized; }

	// debug
	inline void printColor(int x, int y)
	{
		unsigned char pixel[10];
		if (pbo) {
			glReadPixels(x,y, 1,1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
			printf("Pixel(%d,%d)=(%d %d %d %d)\n", x,y, pixel[0], pixel[1], pixel[2], pixel[3]);			
		}		
	}

	float getDepth(int x, int y) ;

	bool checkError(const char *errorMessage);

	void showDeviceProperties();

	inline void setBgColor(const float4 &col) {bgColor = col;}

    void setUnitLen(float FOV, float screen_height=1.f);
};
#endif
