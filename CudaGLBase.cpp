#define OUTPUT_GL_TYPE_ID GL_FLOAT	// GL_UNSIGNED_BYTE
#define OUTPUT_TO_TEXTURE

#include <algorithm>
#include <time.h>
#include <math.h>

#include "CudaGLBase.h"

//<cuda
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
//>


#include "vtk_headers.h"

extern "C"
void vr_render_kernel(dim3 gridSize, dim3 blockSize, float4 *d_output);



//< protected
void CudaGLBase::setViewParam() {
    double m[16], mt[16], mtinv[16];
    glGetDoublev(GL_MODELVIEW_MATRIX, (GLdouble *)m);
    vtkMatrix4x4::Transpose(m, mt);
    vtkMatrix4x4::Invert(mt, mtinv);
    float f[12];
    for (int i=0; i<12; i++) f[i] = mtinv[i];
    //f[0] = mtinv[0]; f[1] = mtinv[4]; f[2] = mtinv[8]; f[3] = mtinv[12];
    //f[4] = mtinv[1]; f[5] = mtinv[5]; f[6] = mtinv[9]; f[7] = mtinv[13];
    //f[8] = mtinv[2]; f[9] = mtinv[6]; f[10]= mtinv[10]; f[11] = mtinv[14];




    memcpy((void *)vrParam.invViewMatrix, (void *)f, sizeof(float)*12);
}

void CudaGLBase::initPixelBuffer(int width, int height){

	if (pbo) {
		// delete old buffer
		CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo)); 
		//clReleaseMemObject(mem_pbo);

		glDeleteBuffersARB(1, &pbo);
	}
	// create pixel buffer object for display
    // need to init glew
	glGenBuffersARB(1, &pbo);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(GLfloat) * 4, 0, GL_STREAM_DRAW_ARB);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(pbo));

	// generate texture for Quad
	glBindTexture(GL_TEXTURE_2D, pbo_texid);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, OUTPUT_GL_TYPE_ID, 0);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindTexture(GL_TEXTURE_2D, 0);

}

void CudaGLBase::executeGpu()
{
	// map PBO to get CUDA device pointer
	float4 *d_output;
	CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_output, pbo));

	// wait for previous kernel calls
	cudaThreadSynchronize();
	checkError("Before Executing GPU");

	//cutilSafeCall(cudaMemset(d_output, 0, vrParam.imgWidth* vrParam.imgHeight* sizeof(float4)));

	dim3 blockSize(BLOCK_SIZE_W, BLOCK_SIZE_H);
	dim3 gridSize(vrParam.imgWidth_coalesced / blockSize.x, getCoalescedBufferSize( vrParam.imgHeight ) / blockSize.y);

#ifdef PRINT_FPS
	clock_t time = clock();
#endif

	// call CUDA kernel, writing results to PBO
	vr_render_kernel(gridSize, blockSize, d_output);

	cudaThreadSynchronize();

#ifdef PRINT_FPS
	time = clock() - time;
#endif

	if (!checkError("Render Kernel"))
		return ;

	CUDA_SAFE_CALL(cudaGLUnmapBufferObject(pbo));

}

void CudaGLBase::executeRender()
{
	glPushAttrib(GL_DEPTH_BUFFER_BIT | GL_TEXTURE_BIT );
	glEnable(GL_TEXTURE_2D);

	glClearColor(bgColor.x, bgColor.y, bgColor.z, bgColor.w);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	// normalize matrix
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0); 


	// draw image from PBO
	glDisable(GL_DEPTH_TEST);
		
	glRasterPos2i(0, 0);

	glBindTexture(GL_TEXTURE_2D, pbo_texid);
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
	{
#ifdef OUTPUT_TO_TEXTURE
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, vrParam.imgWidth, vrParam.imgHeight, GL_RGBA, OUTPUT_GL_TYPE_ID, 0);
#else
		glDrawPixels(vrParam.imgWidth, vrParam.imgHeight, GL_RGBA, GL_FLOAT, 0);
#endif
	}
	glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);


	// draw QUAD, for automatic expanding to the whole screen
#ifdef OUTPUT_TO_TEXTURE
	glColor3f(1,1,1);
	glEnable(GL_TEXTURE_2D);
	
	glBegin(GL_QUADS); 
	{
		glTexCoord2f(0, 1);
		glVertex2f(0, 1);
		glTexCoord2f(0, 0);
		glVertex2f(0, 0);
		glTexCoord2f(1, 0);
		glVertex2f(1, 0);
		glTexCoord2f(1, 1);
		glVertex2f(1, 1);
	}
	glEnd();
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
#endif

	// recall matrix
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glDisable(GL_BLEND);
	glPopAttrib();
}

bool CudaGLBase::checkError(const char *errorMessage)
{
	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		printf("CUDA error : %s : %s.\n",
			errorMessage, cudaGetErrorString( err) );
		return false;
	}
#ifdef _DEBUG
	err = cudaThreadSynchronize();
	if( cudaSuccess != err) {
		printf("cudaThreadSynchronize error: %s : %s.\n",
			errorMessage, cudaGetErrorString( err) );
		return false;
	}
#endif
	return true;
}

//> protected



//< public
CudaGLBase::~CudaGLBase() {
	g_releaseRenderBuffers(&vrParam);
	if (pbo) {
		CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
		glDeleteBuffersARB(1, &pbo);
	}
}

bool CudaGLBase::initDevice() {
	int deviceID=0;
	cudaGetDeviceProperties(&device_prop, deviceID);

	showDeviceProperties();
	bool success = (cudaGLSetGLDevice( deviceID ) == cudaSuccess);
#ifdef PINNED_MEM
	success &= (cudaSetDeviceFlags(cudaDeviceMapHost) == cudaSuccess);
#endif
	return success;
}

void CudaGLBase::executeGpuRender() {
	if (resized) {
		initPixelBuffer(vrParam.imgWidth, vrParam.imgHeight);
	}

	setViewParam();

	g_uploadVRParam(&vrParam);

    // generate image
	executeGpu();
    // draw image
	executeRender();

	resized = false;
}

void CudaGLBase::resize(int width, int height)
{
	vrParam.imgWidth = width, vrParam.imgHeight = height;
	vrParam.imgWidth_coalesced = getCoalescedBufferSize(width);
	//vrParam.unit_pixHeight = (float)vrParam.maxVolWidthInVoxel / (std::min(width, height) * vrParam.unitLen) * vrParam.dof;
	//vrParam.unit_pixHeight =  vrParam.unitLen / (float)(vrParam.dof * std::min(width, height));

	resized = true;
}

float CudaGLBase::getDepth(int x, int y) 
{
	float z;
	int idx = y*vrParam.imgWidth+x;
	cudaMemcpy(&z, vrParam.d_zBuffer+idx, sizeof(float), cudaMemcpyDeviceToHost) ;
	checkError("");
	return z;
}

void CudaGLBase::showDeviceProperties()
{
	printf("Cuda device: %s\n", device_prop.name);
	printf(" canMapHostMemory=%d\n", device_prop.canMapHostMemory);
	printf(" global mem size=%d\n", device_prop.totalGlobalMem);
	printf(" max tex3d size=%d %d %d\n", device_prop.maxTexture3D[0], device_prop.maxTexture3D[1], device_prop.maxTexture3D[2]);
	
}

void CudaGLBase::setUnitLen(float FOV, float screen_height)
{
   vrParam.unitLen = screen_height/tan(FOV/2./180.*M_PI);
}

//> public


