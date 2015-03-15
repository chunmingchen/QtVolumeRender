////////////////////////////////////////////////////////////////////////////////////////////////
//common settings for PC and GPU
////////////////////////////////////////////////////////////////////////////////////////////////
#ifndef HOST_COMMON_H
#define HOST_COMMON_H

#define LOD_DATA_TYPE float

// for debugging:
//#define PRINT_FPS
//#define PRINT_CURSER_COLOR

//#define PINNED_MEM  // not working currently

//////////////////////////////////////////

#ifdef _DEBUG
#define PRINT(...) printf(__VA_ARGS__)
#else
#define PRINT(...) NULL  //<--this should work on C99
#endif

////////////////////////////////////////////////////////////////////////////////////////////////
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdio.h>
#define CUDA_SAFE_CALL(call)\
{\
	cudaError err=call;\
	if(cudaSuccess != err)\
	{\
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        /*exit(-1); */                                                 \
	}\
}
#define BLOCK_SIZE_W 16	//cuda block size per thread
#define BLOCK_SIZE_H 16	//cuda block size per thread

////////////////////////////////////////////////////////////////////////////////////////////////
typedef unsigned int  uint;
typedef unsigned char uchar;


typedef struct {
	int imgWidth, imgHeight;
    int imgWidth_coalesced;
	float step;				// step size
	int maxSteps;			// limit steps per frame
	float clippingVisible;	// clipping plane
	float unitLen;			// UNIT_LEN 2.41421f  //screen_height=1 /tan(45/2/180*PI) = 2.41421  (FOV=45 degree) (in [0..1] space)
	int maxVolWidthInVoxel;	// max volume size to render
	float volBoundry[3];	// vol size in -1..1
	float invViewMatrix[16];
    float dof;				// user defined depth of field (no use here)
	float unit_pixHeight;	// =  (float)c_vrParam.maxVolWidthInVoxel / (minw * c_vrParam.unitLen) * c_vrParam.dof;// todo: save as constant
	//float intensity;
    float *d_zBuffer;
    float intensity;        // tuning opacity
    float value_min;
    float value_dist;

    //char restartRay;		// cannot use bool for kernel!?
	char bFastRender;

}VRParam;

/////////////////////////////


typedef struct {
	unsigned char a,b,g,r;
}TestColor;

//////////////////////////////////////////////////////////////////////////
extern "C"
{
	// render
    void g_uploadVRParam(const VRParam *pVRParam);
	void g_releaseRenderBuffers(VRParam *vrParam);
	void g_createRenderBuffers(VRParam *vrParam);

    // data
    void g_createBrickPool(int w, int h, int d);
    void g_releaseBrickPool();
    void g_uploadBrickPool(void *h_volume, int w, int h, int d, int x, int y, int z);

    // label
    void g_createLabelPool(int w, int h, int d);
    void g_releaseLabelPool();
    void g_uploadLabelPool(void *h_volume, int w, int h, int d, int x, int y, int z);


	// transfer function

	void g_releaseTrFn();
	void g_createTrFn(int w, int h);
	void g_uploadTrFn(void *data, int len);

	// shading
	void g_setPhongShading();
}

#endif //HOST_COMMON_H
