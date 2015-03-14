#ifndef CUDA_RENDER_H
#define CUDA_RENDER_H
#include "HostCommon.h"
#include "CudaCommon.h"

__constant__ VRParam c_vrParam;

extern "C" void g_uploadVRParam(const VRParam *pVRParam)
{
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(c_vrParam, pVRParam, sizeof(VRParam)) );
}


extern "C" void g_releaseRenderBuffers(VRParam *vrParam)
{
	if (NULL==vrParam->d_zBuffer)
	{
		cudaThreadSynchronize();
		CUDA_SAFE_CALL( cudaFree(vrParam->d_zBuffer) );
		vrParam->d_zBuffer = NULL;
	}
}

extern "C" void g_createRenderBuffers(VRParam *vrParam)
{
	g_releaseRenderBuffers(vrParam);

	int pixels = vrParam->imgWidth * vrParam->imgHeight;
	CUDA_SAFE_CALL( cudaMalloc(&vrParam->d_zBuffer, pixels * sizeof(float)) );
}


#endif
