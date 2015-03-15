////////////////////////////////////////
// Volume Renderer
// Chun-Ming Chen
// Ohio State University
////////////////////////////////////////

//#define SHADING

//#include <cutil_inline.h>
#include <cutil_math.h>
#include "HostCommon.h"
#include "CudaCommon.h"
#include "cuda_shader.h"
////////////////////////////////////////////////////////////////////////////////////////////////

struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};

////////////////////////////////////////////////////////////////////////////////////////////////
// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

inline __device__ int intersectBox(Ray &r, float3 &boxmin, float3 &boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / (r.d+1e-30f);
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

	*tnear = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));  // largest_tmin
	*tfar = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));	// smallest_tma

	return *tfar > *tnear;
}

inline __device__ float4 convert_float4(uchar4 &c)
{
	return make_float4(c.x/255.f, c.y/255.f, c.z/255.f, c.w/255.f);
}


__global__ void d_render(float4 *d_output)
{
	uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
	uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

	// ignore out-of-sight pixels.  The image is a little larger than original one for coalesce issue
	if (y>=c_vrParam.imgHeight || x>=c_vrParam.imgWidth)	return;

	// index for uncoalesced buffers
    uint out_idx = (y * c_vrParam.imgWidth) + x;

#if 0 // debug trfn
                                            d_output[out_idx] = tex2D(texTrFn, x/255.f,y/255.f);
											return;
#endif
	// calculate eye ray in world space
	// input: x,y  output: eyeRay
    Ray eyeRay;
	{
        int minw = min(c_vrParam.imgWidth, c_vrParam.imgHeight);
		float u = ((int)(x<<1) - c_vrParam.imgWidth) / (float) minw;
		float v = ((int)(y<<1) - c_vrParam.imgHeight) / (float) minw;
		
		eyeRay.o = make_float3(c_vrParam.invViewMatrix[3], c_vrParam.invViewMatrix[7], c_vrParam.invViewMatrix[11]);

		float3 vec = normalize(make_float3(u, v, -c_vrParam.unitLen));  
        eyeRay.d.x = dot(vec, make_float3(c_vrParam.invViewMatrix[0],c_vrParam.invViewMatrix[1],c_vrParam.invViewMatrix[2]));
		eyeRay.d.y = dot(vec, make_float3(c_vrParam.invViewMatrix[4],c_vrParam.invViewMatrix[5],c_vrParam.invViewMatrix[6]));
		eyeRay.d.z = dot(vec, make_float3(c_vrParam.invViewMatrix[8],c_vrParam.invViewMatrix[9],c_vrParam.invViewMatrix[10]));
		//d_output[d_out_idx] = make_float4(u, v, 0 , 1.f);

	}

#if SHADING_TYPE==PHONG_SHADING
    float3 lightVec;
    // light = M*v
    lightVec.x = dot(c_light.pos, make_float3(c_vrParam.invViewMatrix[0],c_vrParam.invViewMatrix[1],c_vrParam.invViewMatrix[2]));
    lightVec.y = dot(c_light.pos, make_float3(c_vrParam.invViewMatrix[4],c_vrParam.invViewMatrix[5],c_vrParam.invViewMatrix[6]));
    lightVec.z = dot(c_light.pos, make_float3(c_vrParam.invViewMatrix[8],c_vrParam.invViewMatrix[9],c_vrParam.invViewMatrix[10]));
#endif

	// march along ray from back to front, accumulating color 
	{
		float tstep = c_vrParam.step / (float)c_vrParam.maxVolWidthInVoxel; //[0..1]
		eyeRay.d*=tstep;
	}
	float t;

	float tnear;
    float tfar;


    // find intersection with box
    {
        // object space: from 0,0,0 to 1,1,1
        float3 boxMin = make_float3(0.f);
        float3 boxMax = make_float3(c_vrParam.volBoundry[0], c_vrParam.volBoundry[1], c_vrParam.volBoundry[2]);
        int hit = intersectBox(eyeRay, boxMin, boxMax, OUT &tnear, OUT &tfar);
        if (!hit) {
            d_output[out_idx] = make_float4(0); //convert_float4(c_vrParam.bgColor);
            return;
        }
        // clamp to near plane
        tnear = fmaxf(0.f, tnear + EPS);  // pos must be positive

        #if 0
        // clipping plane
        #define invMat c_vrParam.invViewMatrix
        float3 clipplane = make_float3(-invMat[2],-invMat[6],-invMat[10]);
        tnear = max(tnear,  -(dot(eyeRay.o-boxMax*.5f, clipplane)
            + (c_vrParam.clippingVisible-.5f))
            / dot(clipplane, eyeRay.d)  );
        #undef invMat
        #endif

        tfar-=EPS;
    }

    uint count=0;
    float4 outColor;
    float4 col;	//temp color read from texture
    outColor = make_float4(0.f);

    for(t=ceilf(tnear) ;t < tfar && outColor.w<.99f && count < c_vrParam.maxSteps; t += c_vrParam.step)
    {
        float3 pos = eyeRay.o + eyeRay.d * t;						// pos: [0..1]
        ASSERT(pos.x>=0.0 && pos.y>=0.0 && pos.z>=0.0);

        // sampling distance for opacity correction
        //float sampDist =  c_vrParam.step *powf(0.6f, c_lodParam.levels -GET_LEVEL(node));
        const float sampDist = 1.f;

        col = d_shader(pos);

#if SHADING_TYPE==PHONG_SHADING
            col = (d_phong_shading(col, normalize(d_get_normal(pos)), normalize(-eyeRay.d), lightVec));
#endif
        /* mix color */
        mix_color(outColor, col, sampDist);

        count ++;

    }

    d_output[out_idx] =  outColor;

    return;
}

///////////////////////////////////////////////////////////////////////////
/// Render


extern "C"
void vr_render_kernel(dim3 gridSize, dim3 blockSize, float4 *d_output)
{
	d_render<<<gridSize, blockSize>>>( d_output );
}


///////////////////////////////////////////////////////////////////////////
///  Data

cudaArray *da_brickPool = 0;

extern "C"
void g_createBrickPool(int w, int h, int d)
{
    g_releaseBrickPool();

    cudaExtent ext;
    ext.width = w;    ext.height = h;    ext.depth = d;

    // create 3D array
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<LOD_DATA_TYPE>();
    CUDA_SAFE_CALL( cudaMalloc3DArray(&da_brickPool, &channelDesc, ext) );

    // set texture parameters
    texBrickPool.normalized = true; //false;                      // access with normalized texture coordinates?
#ifdef LINTERP_LOD_DATA
    texBrickPool.filterMode = cudaFilterModeLinear;      // linear interpolation
#else
    texBrickPool.filterMode = cudaFilterModePoint;
#endif
    texBrickPool.addressMode[0] = cudaAddressModeClamp;  // wrap texture coordinates
    texBrickPool.addressMode[1] = cudaAddressModeClamp;
    texBrickPool.addressMode[2] = cudaAddressModeClamp;

    // bind array to 3D texture
    CUDA_SAFE_CALL(cudaBindTextureToArray(texBrickPool, da_brickPool, channelDesc));
    cudaThreadSynchronize();
}


extern "C"
void g_uploadBrickPool(void *h_volume, int w, int h, int d, int x, int y, int z)
{
    // copy data to 3D array
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr   = make_cudaPitchedPtr(h_volume, w*sizeof(LOD_DATA_TYPE), w, h);
    copyParams.dstPos	= make_cudaPos(x,y,z);
    copyParams.dstArray = da_brickPool;
    copyParams.extent   = make_cudaExtent(w,h,d);
    copyParams.kind     = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&copyParams) );
}

extern "C"
void g_releaseBrickPool()
{
    if (NULL!=da_brickPool) {
        cudaThreadSynchronize();
        CUDA_SAFE_CALL(cudaFreeArray(da_brickPool));
        da_brickPool = NULL;

    }
}

