/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "rtkConfiguration.h"

/*****************
*  C   #includes *
*****************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*****************
* CUDA #includes *
*****************/
#include <cuda.h>

/*****************
* FDK  #includes *
*****************/
#include "cuda_util.h"
#include "itkCudaFDKBackProjectionImageFilter.hcu"

// P R O T O T Y P E S ////////////////////////////////////////////////////
__global__ void kernel_fdk (float *dev_vol, int2 img_dim, int3 vol_dim, unsigned int Blocks_Y, float invBlocks_Y);
///////////////////////////////////////////////////////////////////////////

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 1, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
///////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__global__
void kernel_fdk_gmem (
    float *dev_vol,
    float *pimg,
    float *pmat,
    int2 img_dim,
    int3 vol_dim,
    unsigned int Blocks_Y,
    float invBlocks_Y)
{
  // CUDA 2.0 does not allow for a 3D grid, which severely
  // limits the manipulation of large 3D arrays of data.  The
  // following code is a hack to bypass this implementation
  // limitation.
  unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z) {
      return; 
  }

  // Index row major into the volume
  long int vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );

  float3 ip;
  int2 ip_r;
  float voxel_data;

  // matrix multiply
  ip.x = pmat[0]*i + pmat[1]*j + pmat[2]*k + pmat[3];
  ip.y = pmat[4]*i + pmat[5]*j + pmat[6]*k + pmat[7];
  ip.z = pmat[8]*i + pmat[9]*j + pmat[10]*k + pmat[11];

  // Change coordinate systems
  ip.z = 1 / ip.z;
  ip.x = ip.x * ip.z;
  ip.y = ip.y * ip.z;

  // Get pixel from 2D image
  ip_r.x = __float2int_rd(ip.x);
  ip_r.y = __float2int_rd(ip.y);

  // Clip against image dimensions
  if (ip_r.x < 0 || ip_r.x >= img_dim.x || ip_r.y < 0 || ip_r.y >= img_dim.y) {
      return;
  }
  voxel_data = pimg[ip_r.x*img_dim.x + ip_r.y];

  // Place it into the volume
  dev_vol[vol_idx] += ip.z * ip.z * voxel_data;
}


__global__
void kernel_fdk (float *dev_vol, int2 img_dim, int3 vol_dim, unsigned int Blocks_Y, float invBlocks_Y)
{
  // CUDA 2.0 does not allow for a 3D grid, which severely
  // limits the manipulation of large 3D arrays of data.  The
  // following code is a hack to bypass this implementation
  // limitation.
  unsigned int blockIdx_z = __float2uint_rd(blockIdx.y * invBlocks_Y);
  unsigned int blockIdx_y = blockIdx.y - __umul24(blockIdx_z, Blocks_Y);
  unsigned int i = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
  unsigned int j = __umul24(blockIdx_y, blockDim.y) + threadIdx.y;
  unsigned int k = __umul24(blockIdx_z, blockDim.z) + threadIdx.z;

  if (i >= vol_dim.x || j >= vol_dim.y || k >= vol_dim.z) {
      return; 
  }

  // Index row major into the volume
  long int vol_idx = i + ( j*(vol_dim.x) ) + ( k*(vol_dim.x)*(vol_dim.y) );

  float3 ip;
  float voxel_data;

  // matrix multiply
  ip.x = tex1Dfetch(tex_matrix, 0)*i + tex1Dfetch(tex_matrix, 1)*j + tex1Dfetch(tex_matrix, 2)*k + tex1Dfetch(tex_matrix, 3);
  ip.y = tex1Dfetch(tex_matrix, 4)*i + tex1Dfetch(tex_matrix, 5)*j + tex1Dfetch(tex_matrix, 6)*k + tex1Dfetch(tex_matrix, 7);
  ip.z = tex1Dfetch(tex_matrix, 8)*i + tex1Dfetch(tex_matrix, 9)*j + tex1Dfetch(tex_matrix, 10)*k + tex1Dfetch(tex_matrix, 11);

  // Change coordinate systems
  ip.z = 1 / ip.z;
  ip.x = ip.x * ip.z;
  ip.y = ip.y * ip.z;

  // Get pixel from 2D image
  ip.x = __float2int_rd(ip.x);
  ip.y = __float2int_rd(ip.y);

  // Clip against image dimensions
  if (ip.x < 0 || ip.x >= img_dim.x || ip.y < 0 || ip.y >= img_dim.y) {
      return;
  }
  voxel_data = tex1Dfetch(tex_img, ip.y*img_dim.x + ip.x);

  // Place it into the volume
  dev_vol[vol_idx] += ip.z * ip.z * voxel_data;
}
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam_init() /////////////////////////////
extern "C"
void
CUDA_reconstruct_conebeam_init (
  kernel_args_fdk *kargs,
  kernel_args_fdk *&dev_kargs, // Holds kernel parameters on device
  float *&dev_vol,             // Holds voxels on device
  float *&dev_img,             // Holds image pixels on device
  float *&dev_matrix           // Holds matrix on device
)
{
  // Size of volume Malloc
  int vol_size_malloc = (kargs->vol_dim.x*kargs->vol_dim.y*kargs->vol_dim.z)*sizeof(float);

  // CUDA device pointers
  cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
  cudaMalloc( (void**)&dev_kargs, sizeof(kernel_args_fdk) );
  cudaMalloc( (void**)&dev_vol, vol_size_malloc);
  cudaMemset( (void *) dev_vol, 0, vol_size_malloc);  
  CUDA_check_error("Unable to allocate data volume");

  // This is just to retrieve the 2D image dimensions
  cudaMalloc ((void**)&dev_img, kargs->img_dim.x*kargs->img_dim.y*sizeof(float));
}



///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam() //////////////////////////////////
extern "C"
void
CUDA_reconstruct_conebeam (
    float *proj,
    kernel_args_fdk *kargs,
    float *dev_vol,
    float *dev_img,
    float *dev_matrix
)
{
  // Thread Block Dimensions
  static int tBlock_x = 16;
  static int tBlock_y = 4;
  static int tBlock_z = 4;

  // Each element in the volume (each voxel) gets 1 thread
  static int blocksInX = (kargs->vol_dim.x+tBlock_x-1)/tBlock_x;
  static int blocksInY = (kargs->vol_dim.y+tBlock_y-1)/tBlock_y;
  static int blocksInZ = (kargs->vol_dim.z+tBlock_z-1)/tBlock_z;
  static dim3 dimGrid  = dim3(blocksInX, blocksInY*blocksInZ);
  static dim3 dimBlock = dim3(tBlock_x, tBlock_y, tBlock_z);

  // Copy image pixel data & projection matrix to device Global Memory
  // and then bind them to the texture hardware.
  cudaMemcpy (dev_img, proj, kargs->img_dim.x*kargs->img_dim.y*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_img, dev_img, kargs->img_dim.x*kargs->img_dim.y*sizeof(float));
  cudaMemcpy (dev_matrix, kargs->matrix, sizeof(kargs->matrix), cudaMemcpyHostToDevice);
  cudaBindTexture (0, tex_matrix, dev_matrix, sizeof(kargs->matrix));

  // Note: cbi->img AND cbi->matrix are passed via texture memory
  //-------------------------------------
  kernel_fdk <<< dimGrid, dimBlock >>> (
      dev_vol,
      kargs->img_dim,
      kargs->vol_dim,
      blocksInY,
      1.0f/(float)blocksInY
  );

  CUDA_check_error("Kernel Panic!");

  // Unbind the image and projection matrix textures
  cudaUnbindTexture (tex_img);
  cudaUnbindTexture (tex_matrix);
}

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam_cleanup() //////////////////////////
extern "C"
void 
CUDA_reconstruct_conebeam_cleanup (
  kernel_args_fdk *kargs,
  kernel_args_fdk *dev_kargs,
  float *vol,
  float *dev_vol,
  float *dev_img,
  float *dev_matrix
)
{
  // Size of volume Malloc
  int vol_size_malloc = (kargs->vol_dim.x*kargs->vol_dim.y*kargs->vol_dim.z)*sizeof(float);

  // Copy reconstructed volume from device to host
  cudaMemcpy (vol, dev_vol, vol_size_malloc, cudaMemcpyDeviceToHost);
  CUDA_check_error ("Error: Unable to retrieve data volume.");

  // Cleanup
  cudaFree (dev_img);
  cudaFree (dev_kargs);
  cudaFree (dev_matrix);
  cudaFree (dev_vol); 
}
