/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
/*****************
*  rtk #includes *
*****************/
#include "rtkConfiguration.h"
#include "itkCudaFDKBackProjectionImageFilter.hcu"
#include "itkCudaUtilities.hcu"

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

// P R O T O T Y P E S ////////////////////////////////////////////////////
__global__ void kernel_fdk (float *dev_vol, int2 img_dim, int3 vol_dim, unsigned int Blocks_Y, float invBlocks_Y);
///////////////////////////////////////////////////////////////////////////

// T E X T U R E S ////////////////////////////////////////////////////////
texture<float, 2, cudaReadModeElementType> tex_img;
texture<float, 1, cudaReadModeElementType> tex_matrix;
///////////////////////////////////////////////////////////////////////////

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

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
  long int vol_idx = i + (j + k*vol_dim.y)*(vol_dim.x);

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

  // Get texture point, clip left to GPU
  voxel_data = tex2D(tex_img, ip.x, ip.y);

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
  cudaArray *&dev_img,         // Holds image pixels on device
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

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMallocArray( &dev_img, &channelDesc, kargs->img_dim.x, kargs->img_dim.y );

  // set texture parameters
  tex_img.addressMode[0] = cudaAddressModeClamp;
  tex_img.addressMode[1] = cudaAddressModeClamp;
  tex_img.filterMode = cudaFilterModeLinear;
  tex_img.normalized = false;    // don't access with normalized texture coordinates
}



///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_reconstruct_conebeam() //////////////////////////////////
extern "C"
void
CUDA_reconstruct_conebeam (
    float *proj,
    kernel_args_fdk *kargs,
    float *dev_vol,
    cudaArray *dev_img,
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

  // copy image data, bind the array to the texture
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMemcpyToArray( dev_img, 0, 0, proj, kargs->img_dim.x*kargs->img_dim.y*sizeof(float), cudaMemcpyHostToDevice);
  cudaBindTextureToArray( tex_img, dev_img, channelDesc);

  // copy matrix, bind data to the texture
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
  cudaArray *dev_img,
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
