/*=========================================================================
 *
 *  Copyright Insight Software Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

//
// pixel by pixel addition of 2D images
//
template<class PixelType>
__device__ void ImageAdd(int2 imSize, PixelType* a, PixelType* b, PixelType* c, unsigned int nElem)
{ 
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imSize.x || j >= imSize.y)
    return;

  unsigned int gidx = i + j * imSize.x;

  if (gidx < nElem)
    {
    c[gidx] = a[gidx] + b[gidx];
    }
}

//
// pixel by pixel subtraction of 2D images
//
template<class PixelType>
__device__ void ImageSub(int2 imSize, PixelType* a, PixelType* b, PixelType* c, unsigned int nElem)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imSize.x || j >= imSize.y)
    return;

  unsigned int gidx = i + j * imSize.x; 

  // bound check
  if (gidx < nElem)
    {
    c[gidx] = a[gidx] - b[gidx];
    }
}

//
// pixel by pixel multiplication of 2D images
//
template<class PixelType>
__device__ void ImageMult(int2 imSize, PixelType* a, PixelType* b, PixelType* c, unsigned int nElem)
{
   unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imSize.x || j >= imSize.y)
    return;

  unsigned int gidx = i + j * imSize.x;

  // bound check
  if (gidx < nElem)
    {
    c[gidx] = a[gidx] * b[gidx];
    }
}

//
// pixel by pixel division of 2D images
//
template<class PixelType>
__device__ void ImageDiv(int2 imSize, PixelType* a, PixelType* b, PixelType* c, unsigned int nElem)
{
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= imSize.x || j >= imSize.y)
    return;

  unsigned int gidx = i + j * imSize.x;

  // bound check
  if (gidx < nElem)
    {
    c[gidx] = a[gidx] / b[gidx];
    }
}

#define MAKE_TEMPLATE_IMPL(_func_, _exportname_, _type_) \
  extern "C" __global__ void _exportname_(int2 imSize, _type_* a, _type_ *b, _type_ *c, unsigned int nElem)\
  {\
    _func_(imSize, a, b, c, nElem);\
  }

#define MAKE_IMPL(_func_)\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_c, char);\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_uc, unsigned char);\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_s, short);\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_i, int);\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_ui, unsigned int);\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_f, float);\
  MAKE_TEMPLATE_IMPL(_func_, _func_##_d, double);

// Export the template implementations
MAKE_IMPL(ImageAdd);
MAKE_IMPL(ImageSub);
MAKE_IMPL(ImageMult);
MAKE_IMPL(ImageDiv);
