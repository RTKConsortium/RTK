/*=========================================================================
 *
 *  Copyright RTK Consortium
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

/*****************
*  rtk #includes *
*****************/
#include "rtkConfiguration.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaForwardProjectionImageFilter.hcu"

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

inline __host__ __device__ float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
inline __host__ __device__ float3 fminf(float3 a, float3 b)
{
  return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}
inline __host__ __device__ float3 fmaxf(float3 a, float3 b)
{
  return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}
inline __host__ __device__ float dot(float3 a, float3 b)
{ 
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline __host__ __device__ float3 operator/(float3 a, float3 b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
inline __host__ __device__ float3 operator/(float3 a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}
inline __host__ __device__ float3 operator*(float3 a, float3 b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline __host__ __device__ float3 operator*(float b, float3 a)
{
    return make_float3(b * a.x, b * a.y, b * a.z);
}
inline __host__ __device__ float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
inline __host__ __device__ float3 operator+(float3 a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}
inline __host__ __device__ void operator+=(float3 &a, float3 b)
{
    a.x += b.x; a.y += b.y; a.z += b.z;
}

// TEXTURES AND CONSTANTS //

//__constant__ float3 spacingSquare;  // inverse view matrix

texture<float, 3, cudaReadModeElementType> tex_vol;
texture<float, 1, cudaReadModeElementType> tex_matrix;
__constant__ int c_ProjectionType;
__constant__ float3 c_sourcePos;
__constant__ int2 c_projSize;
__constant__ float3 c_boxMin;
__constant__ float3 c_boxMax;
__constant__ float3 c_spacing;
__constant__ int c_matSize;
__constant__ float c_tStep;

// primary and compton
texture<float, 1, cudaReadModeElementType> tex_mu;

// primary
__constant__ int c_energySize;
texture<float, 1, cudaReadModeElementType> tex_energy;

// compton
__constant__ float3 c_Direction;
__constant__ float c_energySpacing;
__constant__ float c_InvWlPhoton;
__constant__ float c_E0m;
__constant__ float c_eRadiusOverCrossSectionTerm;
__constant__ float c_Energy;
__constant__ float3 c_DetectorOrientationTimesPixelSurface;


struct Ray {
        float3 o;  // origin
        float3 d;  // direction
};

inline int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_( S T A R T )_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

__device__
double getSolidAngle(const float3 sourceToPixelInVox)
{
  float3 sourceToPixelInMM =  sourceToPixelInVox * c_spacing;
  return abs( dot(sourceToPixelInMM, c_DetectorOrientationTimesPixelSurface)
            / pow( sqrt( dot(sourceToPixelInMM,sourceToPixelInMM) ), 3) );
}

// Intersection function of a ray with a box, followed "slabs" method
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
__device__
int intersectBox(Ray r, float *tnear, float *tfar)
{
    // Compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
    float3 T1;
    T1 = invR * (c_boxMin - r.o);
    float3 T2;
    T2 = invR * (c_boxMax - r.o);

    // Re-order intersections to find smallest and largest on each axis
    float3 tmin;
    tmin = fminf(T2, T1);
    float3 tmax;
    tmax = fmaxf(T2, T1);

    // Find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// Original program
__device__
void interpolateAndAccumulateOriginal(float *dev_proj,
                                      unsigned int numThread,
                                      float3 pos,
                                      float tnear,
                                      float tfar,
                                      float vStep,
                                      float halfVStep,
                                      float3 step)
{
  float  t;
  float  sample = 0.0f;
  float  sum    = 0.0f;
  for(t=tnear; t<=tfar; t+=vStep)
    {
    // Read from 3D texture from volume, and make a trilinear interpolation
    float xtex = pos.x - 0.5; int itex = floor(xtex); float dxtex = xtex - itex;
    float ytex = pos.y - 0.5; int jtex = floor(ytex); float dytex = ytex - jtex;
    float ztex = pos.z - 0.5; int ktex = floor(ztex); float dztex = ztex - ktex;
    sample = (1-dxtex) * (1-dytex) * (1-dztex) * tex3D(tex_vol, itex  , jtex  , ktex)
           + dxtex     * (1-dytex) * (1-dztex) * tex3D(tex_vol, itex+1, jtex  , ktex)
           + (1-dxtex) * dytex     * (1-dztex) * tex3D(tex_vol, itex  , jtex+1, ktex)
           + dxtex     * dytex     * (1-dztex) * tex3D(tex_vol, itex+1, jtex+1, ktex)
           + (1-dxtex) * (1-dytex) * dztex     * tex3D(tex_vol, itex  , jtex  , ktex+1)
           + dxtex     * (1-dytex) * dztex     * tex3D(tex_vol, itex+1, jtex  , ktex+1)
           + (1-dxtex) * dytex     * dztex     * tex3D(tex_vol, itex  , jtex+1, ktex+1)
           + dxtex     * dytex     * dztex     * tex3D(tex_vol, itex+1, jtex+1, ktex+1);

    sum += sample;
    pos += step;
    }

  dev_proj[numThread] = (sum+(tfar-t+halfVStep)*sample) * c_tStep;
}

// New structure
__device__
void interpolateWeights1(float *dev_weights,
                         unsigned int numThread,
                         float3 pos,
                         float tnear,
                         float tfar,
                         float vStep,
                         float halfVStep,
                         float3 step)
{
  float  t;
  for(t=tnear; t<=tfar; t+=vStep)
    {

    float lastStepCoef = 1.;
    if (t+vStep > tfar)
      {
      lastStepCoef += (tfar-t-halfVStep);
      }

    // Read from 3D texture from volume, and make a trilinear interpolation
    float xtex = pos.x - 0.5; int itex = floor(xtex); float dxtex = xtex - itex;
    float ytex = pos.y - 0.5; int jtex = floor(ytex); float dytex = ytex - jtex;
    float ztex = pos.z - 0.5; int ktex = floor(ztex); float dztex = ztex - ktex;

    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex  , ktex  )] += (1-dxtex) * (1-dytex) * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex  , ktex  )] += dxtex     * (1-dytex) * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex+1, ktex  )] += (1-dxtex) * dytex     * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex+1, ktex  )] += dxtex     * dytex     * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex  , ktex+1)] += (1-dxtex) * (1-dytex) * dztex     * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex  , ktex+1)] += dxtex     * (1-dytex) * dztex     * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex+1, ktex+1)] += (1-dxtex) * dytex     * dztex     * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex+1, ktex+1)] += dxtex     * dytex     * dztex     * lastStepCoef;

    pos += step;
    }
}

// Fix the program with /VStep
__device__
void interpolateWeights2(float *dev_weights,
                         unsigned int numThread,
                         float3 pos,
                         float tnear,
                         float tfar,
                         float vStep,
                         float halfVStep,
                         float3 step)
{
  float  t;
  for(t=tnear; t<=tfar; t+=vStep)
    {

    float lastStepCoef = 1.;
    if (t+vStep > tfar)
      {
      lastStepCoef += (tfar-t-halfVStep)/vStep;
      }

    // Read from 3D texture from volume, and make a trilinear interpolation
    float xtex = pos.x - 0.5; int itex = floor(xtex); float dxtex = xtex - itex;
    float ytex = pos.y - 0.5; int jtex = floor(ytex); float dytex = ytex - jtex;
    float ztex = pos.z - 0.5; int ktex = floor(ztex); float dztex = ztex - ktex;

    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex  , ktex  )] += (1-dxtex) * (1-dytex) * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex  , ktex  )] += dxtex     * (1-dytex) * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex+1, ktex  )] += (1-dxtex) * dytex     * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex+1, ktex  )] += dxtex     * dytex     * (1-dztex) * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex  , ktex+1)] += (1-dxtex) * (1-dytex) * dztex     * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex  , ktex+1)] += dxtex     * (1-dytex) * dztex     * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex  , jtex+1, ktex+1)] += (1-dxtex) * dytex     * dztex     * lastStepCoef;
    dev_weights[numThread * c_matSize + (int)tex3D(tex_vol, itex+1, jtex+1, ktex+1)] += dxtex     * dytex     * dztex     * lastStepCoef;

    pos += step;
    }
}

// New structure
__device__
void accumulate1(float *dev_proj,
                 float *dev_weights,
                 unsigned int numThread)
{
  // Loops over energy, multiply weights by mu, accumulate using Beer Lambert
  for(int id = 0; id < c_matSize - 1; id++)
    {
    dev_proj[numThread] += dev_weights[numThread * c_matSize + id] * (float)id;
    }
  dev_proj[numThread] *= c_tStep;
}

// primary projector
__device__
void accumulate2(float *dev_proj,
                 float *dev_weights,
                 const unsigned int numThread,
                 const float3 sourceToPixel,
                 const float3 nearestPoint,
                 const float3 farthestPoint)
{
  // Multiply interpolation weights by step norm in MM to convert voxel
  // intersection length to MM.
  for(int id = 0; id < c_matSize - 1; id++)
    {
    dev_weights[numThread * c_matSize + id] *= c_tStep;
    }

  // The last material is the world material. One must fill the weight with
  // the length from source to nearest point and farthest point to pixel
  // point.
  float3 worldVector = sourceToPixel + nearestPoint - farthestPoint;
  worldVector = worldVector * c_spacing;
  dev_weights[(numThread+1) * c_matSize - 1] = sqrtf(dot(worldVector, worldVector));

  // Loops over energy, multiply weights by mu, accumulate using Beer Lambert
  for(unsigned int e = 0; e < c_energySize; e++)
    {
    float rayIntegral = 0.;
    for(unsigned int id = 0; id < c_matSize; id++)
      {
      rayIntegral += dev_weights[numThread * c_matSize + id] * tex1Dfetch(tex_mu, e * c_matSize + id);
      }
    dev_proj[numThread] += exp(-rayIntegral) * tex1Dfetch(tex_energy, e);
    //m_IntegralOverDetector[threadId] += valueToAccumulate;
    }
}

// compton projector FIXME
__device__
void accumulate3(float *dev_proj,
                 float *dev_weights,
                 const unsigned int numThread,
                 const float3 sourceToPixel,
                 const float3 nearestPoint,
                 const float3 farthestPoint)
{
  // Compute ray length in world material
  // This is used to compute the length in world as well as the direction
  // of the ray in mm.
  float3 worldVector = sourceToPixel + nearestPoint - farthestPoint;
  worldVector = worldVector * c_spacing;
  const float worldVectorNorm = sqrtf(dot(worldVector, worldVector));

  // This is taken from G4LivermoreComptonModel.cc
  float cosT = dot(worldVector, c_Direction) / worldVectorNorm;
  float x = sqrt(1. - cosT) * c_InvWlPhoton; // 1-cosT=2*sin(T/2)^2
  float scatteringFunction = 1.; // TODO m_ScatterFunctionData->FindValue(x, m_Z - 1);

  // This is taken from GateDiffCrossSectionActor.cc and simplified
  float Eratio = 1. / (1. + c_E0m * (1. - cosT));
  //float DCSKleinNishina = m_eRadiusOverCrossSectionTerm *
  //                         Eratio * Eratio *                      // DCSKleinNishinaTerm1
  //                         (Eratio + 1./Eratio - 1. + cosT*cosT); // DCSKleinNishinaTerm2
  float DCSKleinNishina = c_eRadiusOverCrossSectionTerm * Eratio * (1. + Eratio * (Eratio - 1. + cosT * cosT));
  float DCScompton = DCSKleinNishina * scatteringFunction;

  // Multiply interpolation weights by step norm in MM to convert voxel
  // intersection length to MM.
  for(int id = 0; id < c_matSize - 1; id++)
    {
    dev_weights[numThread * c_matSize + id] *= c_tStep;
    }

  // The last material is the world material. One must fill the weight with
  // the length from farthest point to pixel point.
  dev_weights[(numThread+1) * c_matSize - 1] = worldVectorNorm;

 // DEFINITION for Activation/Deactivation log-log interpolation of mu value
//#define INTERP
  const float energy = Eratio * c_Energy;

#ifdef INTERP
  // Pointer to adequate mus
  unsigned int Eceil = ceil(energy / c_energySpacing);
  unsigned int Efloor = floor(energy / c_energySpacing);

  float rayIntegral = 0.;
  float logEnergy   = log(energy / c_energySpacing);
  float logCeil     = log((float)Eceil);
  float logFloor    = log((float)Efloor);

  // log-log interpolation for mu calculation
  for(unsigned int id = 0; id < c_matSize; id++)
    {
    float interp = exp( log( tex1Dfetch(tex_mu, Eceil * c_matSize + id)
                           / tex1Dfetch(tex_mu, Efloor * c_matSize + id) )
                        / (logCeil - logFloor) * (logEnergy - logCeil)
                        + log(tex1Dfetch(tex_mu, Eceil * c_matSize + id))
                      );
    // Ray integral
    rayIntegral += dev_weights[numThread * c_matSize + id] * interp;
    }
#else
  unsigned int e = round(energy / c_energySpacing);

  // Ray integral
  float rayIntegral = 0.;
  for(unsigned int id = 0; id < c_matSize; id++)
    {
    rayIntegral += dev_weights[numThread * c_matSize + id] * tex1Dfetch(tex_mu, e * c_matSize + id);
    }
#endif

  // Final computation
  dev_proj[numThread] += exp(-rayIntegral) * DCScompton * getSolidAngle(sourceToPixel) /* TODO * (*m_ResponseDetector)(energy)*/;
}

// KERNEL kernel_forwardProject
__global__
void kernel_forwardProject(float *dev_proj, float *dev_weights)
{
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
  unsigned int numThread = j*c_projSize.x + i;

  if (i >= c_projSize.x || j >= c_projSize.y)
    return;

  // Setting ray origin
  Ray ray;
  ray.o = c_sourcePos;

  float3 pixelPos;

  pixelPos.x = tex1Dfetch(tex_matrix, 3)  + tex1Dfetch(tex_matrix, 0)*i +
               tex1Dfetch(tex_matrix, 1)*j;
  pixelPos.y = tex1Dfetch(tex_matrix, 7)  + tex1Dfetch(tex_matrix, 4)*i +
               tex1Dfetch(tex_matrix, 5)*j;
  pixelPos.z = tex1Dfetch(tex_matrix, 11) + tex1Dfetch(tex_matrix, 8)*i +
               tex1Dfetch(tex_matrix, 9)*j;

  ray.d = pixelPos - ray.o;
  ray.d = ray.d / sqrtf(dot(ray.d,ray.d));

  // Detect intersection with box
  float tnear, tfar;
  if (intersectBox(ray, &tnear, &tfar) && (tfar > 0.))
    {
    if (tnear < 0.f)
      tnear = 0.f; // clamp to near plane

    // Step length in mm
    float3 dirInMM = c_spacing * ray.d;
    float vStep = c_tStep / sqrtf(dot(dirInMM, dirInMM));
    float3 step = vStep * ray.d;

    // First position in the box
    float3 pos;
    float halfVStep = 0.5f*vStep;
    tnear = tnear + halfVStep;
    pos = ray.o + tnear*ray.d;

    switch (c_ProjectionType)
      {
      case ORIGINAL:
        {
        // Original program
        // interpolateAndAccumulateOriginal(dev_proj, numThread, pos, tnear, tfar, vStep, halfVStep, step);

        // New structure
        //interpolateWeights1(dev_weights, numThread, pos, tnear, tfar, vStep, halfVStep, step);
        //accumulate1(dev_proj, dev_weights, numThread);

        // Fix Bug
        interpolateWeights2(dev_weights, numThread, pos, tnear, tfar, vStep, halfVStep, step);
        accumulate1(dev_proj, dev_weights, numThread);
        break;
        }
      case PRIMARY:
        {
        float3 sourceToPixel = pixelPos - ray.o;
        float3 nearestPoint = ray.o + tnear * ray.d;
        float3 farthestPoint = ray.o + tfar * ray.d;

        interpolateWeights2(dev_weights, numThread, pos, tnear, tfar, vStep, halfVStep, step);
        accumulate2(dev_proj, dev_weights, numThread, sourceToPixel, nearestPoint, farthestPoint);
        break;
        }
      case COMPTON:
        {
        float3 sourceToPixel = pixelPos - ray.o;
        float3 nearestPoint = ray.o + tnear * ray.d;
        float3 farthestPoint = ray.o + tfar * ray.d;

        interpolateWeights2(dev_weights, numThread, pos, tnear, tfar, vStep, halfVStep, step);
        accumulate3(dev_proj, dev_weights, numThread, sourceToPixel, nearestPoint, farthestPoint);
        break;
        }
      }
    }
  else
    {
    switch (c_ProjectionType)
      {
      case ORIGINAL:
        {
        accumulate1(dev_proj, dev_weights, numThread);
        break;
        }
      case PRIMARY:
        {
        float3 sourceToPixel = pixelPos - ray.o;
        float3 nearestPoint = ray.o;
        float3 farthestPoint = ray.o;

        accumulate2(dev_proj, dev_weights, numThread, sourceToPixel, nearestPoint, farthestPoint);
        break;
        }
      case COMPTON:
        {
        float3 sourceToPixel = pixelPos - ray.o;
        float3 nearestPoint = ray.o;
        float3 farthestPoint = ray.o;

        accumulate3(dev_proj, dev_weights, numThread, sourceToPixel, nearestPoint, farthestPoint);
        break;
        }
      }
    }
}

//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
// K E R N E L S -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-( E N D )-_-_
//_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_

///////////////////////////////////////////////////////////////////////////
// FUNCTION: CUDA_forward_project() //////////////////////////////////
void
CUDA_forward_project( int projections_size[2],
                      int vol_size[3],
                      float matrix[12],
                      float *dev_proj,
                      float *dev_vol,
                      float t_step,
                      double source_position[3],
                      float box_min[3],
                      float box_max[3],
                      float spacing[3],
                      CudaAccumulationParameters* params)
{
  float *dev_mu = 0;
  float *dev_energy = 0;

  switch (params->projectionType)
    {
    case ORIGINAL:
      {
      break;
      }
    case PRIMARY:
      {
      // mu
      cudaMalloc( (void**)&dev_mu, params->matSize*params->energySize*sizeof(float) );
      cudaMemcpy (dev_mu, params->mu, params->matSize*params->energySize*sizeof(float), cudaMemcpyHostToDevice);
      CUDA_CHECK_ERROR;
      cudaBindTexture (0, tex_mu, dev_mu, params->matSize*params->energySize*sizeof(float) );
      CUDA_CHECK_ERROR;
      // energy
      cudaMemcpyToSymbol(c_energySize, &params->energySize, sizeof(int));
      cudaMalloc( (void**)&dev_energy, params->energySize*sizeof(float) );
      cudaMemcpy (dev_energy, params->energyWeights, params->energySize*sizeof(float), cudaMemcpyHostToDevice);
      CUDA_CHECK_ERROR;
      cudaBindTexture (0, tex_energy, dev_energy, params->energySize*sizeof(float) );
      CUDA_CHECK_ERROR;
      break;
      }
    case COMPTON:
      {
      // mu
      cudaMalloc( (void**)&dev_mu, params->matSize*params->energySize*sizeof(float) );
      cudaMemcpy (dev_mu, params->mu, params->matSize*params->energySize*sizeof(float), cudaMemcpyHostToDevice);
      CUDA_CHECK_ERROR;
      cudaBindTexture (0, tex_mu, dev_mu, params->matSize*params->energySize*sizeof(float) );
      CUDA_CHECK_ERROR;
      // other params
      float3 dev_direction = make_float3(params->direction[0], params->direction[1], params->direction[2]);
      cudaMemcpyToSymbol(c_Direction, &dev_direction, sizeof(float3));
      cudaMemcpyToSymbol(c_InvWlPhoton, &params->invWlPhoton, sizeof(float));
      cudaMemcpyToSymbol(c_E0m, &params->e0m, sizeof(float));
      cudaMemcpyToSymbol(c_eRadiusOverCrossSectionTerm, &params->eRadiusOverCrossSectionTerm, sizeof(float));
      cudaMemcpyToSymbol(c_Energy, &params->energy, sizeof(float));
      cudaMemcpyToSymbol(c_energySpacing, &params->energy_spacing, sizeof(float));
      float3 dev_DetectorOrientationTimesPixelSurface = make_float3(
        params->detectorOrientationTimesPixelSurface[0],
        params->detectorOrientationTimesPixelSurface[1],
        params->detectorOrientationTimesPixelSurface[2]);
      cudaMemcpyToSymbol(c_DetectorOrientationTimesPixelSurface, &dev_DetectorOrientationTimesPixelSurface, sizeof(float3));
      break;
      }
    }

  // Set texture parameters
  tex_vol.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
  tex_vol.addressMode[1] = cudaAddressModeClamp;
  tex_vol.normalized = false;                     // access with normalized texture coordinates
  tex_vol.filterMode = cudaFilterModePoint;       // no interpolation

  // Copy volume data to array, bind the array to the texture
  cudaExtent volExtent = make_cudaExtent(vol_size[0], vol_size[1], vol_size[2]);
  cudaArray *array_vol;
  static cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaMalloc3DArray((cudaArray**)&array_vol, &channelDesc, volExtent);

  // Copy data to 3D array
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr   = make_cudaPitchedPtr(dev_vol, vol_size[0]*sizeof(float), vol_size[0], vol_size[1]);
  copyParams.dstArray = (cudaArray*)array_vol;
  copyParams.extent   = volExtent;
  copyParams.kind     = cudaMemcpyDeviceToDevice;
  cudaMemcpy3D(&copyParams);

  // Bind 3D array to 3D texture
  cudaBindTextureToArray(tex_vol, (cudaArray*)array_vol, channelDesc);

  // Copy matrix and bind data to the texture
  float *dev_matrix;
  cudaMalloc( (void**)&dev_matrix, 12*sizeof(float) );
  cudaMemcpy (dev_matrix, matrix, 12*sizeof(float), cudaMemcpyHostToDevice);
  CUDA_CHECK_ERROR;
  cudaBindTexture (0, tex_matrix, dev_matrix, 12*sizeof(float) );
  CUDA_CHECK_ERROR;

  // Reset projection
  cudaMemset((void *)dev_proj, 0, projections_size[0]*projections_size[1]*sizeof(float) );
  CUDA_CHECK_ERROR;

  // interpolationWeights
  float *dev_weights;
  cudaMalloc( (void**)&dev_weights, params->matSize*projections_size[0]*projections_size[1]*sizeof(float) );
  cudaMemset((void *)dev_weights, 0, params->matSize*projections_size[0]*projections_size[1]*sizeof(float) );
  CUDA_CHECK_ERROR;

  // constant memory
  float3 dev_sourcePos = make_float3(source_position[0], source_position[1], source_position[2]);
  float3 dev_boxMin = make_float3(box_min[0], box_min[1], box_min[2]);
  int2 dev_projSize = make_int2(projections_size[0], projections_size[1]);
  float3 dev_boxMax = make_float3(box_max[0], box_max[1], box_max[2]);
  float3 dev_spacing = make_float3(spacing[0], spacing[1], spacing[2]);
  cudaMemcpyToSymbol(c_sourcePos, &dev_sourcePos, sizeof(float3));
  cudaMemcpyToSymbol(c_projSize, &dev_projSize, sizeof(int2));
  cudaMemcpyToSymbol(c_boxMin, &dev_boxMin, sizeof(float3));
  cudaMemcpyToSymbol(c_boxMax, &dev_boxMax, sizeof(float3));
  cudaMemcpyToSymbol(c_spacing, &dev_spacing, sizeof(float3));
  cudaMemcpyToSymbol(c_ProjectionType, &params->projectionType, sizeof(int));
  cudaMemcpyToSymbol(c_matSize, &params->matSize, sizeof(int));

  cudaMemcpyToSymbol(c_tStep, &t_step, sizeof(float));

  static dim3 dimBlock  = dim3(16, 16, 1);
  static dim3 dimGrid = dim3(iDivUp(projections_size[0], dimBlock.x), iDivUp(projections_size[1], dimBlock.x));

  // Calling kernel
  kernel_forwardProject <<< dimGrid, dimBlock >>> (dev_proj, dev_weights);

  cudaDeviceSynchronize();
  CUDA_CHECK_ERROR;

  // Unbind the volume and matrix textures
  cudaUnbindTexture (tex_vol);
  CUDA_CHECK_ERROR;
  cudaUnbindTexture (tex_matrix);
  CUDA_CHECK_ERROR;

  // Cleanup
  cudaFreeArray ((cudaArray*)array_vol);
  CUDA_CHECK_ERROR;
  cudaFree (dev_matrix);
  CUDA_CHECK_ERROR;

  switch (params->projectionType)
    {
    case ORIGINAL:
      {
      break;
      }
    case PRIMARY:
      {
      cudaUnbindTexture (tex_mu);
      CUDA_CHECK_ERROR;
      cudaFree (dev_mu);
      CUDA_CHECK_ERROR;
      cudaUnbindTexture (tex_energy);
      CUDA_CHECK_ERROR;
      cudaFree (dev_energy);
      CUDA_CHECK_ERROR;
      break;
      }
    case COMPTON:
      {
      cudaUnbindTexture (tex_mu);
      CUDA_CHECK_ERROR;
      cudaFree (dev_mu);
      CUDA_CHECK_ERROR;
      break;
      }
    }
}
