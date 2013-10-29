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

#ifndef __rtkSiddonForwardProjectionImageFilter_txx
#define __rtkSiddonForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "rtkRayBoxIntersectionFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>
#include "vnl/vnl_math.h"

namespace rtk
{

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication,
          class TProjectedValueAccumulation>
void
SiddonForwardProjectionImageFilter<TInputImage,
                  TOutputImage,
                  TInterpolationWeightMultiplication,
                  TProjectedValueAccumulation>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  const typename Superclass::GeometryType::Pointer geometry = this->GetGeometry();

  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Go over each projection
  for(int iProj=outputRegionForThread.GetIndex(2);
          iProj<outputRegionForThread.GetIndex(2)+(int)outputRegionForThread.GetSize(2);
          iProj++)
  {

    // Account for system rotations
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
    volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInput(1) );

    // Set source position in volume indices
    typename Superclass::GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = volPPToIndex * geometry->GetSourcePosition(iProj);

    // Compute matrix to transform projection index to volume index
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType matrix;
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType FSM;

    FSM = geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix();

    matrix = volPPToIndex.GetVnlMatrix() *
             geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
             GetIndexToPhysicalPointMatrix( this->GetOutput() ).GetVnlMatrix();

//  ***************************************************************************
//  *********************** P L A S T I M A T C H *****************************
//  ***************************************************************************

    // current output image and thread region:
    OutputPixelType *drr = this->GetOutput()->GetBufferPointer();
    typename TOutputImage::IndexType start = outputRegionForThread.GetIndex();
    typename TOutputImage::SizeType  size  = outputRegionForThread.GetSize();
    unsigned int xmax = start[0] + size[0];
    unsigned int ymax = start[1] + size[1];

    // detector size (pixels):
    const typename TOutputImage::SizeType detectorSize = this->GetOutput()->GetLargestPossibleRegion().GetSize();
    // detector spacing (mm/pixel):
    const typename TOutputImage::SpacingType Det_Spacing = this->GetOutput()->GetSpacing();

    // go through all DRR pixels of current thread output region:
    int id_line, id_pixel; // linearized DRR pixel indices

    PointType R;
    PointType Det_O = this->GetOutput()->GetOrigin();
    // Convert degrees to radians
    const double rad = vcl_atan(1.0) / 45.0;
    // Gantry angle
    const double alfa = geometry->GetGantryAngles()[iProj]*rad;
    // Isocenter to Detector Distance
    const double idd  = double(geometry->GetSourceToDetectorDistances()[iProj] - geometry->GetSourceToIsocenterDistances()[iProj]);

    // Direction from isocenter to projector origin
    R[0] = -( ( detectorSize[0]*Det_Spacing[0] + Det_O[0] ) * vcl_cos(-alfa) + idd * vcl_sin(alfa) );
    R[1] = -( detectorSize[1]*Det_Spacing[1] + Det_O[1] );
    R[2] =  ( detectorSize[0]*Det_Spacing[0] + Det_O[0] ) * vcl_sin(alfa) - idd * vcl_cos(alfa);

    // Detector row orientation
    VectorType v1;
    v1[0] = FSM[0][0];
    v1[1] = FSM[1][0];
    v1[2] = FSM[2][0];
    // Detector column orientation
    VectorType v2;
    v2[0] = FSM[0][1];
    v2[1] = FSM[1][1];
    v2[2] = FSM[2][1];
    // Source Position
    PointType Source;
    Source[0] = geometry->GetSourcePosition(iProj)[0];
    Source[1] = geometry->GetSourcePosition(iProj)[1];
    Source[2] = geometry->GetSourcePosition(iProj)[2];

    // output pixel position (WCS) due to y index
    PointType Xh;
    // overall output pixel position (WCS)
    PointType X;
    // direction from S to X
    VectorType ray_direction;
    // effective ray length
    double ray_length;
    // 0-intensity
    OutputPixelType izero = static_cast<OutputPixelType> (0);

    typename TInputImage::SizeType Vol_Size = this->GetInput(1)->GetLargestPossibleRegion().GetSize();
    int vlinepitch = static_cast<int> (Vol_Size[0]);
    int vslicepitch = vlinepitch * static_cast<int> (Vol_Size[1]);

    typename TInputImage::SpacingType Vol_Spacing = this->GetInput(1)->GetSpacing();
    PointType Vol_O = this->GetInput(1)->GetOrigin();

    // NOTE: For the computations below, we need the position of the corner of
    // the first voxel, not its center! However, GetOrigin() normally returns
    // the center position of the first voxel, therefore, we have to correct it.
    typename TInputImage::DirectionType vdir = this->GetInput(1)->GetDirection();
    VectorType vv;
    // volume row direction
    vv[0] = vdir[0][0];
    vv[1] = vdir[1][0];
    vv[2] = vdir[2][0];
    Vol_O = Vol_O - vv * Vol_Spacing[0] * 0.5;
    // volume column direction
    vv[0] = vdir[0][1];
    vv[1] = vdir[1][1];
    vv[2] = vdir[2][1];
    Vol_O = Vol_O - vv * Vol_Spacing[1] * 0.5;
    // volume slicing direction
    vv[0] = vdir[0][2];
    vv[1] = vdir[1][2];
    vv[2] = vdir[2][2];
    Vol_O = Vol_O - vv * Vol_Spacing[2] * 0.5;

    const InputPixelType *volume = this->GetInput(1)->GetBufferPointer();
    // Parametric plane crossings
    double alpha_x_0, alpha_y_0, alpha_z_0;
    double alpha_x_Nx, alpha_y_Ny, alpha_z_Nz;
    double alpha_x_min, alpha_y_min, alpha_z_min;
    double alpha_x_max, alpha_y_max, alpha_z_max;
    // Parametric ray-volume entry / exit coordinates
    double alpha_min, alpha_max;
    // Effective intensity sum along the casted ray
    double d12 = 0.;
    // Basic indices along x-/y-/z-directions
    int i_0, j_0, k_0;
    // Helper variables for main loop
    double alpha_c;
    int i, j, k, i_u, j_u, k_u;
    //double alpha_x_u, alpha_y_u, alpha_z_u;
    double l_i_j_k;
    double alpha_x, alpha_y, alpha_z;
    int vidx;
    double ivol;

    // Avoid warnings
    alpha_x_0 = 0;
    alpha_y_0 = 0;
    alpha_z_0 = 0;

    // Create intersection function
    typedef rtk::RayBoxIntersectionFunction<CoordRepType, TInputImage::ImageDimension> RBIFunctionType;
    typename RBIFunctionType::Pointer rbi[TInputImage::ImageDimension];
    for(unsigned int j=0; j<TInputImage::ImageDimension; j++)
    {
      rbi[j] = RBIFunctionType::New();
      typename RBIFunctionType::VectorType boxMin, boxMax;
      for(unsigned int i=0; i<TInputImage::ImageDimension; i++)
      {
        boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i] + 0.00001;  // To avoid numerical errors
        boxMax[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i] +
                    this->GetInput(1)->GetBufferedRegion().GetSize()[i]  - 1.00001;  // To avoid numerical errors
      }
      rbi[j]->SetBoxMin(boxMin);
      rbi[j]->SetBoxMax(boxMax);
    }

    for(unsigned int i=0; i<TInputImage::ImageDimension; i++)
      rbi[i]->SetRayOrigin( &(Source[0]) );

    for (unsigned int y = start[1]; y < ymax; y++)
    {
      // Projection line pitch
      id_line = y * static_cast<int>(detectorSize[0]);
      Xh = R + v2 * static_cast<double> (y) * Det_Spacing[1];
      for (unsigned int x = start[0]; x < xmax; x++, ++itOut)
      {
        // Pixel index
        id_pixel = id_line + x;

        // current Projection pixel position
        X = Xh + v1 * static_cast<double> (x) * Det_Spacing[0];

        // Ray direction
        ray_direction = X - Source;
        // Ray length
        ray_length = ray_direction.GetNorm();

        // Compute the parametric entry (alpha_min) and exit (alpha_max) point
        // "coordinates" of the ray S-X through the virtually axis-aligned volume:

        // invalid length
        if (ray_length < EPSILON)
        {
          drr[id_pixel] = izero;
          continue;
        }
        // X-component
        if (fabs(ray_direction[0]) > EPSILON)
        {
          alpha_x_0 = (Vol_O[0] - Source[0]) / ray_direction[0]; // 1st x-plane
          alpha_x_Nx = (Vol_O[0] + static_cast<double> (Vol_Size[0]) * Vol_Spacing[0] - Source[0]) / ray_direction[0]; // last x-plane
          alpha_x_min = vnl_math_min(alpha_x_0, alpha_x_Nx);
          alpha_x_max = vnl_math_max(alpha_x_0, alpha_x_Nx);
        }
        // No change along x
        else
        {
          // impossible values
          alpha_x_min = -5.;
          alpha_x_max = 5.;
          // Be sure that ray intersects with volume
          if (rbi[0]->Evaluate(ray_direction))
          {
            drr[id_pixel] = izero;
            continue;
          }
        }
        // Y-component
        if (fabs(ray_direction[1]) > EPSILON)
        {
          alpha_y_0 = (Vol_O[1] - Source[1]) / ray_direction[1]; // 1st y-plane
          alpha_y_Ny = (Vol_O[1] + static_cast<double> (Vol_Size[1]) * Vol_Spacing[1] - Source[1]) / ray_direction[1]; // last y-plane
          alpha_y_min = vnl_math_min(alpha_y_0, alpha_y_Ny);
          alpha_y_max = vnl_math_max(alpha_y_0, alpha_y_Ny);
        }
        // No change along y
        else
        {
          // Impossible values
          alpha_y_min = -5.;
          alpha_y_max = 5.;
          // Be sure that ray intersects with volume
          if (rbi[1]->Evaluate(ray_direction))
          {
            drr[id_pixel] = izero;
            continue;
          }
        }
        // Z-component
        if (fabs(ray_direction[2]) > EPSILON)
        {
          alpha_z_0 = (Vol_O[2] - Source[2]) / ray_direction[2]; // 1st y-plane
          alpha_z_Nz = (Vol_O[2] + static_cast<double> (Vol_Size[2]) * Vol_Spacing[2] - Source[2]) / ray_direction[2]; // last y-plane
          alpha_z_min = vnl_math_min(alpha_z_0, alpha_z_Nz);
          alpha_z_max = vnl_math_max(alpha_z_0, alpha_z_Nz);
        }
        // No change along z
        else
        {
          // Impossible values
          alpha_z_min = -5.;
          alpha_z_max = 5.;
          // - be sure that ray intersects with volume
          if (!rbi[2]->Evaluate(ray_direction))
          {
            drr[id_pixel] = izero;
            continue;
          }
        }
        // Overall min/max
        alpha_min = vnl_math_max(alpha_x_min, vnl_math_max(alpha_y_min, alpha_z_min));
        alpha_max = vnl_math_min(alpha_x_max, vnl_math_min(alpha_y_max, alpha_z_max));

        d12 = 0.;
        // Real intersection
        if (alpha_min < alpha_max)
        {
          // compute index of first intersected voxel: i_0, j_0, k_0
          i_0 = (int) floor((Source[0] + alpha_min * ray_direction[0] - Vol_O[0]) / Vol_Spacing[0]);
          j_0 = (int) floor((Source[1] + alpha_min * ray_direction[1] - Vol_O[1]) / Vol_Spacing[1]);
          k_0 = (int) floor((Source[2] + alpha_min * ray_direction[2] - Vol_O[2]) / Vol_Spacing[2]);
          // now "correct" the indices (there are special cases):
          if(alpha_min == alpha_y_min) // j is special
          {
          if(alpha_y_min == alpha_y_0)
            j_0 = 0;
          else
            // alpha_y_Ny
            j_0 = Vol_Size[1] - 1;
          }
          else if (alpha_min == alpha_x_min) // i is special
          {
            if(alpha_x_min == alpha_x_0)
              i_0 = 0;
            else
              // alpha_x_Nx
              i_0 = Vol_Size[0] - 1;
          }
          else if (alpha_min == alpha_z_min) // k is special
          {
            if(alpha_z_min == alpha_z_0)
              k_0 = 0;
            else
              // alpha_z_Nz
              k_0 = Vol_Size[2] - 1;
          }

        // initialize main variables for incremental Siddon-like approach:
        VectorType stepMM;
        alpha_c = alpha_min;
        i = i_0;
        j = j_0;
        k = k_0;
        stepMM[0] = Vol_Spacing[0] / fabs(ray_direction[0]);
        stepMM[1] = Vol_Spacing[1] / fabs(ray_direction[1]);
        stepMM[2] = Vol_Spacing[2] / fabs(ray_direction[2]);
        i_u = (Source[0] < X[0]) ? 1 : -1;
        j_u = (Source[1] < X[1]) ? 1 : -1;
        k_u = (Source[2] < X[2]) ? 1 : -1;
        // compute alphas of 1st intersection after volume-entry:
        if (fabs(ray_direction[0]) > EPSILON)
        {
          if(Source[0] < X[0])
            alpha_x = (Vol_O[0] + (static_cast<double> (i) + 1.) * Vol_Spacing[0] - Source[0]) / ray_direction[0];
          else
            alpha_x = (Vol_O[0] + static_cast<double> (i) * Vol_Spacing[0] - Source[0]) / ray_direction[0];
        }
        else // no change along x
        {
          alpha_x = alpha_max; // irrelevant
        }
        if (fabs(ray_direction[1]) > EPSILON)
        {
          if(Source[1] < X[1])
            alpha_y = (Vol_O[1] + (static_cast<double> (j) + 1.) * Vol_Spacing[1] - Source[1]) / ray_direction[1];
          else
            alpha_y = (Vol_O[1] + static_cast<double> (j) * Vol_Spacing[1] - Source[1]) / ray_direction[1];
        }
        else // no change along y
        {
          alpha_y = alpha_max; // irrelevant
        }
        if (fabs(ray_direction[2]) > EPSILON)
        {
          if(Source[2] < X[2])
            alpha_z = (Vol_O[2] + (static_cast<double> (k) + 1.) * Vol_Spacing[2] - Source[2]) / ray_direction[2];
          else
            alpha_z = (Vol_O[2] + static_cast<double> (k) * Vol_Spacing[2] - Source[2]) / ray_direction[2];
        }
        else // no change along z
        {
          alpha_z = alpha_max; // irrelevant
        }

        // main loop: go step by step along ray
        while ((alpha_c + EPSILON) < alpha_max) // account for limited precision
        {
          // NOTE: the index is already computed here (index is leading)
          vidx = i + j * vlinepitch + k * vslicepitch;

          ivol = volume[vidx];

          if (alpha_x <= alpha_y && alpha_x <= alpha_z)
          {
            l_i_j_k = alpha_x - alpha_c;
            i += i_u;
            alpha_c = alpha_x;
            alpha_x += stepMM[0];
          }
          else if (alpha_y <= alpha_x && alpha_y <= alpha_z)
          {
            l_i_j_k = alpha_y - alpha_c;
            j += j_u;
            alpha_c = alpha_y;
            alpha_y += stepMM[1];
          }
          // NOTE: if (alpha_z <= alpha_x && alpha_z <= alpha_y)
          else
          {
            l_i_j_k = alpha_z - alpha_c;
            k += k_u;
            alpha_c = alpha_z;
            alpha_z += stepMM[2];
          }
          d12 += (l_i_j_k * ivol);
        }
        //d12*= ray_length; // finally scale with ray length
      }
      // finally write the integrated and weighted intensity
      itOut.Set( m_ProjectedValueAccumulation(threadId,
                                              0,
                                              static_cast<OutputPixelType>(d12),
                                              ray_direction,
                                              &(sourcePosition[0]),
                                              ray_direction,
                                              alpha_min,
                                              alpha_max) );
    }
   }
  }
 }
} // end namespace rtk

#endif
