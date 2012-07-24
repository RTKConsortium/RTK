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

#ifndef __rtkJosephForwardProjectionImageFilter_txx
#define __rtkJosephForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "rtkRayBoxIntersectionFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage, class TOutputImage>
void
JosephForwardProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType threadId )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nPixelPerProj = outputRegionForThread.GetSize(0)*outputRegionForThread.GetSize(1);
  const typename TInputImage::PixelType *beginBuffer = this->GetInput(1)->GetBufferPointer();
  const unsigned int offsets[3] = {1,
                                   this->GetInput(1)->GetBufferedRegion().GetSize()[0],
                                   this->GetInput(1)->GetBufferedRegion().GetSize()[0] *
                                   this->GetInput(1)->GetBufferedRegion().GetSize()[1]};
  const typename Superclass::GeometryType::Pointer geometry = this->GetGeometry();

  // Iterators on volume input and output
  typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Create intersection function
  typedef rtk::RayBoxIntersectionFunction<CoordRepType, Dimension> RBIFunctionType;
  typename RBIFunctionType::Pointer rbi[Dimension];
  for(unsigned int j=0; j<Dimension; j++)
    {
    rbi[j] = RBIFunctionType::New();
    typename RBIFunctionType::VectorType boxMin, boxMax;
    for(unsigned int i=0; i<Dimension; i++)
      {
      boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i]
                + itk::NumericTraits<double>::min(); // To avoid numerical errors
      boxMax[i] = boxMin[i] + this->GetInput(1)->GetBufferedRegion().GetSize()[i]-1
                - itk::NumericTraits<double>::min(); // To avoid numerical errors
      if(i==j)
        {
        boxMin[i] -= 0.5;
        boxMax[i] += 0.5;
        }
      }
    rbi[j]->SetBoxMin(boxMin);
    rbi[j]->SetBoxMax(boxMax);
    }

  // Go over each projection
  for(unsigned int iProj=0; iProj<outputRegionForThread.GetSize(2); iProj++)
    {
    // Account for system rotations
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
    volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInput(1) );

    // Set source position in volume indices
    typename Superclass::GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = volPPToIndex * geometry->GetSourcePosition(iProj);
    for(unsigned int i=0; i<Dimension; i++)
      rbi[i]->SetRayOrigin( &(sourcePosition[0]) );

    // Compute matrix to transform projection index to volume index
    typename Superclass::GeometryType::ThreeDHomogeneousMatrixType matrix;
    matrix = volPPToIndex.GetVnlMatrix() *
             geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
             GetIndexToPhysicalPointMatrix( this->GetInput() ).GetVnlMatrix();

    // Go over each pixel of the projection
    typename RBIFunctionType::VectorType dirVox, stepMM, dirVoxAbs, np, fp;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++, ++itIn, ++itOut)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        dirVox[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          dirVox[i] += matrix[i][j] * itOut.GetIndex()[j];

        // Direction
        dirVox[i] -= sourcePosition[i];
        }

      // Select main direction
      unsigned int mainDir = 0;
      for(unsigned int i=0; i<Dimension; i++)
        {
        dirVoxAbs[i] = vnl_math_abs( dirVox[i] );
        if(dirVoxAbs[i]>dirVoxAbs[mainDir])
          mainDir = i;
        }

      // Test if there is an intersection
      if( rbi[mainDir]->Evaluate(dirVox) )
        {
        dirVox.Normalize();

        // Compute and sort intersections: (n)earest and (f)arthest (p)points
        np = rbi[mainDir]->GetNearestPoint();
        fp = rbi[mainDir]->GetFarthestPoint();
        if(np[mainDir]>fp[mainDir])
          std::swap(np, fp);

        // If the source is in the volume, use source as one of the intersection points
        if(dirVox[mainDir]>0)
          {
          if(np[mainDir]<sourcePosition[mainDir])
            {
            // Source is in volume
            np[0]=sourcePosition[0];
            np[1]=sourcePosition[1];
            np[2]=sourcePosition[2];
            }
          }
        else
          {
          if(fp[mainDir]>sourcePosition[mainDir])
            {
            // Source is in volume
            fp[0]=sourcePosition[0];
            fp[1]=sourcePosition[1];
            fp[2]=sourcePosition[2];
            }
          }

        // Compute main nearest and farthest slice indices
        const int ns = vnl_math_ceil ( np[mainDir] );
        const int fs = vnl_math_floor( fp[mainDir] );

        // If its a corner, we can skip
        if( fs<ns )
          continue;

        // Determine the other two directions
        unsigned int notMainDirInf = (mainDir+1)%Dimension;
        unsigned int notMainDirSup = (mainDir+2)%Dimension;
        if(notMainDirInf>notMainDirSup)
          std::swap(notMainDirInf, notMainDirSup);

        // Init data pointers to first pixel of slice ns (i)nferior and (s)uperior (x|y) corner
        const unsigned int offsetx = offsets[notMainDirInf];
        const unsigned int offsety = offsets[notMainDirSup];
        const unsigned int offsetz = offsets[mainDir];
        const typename TInputImage::PixelType *pxiyi, *pxsyi, *pxiys, *pxsys;
        pxiyi = beginBuffer + ns * offsetz;
        pxsyi = pxiyi + offsetx;
        pxiys = pxiyi + offsety;
        pxsys = pxsyi + offsety;

        // Compute step size and go to first voxel
        const CoordRepType residual = ns-np[mainDir];
        const CoordRepType norm = 1/dirVox[mainDir];
        const CoordRepType stepx = dirVox[notMainDirInf] * norm;
        const CoordRepType stepy = dirVox[notMainDirSup] * norm;
        CoordRepType currentx = np[notMainDirInf] + residual*stepx;
        CoordRepType currenty = np[notMainDirSup] + residual*stepy;

        typename TOutputImage::PixelType value = (residual+0.5) * BilinearInterpolation(pxiyi, pxsyi, pxiys, pxsys,
                                                                                        currentx, currenty,
                                                                                        offsetx, offsety);

        // Middle steps
        typename TOutputImage::PixelType sum = value;
        for(int i=ns; i<fs; i++)
          {
          pxiyi    += offsetz;
          pxsyi    += offsetz;
          pxiys    += offsetz;
          pxsys    += offsetz;
          currentx += stepx;
          currenty += stepy;
          value = BilinearInterpolation(pxiyi, pxsyi, pxiys, pxsys,
                                        currentx, currenty, offsetx, offsety);
          sum      += value;
          }

        // Last step was too long, remove extra
        sum -= (0.5-fp[mainDir]+fs) * value;

        // Convert voxel to millimeters
        stepMM[notMainDirInf] = this->GetInput(1)->GetSpacing()[notMainDirInf] * stepx;
        stepMM[notMainDirSup] = this->GetInput(1)->GetSpacing()[notMainDirSup] * stepy;
        stepMM[mainDir]       = this->GetInput(1)->GetSpacing()[mainDir];
        sum *= stepMM.GetNorm();

        // Accumulate
        itOut.Set( itIn.Get() + sum );
        }
      }
    }
}

template <class TInputImage, class TOutputImage>
typename JosephForwardProjectionImageFilter<TInputImage,TOutputImage>::OutputPixelType
JosephForwardProjectionImageFilter<TInputImage,TOutputImage>
::BilinearInterpolation(const InputPixelType *pxiyi,
                        const InputPixelType *pxsyi,
                        const InputPixelType *pxiys,
                        const InputPixelType *pxsys,
                        const CoordRepType x,
                        const CoordRepType y,
                        const unsigned int ox,
                        const unsigned int oy) const
{
  unsigned int ix = itk::Math::Floor(x);
  unsigned int iy = itk::Math::Floor(y);
  unsigned int idx = ix*ox + iy*oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1.-lx;
  CoordRepType lyc = 1.-ly;
  return lxc * lyc * pxiyi[ idx ] +
         lx  * lyc * pxsyi[ idx ] +
         lxc * ly  * pxiys[ idx ] +
         lx  * ly  * pxsys[ idx ];
/* Alternative slower solution
  const unsigned int ix = itk::Math::Floor(x);
  const unsigned int iy = itk::Math::Floor(y);
  const unsigned int idx = ix*ox + iy*oy;
  const CoordRepType a = p1[idx];
  const CoordRepType b = p2[idx] - a;
  const CoordRepType c = p3[idx] - a;
  const CoordRepType lx = x-ix;
  const CoordRepType ly = y-iy;
  const CoordRepType d = p4[idx] - a - b - c;
  return a + b*lx + c*ly + d*lx*ly;
*/
}

} // end namespace rtk

#endif
