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

#ifndef __rtkJosephBackProjectionImageFilter_txx
#define __rtkJosephBackProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "rtkRayBoxIntersectionFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
void
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::GenerateData()
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  typename TInputImage::RegionType buffReg = this->GetInput(1)->GetBufferedRegion();
  const unsigned int nPixelPerProj = buffReg.GetSize(0) * buffReg.GetSize(1);
  int offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(0)->GetBufferedRegion().GetSize()[0];
  offsets[2] = this->GetInput(0)->GetBufferedRegion().GetSize()[0] * this->GetInput(0)->GetBufferedRegion().GetSize()[1];

  GeometryType *geometry = dynamic_cast<GeometryType*>(this->GetGeometry().GetPointer());
  if( !geometry )
    {
    itkGenericExceptionMacro(<< "Error, ThreeDCircularProjectionGeometry expected");
    }

  // Allocate the output image
  this->AllocateOutputs();

  // beginBuffer is pointing at point with index (0,0,0) in memory, even if
  // it is not in the allocated memory
  typename TOutputImage::PixelType *beginBuffer =
      this->GetOutput()->GetBufferPointer() -
      offsets[0] * this->GetOutput()->GetBufferedRegion().GetIndex()[0] -
      offsets[1] * this->GetOutput()->GetBufferedRegion().GetIndex()[1] -
      offsets[2] * this->GetOutput()->GetBufferedRegion().GetIndex()[2];

  // Iterator on projections input
  typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(1), buffReg);

  // Create intersection functions, one for each possible main direction
  typedef rtk::RayBoxIntersectionFunction<CoordRepType, Dimension> RBIFunctionType;
  typename RBIFunctionType::Pointer rbi[Dimension];
  for(unsigned int j=0; j<Dimension; j++)
    {
    rbi[j] = RBIFunctionType::New();
    typename RBIFunctionType::VectorType boxMin, boxMax;
    for(unsigned int i=0; i<Dimension; i++)
      {
      boxMin[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
      boxMax[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i] +
                  this->GetOutput()->GetRequestedRegion().GetSize()[i]  - 1;
    }
    rbi[j]->SetBoxMin(boxMin);
    rbi[j]->SetBoxMax(boxMax);
    }

  // Go over each projection
  for(int iProj=buffReg.GetIndex(2);
          iProj<buffReg.GetIndex(2)+(int)buffReg.GetSize(2);
          iProj++)
    {
    // Account for system rotations
    // volPPToIndex maps the physical 3D coordinates of a point (in mm) to the
    // corresponding 3D volume index
    typename GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
    volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetOutput() );

    // Set source position in volume indices
    // GetSourcePosition() returns coordinates in mm. Multiplying by 
    // volPPToIndex gives the corresponding volume index
    typename GeometryType::HomogeneousVectorType sourcePosition;
    sourcePosition = volPPToIndex * geometry->GetSourcePosition(iProj);
    for(unsigned int i=0; i<Dimension; i++)
      rbi[i]->SetRayOrigin( &(sourcePosition[0]) );

    // Compute matrix to transform projection index to volume index
    // IndexToPhysicalPointMatrix maps the 2D index of a projection's pixel to its 2D position on the detector (in mm)
    // ProjectionCoordinatesToFixedSystemMatrix maps the 2D position of a pixel on the detector to its 3D coordinates in volume's coordinates (still in mm)
    // volPPToIndex maps 3D volume coordinates to a 3D index
    typename GeometryType::ThreeDHomogeneousMatrixType matrix;
    matrix = volPPToIndex.GetVnlMatrix() *
             geometry->GetProjectionCoordinatesToFixedSystemMatrix(iProj).GetVnlMatrix() *
             GetIndexToPhysicalPointMatrix( this->GetInput(1) ).GetVnlMatrix();

    // Go over each pixel of the projection
    typename RBIFunctionType::VectorType dirVox, stepMM, dirVoxAbs, np, fp;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++, ++itIn)
      {
      // Compute point coordinate in volume depending on projection index
      for(unsigned int i=0; i<Dimension; i++)
        {
        dirVox[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          dirVox[i] += matrix[i][j] * itIn.GetIndex()[j];

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
      if( rbi[mainDir]->Evaluate(dirVox) &&
          rbi[mainDir]->GetFarthestDistance()>=0. && // check if detector after the source
          rbi[mainDir]->GetNearestDistance()<=1.)    // check if detector after or in the volume
        {
        // Clip the casting between source and pixel of the detector
        rbi[mainDir]->SetNearestDistance ( std::max(rbi[mainDir]->GetNearestDistance() , 0.) );
        rbi[mainDir]->SetFarthestDistance( std::min(rbi[mainDir]->GetFarthestDistance(), 1.) );

        // Compute and sort intersections: (n)earest and (f)arthest (p)points
        np = rbi[mainDir]->GetNearestPoint();
        fp = rbi[mainDir]->GetFarthestPoint();
        if(np[mainDir]>fp[mainDir])
          std::swap(np, fp);

        // Compute main nearest and farthest slice indices
        const int ns = vnl_math_rnd( np[mainDir]);
        const int fs = vnl_math_rnd( fp[mainDir]);

        // Determine the other two directions
        unsigned int notMainDirInf = (mainDir+1)%Dimension;
        unsigned int notMainDirSup = (mainDir+2)%Dimension;
        if(notMainDirInf>notMainDirSup)
          std::swap(notMainDirInf, notMainDirSup);

        const CoordRepType minx = rbi[mainDir]->GetBoxMin()[notMainDirInf];
        const CoordRepType miny = rbi[mainDir]->GetBoxMin()[notMainDirSup];
        const CoordRepType maxx = rbi[mainDir]->GetBoxMax()[notMainDirInf];
        const CoordRepType maxy = rbi[mainDir]->GetBoxMax()[notMainDirSup];

        // Init data pointers to first pixel of slice ns (i)nferior and (s)uperior (x|y) corner
        const int offsetx = offsets[notMainDirInf];
        const int offsety = offsets[notMainDirSup];
        const int offsetz = offsets[mainDir];
        OutputPixelType *pxiyi, *pxsyi, *pxiys, *pxsys;

        pxiyi = beginBuffer + ns * offsetz;
        pxsyi = pxiyi + offsetx;
        pxiys = pxiyi + offsety;
        pxsys = pxsyi + offsety;

        // Compute step size and go to first voxel
        const CoordRepType residual = ns - np[mainDir];
        const CoordRepType norm = 1/dirVox[mainDir];
        const CoordRepType stepx = dirVox[notMainDirInf] * norm;
        const CoordRepType stepy = dirVox[notMainDirSup] * norm;
        CoordRepType currentx = np[notMainDirInf] + residual * stepx;
        CoordRepType currenty = np[notMainDirSup] + residual * stepy;

        // Compute voxel to millimeters conversion
        stepMM[notMainDirInf] = this->GetInput(0)->GetSpacing()[notMainDirInf] * stepx;
        stepMM[notMainDirSup] = this->GetInput(0)->GetSpacing()[notMainDirSup] * stepy;
        stepMM[mainDir]       = this->GetInput(0)->GetSpacing()[mainDir];

        if (fs == ns) //If the voxel is a corner, we can skip most steps
          {
            BilinearSplatOnBorders(itIn.Get(), fp[mainDir] - np[mainDir], stepMM.GetNorm(),
                                    pxiyi, pxsyi, pxiys, pxsys, currentx, currenty,
                                    offsetx, offsety, minx, miny, maxx, maxy);
          }
        else
          {

          // First step
            BilinearSplatOnBorders(itIn.Get(), residual + 0.5, stepMM.GetNorm(),
                                 pxiyi, pxsyi, pxiys, pxsys, currentx, currenty,
                                 offsetx, offsety, minx, miny, maxx, maxy);

          // Move to next main direction slice
          pxiyi += offsetz;
          pxsyi += offsetz;
          pxiys += offsetz;
          pxsys += offsetz;
          currentx += stepx;
          currenty += stepy;

          // Middle steps
          for(int i=ns+1; i<fs; i++)
            {

            BilinearSplat(itIn.Get(), 1.0, stepMM.GetNorm(), pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety);

            // Move to next main direction slice
            pxiyi += offsetz;
            pxsyi += offsetz;
            pxiys += offsetz;
            pxsys += offsetz;
            currentx += stepx;
            currenty += stepy;
            }

          // Last step
          BilinearSplatOnBorders(itIn.Get(), fp[mainDir] - fs + 0.5, stepMM.GetNorm(),
                                 pxiyi, pxsyi, pxiys, pxsys, currentx, currenty,
                                 offsetx, offsety, minx, miny, maxx, maxy);
          }
        }

      }
    }
}

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
void
JosephBackProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TSplatWeightMultiplication>
::BilinearSplat(const InputPixelType rayValue,
                                               const double stepLengthInVoxel,
                                               const double voxelSize,
                                               OutputPixelType *pxiyi,
                                               OutputPixelType *pxsyi,
                                               OutputPixelType *pxiys,
                                               OutputPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const int ox,
                                               const int oy)
{
  int ix = vnl_math_floor(x);
  int iy = vnl_math_floor(y);
  int idx = ix*ox + iy*oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1.-lx;
  CoordRepType lyc = 1.-ly;

  pxiyi[idx] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lxc * lyc);
  pxsyi[idx] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lx * lyc);
  pxiys[idx] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lxc * ly);
  pxsys[idx] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lx * ly);

}

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
void
JosephBackProjectionImageFilter<TInputImage,
                                   TOutputImage,
                                   TSplatWeightMultiplication>
::BilinearSplatOnBorders(const InputPixelType rayValue,
                                               const double stepLengthInVoxel,
                                               const double voxelSize,
                                               OutputPixelType *pxiyi,
                                               OutputPixelType *pxsyi,
                                               OutputPixelType *pxiys,
                                               OutputPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const int ox,
                                               const int oy,
                                               const CoordRepType minx,
                                               const CoordRepType miny,
                                               const CoordRepType maxx,
                                               const CoordRepType maxy)
{
  int ix = vnl_math_floor(x);
  int iy = vnl_math_floor(y);
  int idx = ix*ox + iy*oy;
  CoordRepType lx = x - ix;
  CoordRepType ly = y - iy;
  CoordRepType lxc = 1.-lx;
  CoordRepType lyc = 1.-ly;

  int offset_xi = 0;
  int offset_yi = 0;
  int offset_xs = 0;
  int offset_ys = 0;

  if(ix < minx) offset_xi = ox;
  if(iy < miny) offset_yi = oy;
  if(ix >= maxx) offset_xs = -ox;
  if(iy >= maxy) offset_ys = -oy;

  pxiyi[idx + offset_xi + offset_yi] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lxc * lyc);
  pxiys[idx + offset_xi + offset_ys] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lxc * ly);
  pxsyi[idx + offset_xs + offset_yi] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lx * lyc);
  pxsys[idx + offset_xs + offset_ys] += m_SplatWeightMultiplication(rayValue, stepLengthInVoxel, voxelSize, lx * ly);

}


} // end namespace rtk

#endif
