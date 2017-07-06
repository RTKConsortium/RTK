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

#ifndef rtkJosephBackProjectionImageFilter_hxx
#define rtkJosephBackProjectionImageFilter_hxx

#include "rtkHomogeneousMatrix.h"
#include "rtkRayBoxIntersectionFunction.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace rtk
{

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::JosephBackProjectionImageFilter()
{
#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
  // Set the direction along which the requested region should NOT be split
  m_Splitter = rtk::ImageRegionSplitterArbitraryDimension::New();
  m_NumberOfSubsplits = 4;
#else
  // Old versions of ITK (before 4.4) do not have the ImageRegionSplitterDirection
  // and should run this filter with only one thread
  this->SetNumberOfThreads(1);
#endif
}

#if ITK_VERSION_MAJOR > 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR >= 4)
template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
const itk::ImageRegionSplitterBase*
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::GetImageRegionSplitter(void) const
{
  return m_Splitter;
}
#endif

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
unsigned int
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::SplitRequestedRegion(unsigned int i, unsigned int pieces, OutputImageRegionType &splitRegion)
{
const itk::ImageRegionSplitterBase * splitter = this->GetImageRegionSplitter();

splitRegion = this->GetInput(1)->GetBufferedRegion();
return splitter->GetSplit( i, this->GetOptimalNumberOfSplits(), splitRegion );
}

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
void
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::BeforeThreadedGenerateData()
{
  // Allocate the output image
  this->AllocateOutputs();

  // Initialize output region with input region in case the filter is not in
  // place
  if(this->GetInput() != this->GetOutput() )
    {
    // Iterators on volume input and output
    typedef itk::ImageRegionConstIterator<TInputImage> InputRegionIterator;
    InputRegionIterator itVolIn(this->GetInput(0), this->GetInput()->GetBufferedRegion());

    typedef itk::ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
    OutputRegionIterator itVolOut(this->GetOutput(), this->GetInput()->GetBufferedRegion());

    while(!itVolIn.IsAtEnd() )
      {
      itVolOut.Set(itVolIn.Get() );
      ++itVolIn;
      ++itVolOut;
      }
    }

  // Instead of using GetNumberOfThreads, we need to split the image into the
  // number of regions that will actually be returned by
  // itkImageSource::SplitRequestedRegion. Sometimes this number is less than
  // the number of threads requested.
  OutputImageRegionType dummy;
  unsigned int actualThreads = this->SplitRequestedRegion(0, this->GetOptimalNumberOfSplits(), dummy);

  m_Barrier = itk::Barrier::New();
  m_Barrier->Initialize(actualThreads);
}

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
unsigned int
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::GetOptimalNumberOfSplits()
{
  // Very crude approximation at the moment: assumes a flat and centered detector,
  // a centered volume of which we calculate the bounding ball,
  // and neglects the fact that off-center slabs are thinner

  const unsigned int Dimension = TInputImage::ImageDimension;
  typename TInputImage::RegionType buffReg = this->GetInput(1)->GetBufferedRegion();
  buffReg.SetSize(Dimension-1,1);

  // Compute the best splitting axis
  std::vector<float> extent(2);
  std::vector<float> spacing(2);
  std::vector<float> normalizedExtent(2);
  spacing[0] = std::max(this->GetOutput()->GetSpacing()[0], this->GetOutput()->GetSpacing()[2]);
  spacing[1] = this->GetOutput()->GetSpacing()[1];
  for(unsigned int i=0; i<2; i++)
    {
    extent[i] = this->GetInput(1)->GetSpacing()[i] * buffReg.GetSize()[i];
    normalizedExtent[i] = extent[i] / spacing[i];
    }
  if(normalizedExtent[0] > normalizedExtent[1])
    m_SplitAxis = 0;
  else
    m_SplitAxis = 1;
  m_Splitter->SetSplitAxis(m_SplitAxis);

  // Compute the radius of a bounding ball around the volume
  float volumeRadius = 0;
  for (unsigned int dim=0; dim<Dimension; dim++)
    {
    float length = this->GetOutput()->GetRequestedRegion().GetSize()[dim] * this->GetOutput()->GetSpacing()[dim];
    volumeRadius += length * length;
    }
  volumeRadius = sqrt(volumeRadius);

  // Project the distance between two slabs onto the plane parallel to the detector
  // and "at the entrance" of the bounding box ball (i.e. closest to the source)
  float distanceBetweenSlabs, projectedDistanceBetweenSlabs, ratio;

  // Make sure we are not requesting too many threads for the dimension along which we split
  unsigned int requestedThreads = this->GetNumberOfThreads();
  while ((requestedThreads * m_NumberOfSubsplits > buffReg.GetSize()[m_SplitAxis]) && (requestedThreads > 1))
    requestedThreads /= 2;

  // If the size along the dimension we want to split is too small, and we can't use multiple threads, revert to single threading. Otherwise, go on
  if (requestedThreads > 1)
    {
    // Initial estimate, using as many splits as threads available, and m_NumberOfSubsplits subsplits (default is 4)
    distanceBetweenSlabs = buffReg.GetSize()[m_SplitAxis] * this->GetInput(1)->GetSpacing()[m_SplitAxis] * (m_NumberOfSubsplits-1) / (m_NumberOfSubsplits * requestedThreads);
    GeometryType *geometry = dynamic_cast<GeometryType*>(this->GetGeometry().GetPointer());
    if( !geometry )
      {
      itkGenericExceptionMacro(<< "Error, ThreeDCircularProjectionGeometry expected");
      }
    float sid = geometry->GetSourceToIsocenterDistances()[0];
    float sdd = geometry->GetSourceToDetectorDistances()[0];
    projectedDistanceBetweenSlabs = distanceBetweenSlabs * (sid - volumeRadius) / sdd;
    ratio = projectedDistanceBetweenSlabs / spacing[m_SplitAxis];

    // In order to be safe and "compensate" for all the approximations, we should strive to obtain a ratio of 4 or more.
    if (ratio < 4.0)
      requestedThreads = floor((double) requestedThreads * ratio / 4.0);
    }
  return requestedThreads;
}

template <class TInputImage,
          class TOutputImage,
          class TSplatWeightMultiplication>
void
JosephBackProjectionImageFilter<TInputImage,
                                TOutputImage,
                                TSplatWeightMultiplication>
::ThreadedGenerateData( const OutputImageRegionType& inputRegionForThread, ThreadIdType threadId )
{
  // Get the full buffered region in the input projections
  const unsigned int Dimension = TInputImage::ImageDimension;
  int offsets[3];
  offsets[0] = 1;
  offsets[1] = this->GetInput(0)->GetBufferedRegion().GetSize()[0];
  offsets[2] = this->GetInput(0)->GetBufferedRegion().GetSize()[0] * this->GetInput(0)->GetBufferedRegion().GetSize()[1];

  // The current thread will only work on a slab of the input projection data, one projection at a time
  // It will process the first half of its slab, wait for the other threads, and process the second half
  // The following code computes the splitting in half (along the same dimension as SplitRequestedRegion did)
  typename TInputImage::RegionType singleProjectionInputRegionForThread;
  rtk::ImageRegionSplitterArbitraryDimension::Pointer splitter = rtk::ImageRegionSplitterArbitraryDimension::New();

  GeometryType *geometry = dynamic_cast<GeometryType*>(this->GetGeometry().GetPointer());
  if( !geometry )
    {
    itkGenericExceptionMacro(<< "Error, ThreeDCircularProjectionGeometry expected");
    }

  // beginBuffer is pointing at point with index (0,0,0) in memory, even if
  // it is not in the allocated memory
  typename TOutputImage::PixelType *beginBuffer =
      this->GetOutput()->GetBufferPointer() -
      offsets[0] * this->GetOutput()->GetBufferedRegion().GetIndex()[0] -
      offsets[1] * this->GetOutput()->GetBufferedRegion().GetIndex()[1] -
      offsets[2] * this->GetOutput()->GetBufferedRegion().GetIndex()[2];

  // volPPToIndex maps the physical 3D coordinates of a point (in mm) to the
  // corresponding 3D volume index
  typename GeometryType::ThreeDHomogeneousMatrixType volPPToIndex;
  volPPToIndex = GetPhysicalPointToIndexMatrix( this->GetInput(0) );

  // Must process projections one by one, otherwise two threads
  // back projecting different slabs of two opposite projections
  // may collide because of the cone angle
  for (unsigned int proj=inputRegionForThread.GetIndex()[2]; proj<inputRegionForThread.GetIndex()[2]+inputRegionForThread.GetSize()[2]; proj++)
    {
    // Process even slabs first, then odd slabs,
    // to avoid collisions between threads
    for (unsigned int subsplit=0; subsplit<m_NumberOfSubsplits; subsplit++)
      {
      singleProjectionInputRegionForThread = inputRegionForThread;
      singleProjectionInputRegionForThread.SetSize(2,1);
      singleProjectionInputRegionForThread.SetIndex(2,proj);
      splitter->SetSplitAxis(m_SplitAxis);
      splitter->GetSplit(subsplit, m_NumberOfSubsplits, singleProjectionInputRegionForThread);

      // Iterators on projections input
      typedef ProjectionsRegionConstIteratorRayBased<TInputImage> InputRegionIterator;
      InputRegionIterator *itIn;
      itIn = InputRegionIterator::New(this->GetInput(1),
                                      singleProjectionInputRegionForThread,
                                      geometry,
                                      volPPToIndex);

      // Create intersection functions, one for each possible main direction
      typedef rtk::RayBoxIntersectionFunction<CoordRepType, Dimension> RBIFunctionType;
      typename RBIFunctionType::Pointer rbi = RBIFunctionType::New();
      typename RBIFunctionType::VectorType boxMin, boxMax;
      for(unsigned int i=0; i<Dimension; i++)
        {
        boxMin[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
        boxMax[i] = this->GetOutput()->GetRequestedRegion().GetIndex()[i] +
                    this->GetOutput()->GetRequestedRegion().GetSize()[i] - 1;
        }
      rbi->SetBoxMin(boxMin);
      rbi->SetBoxMax(boxMax);

      typename RBIFunctionType::VectorType stepMM, np, fp;


      // Go over each pixel of the projection
      for(unsigned int pix=0; pix<singleProjectionInputRegionForThread.GetNumberOfPixels(); pix++, itIn->Next())
        {
        typename InputRegionIterator::PointType sourcePosition = itIn->GetSourcePosition();
        typename InputRegionIterator::PointType dirVox = itIn->GetSourceToPixel();

        //Set source
        rbi->SetRayOrigin( sourcePosition );

        // Select main direction
        unsigned int mainDir = 0;
        typename RBIFunctionType::VectorType dirVoxAbs;
        for(unsigned int i=0; i<Dimension; i++)
          {
          dirVoxAbs[i] = vnl_math_abs( dirVox[i] );
          if(dirVoxAbs[i]>dirVoxAbs[mainDir])
            mainDir = i;
          }

        // Test if there is an intersection
        if( rbi->Evaluate(&dirVox[0]) &&
            rbi->GetFarthestDistance()>=0. && // check if detector after the source
            rbi->GetNearestDistance()<=1.)    // check if detector after or in the volume
          {
          // Clip the casting between source and pixel of the detector
          rbi->SetNearestDistance ( std::max(rbi->GetNearestDistance() , 0.) );
          rbi->SetFarthestDistance( std::min(rbi->GetFarthestDistance(), 1.) );

          // Compute and sort intersections: (n)earest and (f)arthest (p)points
          np = rbi->GetNearestPoint();
          fp = rbi->GetFarthestPoint();
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

          const CoordRepType minx = rbi->GetBoxMin()[notMainDirInf];
          const CoordRepType miny = rbi->GetBoxMin()[notMainDirSup];
          const CoordRepType maxx = rbi->GetBoxMax()[notMainDirInf];
          const CoordRepType maxy = rbi->GetBoxMax()[notMainDirSup];

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
            BilinearSplatOnBorders(itIn->Get(), fp[mainDir] - np[mainDir], stepMM.GetNorm(),
                                    pxiyi, pxsyi, pxiys, pxsys, currentx, currenty,
                                    offsetx, offsety, minx, miny, maxx, maxy);
            }
          else
            {
            // First step
            BilinearSplatOnBorders(itIn->Get(), residual + 0.5, stepMM.GetNorm(),
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
              BilinearSplat(itIn->Get(), 1.0, stepMM.GetNorm(), pxiyi, pxsyi, pxiys, pxsys, currentx, currenty, offsetx, offsety);

              // Move to next main direction slice
              pxiyi += offsetz;
              pxsyi += offsetz;
              pxiys += offsetz;
              pxsys += offsetz;
              currentx += stepx;
              currenty += stepy;
              }

            // Last step
            BilinearSplatOnBorders(itIn->Get(), fp[mainDir] - fs + 0.5, stepMM.GetNorm(),
                                   pxiyi, pxsyi, pxiys, pxsys, currentx, currenty,
                                   offsetx, offsety, minx, miny, maxx, maxy);
            }
          }
        }
      delete itIn;
      m_Barrier->Wait();
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
