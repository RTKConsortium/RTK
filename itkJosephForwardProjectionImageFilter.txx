#ifndef __itkJosephForwardProjectionImageFilter_txx
#define __itkJosephForwardProjectionImageFilter_txx

#include "rtkHomogeneousMatrix.h"
#include "itkRayBoxIntersectionFunction.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkIdentityTransform.h>

namespace itk
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
  typedef ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Create intersection function
  typedef itk::RayBoxIntersectionFunction<double, Dimension> RBIFunctionType;
  typename RBIFunctionType::Pointer rbi = RBIFunctionType::New();
  typename RBIFunctionType::PointType boxMin, boxMax;
  for(unsigned int i=0; i<Dimension; i++)
    {
    boxMin[i] = this->GetInput(1)->GetBufferedRegion().GetIndex()[i];
    boxMax[i] = boxMin[i] + this->GetInput(1)->GetBufferedRegion().GetSize()[i]-1;
    }
  rbi->SetBoxMin(boxMin);
  rbi->SetBoxMax(boxMax);

  // Go over each projection
  for(unsigned int iProj=outputRegionForThread.GetIndex(2);
                   iProj<outputRegionForThread.GetIndex(2)+outputRegionForThread.GetSize(2);
                   iProj++)
    {
    // Account for system rotations
    itk::Matrix<double, Dimension+1, Dimension+1> rotMatrix;
    rotMatrix = Get3DRigidTransformationHomogeneousMatrix( geometry->GetOutOfPlaneAngles()[iProj],
                                                           geometry->GetGantryAngles()[iProj],
                                                           geometry->GetInPlaneAngles()[iProj],
                                                           0.,0.,0.);
    rotMatrix = GetPhysicalPointToIndexMatrix< TOutputImage >( this->GetInput(1) ) * rotMatrix.GetInverse();

    // Compute source position an change coordinate system
    itk::Vector<double, 4> sourcePosition;
    sourcePosition[0] = geometry->GetSourceOffsetsX()[iProj];
    sourcePosition[1] = geometry->GetSourceOffsetsY()[iProj];
    sourcePosition[2] = -geometry->GetSourceToIsocenterDistances()[iProj];
    sourcePosition[3] = 1.;
    sourcePosition = rotMatrix * sourcePosition;
    rbi->SetRayOrigin( typename RBIFunctionType::PointType(&sourcePosition[0]) );

    // Compute matrix to transform projection index to volume coordinates
    itk::Matrix<double, Dimension+1, Dimension+1> matrix;
    matrix = GetIndexToPhysicalPointMatrix< TOutputImage >( this->GetOutput() );
    matrix[0][3] -= geometry->GetProjectionOffsetsX()[iProj] - geometry->GetSourceOffsetsX()[iProj];
    matrix[1][3] -= geometry->GetProjectionOffsetsY()[iProj] - geometry->GetSourceOffsetsY()[iProj];
    matrix[2][3] = geometry->GetSourceToDetectorDistances()[iProj] -
                   geometry->GetSourceToIsocenterDistances()[iProj];
    matrix[2][2] = 0.; // Force z to axis to detector distance
    matrix = rotMatrix * matrix;

    // Go over each pixel of the projection
    typename TInputImage::SpacingType dirVox, step, stepMM, dirVoxAbs, current, nearest, farthest;
    for(unsigned int pix=0; pix<nPixelPerProj; pix++)
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
      dirVox.Normalize();

      if( rbi->Evaluate(&dirVox[0]) )
        {
        // Select main direction
        unsigned int mainDir = 0;
        for(unsigned int i=0; i<Dimension; i++)
          {
          dirVoxAbs[i] = vnl_math_abs( dirVox[i] );
          if(dirVoxAbs[i]>dirVoxAbs[mainDir])
            mainDir = i;
          }
        unsigned int notMainDirInf = (mainDir+1)%Dimension;
        unsigned int notMainDirSup = (mainDir+2)%Dimension;
        if(notMainDirInf>notMainDirSup)
          std::swap(notMainDirInf, notMainDirSup);

        // Compute main slice indices
        nearest  = rbi->GetNearestPoint().GetVectorFromOrigin();
        farthest = rbi->GetFarthestPoint().GetVectorFromOrigin();
        if(nearest[mainDir]>farthest[mainDir])
          std::swap(nearest, farthest);
        unsigned int nearestMainSlice, farthestMainSlice;
        nearest [mainDir] = vnl_math_max(nearest[mainDir]-1e-10, boxMin[mainDir]);
        nearestMainSlice  = vnl_math_ceil ( nearest [mainDir] );
        farthestMainSlice = vnl_math_floor( farthest[mainDir] );
        const typename TInputImage::PixelType *dataSlice = beginBuffer + nearestMainSlice * offsets[mainDir];
        double residual = nearestMainSlice-nearest[mainDir];
        step = dirVox * ( residual/dirVox[mainDir] );
        current = nearest + step;
        if( boxMax[notMainDirInf]<current[notMainDirInf] || current[notMainDirInf]<boxMin[notMainDirInf] ||
            boxMax[notMainDirSup]<current[notMainDirSup] || current[notMainDirSup]<boxMin[notMainDirSup] )
          {
          ++itIn;
          ++itOut;
          continue; // Corner, skip
          }

        typename TOutputImage::PixelType value = (residual+0.5) *
                                                 BilinearInterpolation(dataSlice,
                                                                       current[notMainDirInf],
                                                                       current[notMainDirSup],
                                                                       offsets[notMainDirInf],
                                                                       offsets[notMainDirSup]);
        typename TOutputImage::PixelType sum = value;

        // Middle steps
        step = dirVox * ( 1/dirVox[mainDir] );
        for(unsigned int i=nearestMainSlice+1; i<farthestMainSlice; i++,
                                                                    dataSlice+=offsets[mainDir],
                                                                    current+=step)
          {

          value = BilinearInterpolation(dataSlice,
                                        current[notMainDirInf],
                                        current[notMainDirSup],
                                        offsets[notMainDirInf],
                                        offsets[notMainDirSup]);
          sum += value;
          }

        // Last step was too long, remove extra
        sum -= (0.5-farthest[mainDir]+farthestMainSlice) * value;

        // Convert voxel to millimeters
        for(unsigned int i=0; i<Dimension; i++)
          stepMM[i] = this->GetInput(1)->GetSpacing()[i] * step[i];
        sum *= stepMM.GetNorm();
        itOut.Set( itIn.Get() + sum );
        }
      ++itIn;
      ++itOut;
      }
    }
}

template <class TInputImage, class TOutputImage>
typename JosephForwardProjectionImageFilter<TInputImage,TOutputImage>::OutputPixelType
JosephForwardProjectionImageFilter<TInputImage,TOutputImage>
::BilinearInterpolation(const InputPixelType *data, double x, double y, unsigned int ox, unsigned int oy) const
{
  unsigned int ix = itk::Math::Floor(x);
  unsigned int iy = itk::Math::Floor(y);
  unsigned int idxX = ix*ox;
  unsigned int idxY = iy*oy;
  double lx = x - ix;
  double ly = y - iy;
  double lxc = 1.-lx;
  double lyc = 1.-ly;
  return lx  * ly  * data[ idxX+ox + idxY+oy ] +
         lxc * ly  * data[ idxX    + idxY+oy ] +
         lx  * lyc * data[ idxX+ox + idxY ] +
         lxc * lyc * data[ idxX    + idxY ];
}

} // end namespace itk

#endif
