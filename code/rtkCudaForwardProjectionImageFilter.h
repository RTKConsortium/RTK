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

#ifndef __rtkCudaForwardProjectionImageFilter_h
#define __rtkCudaForwardProjectionImageFilter_h

#include "rtkJosephForwardProjectionImageFilter.h"
#include "itkCudaInPlaceImageFilter.h"
#include "itkCudaUtil.h"
#include "itkCudaKernelManager.h"
#include "rtkWin32Header.h"

#include "itkImage.h"

/** \class CudaForwardProjectionImageFilter
 * \brief Trilinear interpolation forward projection implemented in CUDA
 *
 * CudaForwardProjectionImageFilter is similar to
 * JosephForwardProjectionImageFilter, except it uses a
 * fixed step between sampling points instead of placing these
 * sampling points only on the main direction slices.
 *
 * The code was developed based on the file tt_project_ray_gpu_kernels.cu of
 * NiftyRec (http://sourceforge.net/projects/niftyrec/) which is distributed under a BSD
 * license. See COPYRIGHT.TXT.
 *
 * \author Marc Vila, updated by Simon Rit and Cyril Mory
 *
 * \ingroup Projector CudaImageToImageFilter
 */

namespace rtk
{
namespace Functor
{
/** \class CudaInterpolationWeightMultiplication
 * \brief Function to multiply the interpolation weights with the projected
 * volume values.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template< class TInput, class TCoordRepType, class TOutput=TCoordRepType >
class CudaInterpolationWeightMultiplication
{
public:
  CudaInterpolationWeightMultiplication() {};
  ~CudaInterpolationWeightMultiplication() {};
  bool operator!=( const CudaInterpolationWeightMultiplication & ) const {
    return false;
  }
  bool operator==(const CudaInterpolationWeightMultiplication & other) const
  {
    return !( *this != other );
  }

  inline TOutput operator()( const ThreadIdType itkNotUsed(threadId),
                             const double itkNotUsed(stepLengthInVoxel),
                             const TCoordRepType weight,
                             const TInput *p,
                             const int i ) const
  {
    return weight*p[i];
  }
};

/** \class CudaProjectedValueAccumulation
 * \brief Function to accumulate the ray casting on the projection.
 *
 * \author Simon Rit
 *
 * \ingroup Functions
 */
template< class TInput, class TOutput >
class CudaProjectedValueAccumulation
{
public:
  typedef itk::Vector<double, 3> VectorType;
  typedef itk::Image<float, 2> MaterialMuImageType;

  CudaProjectedValueAccumulation()
  {
    MaterialMuImageType::RegionType region;
    MaterialMuImageType::SizeType size;
    size[0] = 20; //mat
    size[1] = 10; //e
    region.SetSize(size);
    m_MaterialMu = MaterialMuImageType::New();
    m_MaterialMu->SetRegions(region);
    m_MaterialMu->Allocate();
    itk::ImageRegionIterator< MaterialMuImageType > it(m_MaterialMu, region);
    for(unsigned int e = 0; e < size[1]; e++)
      {
      for(unsigned int i = 0; i < size[0]; i++)
        {
        it.Set(e*i);
        ++it;
        }
      }

    m_EnergyWeightList = new float[10];
    for (int i = 0; i < 10; i++)
      {
      m_EnergyWeightList[i] = (float)i;
      }
  };
  ~CudaProjectedValueAccumulation()
  {
  delete[] m_EnergyWeightList;
  };
  bool operator!=( const CudaProjectedValueAccumulation & ) const
    {
    return false;
    }
  bool operator==(const CudaProjectedValueAccumulation & other) const
    {
    return !( *this != other );
    }
  
  MaterialMuImageType::Pointer GetMaterialMu()
  {
    return m_MaterialMu;
  };
  
  float* GetEnergyWeightList()
  {
    return m_EnergyWeightList;
  }

  inline TOutput operator()( const ThreadIdType itkNotUsed(threadId),
                             const TInput &input,
                             const TOutput &rayCastValue,
                             const VectorType &stepInMM,
                             const VectorType &itkNotUsed(source),
                             const VectorType &itkNotUsed(sourceToPixel),
                             const VectorType &itkNotUsed(nearestPoint),
                             const VectorType &itkNotUsed(farthestPoint)) const
    {
    return input + rayCastValue * stepInMM.GetNorm();
    }
  MaterialMuImageType::Pointer m_MaterialMu;
  float* m_EnergyWeightList;
};

} // end namespace Functor


/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(rtkCudaForwardProjectionImageFilterKernel);

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::CudaInterpolationWeightMultiplication<typename TInputImage::PixelType, double>,
          class TProjectedValueAccumulation        = Functor::CudaProjectedValueAccumulation<typename TInputImage::PixelType, typename TOutputImage::PixelType>
          >
class ITK_EXPORT CudaForwardProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< TInputImage, TOutputImage,
  ForwardProjectionImageFilter< TInputImage, TOutputImage > >
{
public:
  /** Standard class typedefs. */
  typedef CudaForwardProjectionImageFilter                                    Self;
  typedef ForwardProjectionImageFilter<TInputImage, TOutputImage>             Superclass;
  typedef itk::CudaInPlaceImageFilter<TInputImage, TOutputImage, Superclass > GPUSuperclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;
  
  typedef itk::CudaImage<float,3>                                        ImageType;
  typedef ImageType::RegionType        OutputImageRegionType;
  typedef itk::Vector<float,3>         VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaForwardProjectionImageFilter, ImageToImageFilter);

  /** Get/Set the functor that is used to multiply each interpolation value with a volume value */
  TInterpolationWeightMultiplication &       GetInterpolationWeightMultiplication() { return m_InterpolationWeightMultiplication; }
  const TInterpolationWeightMultiplication & GetInterpolationWeightMultiplication() const { return m_InterpolationWeightMultiplication; }
  void SetInterpolationWeightMultiplication(const TInterpolationWeightMultiplication & _arg)
    {
    if ( m_InterpolationWeightMultiplication != _arg )
      {
      m_InterpolationWeightMultiplication = _arg;
      this->Modified();
      }
    }

  /** Get/Set the functor that is used to accumulate values in the projection image after the ray
   * casting has been performed. */
  TProjectedValueAccumulation &       GetProjectedValueAccumulation() { return m_ProjectedValueAccumulation; }
  const TProjectedValueAccumulation & GetProjectedValueAccumulation() const { return m_ProjectedValueAccumulation; }
  void SetProjectedValueAccumulation(const TProjectedValueAccumulation & _arg)
    {
    if ( m_ProjectedValueAccumulation != _arg )
      {
      m_ProjectedValueAccumulation = _arg;
      this->Modified();
      }
    }

protected:
  rtkcuda_EXPORT CudaForwardProjectionImageFilter() {};
  ~CudaForwardProjectionImageFilter() {};
  
  void GPUGenerateData();

private:
  //purposely not implemented
  CudaForwardProjectionImageFilter(const Self&);
  void operator=(const Self&);

  int                m_VolumeDimension[3];
  int                m_ProjectionDimension[2];
  float *            m_DeviceVolume;
  float *            m_DeviceProjection;
  float *            m_DeviceMatrix;
  float *            m_DeviceMu;

  TInterpolationWeightMultiplication m_InterpolationWeightMultiplication;
  TProjectedValueAccumulation        m_ProjectedValueAccumulation;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkCudaForwardProjectionImageFilter.txx"
#endif

#endif
