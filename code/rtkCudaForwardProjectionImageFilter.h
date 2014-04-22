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

/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(rtkCudaForwardProjectionImageFilterKernel);

template <class TInputImage,
          class TOutputImage,
          class TInterpolationWeightMultiplication = Functor::InterpolationWeightMultiplication<typename TInputImage::PixelType, double>,
          class TProjectedValueAccumulation        = Functor::ProjectedValueAccumulation<typename TInputImage::PixelType, typename TOutputImage::PixelType>
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
