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

#include "rtkForwardProjectionImageFilter.h"
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
 * \author Simon Rit, updated by Cyril Mory
 *
 * \ingroup Projector CudaImageToImageFilter
 */

namespace rtk
{

/** Create a helper Cuda Kernel class for CudaImageOps */
itkCudaKernelClassMacro(rtkCudaForwardProjectionImageFilterKernel);

class ITK_EXPORT CudaForwardProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  ForwardProjectionImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float,3>                                        ImageType;
  typedef CudaForwardProjectionImageFilter                               Self;
  typedef ForwardProjectionImageFilter<ImageType, ImageType>             Superclass;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType, Superclass > GPUSuperclass;
  typedef itk::SmartPointer<Self>                                        Pointer;
  typedef itk::SmartPointer<const Self>                                  ConstPointer;

  typedef ImageType::RegionType        OutputImageRegionType;
  typedef itk::Vector<float,3>         VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaForwardProjectionImageFilter, ImageToImageFilter);

protected:
  rtkcuda_EXPORT CudaForwardProjectionImageFilter();
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
}; // end of class

} // end namespace rtk

#endif
