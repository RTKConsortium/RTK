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

#ifndef rtkCudaForwardProjectionImageFilter_h
#define rtkCudaForwardProjectionImageFilter_h

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

template <class TInputImage = itk::CudaImage<float,3>,
          class TOutputImage = itk::CudaImage<float,3> >
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
  typedef itk::Vector<float,3>                                                VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaForwardProjectionImageFilter, ImageToImageFilter);

  /** Set step size along ray (in mm). Default is 1 mm. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

protected:
  CudaForwardProjectionImageFilter();
  ~CudaForwardProjectionImageFilter() {};

  virtual void GPUGenerateData();

private:
  //purposely not implemented
  CudaForwardProjectionImageFilter(const Self&);
  void operator=(const Self&);

  double             m_StepSize;
  bool               m_UseCudaTexture;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkCudaForwardProjectionImageFilter.hxx"
#endif

#endif
