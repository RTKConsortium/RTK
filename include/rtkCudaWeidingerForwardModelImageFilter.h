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

#ifndef rtkCudaWeidingerForwardModelImageFilter_h
#define rtkCudaWeidingerForwardModelImageFilter_h

#include "rtkConfiguration.h"
//Conditional definition of the class to pass ITKHeaderTest
#ifdef RTK_USE_CUDA

#include "rtkWeidingerForwardModelImageFilter.h"
#include "itkCudaImageToImageFilter.h"
#include "itkCudaUtil.h"
#include "itkCudaKernelManager.h"
#include "rtkWin32Header.h"

/** \class CudaWeidingerForwardModelImageFilter
 * \brief CUDA implementation of the Weidinger forward model filter
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 */

namespace rtk
{
template <class TMaterialProjections,
          class TPhotonCounts,
          class TSpectrum,
          class TProjections=itk::CudaImage<typename TMaterialProjections::PixelType::ValueType, TMaterialProjections::ImageDimension> >
class ITK_EXPORT CudaWeidingerForwardModelImageFilter :
  public itk::CudaImageToImageFilter< TMaterialProjections, TMaterialProjections,
  WeidingerForwardModelImageFilter< TMaterialProjections, TPhotonCounts, TSpectrum, TProjections > >
{
public:
  /** Standard class typedefs. */
  typedef CudaWeidingerForwardModelImageFilter                                                            Self;
  typedef WeidingerForwardModelImageFilter<TMaterialProjections, TPhotonCounts, TSpectrum, TProjections>  Superclass;
  typedef itk::CudaImageToImageFilter<TMaterialProjections, TMaterialProjections, Superclass >            GPUSuperclass;
  typedef itk::SmartPointer<Self>                                                                         Pointer;
  typedef itk::SmartPointer<const Self>                                                                   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaWeidingerForwardModelImageFilter, ImageToImageFilter);

protected:
  CudaWeidingerForwardModelImageFilter();
  ~CudaWeidingerForwardModelImageFilter() {};

  virtual void GPUGenerateData();

private:
  //purposely not implemented
  CudaWeidingerForwardModelImageFilter(const Self&);
  void operator=(const Self&);
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkCudaWeidingerForwardModelImageFilter.hxx"
#endif

#endif //end conditional definition of the class

#endif
