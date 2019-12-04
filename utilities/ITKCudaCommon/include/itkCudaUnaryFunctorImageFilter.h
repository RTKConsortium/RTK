/*=========================================================================
 *
 *  Copyright Insight Software Consortium
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
#ifndef itkCudaUnaryFunctorImageFilter_h
#define itkCudaUnaryFunctorImageFilter_h

#include "itkCudaFunctorBase.h"
#include "itkCudaInPlaceImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"

namespace itk
{
/** \class CudaUnaryFunctorImageFilter
 * \brief Implements pixel-wise generic operation on one image using Cuda.
 *
 * Cuda version of unary functor image filter.
 * Cuda Functor handles parameter setup for the Cuda kernel.
 *
 * \ingroup   ITKCudaCommon
 */
template <class TInputImage,
          class TOutputImage,
          class TFunction,
          class TParentImageFilter = InPlaceImageFilter<TInputImage, TOutputImage>>
class ITK_EXPORT CudaUnaryFunctorImageFilter
  : public CudaInPlaceImageFilter<TInputImage, TOutputImage, TParentImageFilter>
{
public:
  /** Standard class type alias. */
  using Self = CudaUnaryFunctorImageFilter;
  using CPUSuperclass = TParentImageFilter;
  using GPUSuperclass = CudaInPlaceImageFilter<TInputImage, TOutputImage>;
  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaUnaryFunctorImageFilter, CudaInPlaceImageFilter);

  /** Some type alias. */
  using FunctorType = TFunction;

  using InputImageType = TInputImage;
  using InputImagePointer = typename InputImageType::ConstPointer;
  using InputImageRegionType = typename InputImageType::RegionType;
  using InputImagePixelType = typename InputImageType::PixelType;

  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using OutputImageRegionType = typename OutputImageType::RegionType;
  using OutputImagePixelType = typename OutputImageType::PixelType;

  FunctorType &
  GetFunctor()
  {
    return m_Functor;
  }
  const FunctorType &
  GetFunctor() const
  {
    return m_Functor;
  }

  /** Set the functor object. */
  void
  SetFunctor(const FunctorType & functor)
  {
    if (m_Functor != functor)
    {
      m_Functor = functor;
      this->Modified();
    }
  }

protected:
  CudaUnaryFunctorImageFilter() {}
  virtual ~CudaUnaryFunctorImageFilter() {}

  virtual void
  GenerateOutputInformation();

  virtual void
  GPUGenerateData();

  /** Cuda kernel handle is defined here instead of in the child class
   * because GPUGenerateData() in this base class is used. */
  int m_UnaryFunctorImageFilterCudaKernelHandle;

private:
  CudaUnaryFunctorImageFilter(const Self &); // purposely not implemented
  void
  operator=(const Self &); // purposely not implemented

  FunctorType m_Functor;
};

} // end of namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkCudaUnaryFunctorImageFilter.hxx"
#endif

#endif
