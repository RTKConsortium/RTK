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

#ifndef rtkNesterovUpdateImageFilter_h
#define rtkNesterovUpdateImageFilter_h

#include "itkInPlaceImageFilter.h"

namespace rtk
{

/** \class NesterovUpdateImageFilter
 * \brief Applies Nesterov's momentum technique
 *
 * NesterovUpdateImageFilter implements Nesterov's momentum technique
 * in order to accelerate the convergence rate of Newton's method, or
 * other optimization algorithms. The first input is the current iterate,
 * the second input is the product of the inverse hessian matrix
 * by the gradient vector (the Newton's update, before applying a minus sign)
 *
*/

template< typename TImage>
class NesterovUpdateImageFilter : public itk::InPlaceImageFilter< TImage,  TImage>
{
public:
   
  /** Standard class typedefs. */
  typedef NesterovUpdateImageFilter                        Self;
  typedef itk::InPlaceImageFilter< TImage, TImage>         Superclass;
  typedef itk::SmartPointer< Self >                        Pointer;

  /** Convenient typedef */
  typedef typename Superclass::OutputImageType::RegionType OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(NesterovUpdateImageFilter, itk::InPlaceImageFilter)

  /** Get and Set macro*/
  itkGetMacro(NumberOfIterations, int)
  itkSetMacro(NumberOfIterations, int)

  /** Method to reset the internal counter of iterations
   * (Nesterov's method uses coefficients that change at each iteration) */
  void ResetIterations();

protected:
  NesterovUpdateImageFilter();
  virtual ~NesterovUpdateImageFilter() ITK_OVERRIDE {}

  /** Does the real work. */
#if ITK_VERSION_MAJOR<5
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;
#else
  void DynamicThreadedGenerateData(const OutputImageRegionType& outputRegionForThread) ITK_OVERRIDE;
#endif
  void AfterThreadedGenerateData() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  int                 m_NumberOfIterations;
  int                 m_CurrentIteration;
  bool                m_MustInitializeIntermediateImages;
  std::vector<typename TImage::PixelType::ValueType> m_tCoeffs;
  std::vector<typename TImage::PixelType::ValueType> m_Sums;
  std::vector<typename TImage::PixelType::ValueType> m_Ratios;

  // Internal images
  typename TImage::Pointer m_Vk;
  typename TImage::Pointer m_Zk;

private:
  NesterovUpdateImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkNesterovUpdateImageFilter.hxx"
#endif

#endif
