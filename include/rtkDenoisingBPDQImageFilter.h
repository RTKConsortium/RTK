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

#ifndef rtkDenoisingBPDQImageFilter_h
#define rtkDenoisingBPDQImageFilter_h

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

#include <itkCastImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkInPlaceImageFilter.h>

namespace rtk
{
/** \class DenoisingBPDQImageFilter
 * \brief Base class for Basis Pursuit DeQuantization denoising filters
 *
 * \author Cyril Mory
 *
 * \ingroup IntensityImageFilters
 */

template< typename TOutputImage, typename TGradientImage>
class DenoisingBPDQImageFilter :
        public itk::InPlaceImageFilter< TOutputImage, TOutputImage >
{
public:

  /** Standard class typedefs. */
  typedef DenoisingBPDQImageFilter                              Self;
  typedef itk::InPlaceImageFilter< TOutputImage, TOutputImage>  Superclass;
  typedef itk::SmartPointer<Self>                               Pointer;
  typedef itk::SmartPointer<const Self>                         ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(DenoisingBPDQImageFilter, ImageToImageFilter)

  /** Sub filter type definitions */
  typedef ForwardDifferenceGradientImageFilter
            <TOutputImage,
             typename TOutputImage::ValueType,
             typename TOutputImage::ValueType,
             TGradientImage>                                                      GradientFilterType;
  typedef itk::MultiplyImageFilter<TOutputImage>                                  MultiplyFilterType;
  typedef itk::SubtractImageFilter<TOutputImage>                                  SubtractImageFilterType;
  typedef itk::SubtractImageFilter<TGradientImage>                                SubtractGradientFilterType;
  typedef itk::InPlaceImageFilter<TGradientImage>                                 ThresholdFilterType;
  typedef BackwardDifferenceDivergenceImageFilter<TGradientImage, TOutputImage>   DivergenceFilterType;

  itkGetMacro(NumberOfIterations, int)
  itkSetMacro(NumberOfIterations, int)

  itkSetMacro(Gamma, double)
  itkGetMacro(Gamma, double)

protected:
  DenoisingBPDQImageFilter();
  ~DenoisingBPDQImageFilter() {}

  void GenerateData() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

  virtual ThresholdFilterType* GetThresholdFilter(){return ITK_NULLPTR;}

  /** Sub filter pointers */
  typename GradientFilterType::Pointer                  m_GradientFilter;
  typename MultiplyFilterType::Pointer                  m_MultiplyFilter;
  typename SubtractImageFilterType::Pointer             m_SubtractFilter;
  typename SubtractGradientFilterType::Pointer          m_SubtractGradientFilter;
  typename DivergenceFilterType::Pointer                m_DivergenceFilter;

  double m_Gamma;
  double m_Beta;
  double m_MinSpacing;
  int    m_NumberOfIterations;
  bool   m_DimensionsProcessed[TOutputImage::ImageDimension];

private:
  DenoisingBPDQImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  virtual void SetPipelineForFirstIteration();
  virtual void SetPipelineAfterFirstIteration();
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDenoisingBPDQImageFilter.hxx"
#endif

#endif //__rtkDenoisingBPDQImageFilter__
