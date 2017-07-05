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

#ifndef rtkConjugateGradientGetP_kPlusOneImageFilter_h
#define rtkConjugateGradientGetP_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

namespace rtk
{
template< typename TInputImage>
class ConjugateGradientGetP_kPlusOneImageFilter : public itk::ImageToImageFilter< TInputImage, TInputImage>
{
public:
    
  /** Standard class typedefs. */
  typedef ConjugateGradientGetP_kPlusOneImageFilter           Self;
  typedef itk::ImageToImageFilter< TInputImage, TInputImage>  Superclass;
  typedef itk::SmartPointer< Self >                           Pointer;
  typedef typename TInputImage::RegionType                    OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientGetP_kPlusOneImageFilter, itk::ImageToImageFilter)

  /** Functions to set the inputs */
  void SetR_kPlusOne(const TInputImage* R_kPlusOne);
  void SetRk(const TInputImage* Rk);
  void SetPk(const TInputImage* Pk);

  itkSetMacro(SquaredNormR_k, float)
  itkSetMacro(SquaredNormR_kPlusOne, float)

  /** Typedefs for sub filters */
  typedef itk::AddImageFilter<TInputImage>      AddFilterType;
  typedef itk::MultiplyImageFilter<TInputImage> MultiplyFilterType;

protected:
  ConjugateGradientGetP_kPlusOneImageFilter();
  ~ConjugateGradientGetP_kPlusOneImageFilter() {}

  typename TInputImage::Pointer GetR_kPlusOne();
  typename TInputImage::Pointer GetRk();
  typename TInputImage::Pointer GetPk();

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;

private:
  ConjugateGradientGetP_kPlusOneImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

  float m_SquaredNormR_k;
  float m_SquaredNormR_kPlusOne;
  float m_Betak;

  /** Pointers to sub filters */
  typename AddFilterType::Pointer       m_AddFilter;
  typename MultiplyFilterType::Pointer  m_MultiplyFilter;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientGetP_kPlusOneImageFilter.hxx"
#endif

#endif
