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

#ifndef rtkConjugateGradientGetX_kPlusOneImageFilter_h
#define rtkConjugateGradientGetX_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

namespace rtk
{
template< typename TInputImage>
class ConjugateGradientGetX_kPlusOneImageFilter : public itk::ImageToImageFilter< TInputImage, TInputImage>
{
public:
    
  /** Standard class typedefs. */
  typedef ConjugateGradientGetX_kPlusOneImageFilter          Self;
  typedef itk::ImageToImageFilter< TInputImage, TInputImage> Superclass;
  typedef itk::SmartPointer< Self >                          Pointer;
  typedef typename TInputImage::RegionType                   OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConjugateGradientGetX_kPlusOneImageFilter, itk::ImageToImageFilter)

  /** Functions to set the inputs */
  void SetXk(const TInputImage* Xk);
  void SetPk(const TInputImage* Pk);

  itkGetMacro(Alphak, float)
  itkSetMacro(Alphak, float)

  /** Typedefs for sub filters */
  typedef itk::AddImageFilter<TInputImage>      AddFilterType;
  typedef itk::MultiplyImageFilter<TInputImage> MultiplyFilterType;

protected:
  ConjugateGradientGetX_kPlusOneImageFilter();
  ~ConjugateGradientGetX_kPlusOneImageFilter() {}

  typename TInputImage::Pointer GetXk();
  typename TInputImage::Pointer GetPk();

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  void GenerateOutputInformation() ITK_OVERRIDE;
  
private:
  ConjugateGradientGetX_kPlusOneImageFilter(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented
  float m_Alphak;

  /** Pointers to sub filters */
  typename AddFilterType::Pointer       m_AddFilter;
  typename MultiplyFilterType::Pointer  m_MultiplyFilter;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientGetX_kPlusOneImageFilter.hxx"
#endif

#endif
