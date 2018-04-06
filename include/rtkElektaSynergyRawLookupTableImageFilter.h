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

#ifndef rtkElektaSynergyRawLookupTableImageFilter_h
#define rtkElektaSynergyRawLookupTableImageFilter_h

#include "rtkLookupTableImageFilter.h"
#include <itkNumericTraits.h>

namespace rtk
{

/** \class ElektaSynergyRawLookupTableImageFilter
 * \brief First part of ElektaSynergyLookupTableImageFilter.
 *
 * The lookup table has been split in two to allow application of the scatter
 * correction algorithm in between. The first part only set values to have max
 * value in air.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template < class TInputImage  = itk::Image<unsigned short, 2>,
           class TOutputImage = itk::Image<unsigned short, 2> >
class ITK_EXPORT ElektaSynergyRawLookupTableImageFilter :
    public LookupTableImageFilter< TInputImage, TOutputImage >
{

public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawLookupTableImageFilter Self;
  typedef LookupTableImageFilter< TInputImage,
                                  TOutputImage > Superclass;
  typedef itk::SmartPointer<Self>                Pointer;
  typedef itk::SmartPointer<const Self>          ConstPointer;

  typedef typename TInputImage::PixelType        InputImagePixelType;
  typedef typename TOutputImage::PixelType       OutputImagePixelType;
  typedef typename Superclass::FunctorType::LookupTableType LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawLookupTableImageFilter, LookupTableImageFilter);

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro( SameTypeCheck,
                   ( itk::Concept::SameType< InputImagePixelType, unsigned short > ) );
#endif

protected:
  ElektaSynergyRawLookupTableImageFilter();
  ~ElektaSynergyRawLookupTableImageFilter() {}

private:
  ElektaSynergyRawLookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                         //purposely not implemented

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkElektaSynergyRawLookupTableImageFilter.hxx"
#endif

#endif
