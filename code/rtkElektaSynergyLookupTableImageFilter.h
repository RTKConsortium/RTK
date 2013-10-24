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

#ifndef __rtkElektaSynergyLookupTableImageFilter_h
#define __rtkElektaSynergyLookupTableImageFilter_h

#include "rtkLookupTableImageFilter.h"
#include <itkNumericTraits.h>

namespace rtk
{

/** \class ElektaSynergyLookupTableImageFilter
 * \brief Lookup table for Elekta Synergy data.
 *
 * The lookup table converts the raw values measured by the panel to the
 * logarithm of the value divided by the maximum numerical value. This could
 * be improved with a calibration of the air value.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template <class TOutputImage>
class ITK_EXPORT ElektaSynergyLookupTableImageFilter:
    public LookupTableImageFilter< itk::Image<unsigned short, TOutputImage::ImageDimension>,
                                   TOutputImage >
{

public:
  /** Standard class typedefs. */
  typedef ElektaSynergyLookupTableImageFilter                                 Self;
  typedef LookupTableImageFilter<itk::Image<unsigned short, 
                                            TOutputImage::ImageDimension>,
                                 TOutputImage>                                Superclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;

  typedef unsigned short                                    InputImagePixelType;
  typedef typename TOutputImage::PixelType                  OutputImagePixelType;
  typedef typename Superclass::FunctorType::LookupTableType LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyLookupTableImageFilter, LookupTableImageFilter);

protected:
  ElektaSynergyLookupTableImageFilter();
  virtual ~ElektaSynergyLookupTableImageFilter() {
  }

private:
  ElektaSynergyLookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);              //purposely not implemented

};

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
template <unsigned int VImageDimension = 2>
class ITK_EXPORT ElektaSynergyRawLookupTableImageFilter :
    public LookupTableImageFilter< itk::Image<unsigned short, VImageDimension>,
                                   itk::Image<unsigned short, VImageDimension> >
{

public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawLookupTableImageFilter                                Self;
  typedef LookupTableImageFilter< itk::Image<unsigned short, VImageDimension>,
                                  itk::Image<unsigned short, VImageDimension> > Superclass;
  typedef itk::SmartPointer<Self>                                 Pointer;
  typedef itk::SmartPointer<const Self>                           ConstPointer;

  typedef unsigned short                           InputImagePixelType;
  typedef unsigned short                           OutputImagePixelType;
  typedef typename Superclass::FunctorType::LookupTableType LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawLookupTableImageFilter, LookupTableImageFilter);

protected:
  ElektaSynergyRawLookupTableImageFilter();
  virtual ~ElektaSynergyRawLookupTableImageFilter() {
  }

private:
  ElektaSynergyRawLookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                         //purposely not implemented

};

/** \class ElektaSynergyLogLookupTableImageFilter
 * \brief Second part of ElektaSynergyLookupTableImageFilter.
 *
 * The lookup table has been split in two to allow application of the scatter
 * correction algorithm in between. This second part takes the log of the ratio
 * with the reference value times -1.
 *
 * \test rtkelektatest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup ImageToImageFilter
 */
template <class TOutputImage>
class ITK_EXPORT ElektaSynergyLogLookupTableImageFilter :
    public LookupTableImageFilter< itk::Image<unsigned short, TOutputImage::ImageDimension>,
                                   TOutputImage >
{

public:
  /** Standard class typedefs. */
  typedef ElektaSynergyLogLookupTableImageFilter                              Self;
  typedef LookupTableImageFilter<itk::Image<unsigned short,
                                            TOutputImage::ImageDimension>,
                                 TOutputImage>                                Superclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;

  typedef unsigned short                                    InputImagePixelType;
  typedef typename TOutputImage::PixelType                  OutputImagePixelType;
  typedef typename Superclass::FunctorType::LookupTableType LookupTableType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyLogLookupTableImageFilter, LookupTableImageFilter);

protected:
  ElektaSynergyLogLookupTableImageFilter();
  virtual ~ElektaSynergyLogLookupTableImageFilter() {
  }

private:
  ElektaSynergyLogLookupTableImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                         //purposely not implemented

};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkElektaSynergyLookupTableImageFilter.txx"
#endif

#endif
