/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFFTWRealToComplexConjugateImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2010-02-26 05:28:24 $
  Version:   $Revision: 1.12 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __rtkFFTWRealToComplexConjugateImageFilter_h
#define __rtkFFTWRealToComplexConjugateImageFilter_h

#include <itkFFTRealToComplexConjugateImageFilter.h>
#include "rtkFFTWCommon.h"

namespace rtk
{
/** \class FFTWRealToComplexConjugateImageFilter
 * \brief
 *
 * \ingroup
 */

#if ITK_VERSION_MAJOR >= 4
  template< class TInputImage, class TOutputImage=Image< std::complex<typename TInputImage::PixelType>, TInputImage::ImageDimension> >
  class ITK_EXPORT FFTWRealToComplexConjugateImageFilter :
    public FFTRealToComplexConjugateImageFilter<TInputImage,TOutputImage>
#else
  template <class TPixel, unsigned int VDimension = 3>
  class ITK_EXPORT FFTWRealToComplexConjugateImageFilter :
    public itk::FFTRealToComplexConjugateImageFilter<TPixel,VDimension>
#endif
{
public:
  typedef FFTWRealToComplexConjugateImageFilter                   Self;
#if ITK_VERSION_MAJOR >= 4
  typedef itk::FFTRealToComplexConjugateImageFilter<TInputImage,TOutputImage>  Superclass;
#else
  typedef itk:: FFTRealToComplexConjugateImageFilter<TPixel,VDimension> Superclass;
#endif
  typedef itk::SmartPointer<Self>                                      Pointer;
  typedef itk::SmartPointer<const Self>                                ConstPointer;

  /** Standard class typedefs. */
  typedef typename Superclass::TInputImageType  InputImageType;
  typedef typename InputImageType::PixelType    InputPixelType;
  typedef typename Superclass::TOutputImageType OutputImageType;

  /**
   * the proxy type is a wrapper for the fftw API
   * since the proxy is only defined over double and float,
   * trying to use any other pixel type is inoperative, as
   * is trying to use double if only the float FFTW version is
   * configured in, or float if only double is configured.
   */
  typedef typename fftw::Proxy<typename InputImageType::PixelType> FFTWProxyType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FFTWRealToComplexConjugateImageFilter,
               FFTRealToComplexConjugateImageFilter);

  //
  // these should be defined in every FFT filter class
  virtual void GenerateData();  // generates output from input

protected:
  FFTWRealToComplexConjugateImageFilter() {}
  ~FFTWRealToComplexConjugateImageFilter() {}

  virtual bool FullMatrix();

  virtual void UpdateOutputData(itk::DataObject *output);

private:
   //purposely not implemented
  FFTWRealToComplexConjugateImageFilter(const Self&);
  void operator=(const Self&);

  bool m_CanUseDestructiveAlgorithm;

};
} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFFTWRealToComplexConjugateImageFilter.txx"
#endif

#endif //__rtkFFTWRealToComplexConjugateImageFilter_h
