/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFFTWComplexConjugateToRealImageFilter.h,v $
  Language:  C++
  Date:      $Date: 2009-05-12 17:26:20 $
  Version:   $Revision: 1.10 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __itkFFTWComplexConjugateToRealImageFilter_h
#define __itkFFTWComplexConjugateToRealImageFilter_h

#include "itkFFTComplexConjugateToRealImageFilter.h"

//
// FFTWCommon defines proxy classes based on data types
#include "itkFFTWCommon.h"


namespace itk
{

template <typename TPixel,unsigned int VDimension>
class ITK_EXPORT FFTWComplexConjugateToRealImageFilter :
    public FFTComplexConjugateToRealImageFilter<TPixel,VDimension>
{
public:
  typedef FFTWComplexConjugateToRealImageFilter                   Self;
  typedef FFTComplexConjugateToRealImageFilter<TPixel,VDimension> Superclass;
  typedef SmartPointer<Self>                                      Pointer;
  typedef SmartPointer<const Self>                                ConstPointer;
  //
  // the proxy type is a wrapper for the fftw API
  // since the proxy is only defined over double and float,
  // trying to use any other pixel type is inoperative, as
  // is trying to use double if only the float FFTW version is 
  // configured in, or float if only double is configured.
  //
  typedef typename fftw::Proxy<TPixel> FFTWProxyType;

  /** Standard class typedefs. */
  typedef typename Superclass::TInputImageType  TInputImageType;
  typedef typename Superclass::TOutputImageType TOutputImageType;
  typedef typename TOutputImageType::RegionType OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FFTWComplexConjugateToRealImageFilter,
               FFTComplexConjugateToRealImageFilter);

  /** Image type typedef support. */
  typedef TInputImageType              ImageType;
  typedef typename ImageType::SizeType ImageSizeType;

  //
  // these should be defined in every FFT filter class
  virtual bool FullMatrix();
protected:
  FFTWComplexConjugateToRealImageFilter()
    {
    }
  virtual ~FFTWComplexConjugateToRealImageFilter()
    {
    }

  virtual void UpdateOutputData(DataObject *output);
  
  virtual void BeforeThreadedGenerateData();
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId );
   
private:
  FFTWComplexConjugateToRealImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
  
  bool m_CanUseDestructiveAlgorithm;

};


} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFTWComplexConjugateToRealImageFilter.txx"
#endif

#endif //__itkFFTWComplexConjugateToRealImageFilter_h
