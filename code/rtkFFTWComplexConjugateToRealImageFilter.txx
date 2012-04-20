/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFFTWComplexConjugateToRealImageFilter.txx,v $
  Language:  C++
  Date:      $Date: 2010-02-26 23:50:55 $
  Version:   $Revision: 1.15 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __rtkFFTWComplexConjugateToRealImageFilter_txx
#define __rtkFFTWComplexConjugateToRealImageFilter_txx

#include "rtkFFTWComplexConjugateToRealImageFilter.h"
#if ITK_VERSION_MAJOR <= 3
#  include "itkFFTComplexConjugateToRealImageFilter.txx"
#else
#  include "itkFFTComplexConjugateToRealImageFilter.hxx"
#endif
#include <iostream>
#include <itkIndent.h>
#include <itkMetaDataObject.h>
#include <itkImageRegionIterator.h>
#include <itkProgressReporter.h>

namespace rtk
{

#if ITK_VERSION_MAJOR >= 4
  template< class TInputImage, class TOutputImage >
  void
  FFTWComplexConjugateToRealImageFilter<TInputImage,TOutputImage>::
#else
  template <typename TPixel, unsigned int VDimension>
  void
  FFTWComplexConjugateToRealImageFilter<TPixel,VDimension>::
#endif
BeforeThreadedGenerateData()
{
  unsigned int ImageDimension = InputImageType::ImageDimension;

  // get pointers to the input and output
  typename InputImageType::ConstPointer  inputPtr  = this->GetInput();
  typename OutputImageType::Pointer      outputPtr = this->GetOutput();

  if ( !inputPtr || !outputPtr )
    {
    return;
    }

  // we don't have a nice progress to report, but at least this simple line
  // reports the begining and the end of the process
  itk::ProgressReporter progress(this, 0, 1);

  // allocate output buffer memory
  outputPtr->SetBufferedRegion( outputPtr->GetRequestedRegion() );
  outputPtr->Allocate();

  const typename InputImageType::SizeType&   outputSize
    = outputPtr->GetLargestPossibleRegion().GetSize();
  const typename OutputImageType::SizeType& inputSize
    = inputPtr->GetLargestPossibleRegion().GetSize();

  // figure out sizes
  // size of input and output aren't the same which is handled in the
  // superclass, sort of.
  // the input size and output size only differ in the fastest moving dimension
  unsigned int total_outputSize = 1;
  unsigned int total_inputSize = 1;

  for(unsigned i = 0; i <ImageDimension; i++)
    {
    total_outputSize *= outputSize[i];
    total_inputSize *= inputSize[i];
    }

  typename FFTWProxyType::ComplexType * in;
  // complex to real transform don't have any algorithm which support the
  // FFTW_PRESERVE_INPUT at this time. So if the input can't be destroyed,
  // we have to copy the input data to a buffer before running the ifft.
  if( m_CanUseDestructiveAlgorithm )
    {
    // ok, so lets use the input buffer directly, to save some memory
    in = (typename FFTWProxyType::ComplexType*)inputPtr->GetBufferPointer();
    }
  else
    {
    // we must use a buffer where fftw can work and destroy what it wants
    in = new typename FFTWProxyType::ComplexType[total_inputSize];
    // no need to copy the data after the plan creation: FFTW_ESTIMATE ensure
    // that the input in not destroyed during this step
    memcpy(in,
           inputPtr->GetBufferPointer(),
           total_inputSize * sizeof(typename FFTWProxyType::ComplexType) );
    }
  OutputPixelType * out = outputPtr->GetBufferPointer();
  typename FFTWProxyType::PlanType plan;

  switch(ImageDimension)
    {
    case 1:
      plan = FFTWProxyType::Plan_dft_c2r_1d(outputSize[0],
                                            in,
                                            out,
                                            FFTW_ESTIMATE,
                                            this->GetNumberOfThreads() );
      break;
    case 2:
      plan = FFTWProxyType::Plan_dft_c2r_2d(outputSize[1],outputSize[0],
                                            in,
                                            out,
                                            FFTW_ESTIMATE,
                                            this->GetNumberOfThreads() );
      break;
    case 3:
      plan = FFTWProxyType::Plan_dft_c2r_3d(outputSize[2],outputSize[1],outputSize[0],
                                            in,
                                            out,
                                            FFTW_ESTIMATE,
                                            this->GetNumberOfThreads() );
      break;
    default:
      int *sizes = new int[ImageDimension];
      for(unsigned int i = 0; i < ImageDimension; i++)
        {
        sizes[(ImageDimension - 1) - i] = outputSize[i];
        }
      plan = FFTWProxyType::Plan_dft_c2r(ImageDimension,sizes,
                                         in,
                                         out,
                                         FFTW_ESTIMATE,
                                         this->GetNumberOfThreads() );
      delete [] sizes;
    }
  FFTWProxyType::Execute(plan);

  // some cleanup
  FFTWProxyType::DestroyPlan(plan);
  if( !m_CanUseDestructiveAlgorithm )
    {
    delete [] in;
    }
}

#if ITK_VERSION_MAJOR >= 4
  template< class TInputImage, class TOutputImage >
  void
  FFTWComplexConjugateToRealImageFilter<TInputImage,TOutputImage>::
#else
  template <typename TPixel, unsigned int VDimension>
  void
  FFTWComplexConjugateToRealImageFilter<TPixel,VDimension>::
#endif
ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId )
{
  typedef itk::ImageRegionIterator< OutputImageType > IteratorType;
  unsigned long total_outputSize = this->GetOutput()->GetRequestedRegion().GetNumberOfPixels();
  IteratorType  it(this->GetOutput(), outputRegionForThread);
  while( !it.IsAtEnd() )
    {
    it.Set( it.Value() / total_outputSize );
    ++it;
    }
}

#if ITK_VERSION_MAJOR >= 4
  template< class TInputImage, class TOutputImage >
  bool
  FFTWComplexConjugateToRealImageFilter<TInputImage,TOutputImage>::
#else
  template <typename TPixel,unsigned int VDimension>
  bool
  FFTWComplexConjugateToRealImageFilter<TPixel,VDimension>::
#endif
FullMatrix()
{
  return false;
}

#if ITK_VERSION_MAJOR >= 4
  template< class TInputImage, class TOutputImage >
  void
  FFTWComplexConjugateToRealImageFilter<TInputImage,TOutputImage>::
#else
  template <typename TPixel,unsigned int VDimension>
  void
  FFTWComplexConjugateToRealImageFilter<TPixel,VDimension>::
#endif
UpdateOutputData(itk::DataObject * output)
{
  // we need to catch that information now, because it is changed later
  // during the pipeline execution, and thus can't be grabbed in
  // GenerateData().
  m_CanUseDestructiveAlgorithm = this->GetInput()->GetReleaseDataFlag();
  Superclass::UpdateOutputData( output );
}

} // namespace rtk
#endif // _rtkFFTWComplexConjugateToRealImageFilter_txx
