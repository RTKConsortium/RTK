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

#ifndef rtkLUTbasedVariableI0RawToAttenuationImageFilter_hxx
#define rtkLUTbasedVariableI0RawToAttenuationImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>
#include "rtkI0EstimationProjectionFilter.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
LUTbasedVariableI0RawToAttenuationImageFilter<TInputImage, TOutputImage>
::LUTbasedVariableI0RawToAttenuationImageFilter()
{
  // Create the lut
  typename LookupTableType::Pointer lut = LookupTableType::New();
  typename LookupTableType::SizeType size;
  size[0] = itk::NumericTraits<InputImagePixelType>::max()-itk::NumericTraits<InputImagePixelType>::NonpositiveMin()+1;
  lut->SetRegions( size );
  lut->Allocate();

  // Iterate and set lut
  itk::ImageRegionIteratorWithIndex<LookupTableType> it( lut, lut->GetBufferedRegion() );
  it.Set(0);
  ++it;
  while( !it.IsAtEnd() )
    {
    it.Set( it.GetIndex()[0] );
    ++it;
    }

  // Default value for I0 is the numerical max
  m_I0 = size[0]-1;
  m_IDark = 0;

  // Mini pipeline for creating the lut.
  m_SubtractRampFilter = SubtractLUTFilterType::New();
  m_SubtractLUTFilter = SubtractLUTFilterType::New();
  m_ThresholdRampFilter = ThresholdLUTFilterType::New();
  m_LogRampFilter = LogLUTFilterType::New();

  m_SubtractRampFilter->SetInput1(lut);
  m_SubtractRampFilter->SetConstant2(m_IDark);
  m_SubtractRampFilter->InPlaceOff();
  m_ThresholdRampFilter->SetInput(m_SubtractRampFilter->GetOutput());
  m_ThresholdRampFilter->ThresholdBelow(1.);
  m_ThresholdRampFilter->SetOutsideValue(1.);
  m_LogRampFilter->SetInput(m_ThresholdRampFilter->GetOutput());

  m_SubtractLUTFilter->SetConstant1((OutputImagePixelType) log( std::max(m_I0-m_IDark, 1.) ) );
  m_SubtractLUTFilter->SetInput2(m_LogRampFilter->GetOutput());

  // Set the lut to member and functor
  this->SetLookupTable( m_SubtractLUTFilter->GetOutput() );
}

template <class TInputImage, class TOutputImage>
void
LUTbasedVariableI0RawToAttenuationImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  typedef rtk::I0EstimationProjectionFilter<TInputImage> I0EstimationType;
  I0EstimationType * i0est = dynamic_cast<I0EstimationType*>( this->GetInput()->GetSource().GetPointer() );
  if(i0est)
    {
    m_SubtractLUTFilter->SetConstant1((OutputImagePixelType) log( std::max((double)i0est->GetI0()-m_IDark, 1.) ) );
    }
  else
    {
    m_SubtractLUTFilter->SetConstant1((OutputImagePixelType) log( std::max(m_I0-m_IDark, 1.) ) );
    }
  m_SubtractRampFilter->SetInput2(m_IDark);

  Superclass::BeforeThreadedGenerateData(); // Update the LUT
}

}

#endif
