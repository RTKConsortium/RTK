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

#ifndef __rtkLUTbasedVariableI0RawToAttenuationImageFilter_txx
#define __rtkLUTbasedVariableI0RawToAttenuationImageFilter_txx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

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
    it.Set( -log( double(it.GetIndex()[0]) ) );
    ++it;
    }

  // Default value for I0 is the numerical max
  m_I0 = size[0]-1;

  // Mini pipeline for creating the lut.
  m_AddLUTFilter = AddLUTFilterType::New();
  m_ClampLUTFilter = ClampLUTFilterType::New();
  m_AddLUTFilter->SetInput1(lut);
  m_AddLUTFilter->SetConstant2((OutputImagePixelType) log( std::max((double)m_I0, 1.) ) );
  m_ClampLUTFilter->SetInput( m_AddLUTFilter->GetOutput() );
  m_ClampLUTFilter->SetBounds(0., itk::NumericTraits<OutputImagePixelType>::max() );
  m_ClampLUTFilter->Update();

  // Set the lut to member and functor
  this->SetLookupTable( m_ClampLUTFilter->GetOutput() );
}

template <class TInputImage, class TOutputImage>
void
LUTbasedVariableI0RawToAttenuationImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  m_AddLUTFilter->SetConstant2((OutputImagePixelType) log( std::max((double)m_I0, 1.) ) );
  Superclass::BeforeThreadedGenerateData(); // Update the LUT
}

}

#endif
