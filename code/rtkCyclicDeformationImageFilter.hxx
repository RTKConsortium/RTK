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

#ifndef rtkCyclicDeformationImageFilter_hxx
#define rtkCyclicDeformationImageFilter_hxx

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIterator.h>

#include <fstream>

namespace rtk
{

template <class TOutputImage>
void
CyclicDeformationImageFilter<TOutputImage>
::GenerateOutputInformation()
{
  typename OutputImageType::PointType   origin;
  typename OutputImageType::SpacingType spacing;
  OutputImageRegionType                 region;
  for(unsigned int i=0; i<OutputImageType::ImageDimension; i++)
    {
    origin[i] = this->GetInput()->GetOrigin()[i];
    spacing[i] = this->GetInput()->GetSpacing()[i];
    region.SetIndex(i, this->GetInput()->GetLargestPossibleRegion().GetIndex(i));
    region.SetSize(i, this->GetInput()->GetLargestPossibleRegion().GetSize(i));
    }
  this->GetOutput()->SetOrigin( origin );
  this->GetOutput()->SetSpacing( spacing );
  this->GetOutput()->SetLargestPossibleRegion( region );
}

template <class TOutputImage>
void
CyclicDeformationImageFilter<TOutputImage>
::GenerateInputRequestedRegion()
{
  typename InputImageType::Pointer inputPtr = const_cast< InputImageType * >( this->GetInput() );
  if ( !inputPtr )
    return;
  inputPtr->SetRequestedRegion( inputPtr->GetLargestPossibleRegion() );
}

template <class TOutputImage>
void
CyclicDeformationImageFilter<TOutputImage>
::BeforeThreadedGenerateData()
{
  unsigned int nframe = this->GetInput()->GetLargestPossibleRegion().GetSize(OutputImageType::ImageDimension);
  if( this->GetFrame() > m_Signal.size() )
    itkGenericExceptionMacro(<< "Frame number #"
                             << this->GetFrame()
                             << " is larger than phase signal which has size "
                             << m_SignalFilename);

  double sigValue = m_Signal[this->GetFrame()];
  if( sigValue<0. || sigValue >=1. )
    itkGenericExceptionMacro(<< "Signal value #"
                             << this->GetFrame()
                             << " is " << sigValue
                             << " which is not in [0,1)");

  sigValue *= nframe;
  m_FrameInf = itk::Math::Floor<unsigned int, double>(sigValue);
  m_FrameSup = itk::Math::Floor<unsigned int, double>(sigValue + 1.);
  m_WeightInf = m_FrameSup - sigValue;
  m_WeightSup = sigValue - m_FrameInf;
  m_FrameInf = m_FrameInf % nframe;
  m_FrameSup = m_FrameSup % nframe;
}

template <class TOutputImage>
void
CyclicDeformationImageFilter<TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread,
                       ThreadIdType itkNotUsed(threadId) )
{
  // Prepare inferior input iterator
  typename InputImageType::RegionType inputRegionForThreadInf;
  inputRegionForThreadInf = this->GetInput()->GetLargestPossibleRegion();
  for(unsigned int i=0; i<OutputImageType::ImageDimension; i++)
    {
    inputRegionForThreadInf.SetIndex(i, outputRegionForThread.GetIndex(i));
    inputRegionForThreadInf.SetSize(i, outputRegionForThread.GetSize(i));
    }
  inputRegionForThreadInf.SetSize(OutputImageType::ImageDimension, 1);
  inputRegionForThreadInf.SetIndex(OutputImageType::ImageDimension, m_FrameInf);
  typename itk::ImageRegionConstIterator<InputImageType> itInf(this->GetInput(), inputRegionForThreadInf);

  // Prepare superior input iterator
  typename InputImageType::RegionType inputRegionForThreadSup = inputRegionForThreadInf;
  inputRegionForThreadSup.SetIndex(OutputImageType::ImageDimension, m_FrameSup);
  typename itk::ImageRegionConstIterator<InputImageType> itSup(this->GetInput(), inputRegionForThreadSup);

  // Output iterator
  itk::ImageRegionIterator<OutputImageType> itOut(this->GetOutput(), outputRegionForThread);
  while( !itOut.IsAtEnd() )
    {
    itOut.Set(itInf.Get()*m_WeightInf + itSup.Get()*m_WeightSup);
    ++itOut;
    ++itInf;
    ++itSup;
    }
}

template <class TOutputImage>
void
CyclicDeformationImageFilter<TOutputImage>
::SetSignalFilename (const std::string _arg)
{
  itkDebugMacro("setting SignalFilename to " << _arg);
  if ( this->m_SignalFilename != _arg )
    {
    this->m_SignalFilename = _arg;
    this->Modified();

    std::ifstream is( _arg.c_str() );
    if( !is.is_open() )
      {
      itkGenericExceptionMacro(<< "Could not open signal file " << m_SignalFilename);
      }

    double value;
    while( !is.eof() )
      {
      is >> value;
      m_Signal.push_back(value);
      }
    }
}

template <class TOutputImage>
void
CyclicDeformationImageFilter<TOutputImage>
::SetSignalVector (std::vector<double> _arg)
{
  if ( m_Signal != _arg )
    {
    m_Signal = _arg;
    this->Modified();
    }
}

} // end namespace rtk

#endif
