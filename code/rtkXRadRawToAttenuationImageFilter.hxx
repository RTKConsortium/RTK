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

#ifndef rtkXRadRawToAttenuationImageFilter_hxx
#define rtkXRadRawToAttenuationImageFilter_hxx

#include <itkImageFileReader.h>
#include <itkImageRegionIterator.h>
#include "rtkConfiguration.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
XRadRawToAttenuationImageFilter<TInputImage, TOutputImage>
::XRadRawToAttenuationImageFilter() :
  m_DarkImageFileName("dark.header"),
  m_FlatImageFileName("flat.header")
{
}

template<class TInputImage, class TOutputImage>
void
XRadRawToAttenuationImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  typedef itk::ImageFileReader<OutputImageType> ReaderType;
  typename ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(m_DarkImageFileName);
  reader->Update();
  m_DarkImage = reader->GetOutput();
  m_DarkImage->DisconnectPipeline();

  reader->SetFileName(m_FlatImageFileName);
  reader->Update();
  m_FlatImage = reader->GetOutput();
  m_FlatImage->DisconnectPipeline();
}

template<class TInputImage, class TOutputImage>
void
XRadRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  // Dark and flat image iterator
  OutputImageRegionType darkRegion = outputRegionForThread;
  darkRegion.SetSize(2,1);
  darkRegion.SetIndex(2,1);
  itk::ImageRegionConstIterator<OutputImageType> itDark(m_DarkImage, darkRegion);
  itk::ImageRegionConstIterator<OutputImageType> itFlat(m_FlatImage, darkRegion);

  // Projection regions
  OutputImageRegionType outputRegionSlice = outputRegionForThread;
  outputRegionSlice.SetSize(2,1);
  for(int k = outputRegionForThread.GetIndex(2);
          k < outputRegionForThread.GetIndex(2) + (int)outputRegionForThread.GetSize(2);
          k++)
    {
    outputRegionSlice.SetIndex(2,k);

    // Create iterators
    itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionSlice);
    itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionSlice);

    itDark.GoToBegin();
    itFlat.GoToBegin();
    while( !itIn.IsAtEnd() )
      {
      double den = itFlat.Get() - (double)itDark.Get();
      if(den==0.)
        itOut.Set(0.);
      else
        {
        double ratio = (itIn.Get()  - (double)itDark.Get()) / den;
        if(ratio<=0.)
          itOut.Set(0.);
        else
          itOut.Set( -log(ratio) );
        }
      ++itIn;
      ++itOut;
      ++itDark;
      ++itFlat;
      }
    }
}

} // end namespace rtk

#endif
