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

#ifndef rtkEdfRawToAttenuationImageFilter_hxx
#define rtkEdfRawToAttenuationImageFilter_hxx

#include <itkImageFileWriter.h>
#include <itksys/SystemTools.hxx>
#include <itkRegularExpressionSeriesFileNames.h>

#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
EdfRawToAttenuationImageFilter<TInputImage, TOutputImage>
::EdfRawToAttenuationImageFilter() :
  m_DarkProjectionsReader( EdfImageSeries::New() ),
  m_ReferenceProjectionsReader( EdfImageSeries::New() )
{
}

template<class TInputImage, class TOutputImage>
void
EdfRawToAttenuationImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  if( m_FileNames.size() != this->GetInput()->GetLargestPossibleRegion().GetSize()[2] )
    {
    itkGenericExceptionMacro(<< "Error, file names do not correspond to input");
    }

  std::string path = itksys::SystemTools::GetFilenamePath(m_FileNames[0]);
  std::vector<std::string> pathComponents;
  itksys::SystemTools::SplitPath(m_FileNames[0].c_str(), pathComponents);
  std::string fileName = pathComponents.back();

  // Reference images (flood field)
  itk::RegularExpressionSeriesFileNames::Pointer refNames = itk::RegularExpressionSeriesFileNames::New();
  refNames->SetDirectory(path.c_str() );
  refNames->SetNumericSort(false);
  refNames->SetRegularExpression("refHST[0-9]*.edf$");
  refNames->SetSubMatch(0);

  m_ReferenceProjectionsReader->SetFileNames( refNames->GetFileNames() );
  m_ReferenceProjectionsReader->Update();

  m_ReferenceIndices.clear();
  for(unsigned int i=0; i<refNames->GetFileNames().size(); i++)
    {
    const std::string            name = refNames->GetFileNames()[i];
    const std::string::size_type nameSize = name.size();
    std::string                  indexStr(name, nameSize-8, 4);
    m_ReferenceIndices.push_back( atoi( indexStr.c_str() ) );
    }

  // Dark images
  FileNamesContainer darkFilenames;
  darkFilenames.push_back( path + std::string("/dark.edf") );
//  darkFilenames.push_back( path + std::string("/darkend0000.edf") );
  m_DarkProjectionsReader->SetFileNames( darkFilenames );
  m_DarkProjectionsReader->Update();
}

template<class TInputImage, class TOutputImage>
void
EdfRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  // Dark image iterator
  OutputImageRegionType darkRegion = outputRegionForThread;

  darkRegion.SetSize(2,1);
  darkRegion.SetIndex(2,0);
  itk::ImageRegionConstIterator<InputImageType> itDark(m_DarkProjectionsReader->GetOutput(), darkRegion);

  // Ref and projection regions
  OutputImageRegionType refRegion1 = outputRegionForThread;
  OutputImageRegionType refRegion2 = outputRegionForThread;
  OutputImageRegionType outputRegionSlice = outputRegionForThread;
  refRegion1.SetSize(2,1);
  refRegion2.SetSize(2,1);
  outputRegionSlice.SetSize(2,1);

  for(int k = outputRegionForThread.GetIndex(2);
      k < outputRegionForThread.GetIndex(2) + (int)outputRegionForThread.GetSize(2);
      k++)
    {
    outputRegionSlice.SetIndex(2,k);

    // Create iterators
    itk::ImageRegionConstIterator<InputImageType> itIn(this->GetInput(), outputRegionSlice);
    itk::ImageRegionIterator<OutputImageType>     itOut(this->GetOutput(), outputRegionSlice);

    // Find index of reference images
    for(unsigned int i=0; i<m_ReferenceIndices.size(); i++)
      {
      itDark.GoToBegin();
      if(k == m_ReferenceIndices[i])
        {
        refRegion1.SetIndex(2, i);
        itk::ImageRegionConstIterator<InputImageType> itRef(m_ReferenceProjectionsReader->GetOutput(), refRegion1);
        while( !itDark.IsAtEnd() )
          {
          // The reference has been exactly acquired at the same position
          itOut.Set( -log( ( (double)itIn.Get()  - (double)itDark.Get() ) /
                           ( (double)itRef.Get() - (double)itDark.Get() ) ) );
          ++itIn;
          ++itOut;
          ++itDark;
          ++itRef;
          }
        }
      else if(i>0 && k>m_ReferenceIndices[i-1] && k<m_ReferenceIndices[i])
        {
        // The reference must be interpolated
        refRegion1.SetIndex(2, i-1);
        refRegion2.SetIndex(2, i);
        itk::ImageRegionConstIterator<InputImageType> itRef1(m_ReferenceProjectionsReader->GetOutput(), refRegion1);
        itk::ImageRegionConstIterator<InputImageType> itRef2(m_ReferenceProjectionsReader->GetOutput(), refRegion2);

        double w1 = 1./(m_ReferenceIndices[i]-m_ReferenceIndices[i-1]);
        double w2 = w1 * (k - m_ReferenceIndices[i-1]);

        w1 *= (m_ReferenceIndices[i] - k);
        while( !itDark.IsAtEnd() )
          {
          double ref = w1*itRef1.Get() + w2*itRef2.Get();
          itOut.Set( -log( ( (double)itIn.Get()  - (double)itDark.Get() ) /
                           (        ref          - (double)itDark.Get() ) ) );
          ++itIn;
          ++itOut;
          ++itDark;
          ++itRef1;
          ++itRef2;
          }
        }
      }
    }
}

} // end namespace rtk

#endif
