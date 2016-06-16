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

#ifndef __rtkVarianObiHncRawToAttenuationImageFilter_hxx
#define __rtkVarianObiHncRawToAttenuationImageFilter_hxx

#include <itkImageFileReader.h>
#include <itksys/SystemTools.hxx>

#include "rtkHncImageIOFactory.h"
#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
VarianObiHncRawToAttenuationImageFilter<TInputImage, TOutputImage>
::VarianObiHncRawToAttenuationImageFilter() :
  m_FloodImageFileName( "norm.hnc" ),
  m_ProjectionFileName( "./norm.hnc" )
{
}

template<class TInputImage, class TOutputImage>
void
VarianObiHncRawToAttenuationImageFilter<TInputImage, TOutputImage>
::BeforeThreadedGenerateData()
{
  std::string path = itksys::SystemTools::GetFilenamePath(m_ProjectionFileName);
  std::vector<std::string> pathComponents;
  itksys::SystemTools::SplitPath(m_ProjectionFileName.c_str(), pathComponents);
  //std::string fileName = pathComponents.back();

  pathComponents.pop_back();
  pathComponents.push_back(m_FloodImageFileName);
  std::string fullPathFileName = itksys::SystemTools::JoinPath(pathComponents);

  // Reference image (flood field)
  HncImageIOFactory::RegisterOneFactory();

  typedef itk::ImageFileReader< InputImageType > HncImageType;
  typename HncImageType::Pointer floodProjectionReader = HncImageType::New();

  floodProjectionReader->SetFileName( fullPathFileName );
  floodProjectionReader->Update();
  m_FloodImage = floodProjectionReader->GetOutput();
  m_FloodImage->DisconnectPipeline();
}

template<class TInputImage, class TOutputImage>
void
VarianObiHncRawToAttenuationImageFilter<TInputImage, TOutputImage>
::ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId) )
{
  // Flood image iterator
  OutputImageRegionType floodRegion = outputRegionForThread;

  floodRegion.SetSize(2,1);
  floodRegion.SetIndex(2,0);
  itk::ImageRegionConstIterator<InputImageType> itFlood(m_FloodImage, floodRegion);

  // Projection region
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

    itFlood.GoToBegin();
    while( !itFlood.IsAtEnd() )
      {
      // The reference has been exactly acquired at the same position
      if( !itIn.Get() )
      {
        itOut.Set(0.0F);
      }
      else
      {
        double val = vcl_log( static_cast<double>(itFlood.Get()) ) - vcl_log( static_cast<double>(itIn.Get()) );
        if(val > 0)
          itOut.Set(val);
          //itOut.Set(val * 10000.0F); JLu code applies a scaling factor of 10000.0F. Not sure why.
        else
          itOut.Set(0.0F);
      }
      ++itIn;
      ++itOut;
      ++itFlood;
      }
    }

}

} // end namespace rtk

#endif
