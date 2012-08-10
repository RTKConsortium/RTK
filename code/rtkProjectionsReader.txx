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

#ifndef __rtkProjectionsReader_txx
#define __rtkProjectionsReader_txx

// ITK
#include <itkImageSeriesReader.h>
#include <itkConfigure.h>

// Varian Obi includes
#include "rtkHndImageIOFactory.h"
#include "rtkVarianObiRawImageFilter.h"

// Elekta Synergy includes
#include "rtkHisImageIOFactory.h"
#include "rtkElektaSynergyRawToAttenuationImageFilter.h"

// Tiff includes
#include "rtkTiffLookupTableImageFilter.h"

namespace rtk
{

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  os << indent << "RawDataReader: " << m_RawDataReader->GetNameOfClass() << std::endl;
  os << indent << "RawToProjectionsFilter: " << m_RawToProjectionsFilter->GetNameOfClass() << std::endl;
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::GenerateOutputInformation(void)
{
  if (m_FileNames.size() == 0)
    return;

  static bool firstTime = true;
  if(firstTime)
    {
    rtk::HndImageIOFactory::RegisterOneFactory();
    rtk::HisImageIOFactory::RegisterOneFactory();
#if ITK_VERSION_MAJOR <= 3
    itk::ImageIOFactory::RegisterBuiltInFactories();
#endif
    firstTime = false;
    }
  itk::ImageIOBase::Pointer imageIO = itk::ImageIOFactory::CreateImageIO( m_FileNames[0].c_str(), itk::ImageIOFactory::ReadMode );

  if(m_ImageIO != imageIO)
    {
    // In this block, we create a specific pipe depending on the type
    if( !strcmp(imageIO->GetNameOfClass(), "HndImageIO") )
      {
      /////////// Varian OBI
      typedef unsigned int                                       InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;

      // Convert raw to Projections
      typedef rtk::VarianObiRawImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      rawFilter->SetInput( reader->GetOutput() );
      m_RawToProjectionsFilter = rawFilter;
      }
    else if( !strcmp(imageIO->GetNameOfClass(), "HisImageIO") )
      {
      /////////// Elekta synergy
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;

      // Convert raw to Projections
      typedef rtk::ElektaSynergyRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      rawFilter->SetInput( reader->GetOutput() );
      m_RawToProjectionsFilter = rawFilter;
      }
    else if( !strcmp(imageIO->GetNameOfClass(), "TIFFImageIO") )
      {
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;

      // Convert raw to Projections
      typedef rtk::TiffLookupTableImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      rawFilter->SetInput( reader->GetOutput() );
      m_RawToProjectionsFilter = rawFilter;
      }
    else
      {
      ///////////// Default: whatever the format, we assume that we directly
      // read the Projections
      typedef itk::ImageSeriesReader< OutputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;
      m_RawToProjectionsFilter = reader;
      }

    //Store imageIO to avoid creating the pipe more than necessary
    m_ImageIO = imageIO;
    }

  // Set output information as provided by the pipe
  m_RawToProjectionsFilter->UpdateOutputInformation();
  TOutputImage * output = this->GetOutput();
  output->SetOrigin( m_RawToProjectionsFilter->GetOutput()->GetOrigin() );
  output->SetSpacing( m_RawToProjectionsFilter->GetOutput()->GetSpacing() );
  output->SetDirection( m_RawToProjectionsFilter->GetOutput()->GetDirection() );
  output->SetLargestPossibleRegion( m_RawToProjectionsFilter->GetOutput()->GetLargestPossibleRegion() );
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::GenerateData()
{
  TOutputImage * output = this->GetOutput();

  m_RawToProjectionsFilter->GetOutput()->SetRequestedRegion( output->GetRequestedRegion() );
  m_RawToProjectionsFilter->Update();
  this->GraftOutput( m_RawToProjectionsFilter->GetOutput() );
}

} //namespace rtk

#endif
