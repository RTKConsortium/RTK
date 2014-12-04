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
#include <itkMetaDataObject.h>
#include <itkGDCMImageIO.h>

// RTK
#include "rtkIOFactories.h"

// Varian Obi includes
#include "rtkHndImageIOFactory.h"
#include "rtkVarianObiRawImageFilter.h"

// Elekta Synergy includes
#include "rtkHisImageIOFactory.h"
#include "rtkElektaSynergyRawToAttenuationImageFilter.h"

// ImagX includes
#include "rtkImagXImageIOFactory.h"
#include "rtkImagXRawToAttenuationImageFilter.h"

// Tiff includes
#include "rtkTiffLookupTableImageFilter.h"

// European Synchrotron Radiation Facility
#include "rtkEdfImageIOFactory.h"
#include "rtkEdfRawToAttenuationImageFilter.h"

// Xrad small animal scanner
#include "rtkXRadImageIOFactory.h"
#include "rtkXRadRawToAttenuationImageFilter.h"

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
    rtk::RegisterIOFactories();

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
    else if( !strcmp(imageIO->GetNameOfClass(), "ImagXImageIO") )
      {
      /////////// ImagX
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;

      // Convert raw to Projections
      typedef rtk::ImagXRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
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
    else if( !strcmp(imageIO->GetNameOfClass(), "EdfImageIO") )
      {
      /////////// ESRF
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;

      // Convert raw to Projections
      typedef rtk::EdfRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      rawFilter->SetInput( reader->GetOutput() );
      rawFilter->SetFileNames( this->GetFileNames() );
      m_RawToProjectionsFilter = rawFilter;
      }
    else if( !strcmp(imageIO->GetNameOfClass(), "XRadImageIO") )
      {
      /////////// XRad
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;

      // Convert raw to Projections
      typedef rtk::XRadRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      rawFilter->SetInput( reader->GetOutput() );
      m_RawToProjectionsFilter = rawFilter;
      }
    else
      {
      ///////////// Default: whatever the format, we assume that we directly
      // read the Projections
      std::string tagkey, labelId, value;                 // Tag for Manufacturer's name
      if (m_FileNames[0].find(".dcm") != std::string::npos) // if dicom image case
        {
        imageIO = itk::GDCMImageIO::New();
        // Reading manufacturer's name (iMagX case)
        tagkey = "0008|0070";
        itk::GDCMImageIO::GetLabelFromTag( tagkey, labelId );
        }

      typedef itk::ImageSeriesReader< OutputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      reader->SetImageIO( imageIO );
      reader->SetFileNames( this->GetFileNames() );
      m_RawDataReader = reader;
      m_RawToProjectionsFilter = reader;

      if(dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer()))
        {
        // Read just first projection
        std::vector<std::string> firstProj;
        firstProj.push_back(m_FileNames[0]);
        reader->SetFileNames( firstProj );
        // Necessary Update() to have access to tag value
        reader->Update();

        dynamic_cast<itk::GDCMImageIO*>(imageIO.GetPointer())->GetValueFromTag(tagkey, value);
        if (value=="IBA ")
          {
          // Reading all iMagX projections
          typedef unsigned short                                     InputPixelType;
          typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

          typedef itk::ImageSeriesReader< InputImageType > ReaderType;
          typename ReaderType::Pointer reader = ReaderType::New();
          reader->SetImageIO( imageIO );
          reader->SetFileNames( this->GetFileNames() );
          m_RawDataReader = reader;

          // Convert raw to Projections
          typedef rtk::ImagXRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
          typename RawFilterType::Pointer rawFilter = RawFilterType::New();
          rawFilter->SetInput( reader->GetOutput() );
          rawFilter->UpdateOutputInformation();
          // Calculate Origin
          float originX = (reader->GetOutput()->GetLargestPossibleRegion().GetSize()[0]-1)*(-0.5)*reader->GetOutput()->GetSpacing()[0];
          float originY = (reader->GetOutput()->GetLargestPossibleRegion().GetSize()[1]-1)*(-0.5)*reader->GetOutput()->GetSpacing()[1];
          // Set Origin
          typename OutputImageType::PointType origin;
          origin[0] = originX;
          origin[1] = originY;
          origin[2] = 0.f;
          rawFilter->GetOutput()->SetOrigin(origin);
          m_RawToProjectionsFilter = rawFilter;
          }
        else // Other manufacturers
          // Reading all projections
          reader->SetFileNames( this->GetFileNames() );
        }
      }
    //Store imageIO to avoid creating the pipe more than necessary
    m_ImageIO = imageIO;
  }
  // Release output data of m_RawDataReader if conversion occurs
  if ( m_RawDataReader != m_RawToProjectionsFilter )
    m_RawDataReader->ReleaseDataFlagOn();

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
