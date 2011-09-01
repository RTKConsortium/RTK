#ifndef __itkProjectionsReader_txx
#define __itkProjectionsReader_txx

// ITK
#include <itkImageSeriesReader.h>
#include <itkConfigure.h>

// Varian Obi includes
#include "itkHndImageIOFactory.h"
#include "itkVarianObiRawImageFilter.h"

// Elekta Synergy includes
#include "itkHisImageIOFactory.h"
#include "itkElektaSynergyRawToAttenuationImageFilter.h"

// Tiff includes
#include "itkTiffLutImageFilter.h"

namespace itk
{

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::PrintSelf(std::ostream& os, Indent indent) const
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
  if (m_FileNames.size()==0)
    return;

  static bool firstTime = true;
  if(firstTime)
    {
    itk::HndImageIOFactory::RegisterOneFactory();
    itk::HisImageIOFactory::RegisterOneFactory();
#if ITK_VERSION_MAJOR <= 3
    itk::ImageIOFactory::RegisterBuiltInFactories();
#endif
    firstTime = false;
    }
  itk::ImageIOBase::Pointer imageIO = ImageIOFactory::CreateImageIO( m_FileNames[0].c_str(), ImageIOFactory::ReadMode );

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
      typedef itk::VarianObiRawImageFilter<InputImageType, OutputImageType> RawFilterType;
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
      typedef itk::ElektaSynergyRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
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
      typedef itk::TiffLutImageFilter<InputImageType, OutputImageType> RawFilterType;
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

} //namespace ITK

#endif
