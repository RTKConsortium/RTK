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
#include <itkConfigure.h>
#include <itkImageSeriesReader.h>
#include <itkCropImageFilter.h>
#include <itkBinShrinkImageFilter.h>
#include <itkNumericTraits.h>
#include <itkChangeInformationImageFilter.h>

// RTK
#include "rtkIOFactories.h"
#include "rtkBoellaardScatterCorrectionImageFilter.h"
#include "rtkLUTbasedVariableI0RawToAttenuationImageFilter.h"

// Varian Obi includes
#include "rtkHndImageIOFactory.h"
#include "rtkVarianObiRawImageFilter.h"

// Elekta Synergy includes
#include "rtkHisImageIOFactory.h"
#include "rtkElektaSynergyLookupTableImageFilter.h"

// ImagX includes
#include "rtkImagXImageIOFactory.h"

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
ProjectionsReader<TOutputImage>
::ProjectionsReader():
  m_ImageIO(NULL),
  m_AirThreshold(32000),
  m_ScatterToPrimaryRatio(0.),
  m_NonNegativityConstraintThreshold( itk::NumericTraits<double>::NonpositiveMin() ),
  m_I0( itk::NumericTraits<double>::NonpositiveMin() )
{
  // Filters common to all input types and that do not depend on the input image type.
  m_WaterPrecorrectionFilter = WaterPrecorrectionType::New();
  m_StreamingFilter = StreamingType::New();

  // Default values of parameters
  m_Spacing.Fill( itk::NumericTraits<typename OutputImageType::SpacingValueType>::max() );
  m_Origin.Fill( itk::NumericTraits<typename OutputImageType::PointValueType>::max() );
  m_Direction.Fill( itk::NumericTraits<typename OutputImageType::PointValueType>::max() );
  m_LowerBoundaryCropSize.Fill(0);
  m_UpperBoundaryCropSize.Fill(0);
  m_ShrinkFactors.Fill(1);
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::PrintSelf(std::ostream& os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  if(m_RawDataReader.GetPointer())
    os << indent << "RawDataReader: " << m_RawDataReader->GetNameOfClass() << std::endl;
  if(m_RawToAttenuationFilter.GetPointer())
    os << indent << "RawToProjectionsFilter: " << m_RawToAttenuationFilter->GetNameOfClass() << std::endl;
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
    imageIO->SetFileName( m_FileNames[0].c_str() );
    imageIO->ReadImageInformation();

    // In this block, we create the filters used depending on the input type

    // Reset
    m_RawDataReader = NULL;
    m_ChangeInformationFilter = NULL;
    m_ElektaRawFilter = NULL;
    m_CropFilter = NULL;
    m_BinningFilter = NULL;
    m_ScatterFilter = NULL;
    m_I0EstimationFilter = NULL;
    m_RawToAttenuationFilter = NULL;

    // Start creation
    if( (!strcmp(imageIO->GetNameOfClass(), "EdfImageIO") &&
               imageIO->GetComponentType() == itk::ImageIOBase::USHORT) ||
             !strcmp(imageIO->GetNameOfClass(), "XRadImageIO"))
      {
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      typedef itk::ChangeInformationImageFilter< InputImageType > ChangeInfoType;
      typename ChangeInfoType::Pointer cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      if( !strcmp(imageIO->GetNameOfClass(), "EdfImageIO") )
        {
        /////////// ESRF
        // Convert raw to Projections
        typedef rtk::EdfRawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
        typename RawFilterType::Pointer rawFilter = RawFilterType::New();
        m_RawToAttenuationFilter = rawFilter;
        }
      if( !strcmp(imageIO->GetNameOfClass(), "XRadImageIO") )
        {
        /////////// XRad
        // Convert raw to Projections
        typedef rtk::XRadRawToAttenuationImageFilter<InputImageType, OutputImageType> XRadRawFilterType;
        typename XRadRawFilterType::Pointer rawFilterXRad = XRadRawFilterType::New();
        m_RawToAttenuationFilter = rawFilterXRad;
        }
      }
    else if( !strcmp(imageIO->GetNameOfClass(), "HndImageIO") )
      {
      /////////// Varian OBI
      typedef unsigned int                                       InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      typedef itk::ChangeInformationImageFilter< InputImageType > ChangeInfoType;
      typename ChangeInfoType::Pointer cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      typedef itk::CropImageFilter< InputImageType, InputImageType > CropType;
      typename CropType::Pointer crop = CropType::New();
      m_CropFilter = crop;

      // Bin
      typedef itk::BinShrinkImageFilter< InputImageType, InputImageType > BinType;
      typename BinType::Pointer bin = BinType::New();
      m_BinningFilter = bin;

      // Scatter correction
      typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType>  ScatterFilterType;
      typename ScatterFilterType::Pointer scatter = ScatterFilterType::New();
      m_ScatterFilter = scatter;

      // Convert raw to Projections
      typedef rtk::VarianObiRawImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      m_RawToAttenuationFilter = rawFilter;
      }
    else if( !strcmp(imageIO->GetNameOfClass(), "HisImageIO") ||
             !strcmp(imageIO->GetNameOfClass(), "DCMImagXImageIO") ||
             !strcmp(imageIO->GetNameOfClass(), "ImagXImageIO") ||
             !strcmp(imageIO->GetNameOfClass(), "TIFFImageIO") ||
             imageIO->GetComponentType() == itk::ImageIOBase::USHORT )
      {
      /////////// Elekta synergy, IBA / iMagX, TIFF
      typedef unsigned short                                     InputPixelType;
      typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType;

      // Reader
      typedef itk::ImageSeriesReader< InputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      typedef itk::ChangeInformationImageFilter< InputImageType > ChangeInfoType;
      typename ChangeInfoType::Pointer cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      typedef itk::CropImageFilter< InputImageType, InputImageType > CropType;
      typename CropType::Pointer crop = CropType::New();
      m_CropFilter = crop;

      // Elekta specific conversion of input raw data
      if( !strcmp(imageIO->GetNameOfClass(), "HisImageIO") )
        {
        typedef rtk::ElektaSynergyRawLookupTableImageFilter<OutputImageDimension> ElektaRawType;
        typename ElektaRawType::Pointer elekta = ElektaRawType::New();
        m_ElektaRawFilter = elekta;

        // Backward compatibility for default Elekta parameters
        m_LowerBoundaryCropSize.Fill(4);
        m_LowerBoundaryCropSize[2] = 0;
        m_UpperBoundaryCropSize.Fill(4);
        m_UpperBoundaryCropSize[2] = 0;
        m_I0 = 65536;
        }

      // Bin
      typedef itk::BinShrinkImageFilter< InputImageType, InputImageType > BinType;
      typename BinType::Pointer bin = BinType::New();
      m_BinningFilter = bin;

      // Scatter correction
      typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType>  ScatterFilterType;
      typename ScatterFilterType::Pointer scatter = ScatterFilterType::New();
      m_ScatterFilter = scatter;

      // I0 estimation filter (shunt from pipeline by default)
      typedef rtk::I0EstimationProjectionFilter<InputImageType, InputImageType> I0EstimationFilterType;
      typename I0EstimationFilterType::Pointer i0est = I0EstimationFilterType::New();
      m_I0EstimationFilter = i0est;

      // Convert raw to Projections
      typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      m_RawToAttenuationFilter = rawFilter;
      }
    else
      {
      ///////////// Default: whatever the format, we assume that we directly
      // read the Projections

      typedef itk::ImageSeriesReader< OutputImageType > ReaderType;
      typename ReaderType::Pointer reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      typedef itk::ChangeInformationImageFilter< OutputImageType > ChangeInfoType;
      typename ChangeInfoType::Pointer cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      typedef itk::CropImageFilter< OutputImageType, OutputImageType > CropType;
      typename CropType::Pointer crop = CropType::New();
      m_CropFilter = crop;

      // Bin
      typedef itk::BinShrinkImageFilter< OutputImageType, OutputImageType > BinType;
      typename BinType::Pointer bin = BinType::New();
      m_BinningFilter = bin;
      }

    //Store imageIO to avoid creating the pipe more than necessary
    m_ImageIO = imageIO;
    }

  // Parameter propagation
  if( !strcmp(imageIO->GetNameOfClass(), "XRadImageIO") ||
      !strcmp(imageIO->GetNameOfClass(), "HisImageIO") ||
      !strcmp(imageIO->GetNameOfClass(), "DCMImagXImageIO") ||
      !strcmp(imageIO->GetNameOfClass(), "ImagXImageIO") ||
      !strcmp(imageIO->GetNameOfClass(), "TIFFImageIO") ||
      imageIO->GetComponentType() == itk::ImageIOBase::USHORT )
    PropagateParametersToMiniPipeline< itk::Image<unsigned short, OutputImageDimension> >();
  else if( !strcmp(imageIO->GetNameOfClass(), "HndImageIO") )
    PropagateParametersToMiniPipeline< itk::Image<unsigned int, OutputImageDimension> >();
  else
    PropagateParametersToMiniPipeline< OutputImageType >();

  // Set output information as provided by the pipeline
  m_StreamingFilter->UpdateOutputInformation();
  TOutputImage * output = this->GetOutput();
  output->SetOrigin( m_StreamingFilter->GetOutput()->GetOrigin() );
  output->SetSpacing( m_StreamingFilter->GetOutput()->GetSpacing() );
  output->SetDirection( m_StreamingFilter->GetOutput()->GetDirection() );
  output->SetLargestPossibleRegion( m_StreamingFilter->GetOutput()->GetLargestPossibleRegion() );
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::GenerateData()
{
  TOutputImage * output = this->GetOutput();
  m_StreamingFilter->SetNumberOfStreamDivisions( output->GetRequestedRegion().GetSize(TOutputImage::ImageDimension-1) );
  m_StreamingFilter->GetOutput()->SetRequestedRegion( output->GetRequestedRegion() );
  m_StreamingFilter->Update();
  this->GraftOutput( m_StreamingFilter->GetOutput() );
}

//--------------------------------------------------------------------
template <class TOutputImage>
template <class TInputImage>
void ProjectionsReader<TOutputImage>
::PropagateParametersToMiniPipeline()
{
  // Raw
  typedef typename itk::ImageSeriesReader< TInputImage> RawType;
  RawType *raw = dynamic_cast<RawType*>(m_RawDataReader.GetPointer());
  assert(raw != NULL);
  raw->SetFileNames( this->GetFileNames() );
  raw->SetImageIO( m_ImageIO );
  TInputImage *nextInput = raw->GetOutput();

  // Image information
  OutputImageSpacingType defaultSpacing;
  defaultSpacing.Fill( itk::NumericTraits<typename OutputImageType::SpacingValueType>::max() );
  OutputImagePointType defaultOrigin;
  defaultOrigin.Fill( itk::NumericTraits<typename OutputImageType::PointValueType>::max() );
  OutputImageDirectionType defaultDirection;
  defaultDirection.Fill( itk::NumericTraits<typename OutputImageType::PointValueType>::max() );
  if(m_Spacing != defaultSpacing || m_Origin != defaultOrigin || m_Direction != defaultDirection)
    {
    if(m_ChangeInformationFilter.GetPointer() == NULL)
      {
        itkGenericExceptionMacro(<< "Can not change image information with this input (not implemented)");
      }
    else
      {
      typedef itk::ChangeInformationImageFilter< TInputImage > ChangeInfoType;
      ChangeInfoType *cif = dynamic_cast<ChangeInfoType*>(m_ChangeInformationFilter.GetPointer());
      assert(cif != NULL);
      if(m_Spacing != defaultSpacing)
        {
        cif->SetOutputSpacing(m_Spacing);
        cif->ChangeSpacingOn();
        }
      if(m_Origin != defaultOrigin)
        {
        cif->SetOutputOrigin(m_Origin);
        cif->ChangeOriginOn();
        }
      if(m_Direction != defaultDirection)
        {
        cif->SetOutputDirection(m_Direction);
        cif->ChangeDirectionOn();
        }
      cif->SetInput(nextInput);
      nextInput = cif->GetOutput();
      }
    }

  // Crop
  OutputImageSizeType defaultCropSize;
  defaultCropSize.Fill(0);
  if(m_LowerBoundaryCropSize != defaultCropSize || m_UpperBoundaryCropSize != defaultCropSize)
    {
    if(m_CropFilter.GetPointer() == NULL)
      {
        itkGenericExceptionMacro(<< "Can not crop images read with this input (not implemented)");
      }
    else
      {
      typedef itk::CropImageFilter< TInputImage, TInputImage > CropType;
      CropType *crop = dynamic_cast<CropType*>(m_CropFilter.GetPointer());
      assert(crop != NULL);
      crop->SetLowerBoundaryCropSize(m_LowerBoundaryCropSize);
      crop->SetUpperBoundaryCropSize(m_UpperBoundaryCropSize);
      crop->SetInput(nextInput);
      nextInput = crop->GetOutput();
      }
    }

  // Elekta raw data converter
  itk::ImageBase<OutputImageDimension> *nextInputBase = dynamic_cast<itk::ImageBase<OutputImageDimension> *>(nextInput);
  assert(nextInputBase != NULL);
  ConnectElektaRawFilter(&nextInputBase);
  nextInput = dynamic_cast<TInputImage *>(nextInputBase);
  assert(nextInput != NULL);

  // Binning
  ShrinkFactorsType defaultShrinkFactors;
  defaultShrinkFactors.Fill(1);
  if(m_ShrinkFactors != defaultShrinkFactors)
    {
    if(m_BinningFilter.GetPointer() == NULL)
      {
        itkGenericExceptionMacro(<< "Can not bin / shrink images read with this input (not implemented)");
      }
    else
      {
      typedef itk::BinShrinkImageFilter< TInputImage, TInputImage > BinType;
      BinType *bin = dynamic_cast<BinType*>(m_BinningFilter.GetPointer());
      assert(bin != NULL);
      bin->SetShrinkFactors(m_ShrinkFactors);
      bin->SetInput(nextInput);
      nextInput = bin->GetOutput();
      }
    }

  // Boellaard scatter correction
  if(m_NonNegativityConstraintThreshold != itk::NumericTraits<double>::NonpositiveMin() ||
     m_ScatterToPrimaryRatio != 0.)
    {
    if(m_ScatterFilter.GetPointer() == NULL)
      {
        itkGenericExceptionMacro(<< "Can not use Boellaard scatter correction with this input (not implemented)");
      }
    else
      {
      typedef rtk::BoellaardScatterCorrectionImageFilter<TInputImage, TInputImage>  ScatterFilterType;
      ScatterFilterType *scatter = dynamic_cast<ScatterFilterType*>(m_ScatterFilter.GetPointer());
      assert(scatter != NULL);
      scatter->SetAirThreshold(m_AirThreshold);
      scatter->SetScatterToPrimaryRatio(m_ScatterToPrimaryRatio);
      if(m_NonNegativityConstraintThreshold != itk::NumericTraits<double>::NonpositiveMin())
        scatter->SetNonNegativityConstraintThreshold(m_NonNegativityConstraintThreshold);
      scatter->SetInput(nextInput);
      nextInput = scatter->GetOutput();
      }
    }

  // LUTbasedVariableI0RawToAttenuationImageFilter
  if( m_I0 != itk::NumericTraits<double>::NonpositiveMin() )
    {
    if(m_RawToAttenuationFilter.GetPointer() == NULL)
      {
      itkGenericExceptionMacro(<< "Can not use I0 in LUTbasedVariableI0RawToAttenuationImageFilter with this input (not implemented)");
      }
    else
      {
      itk::ImageBase<OutputImageDimension> *nextInputBase;
      nextInputBase = dynamic_cast<itk::ImageBase<OutputImageDimension> *>(nextInput);
      assert(nextInputBase != NULL);
      PropagateI0(&nextInputBase);
      nextInput = dynamic_cast<TInputImage *>(nextInputBase);
      assert(nextInput != NULL);
      }
    }

  // Raw to attenuation filter, change of type
  OutputImageType *output = NULL;
  if(m_RawToAttenuationFilter.GetPointer() != NULL)
    {
    typedef itk::ImageToImageFilter<TInputImage, OutputImageType> IToIFilterType;
    IToIFilterType * itoi = dynamic_cast<IToIFilterType*>( m_RawToAttenuationFilter.GetPointer() );
    assert(itoi != NULL);
    itoi->SetInput(nextInput);
    output = itoi->GetOutput();

    // Release output data of m_RawDataReader if conversion occurs
    itoi->ReleaseDataFlagOn();
    }
  else
    {
    output = dynamic_cast<OutputImageType *>(nextInput);
    assert(output != NULL);
    }

  // ESRF raw to attenuation converter also needs the filenames
  typedef rtk::EdfRawToAttenuationImageFilter<TInputImage, OutputImageType> EdfRawFilterType;
  EdfRawFilterType *edf = dynamic_cast<EdfRawFilterType*>( m_RawToAttenuationFilter.GetPointer() );
  if(edf)
    edf->SetFileNames( this->GetFileNames() );

  // Water coefficients
  if(m_WaterPrecorrectionCoefficients.size() != 0)
    {
    m_WaterPrecorrectionFilter->SetCoefficients(m_WaterPrecorrectionCoefficients);
    m_WaterPrecorrectionFilter->SetInput(output);
    output = m_WaterPrecorrectionFilter->GetOutput();
    }

  // Streaming image filter
  m_StreamingFilter->SetInput( output );
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::ConnectElektaRawFilter(itk::ImageBase<OutputImageDimension> **nextInputBase)
{
  if(m_ElektaRawFilter.GetPointer() != NULL)
    {
    typedef rtk::ElektaSynergyRawLookupTableImageFilter<OutputImageDimension> ElektaRawType;
    ElektaRawType *elektaRaw = dynamic_cast<ElektaRawType*>(m_ElektaRawFilter.GetPointer());
    assert(elektaRaw != NULL);
    typedef typename itk::Image<unsigned short, OutputImageDimension> InputImageType;
    InputImageType *nextInput = dynamic_cast<InputImageType*>(*nextInputBase);
    elektaRaw->SetInput(nextInput);
    *nextInputBase = elektaRaw->GetOutput();
    }
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::PropagateI0(itk::ImageBase<OutputImageDimension> **nextInputBase)
{
  typedef itk::Image<unsigned short, OutputImageDimension> InputImageType;
  InputImageType *nextInput = dynamic_cast<InputImageType*>(*nextInputBase);
  assert(nextInput != NULL);
  if(m_I0==0)
    {
    typedef rtk::I0EstimationProjectionFilter< InputImageType, InputImageType > I0EstimationType;
    I0EstimationType *i0est = dynamic_cast<I0EstimationType*>(m_I0EstimationFilter.GetPointer());
    assert(i0est != NULL);
    i0est->SetInput(nextInput);
    *nextInputBase = i0est->GetOutput();
    }

  typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter< InputImageType, OutputImageType > I0Type;
  I0Type *i0 = dynamic_cast<I0Type*>(m_RawToAttenuationFilter.GetPointer());
  assert(i0 != NULL);
  i0->SetI0(m_I0);
  // Pipeline connection for m_RawToAttenuationFilter is done after the call to this function
}

} //namespace rtk

#endif
