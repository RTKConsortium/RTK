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

#ifndef rtkProjectionsReader_hxx
#define rtkProjectionsReader_hxx

// ITK
#include <itkConfigure.h>
#include <itkImageSeriesReader.h>
#include <itkCropImageFilter.h>
#include <itkBinShrinkImageFilter.h>
#include <itkNumericTraits.h>
#include <itkChangeInformationImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkVectorIndexSelectionCastImageFilter.h>

// RTK
#include "rtkIOFactories.h"
#include "rtkBoellaardScatterCorrectionImageFilter.h"
#include "rtkLUTbasedVariableI0RawToAttenuationImageFilter.h"
#include "rtkConditionalMedianImageFilter.h"

// Varian Obi includes
#include "rtkHndImageIOFactory.h"
#include "rtkXimImageIOFactory.h"
#include "rtkVarianObiRawImageFilter.h"

// Elekta Synergy includes
#include "rtkHisImageIOFactory.h"
#include "rtkElektaSynergyRawLookupTableImageFilter.h"
#include "rtkElektaSynergyLookupTableImageFilter.h"

// ImagX includes
#include "rtkImagXImageIOFactory.h"

// European Synchrotron Radiation Facility
#include "rtkEdfImageIOFactory.h"
#include "rtkEdfRawToAttenuationImageFilter.h"

// Xrad small animal scanner
#include "rtkXRadImageIOFactory.h"
#include "rtkXRadRawToAttenuationImageFilter.h"

// Ora (medPhoton) image files
#include "rtkOraLookupTableImageFilter.h"

// Macro to handle input images with vector pixel type in GenerateOutputInformation()
#define SET_INPUT_IMAGE_VECTOR_TYPE(componentType, numberOfComponents) \
if ( !strcmp(imageIO->GetComponentTypeAsString(imageIO->GetComponentType()).c_str(), #componentType) \
  && (imageIO->GetNumberOfComponents() == numberOfComponents) ) \
  { \
  typedef itk::Vector< componentType, numberOfComponents >     InputPixelType; \
  typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType; \
  typedef itk::ImageSeriesReader< InputImageType > ReaderType; \
  typename ReaderType::Pointer reader = ReaderType::New(); \
  m_RawDataReader = reader; \
  typedef itk::VectorIndexSelectionCastImageFilter<InputImageType, OutputImageType> VectorComponentSelectionType; \
  typename VectorComponentSelectionType::Pointer vectorComponentSelectionFilter = VectorComponentSelectionType::New(); \
  if (m_VectorComponent < numberOfComponents ) \
    vectorComponentSelectionFilter->SetIndex(m_VectorComponent); \
  else \
    itkGenericExceptionMacro(<< "Cannot extract " << m_VectorComponent << "-th component from vector of size " << numberOfComponents) \
  m_VectorComponentSelectionFilter = vectorComponentSelectionFilter; \
  }

// Macro to handle input images with vector pixel type in PropagateParametersToMiniPipeline()
#define PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(componentType, numberOfComponents) \
if ( !strcmp(m_ImageIO->GetComponentTypeAsString(m_ImageIO->GetComponentType()).c_str(), #componentType) \
  && (m_ImageIO->GetNumberOfComponents() == numberOfComponents) ) \
  { \
  typedef itk::Vector< componentType, numberOfComponents >     InputPixelType; \
  typedef itk::Image< InputPixelType, OutputImageDimension > InputImageType; \
  typedef typename itk::ImageSeriesReader< InputImageType > RawType; \
  RawType *raw = dynamic_cast<RawType*>(m_RawDataReader.GetPointer()); \
  assert(raw != ITK_NULLPTR); \
  raw->SetFileNames( this->GetFileNames() ); \
  raw->SetImageIO( m_ImageIO ); \
  typedef itk::VectorIndexSelectionCastImageFilter<InputImageType, OutputImageType> VectorComponentSelectionType; \
  VectorComponentSelectionType *vectorComponentSelectionFilter = dynamic_cast<VectorComponentSelectionType*>(m_VectorComponentSelectionFilter.GetPointer()); \
  assert(vectorComponentSelectionFilter != ITK_NULLPTR); \
  vectorComponentSelectionFilter->SetInput(raw->GetOutput()); \
  output = vectorComponentSelectionFilter->GetOutput(); \
  }

namespace rtk
{

//--------------------------------------------------------------------
template <class TOutputImage>
ProjectionsReader<TOutputImage>
::ProjectionsReader():
  m_ImageIO(ITK_NULLPTR),
  m_AirThreshold(32000),
  m_ScatterToPrimaryRatio(0.),
  m_NonNegativityConstraintThreshold( itk::NumericTraits<double>::NonpositiveMin() ),
  m_I0( itk::NumericTraits<double>::NonpositiveMin() ),
  m_IDark( 0. ),
  m_ConditionalMedianThresholdMultiplier( 1. ),
  m_ComputeLineIntegral(true),
  m_VectorComponent(0)
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
  m_MedianRadius.Fill(0);
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
    m_RawDataReader = ITK_NULLPTR;
    m_VectorComponentSelectionFilter = ITK_NULLPTR;
    m_ChangeInformationFilter = ITK_NULLPTR;
    m_ElektaRawFilter = ITK_NULLPTR;
    m_CropFilter = ITK_NULLPTR;
    m_ConditionalMedianFilter = ITK_NULLPTR;
    m_BinningFilter = ITK_NULLPTR;
    m_ScatterFilter = ITK_NULLPTR;
    m_I0EstimationFilter = ITK_NULLPTR;
    m_RawToAttenuationFilter = ITK_NULLPTR;
    m_RawCastFilter = ITK_NULLPTR;

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

        // Or just cast to OutputImageType
        typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
        typename CastFilterType::Pointer castFilter = CastFilterType::New();
        m_RawCastFilter = castFilter;
        }
      if( !strcmp(imageIO->GetNameOfClass(), "XRadImageIO") )
        {
        /////////// XRad
        // Convert raw to Projections
        typedef rtk::XRadRawToAttenuationImageFilter<InputImageType, OutputImageType> XRadRawFilterType;
        typename XRadRawFilterType::Pointer rawFilterXRad = XRadRawFilterType::New();
        m_RawToAttenuationFilter = rawFilterXRad;

        // Or just cast to OutputImageType
        typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
        typename CastFilterType::Pointer castFilter = CastFilterType::New();
        m_RawCastFilter = castFilter;
        }
      }
    else if( !strcmp(imageIO->GetNameOfClass(), "HndImageIO") ||
             !strcmp(imageIO->GetNameOfClass(), "XimImageIO") )
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

      // Conditional median
      typedef rtk::ConditionalMedianImageFilter< InputImageType > ConditionalMedianType;
      typename ConditionalMedianType::Pointer cond = ConditionalMedianType::New();
      m_ConditionalMedianFilter = cond;

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
      typedef rtk::VarianObiRawImageFilter<InputImageType, OutputImageType> RawFilterType;
      typename RawFilterType::Pointer rawFilter = RawFilterType::New();
      m_RawToAttenuationFilter = rawFilter;

      // Or just cast to OutputImageType
      typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
      typename CastFilterType::Pointer castFilter = CastFilterType::New();
      m_RawCastFilter = castFilter;
      }
    else if( imageIO->GetComponentType() == itk::ImageIOBase::USHORT )
      {
      /////////// Ora, Elekta synergy, IBA / iMagX, unsigned short
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
        typedef rtk::ElektaSynergyRawLookupTableImageFilter< itk::Image<unsigned short, OutputImageDimension>,
                                                             itk::Image<unsigned short, OutputImageDimension> > ElektaRawType;
        typename ElektaRawType::Pointer elekta = ElektaRawType::New();
        m_ElektaRawFilter = elekta;

        // Backward compatibility for default Elekta parameters
        OutputImageSizeType defaultCropSize;
        defaultCropSize.Fill(0);
        if(m_LowerBoundaryCropSize == defaultCropSize && m_UpperBoundaryCropSize == defaultCropSize)
          {
          m_LowerBoundaryCropSize.Fill(4);
          m_LowerBoundaryCropSize[2] = 0;
          m_UpperBoundaryCropSize.Fill(4);
          m_UpperBoundaryCropSize[2] = 0;
          }
        if( m_I0 == itk::NumericTraits<double>::NonpositiveMin() )
          m_I0 = 65536;
        }

      // Conditional median
      typedef rtk::ConditionalMedianImageFilter< InputImageType > ConditionalMedianType;
      typename ConditionalMedianType::Pointer cond = ConditionalMedianType::New();
      m_ConditionalMedianFilter = cond;

      // Bin
      typedef itk::BinShrinkImageFilter< InputImageType, InputImageType > BinType;
      typename BinType::Pointer bin = BinType::New();
      m_BinningFilter = bin;

      // Ora & ushort specific conversion of input raw data
      if( !strcmp(imageIO->GetNameOfClass(), "OraImageIO") )
        {
        typedef rtk::OraLookupTableImageFilter< OutputImageType > OraRawType;
        typename OraRawType::Pointer oraraw = OraRawType::New();
        m_RawToAttenuationFilter = oraraw;
        }
      else
        {
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

        // Or just casts to OutputImageType
        typedef itk::CastImageFilter<InputImageType, OutputImageType> CastFilterType;
        typename CastFilterType::Pointer castFilter = CastFilterType::New();
        m_RawCastFilter = castFilter;
        }
      }
    else
      {
      // If ImageIO has vector pixels, extract one component from it
      if (!strcmp(imageIO->GetPixelTypeAsString(imageIO->GetPixelType()).c_str(), "vector"))
        {
        SET_INPUT_IMAGE_VECTOR_TYPE(float, 1)
        SET_INPUT_IMAGE_VECTOR_TYPE(float, 2)
        SET_INPUT_IMAGE_VECTOR_TYPE(float, 3)
        SET_INPUT_IMAGE_VECTOR_TYPE(float, 4)
        SET_INPUT_IMAGE_VECTOR_TYPE(float, 5)
        SET_INPUT_IMAGE_VECTOR_TYPE(float, 6)
        SET_INPUT_IMAGE_VECTOR_TYPE(double, 1)
        SET_INPUT_IMAGE_VECTOR_TYPE(double, 2)
        SET_INPUT_IMAGE_VECTOR_TYPE(double, 3)
        SET_INPUT_IMAGE_VECTOR_TYPE(double, 4)
        SET_INPUT_IMAGE_VECTOR_TYPE(double, 5)
        SET_INPUT_IMAGE_VECTOR_TYPE(double, 6)
        }
      else
        {
        ///////////// Default: whatever the format, we assume that we directly
        //// read the Projections
        typedef itk::ImageSeriesReader< OutputImageType > ReaderType;
        typename ReaderType::Pointer reader = ReaderType::New();
        m_RawDataReader = reader;
        }

      // Change information
      typedef itk::ChangeInformationImageFilter< OutputImageType > ChangeInfoType;
      typename ChangeInfoType::Pointer cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      typedef itk::CropImageFilter< OutputImageType, OutputImageType > CropType;
      typename CropType::Pointer crop = CropType::New();
      m_CropFilter = crop;

      // Conditional median
      typedef rtk::ConditionalMedianImageFilter< OutputImageType > ConditionalMedianType;
      typename ConditionalMedianType::Pointer cond = ConditionalMedianType::New();
      m_ConditionalMedianFilter = cond;

      // Bin
      typedef itk::BinShrinkImageFilter< OutputImageType, OutputImageType > BinType;
      typename BinType::Pointer bin = BinType::New();
      m_BinningFilter = bin;
      }

    //Store imageIO to avoid creating the pipe more than necessary
    m_ImageIO = imageIO;
    }

  // Parameter propagation
  if( imageIO->GetComponentType() == itk::ImageIOBase::USHORT )
    PropagateParametersToMiniPipeline< itk::Image<unsigned short, OutputImageDimension> >();
  else if( !strcmp(imageIO->GetNameOfClass(), "HndImageIO") ||
           !strcmp(imageIO->GetNameOfClass(), "XimImageIO"))
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
  TInputImage *nextInput;
  OutputImageType *output = ITK_NULLPTR;

  // Vector component selection
  if(m_VectorComponentSelectionFilter.GetPointer() != ITK_NULLPTR)
    {
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(float, 1)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(float, 2)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(float, 3)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(float, 4)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(float, 5)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(float, 6)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(double, 1)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(double, 2)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(double, 3)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(double, 4)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(double, 5)
    PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(double, 6)
    }
  else // Regular case
    {
    // Raw
    typedef typename itk::ImageSeriesReader< TInputImage> RawType;
    RawType *raw = dynamic_cast<RawType*>(m_RawDataReader.GetPointer());
    assert(raw != ITK_NULLPTR);
    raw->SetFileNames( this->GetFileNames() );
    raw->SetImageIO( m_ImageIO );
    nextInput = raw->GetOutput();

    // Image information
    OutputImageSpacingType defaultSpacing;
    defaultSpacing.Fill( itk::NumericTraits<typename OutputImageType::SpacingValueType>::max() );
    OutputImagePointType defaultOrigin;
    defaultOrigin.Fill( itk::NumericTraits<typename OutputImageType::PointValueType>::max() );
    OutputImageDirectionType defaultDirection;
    defaultDirection.Fill( itk::NumericTraits<typename OutputImageType::PointValueType>::max() );
    if(m_Spacing != defaultSpacing || m_Origin != defaultOrigin || m_Direction != defaultDirection)
      {
      if(m_ChangeInformationFilter.GetPointer() == ITK_NULLPTR)
        {
          itkGenericExceptionMacro(<< "Can not change image information with this input (not implemented)");
        }
      else
        {
        typedef itk::ChangeInformationImageFilter< TInputImage > ChangeInfoType;
        ChangeInfoType *cif = dynamic_cast<ChangeInfoType*>(m_ChangeInformationFilter.GetPointer());
        assert(cif != ITK_NULLPTR);
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
      if(m_CropFilter.GetPointer() == ITK_NULLPTR)
        {
          itkGenericExceptionMacro(<< "Can not crop images read with this input (not implemented)");
        }
      else
        {
        typedef itk::CropImageFilter< TInputImage, TInputImage > CropType;
        CropType *crop = dynamic_cast<CropType*>(m_CropFilter.GetPointer());
        assert(crop != ITK_NULLPTR);
        crop->SetLowerBoundaryCropSize(m_LowerBoundaryCropSize);
        crop->SetUpperBoundaryCropSize(m_UpperBoundaryCropSize);
        crop->SetInput(nextInput);
        nextInput = crop->GetOutput();
        }
      }

    // Elekta raw data converter
    itk::ImageBase<OutputImageDimension> *nextInputBase = dynamic_cast<itk::ImageBase<OutputImageDimension> *>(nextInput);
    assert(nextInputBase != ITK_NULLPTR);
    ConnectElektaRawFilter(&nextInputBase);
    nextInput = dynamic_cast<TInputImage *>(nextInputBase);
    assert(nextInput != ITK_NULLPTR);

    // Conditional median
    MedianRadiusType defaultMedianRadius;
    defaultMedianRadius.Fill(0);
    if(m_MedianRadius != defaultMedianRadius)
      {
      if(m_ConditionalMedianFilter.GetPointer() == ITK_NULLPTR)
        {
          itkGenericExceptionMacro(<< "Can not apply conditional median filter on this input (not implemented)");
        }
      else
        {
        typedef rtk::ConditionalMedianImageFilter< TInputImage > ConditionalMedianType;
        ConditionalMedianType *cond = dynamic_cast<ConditionalMedianType*>(m_ConditionalMedianFilter.GetPointer());
        assert(cond != ITK_NULLPTR);
        cond->SetRadius(m_MedianRadius);
        cond->SetInput(nextInput);
        cond->SetThresholdMultiplier(m_ConditionalMedianThresholdMultiplier);
        nextInput = cond->GetOutput();
        }
      }

    // Binning
    ShrinkFactorsType defaultShrinkFactors;
    defaultShrinkFactors.Fill(1);
    if(m_ShrinkFactors != defaultShrinkFactors)
      {
      if(m_BinningFilter.GetPointer() == ITK_NULLPTR)
        {
          itkGenericExceptionMacro(<< "Can not bin / shrink images read with this input (not implemented)");
        }
      else
        {
        typedef itk::BinShrinkImageFilter< TInputImage, TInputImage > BinType;
        BinType *bin = dynamic_cast<BinType*>(m_BinningFilter.GetPointer());
        assert(bin != ITK_NULLPTR);
        bin->SetShrinkFactors(m_ShrinkFactors);
        bin->SetInput(nextInput);
        nextInput = bin->GetOutput();
        }
      }

    // Boellaard scatter correction
    if(m_NonNegativityConstraintThreshold != itk::NumericTraits<double>::NonpositiveMin() ||
       m_ScatterToPrimaryRatio != 0.)
      {
      if(m_ScatterFilter.GetPointer() == ITK_NULLPTR)
        {
          itkGenericExceptionMacro(<< "Can not use Boellaard scatter correction with this input (not implemented)");
        }
      else
        {
        typedef rtk::BoellaardScatterCorrectionImageFilter<TInputImage, TInputImage>  ScatterFilterType;
        ScatterFilterType *scatter = dynamic_cast<ScatterFilterType*>(m_ScatterFilter.GetPointer());
        assert(scatter != ITK_NULLPTR);
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
      if(m_RawToAttenuationFilter.GetPointer() == ITK_NULLPTR)
        {
        itkGenericExceptionMacro(<< "Can not use I0 in LUTbasedVariableI0RawToAttenuationImageFilter with this input (not implemented)");
        }
      else
        {
        nextInputBase = dynamic_cast<itk::ImageBase<OutputImageDimension> *>(nextInput);
        assert(nextInputBase != ITK_NULLPTR);
        PropagateI0(&nextInputBase);
        nextInput = dynamic_cast<TInputImage *>(nextInputBase);
        assert(nextInput != ITK_NULLPTR);
        }
      }

    // Raw to attenuation or cast filter, change of type
    if(m_RawToAttenuationFilter.GetPointer() != ITK_NULLPTR)
      {
      // Check if Ora pointer
      typedef rtk::OraLookupTableImageFilter< OutputImageType > OraRawType;
      OraRawType *oraraw = dynamic_cast<OraRawType*>( m_RawToAttenuationFilter.GetPointer() );
      if(oraraw != ITK_NULLPTR)
        {
        oraraw->SetComputeLineIntegral(m_ComputeLineIntegral);
        oraraw->SetFileNames(m_FileNames);
        }

      // Cast or convert to line integral depending on m_ComputeLineIntegral
      typedef itk::ImageToImageFilter<TInputImage, OutputImageType> IToIFilterType;
      IToIFilterType * itoi = ITK_NULLPTR;
      if(m_ComputeLineIntegral || oraraw != ITK_NULLPTR)
        itoi = dynamic_cast<IToIFilterType*>( m_RawToAttenuationFilter.GetPointer() );
      else
        itoi = dynamic_cast<IToIFilterType*>( m_RawCastFilter.GetPointer() );
      assert(itoi != ITK_NULLPTR);
      itoi->SetInput(nextInput);
      output = itoi->GetOutput();

      // Release output data of m_RawDataReader if conversion occurs
      itoi->ReleaseDataFlagOn();
      }
    else
      {
      output = dynamic_cast<OutputImageType *>(nextInput);
      assert(output != ITK_NULLPTR);
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
    }
  // Streaming image filter
  m_StreamingFilter->SetInput( output );
}

//--------------------------------------------------------------------
template <class TOutputImage>
void ProjectionsReader<TOutputImage>
::ConnectElektaRawFilter(itk::ImageBase<OutputImageDimension> **nextInputBase)
{
  if(m_ElektaRawFilter.GetPointer() != ITK_NULLPTR)
    {
    typedef rtk::ElektaSynergyRawLookupTableImageFilter< itk::Image<unsigned short, OutputImageDimension>,
                                                         itk::Image<unsigned short, OutputImageDimension> > ElektaRawType;
    ElektaRawType *elektaRaw = dynamic_cast<ElektaRawType*>(m_ElektaRawFilter.GetPointer());
    assert(elektaRaw != ITK_NULLPTR);
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
  typedef itk::Image<unsigned short, OutputImageDimension> UnsignedShortImageType;
  UnsignedShortImageType *nextInputUShort = dynamic_cast<UnsignedShortImageType*>(*nextInputBase);
  if(nextInputUShort != ITK_NULLPTR)
    {
    if(m_I0==0)
      {
      typedef rtk::I0EstimationProjectionFilter< UnsignedShortImageType, UnsignedShortImageType > I0EstimationType;
      I0EstimationType *i0est = dynamic_cast<I0EstimationType*>(m_I0EstimationFilter.GetPointer());
      assert(i0est != ITK_NULLPTR);
      i0est->SetInput(nextInputUShort);
      *nextInputBase = i0est->GetOutput();
      }
    typedef rtk::LUTbasedVariableI0RawToAttenuationImageFilter< UnsignedShortImageType, OutputImageType > I0Type;
    I0Type *i0 = dynamic_cast<I0Type*>(m_RawToAttenuationFilter.GetPointer());
    i0->SetI0(m_I0);
    i0->SetIDark(m_IDark);
    }

  typedef itk::Image<unsigned int, OutputImageDimension> UnsignedIntImageType;
  UnsignedIntImageType *nextInputUInt = dynamic_cast<UnsignedIntImageType*>(*nextInputBase);
  if(nextInputUInt != ITK_NULLPTR)
    {
    if(m_I0==0)
      {
      typedef rtk::I0EstimationProjectionFilter< UnsignedIntImageType, UnsignedIntImageType > I0EstimationType;
      I0EstimationType *i0est = dynamic_cast<I0EstimationType*>(m_I0EstimationFilter.GetPointer());
      assert(i0est != ITK_NULLPTR);
      i0est->SetInput(nextInputUInt);
      *nextInputBase = i0est->GetOutput();
      }
    typedef rtk::VarianObiRawImageFilter<UnsignedIntImageType, OutputImageType> I0Type;
    I0Type *i0 = dynamic_cast<I0Type*>(m_RawToAttenuationFilter.GetPointer());
    i0->SetI0(m_I0);
    i0->SetIDark(m_IDark);
    }
  // Pipeline connection for m_RawToAttenuationFilter is done after the call to this function
}

} //namespace rtk

#endif
