/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
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

// Macro to handle input images with vector pixel type in GenerateOutputInformation();
#define SET_INPUT_IMAGE_VECTOR_TYPE(componentType, numberOfComponents)                                              \
  if (!strcmp(imageIO->GetComponentTypeAsString(imageIO->GetComponentType()).c_str(), #componentType) &&            \
      (imageIO->GetNumberOfComponents() == numberOfComponents))                                                     \
  {                                                                                                                 \
    using InputPixelType = itk::Vector<componentType, numberOfComponents>;                                          \
    using InputImageType = itk::Image<InputPixelType, OutputImageDimension>;                                        \
    using ReaderType = itk::ImageSeriesReader<InputImageType>;                                                      \
    auto reader = ReaderType::New();                                                                                \
    m_RawDataReader = reader;                                                                                       \
    using VectorComponentSelectionType = itk::VectorIndexSelectionCastImageFilter<InputImageType, OutputImageType>; \
    auto vectorComponentSelectionFilter = VectorComponentSelectionType::New();                                      \
    if (m_VectorComponent < numberOfComponents)                                                                     \
      vectorComponentSelectionFilter->SetIndex(m_VectorComponent);                                                  \
    else                                                                                                            \
      itkGenericExceptionMacro(<< "Cannot extract " << m_VectorComponent << "-th component from vector of size "    \
                               << numberOfComponents);                                                              \
    m_VectorComponentSelectionFilter = vectorComponentSelectionFilter;                                              \
  }

// Macro to handle input images with vector pixel type in PropagateParametersToMiniPipeline();
#define PROPAGATE_INPUT_IMAGE_VECTOR_TYPE(componentType, numberOfComponents)                                        \
  if (!strcmp(m_ImageIO->GetComponentTypeAsString(m_ImageIO->GetComponentType()).c_str(), #componentType) &&        \
      (m_ImageIO->GetNumberOfComponents() == numberOfComponents))                                                   \
  {                                                                                                                 \
    using InputPixelType = itk::Vector<componentType, numberOfComponents>;                                          \
    using InputImageType = itk::Image<InputPixelType, OutputImageDimension>;                                        \
    using RawType = typename itk::ImageSeriesReader<InputImageType>;                                                \
    RawType * raw = dynamic_cast<RawType *>(m_RawDataReader.GetPointer());                                          \
    assert(raw != nullptr);                                                                                         \
    raw->SetFileNames(this->GetFileNames());                                                                        \
    raw->SetImageIO(m_ImageIO);                                                                                     \
    using VectorComponentSelectionType = itk::VectorIndexSelectionCastImageFilter<InputImageType, OutputImageType>; \
    VectorComponentSelectionType * vectorComponentSelectionFilter =                                                 \
      dynamic_cast<VectorComponentSelectionType *>(m_VectorComponentSelectionFilter.GetPointer());                  \
    assert(vectorComponentSelectionFilter != nullptr);                                                              \
    vectorComponentSelectionFilter->SetInput(raw->GetOutput());                                                     \
    output = vectorComponentSelectionFilter->GetOutput();                                                           \
  }

namespace rtk
{

//--------------------------------------------------------------------
template <class TOutputImage>
ProjectionsReader<TOutputImage>::ProjectionsReader()
{
  // Filters common to all input types and that do not depend on the input image type.
  m_WaterPrecorrectionFilter = WaterPrecorrectionType::New();
  m_StreamingFilter = StreamingType::New();

  // Default values of parameters
  m_Spacing.Fill(itk::NumericTraits<typename OutputImageType::SpacingValueType>::max());
  m_Origin.Fill(itk::NumericTraits<typename OutputImageType::PointValueType>::max());
  m_Direction.Fill(itk::NumericTraits<typename OutputImageType::PointValueType>::max());
  m_LowerBoundaryCropSize.Fill(0);
  m_UpperBoundaryCropSize.Fill(0);
  m_ShrinkFactors.Fill(1);
  m_MedianRadius.Fill(0);
}

//--------------------------------------------------------------------
template <class TOutputImage>
void
ProjectionsReader<TOutputImage>::PrintSelf(std::ostream & os, itk::Indent indent) const
{
  Superclass::PrintSelf(os, indent);

  if (m_RawDataReader.GetPointer())
    os << indent << "RawDataReader: " << m_RawDataReader->GetNameOfClass() << std::endl;
  if (m_RawToAttenuationFilter.GetPointer())
    os << indent << "RawToProjectionsFilter: " << m_RawToAttenuationFilter->GetNameOfClass() << std::endl;
}

//--------------------------------------------------------------------
template <class TOutputImage>
void
ProjectionsReader<TOutputImage>::GenerateOutputInformation()
{
  if (m_FileNames.empty())
    return;

  static bool firstTime = true;
  if (firstTime)
    rtk::RegisterIOFactories();
  firstTime = false;

  itk::ImageIOBase::Pointer imageIO =
    itk::ImageIOFactory::CreateImageIO(m_FileNames[0].c_str(), itk::ImageIOFactory::IOFileModeEnum::ReadMode);

  if (imageIO == nullptr)
  {
    if (m_ImageIO != nullptr)
    {
      // Can only occur if the image IO has been set manually
      std::swap(m_ImageIO, imageIO);
    }
    else
    {
      itkGenericExceptionMacro(<< "Cannot create ImageIOFactory for file " << m_FileNames[0].c_str());
    }
  }

  if (m_ImageIO != imageIO)
  {
    imageIO->SetFileName(m_FileNames[0].c_str());
    imageIO->ReadImageInformation();

    // In this block, we create the filters used depending on the input type

    // Reset
    m_RawDataReader = nullptr;
    m_VectorComponentSelectionFilter = nullptr;
    m_ChangeInformationFilter = nullptr;
    m_ElektaRawFilter = nullptr;
    m_CropFilter = nullptr;
    m_ConditionalMedianFilter = nullptr;
    m_BinningFilter = nullptr;
    m_ScatterFilter = nullptr;
    m_I0EstimationFilter = nullptr;
    m_RawToAttenuationFilter = nullptr;
    m_RawCastFilter = nullptr;

    // Start creation
    if ((!strcmp(imageIO->GetNameOfClass(), "EdfImageIO") &&
         imageIO->GetComponentType() == itk::ImageIOBase::IOComponentEnum::USHORT) ||
        !strcmp(imageIO->GetNameOfClass(), "XRadImageIO"))
    {
      using InputPixelType = unsigned short;
      using InputImageType = itk::Image<InputPixelType, OutputImageDimension>;

      // Reader
      using ReaderType = itk::ImageSeriesReader<InputImageType>;
      auto reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      using ChangeInfoType = itk::ChangeInformationImageFilter<InputImageType>;
      auto cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      if (!strcmp(imageIO->GetNameOfClass(), "EdfImageIO"))
      {
        /////////// ESRF
        // Convert raw to Projections
        using RawFilterType = rtk::EdfRawToAttenuationImageFilter<InputImageType, OutputImageType>;
        auto rawFilter = RawFilterType::New();
        m_RawToAttenuationFilter = rawFilter;

        // Or just cast to OutputImageType
        using CastFilterType = itk::CastImageFilter<InputImageType, OutputImageType>;
        auto castFilter = CastFilterType::New();
        m_RawCastFilter = castFilter;
      }
      if (!strcmp(imageIO->GetNameOfClass(), "XRadImageIO"))
      {
        /////////// XRad
        // Convert raw to Projections
        using XRadRawFilterType = rtk::XRadRawToAttenuationImageFilter<InputImageType, OutputImageType>;
        auto rawFilterXRad = XRadRawFilterType::New();
        m_RawToAttenuationFilter = rawFilterXRad;

        // Or just cast to OutputImageType
        using CastFilterType = itk::CastImageFilter<InputImageType, OutputImageType>;
        auto castFilter = CastFilterType::New();
        m_RawCastFilter = castFilter;
      }
    }
    else if (!strcmp(imageIO->GetNameOfClass(), "HndImageIO") || !strcmp(imageIO->GetNameOfClass(), "XimImageIO"))
    {
      /////////// Varian OBI
      using InputPixelType = unsigned int;
      using InputImageType = itk::Image<InputPixelType, OutputImageDimension>;

      // Reader
      using ReaderType = itk::ImageSeriesReader<InputImageType>;
      auto reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      using ChangeInfoType = itk::ChangeInformationImageFilter<InputImageType>;
      auto cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      using CropType = itk::CropImageFilter<InputImageType, InputImageType>;
      auto crop = CropType::New();
      m_CropFilter = crop;

      // Conditional median
      using ConditionalMedianType = rtk::ConditionalMedianImageFilter<InputImageType>;
      auto cond = ConditionalMedianType::New();
      m_ConditionalMedianFilter = cond;

      // Bin
      using BinType = itk::BinShrinkImageFilter<InputImageType, InputImageType>;
      auto bin = BinType::New();
      m_BinningFilter = bin;

      // Scatter correction
      using ScatterFilterType = rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType>;
      auto scatter = ScatterFilterType::New();
      m_ScatterFilter = scatter;

      // I0 estimation filter (shunt from pipeline by default)
      using I0EstimationFilterType = rtk::I0EstimationProjectionFilter<InputImageType, InputImageType>;
      auto i0est = I0EstimationFilterType::New();
      m_I0EstimationFilter = i0est;

      // Convert raw to Projections
      using RawFilterType = rtk::VarianObiRawImageFilter<InputImageType, OutputImageType>;
      auto rawFilter = RawFilterType::New();
      m_RawToAttenuationFilter = rawFilter;

      // Or just cast to OutputImageType
      using CastFilterType = itk::CastImageFilter<InputImageType, OutputImageType>;
      auto castFilter = CastFilterType::New();
      m_RawCastFilter = castFilter;
    }
    else if (imageIO->GetComponentType() == itk::ImageIOBase::IOComponentEnum::USHORT)
    {
      /////////// Ora, Elekta synergy, IBA / iMagX, unsigned short
      using InputPixelType = unsigned short;
      using InputImageType = itk::Image<InputPixelType, OutputImageDimension>;

      // Reader
      using ReaderType = itk::ImageSeriesReader<InputImageType>;
      auto reader = ReaderType::New();
      m_RawDataReader = reader;

      // Change information
      using ChangeInfoType = itk::ChangeInformationImageFilter<InputImageType>;
      auto cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      using CropType = itk::CropImageFilter<InputImageType, InputImageType>;
      auto crop = CropType::New();
      m_CropFilter = crop;

      // Elekta specific conversion of input raw data
      if (!strcmp(imageIO->GetNameOfClass(), "HisImageIO"))
      {
        using ElektaRawType =
          rtk::ElektaSynergyRawLookupTableImageFilter<itk::Image<unsigned short, OutputImageDimension>,
                                                      itk::Image<unsigned short, OutputImageDimension>>;
        auto elekta = ElektaRawType::New();
        m_ElektaRawFilter = elekta;

        // Backward compatibility for default Elekta parameters
        OutputImageSizeType defaultCropSize;
        defaultCropSize.Fill(0);
        if (m_LowerBoundaryCropSize == defaultCropSize && m_UpperBoundaryCropSize == defaultCropSize)
        {
          m_LowerBoundaryCropSize.Fill(4);
          m_LowerBoundaryCropSize[2] = 0;
          m_UpperBoundaryCropSize.Fill(4);
          m_UpperBoundaryCropSize[2] = 0;
        }
        if (m_I0 == itk::NumericTraits<double>::NonpositiveMin())
          m_I0 = 65536;
      }

      // Conditional median
      using ConditionalMedianType = rtk::ConditionalMedianImageFilter<InputImageType>;
      auto cond = ConditionalMedianType::New();
      m_ConditionalMedianFilter = cond;

      // Bin
      using BinType = itk::BinShrinkImageFilter<InputImageType, InputImageType>;
      auto bin = BinType::New();
      m_BinningFilter = bin;

      // Ora & ushort specific conversion of input raw data
      if (!strcmp(imageIO->GetNameOfClass(), "OraImageIO"))
      {
        using OraRawType = rtk::OraLookupTableImageFilter<OutputImageType>;
        auto oraraw = OraRawType::New();
        m_RawToAttenuationFilter = oraraw;
      }
      else
      {
        // Scatter correction
        using ScatterFilterType = rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType>;
        auto scatter = ScatterFilterType::New();
        m_ScatterFilter = scatter;

        // I0 estimation filter (shunt from pipeline by default)
        using I0EstimationFilterType = rtk::I0EstimationProjectionFilter<InputImageType, InputImageType>;
        auto i0est = I0EstimationFilterType::New();
        m_I0EstimationFilter = i0est;

        // Convert raw to Projections
        using RawFilterType = rtk::LUTbasedVariableI0RawToAttenuationImageFilter<InputImageType, OutputImageType>;
        auto rawFilter = RawFilterType::New();
        m_RawToAttenuationFilter = rawFilter;

        // Or just casts to OutputImageType
        using CastFilterType = itk::CastImageFilter<InputImageType, OutputImageType>;
        auto castFilter = CastFilterType::New();
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
        using ReaderType = itk::ImageSeriesReader<OutputImageType>;
        auto reader = ReaderType::New();
        m_RawDataReader = reader;
      }

      // Change information
      using ChangeInfoType = itk::ChangeInformationImageFilter<OutputImageType>;
      auto cif = ChangeInfoType::New();
      m_ChangeInformationFilter = cif;

      // Crop
      using CropType = itk::CropImageFilter<OutputImageType, OutputImageType>;
      auto crop = CropType::New();
      m_CropFilter = crop;

      // Conditional median
      using ConditionalMedianType = rtk::ConditionalMedianImageFilter<OutputImageType>;
      auto cond = ConditionalMedianType::New();
      m_ConditionalMedianFilter = cond;

      // Bin
      using BinType = itk::BinShrinkImageFilter<OutputImageType, OutputImageType>;
      auto bin = BinType::New();
      m_BinningFilter = bin;
    }

    // Store imageIO to avoid creating the pipe more than necessary
    m_ImageIO = imageIO;
  }

  // Parameter propagation
  if (imageIO->GetComponentType() == itk::ImageIOBase::IOComponentEnum::USHORT)
    PropagateParametersToMiniPipeline<itk::Image<unsigned short, OutputImageDimension>>();
  else if (!strcmp(imageIO->GetNameOfClass(), "HndImageIO") || !strcmp(imageIO->GetNameOfClass(), "XimImageIO"))
    PropagateParametersToMiniPipeline<itk::Image<unsigned int, OutputImageDimension>>();
  else
    PropagateParametersToMiniPipeline<OutputImageType>();

  // Set output information as provided by the pipeline
  m_StreamingFilter->UpdateOutputInformation();
  TOutputImage * output = this->GetOutput();
  output->SetOrigin(m_StreamingFilter->GetOutput()->GetOrigin());
  output->SetSpacing(m_StreamingFilter->GetOutput()->GetSpacing());
  output->SetDirection(m_StreamingFilter->GetOutput()->GetDirection());
  output->SetLargestPossibleRegion(m_StreamingFilter->GetOutput()->GetLargestPossibleRegion());
}

//--------------------------------------------------------------------
template <class TOutputImage>
void
ProjectionsReader<TOutputImage>::GenerateData()
{
  TOutputImage * output = this->GetOutput();
  m_StreamingFilter->SetNumberOfStreamDivisions(output->GetRequestedRegion().GetSize(TOutputImage::ImageDimension - 1));
  m_StreamingFilter->GetOutput()->SetRequestedRegion(output->GetRequestedRegion());
  m_StreamingFilter->Update();
  this->GraftOutput(m_StreamingFilter->GetOutput());
}

//--------------------------------------------------------------------
template <class TOutputImage>
template <class TInputImage>
void
ProjectionsReader<TOutputImage>::PropagateParametersToMiniPipeline()
{
  TInputImage *     nextInput = nullptr;
  OutputImageType * output = nullptr;

  // Vector component selection
  if (m_VectorComponentSelectionFilter.GetPointer() != nullptr)
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
    using RawType = typename itk::ImageSeriesReader<TInputImage>;
    auto * raw = dynamic_cast<RawType *>(m_RawDataReader.GetPointer());
    assert(raw != nullptr);
    raw->SetFileNames(this->GetFileNames());
    raw->SetImageIO(m_ImageIO);
    nextInput = raw->GetOutput();

    // Image information
    OutputImageSpacingType defaultSpacing;
    defaultSpacing.Fill(itk::NumericTraits<typename OutputImageType::SpacingValueType>::max());
    OutputImagePointType defaultOrigin;
    defaultOrigin.Fill(itk::NumericTraits<typename OutputImageType::PointValueType>::max());
    OutputImageDirectionType defaultDirection;
    defaultDirection.Fill(itk::NumericTraits<typename OutputImageType::PointValueType>::max());
    if (m_Spacing != defaultSpacing || m_Origin != defaultOrigin || m_Direction != defaultDirection)
    {
      if (m_ChangeInformationFilter.GetPointer() == nullptr)
      {
        itkGenericExceptionMacro(<< "Can not change image information with this input (not implemented)");
      }
      else
      {
        using ChangeInfoType = itk::ChangeInformationImageFilter<TInputImage>;
        auto * cif = dynamic_cast<ChangeInfoType *>(m_ChangeInformationFilter.GetPointer());
        assert(cif != nullptr);
        if (m_Spacing != defaultSpacing)
        {
          cif->SetOutputSpacing(m_Spacing);
          cif->ChangeSpacingOn();
        }
        if (m_Origin != defaultOrigin)
        {
          cif->SetOutputOrigin(m_Origin);
          cif->ChangeOriginOn();
        }
        if (m_Direction != defaultDirection)
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
    if (m_LowerBoundaryCropSize != defaultCropSize || m_UpperBoundaryCropSize != defaultCropSize)
    {
      if (m_CropFilter.GetPointer() == nullptr)
      {
        itkGenericExceptionMacro(<< "Can not crop images read with this input (not implemented)");
      }
      else
      {
        using CropType = itk::CropImageFilter<TInputImage, TInputImage>;
        auto * crop = dynamic_cast<CropType *>(m_CropFilter.GetPointer());
        assert(crop != nullptr);
        crop->SetLowerBoundaryCropSize(m_LowerBoundaryCropSize);
        crop->SetUpperBoundaryCropSize(m_UpperBoundaryCropSize);
        crop->SetInput(nextInput);
        nextInput = crop->GetOutput();
      }
    }

    // Elekta raw data converter
    auto * nextInputBase = dynamic_cast<itk::ImageBase<OutputImageDimension> *>(nextInput);
    assert(nextInputBase != nullptr);
    ConnectElektaRawFilter(&nextInputBase);
    nextInput = dynamic_cast<TInputImage *>(nextInputBase);
    assert(nextInput != nullptr);

    // Conditional median
    MedianRadiusType defaultMedianRadius;
    defaultMedianRadius.Fill(0);
    if (m_MedianRadius != defaultMedianRadius)
    {
      if (m_ConditionalMedianFilter.GetPointer() == nullptr)
      {
        itkGenericExceptionMacro(<< "Can not apply conditional median filter on this input (not implemented)");
      }
      else
      {
        using ConditionalMedianType = rtk::ConditionalMedianImageFilter<TInputImage>;
        auto * cond = dynamic_cast<ConditionalMedianType *>(m_ConditionalMedianFilter.GetPointer());
        assert(cond != nullptr);
        cond->SetRadius(m_MedianRadius);
        cond->SetInput(nextInput);
        cond->SetThresholdMultiplier(m_ConditionalMedianThresholdMultiplier);
        nextInput = cond->GetOutput();
      }
    }

    // Binning
    ShrinkFactorsType defaultShrinkFactors;
    defaultShrinkFactors.Fill(1);
    if (m_ShrinkFactors != defaultShrinkFactors)
    {
      if (m_BinningFilter.GetPointer() == nullptr)
      {
        itkGenericExceptionMacro(<< "Can not bin / shrink images read with this input (not implemented)");
      }
      else
      {
        using BinType = itk::BinShrinkImageFilter<TInputImage, TInputImage>;
        auto * bin = dynamic_cast<BinType *>(m_BinningFilter.GetPointer());
        assert(bin != nullptr);
        bin->SetShrinkFactors(m_ShrinkFactors);
        bin->SetInput(nextInput);
        nextInput = bin->GetOutput();
      }
    }

    // Boellaard scatter correction
    if (m_NonNegativityConstraintThreshold != itk::NumericTraits<double>::NonpositiveMin() ||
        m_ScatterToPrimaryRatio != 0.)
    {
      if (m_ScatterFilter.GetPointer() == nullptr)
      {
        itkGenericExceptionMacro(<< "Can not use Boellaard scatter correction with this input (not implemented)");
      }
      else
      {
        using ScatterFilterType = rtk::BoellaardScatterCorrectionImageFilter<TInputImage, TInputImage>;
        auto * scatter = dynamic_cast<ScatterFilterType *>(m_ScatterFilter.GetPointer());
        assert(scatter != nullptr);
        scatter->SetAirThreshold(m_AirThreshold);
        scatter->SetScatterToPrimaryRatio(m_ScatterToPrimaryRatio);
        if (m_NonNegativityConstraintThreshold != itk::NumericTraits<double>::NonpositiveMin())
          scatter->SetNonNegativityConstraintThreshold(m_NonNegativityConstraintThreshold);
        scatter->SetInput(nextInput);
        nextInput = scatter->GetOutput();
      }
    }

    // LUTbasedVariableI0RawToAttenuationImageFilter
    if (m_I0 != itk::NumericTraits<double>::NonpositiveMin())
    {
      if (m_RawToAttenuationFilter.GetPointer() == nullptr)
      {
        itkGenericExceptionMacro(
          << "Can not use I0 in LUTbasedVariableI0RawToAttenuationImageFilter with this input (not implemented)");
      }
      else
      {
        nextInputBase = dynamic_cast<itk::ImageBase<OutputImageDimension> *>(nextInput);
        assert(nextInputBase != nullptr);
        PropagateI0(&nextInputBase);
        nextInput = dynamic_cast<TInputImage *>(nextInputBase);
        assert(nextInput != nullptr);
      }
    }

    // Raw to attenuation or cast filter, change of type
    if (m_RawToAttenuationFilter.GetPointer() != nullptr)
    {
      // Check if Ora pointer
      using OraRawType = rtk::OraLookupTableImageFilter<OutputImageType>;
      auto * oraraw = dynamic_cast<OraRawType *>(m_RawToAttenuationFilter.GetPointer());
      if (oraraw != nullptr)
      {
        oraraw->SetComputeLineIntegral(m_ComputeLineIntegral);
        oraraw->SetFileNames(m_FileNames);
      }

      // Cast or convert to line integral depending on m_ComputeLineIntegral
      using IToIFilterType = itk::ImageToImageFilter<TInputImage, OutputImageType>;
      IToIFilterType * itoi = nullptr;
      if (m_ComputeLineIntegral || oraraw != nullptr)
        itoi = dynamic_cast<IToIFilterType *>(m_RawToAttenuationFilter.GetPointer());
      else
        itoi = dynamic_cast<IToIFilterType *>(m_RawCastFilter.GetPointer());
      assert(itoi != nullptr);
      itoi->SetInput(nextInput);
      output = itoi->GetOutput();

      // Release output data of m_RawDataReader if conversion occurs
      itoi->ReleaseDataFlagOn();
    }
    else
    {
      output = dynamic_cast<OutputImageType *>(nextInput);
      assert(output != nullptr);
    }

    // ESRF raw to attenuation converter also needs the filenames
    using EdfRawFilterType = rtk::EdfRawToAttenuationImageFilter<TInputImage, OutputImageType>;
    auto * edf = dynamic_cast<EdfRawFilterType *>(m_RawToAttenuationFilter.GetPointer());
    if (edf)
      edf->SetFileNames(this->GetFileNames());

    // Water coefficients
    if (!m_WaterPrecorrectionCoefficients.empty())
    {
      m_WaterPrecorrectionFilter->SetCoefficients(m_WaterPrecorrectionCoefficients);
      m_WaterPrecorrectionFilter->SetInput(output);
      output = m_WaterPrecorrectionFilter->GetOutput();
    }
  }
  // Streaming image filter
  m_StreamingFilter->SetInput(output);
}

//--------------------------------------------------------------------
template <class TOutputImage>
void
ProjectionsReader<TOutputImage>::ConnectElektaRawFilter(itk::ImageBase<OutputImageDimension> ** nextInputBase)
{
  if (m_ElektaRawFilter.GetPointer() != nullptr)
  {
    using ElektaRawType = rtk::ElektaSynergyRawLookupTableImageFilter<itk::Image<unsigned short, OutputImageDimension>,
                                                                      itk::Image<unsigned short, OutputImageDimension>>;
    auto * elektaRaw = dynamic_cast<ElektaRawType *>(m_ElektaRawFilter.GetPointer());
    assert(elektaRaw != nullptr);
    using InputImageType = typename itk::Image<unsigned short, OutputImageDimension>;
    auto * nextInput = dynamic_cast<InputImageType *>(*nextInputBase);
    elektaRaw->SetInput(nextInput);
    *nextInputBase = elektaRaw->GetOutput();
  }
}

//--------------------------------------------------------------------
template <class TOutputImage>
void
ProjectionsReader<TOutputImage>::PropagateI0(itk::ImageBase<OutputImageDimension> ** nextInputBase)
{
  using UnsignedShortImageType = itk::Image<unsigned short, OutputImageDimension>;
  auto * nextInputUShort = dynamic_cast<UnsignedShortImageType *>(*nextInputBase);
  if (nextInputUShort != nullptr)
  {
    if (m_I0 == 0)
    {
      using I0EstimationType = rtk::I0EstimationProjectionFilter<UnsignedShortImageType, UnsignedShortImageType>;
      auto * i0est = dynamic_cast<I0EstimationType *>(m_I0EstimationFilter.GetPointer());
      assert(i0est != nullptr);
      i0est->SetInput(nextInputUShort);
      *nextInputBase = i0est->GetOutput();
    }
    using I0Type = rtk::LUTbasedVariableI0RawToAttenuationImageFilter<UnsignedShortImageType, OutputImageType>;
    auto * i0 = dynamic_cast<I0Type *>(m_RawToAttenuationFilter.GetPointer());
    i0->SetI0(m_I0);
    i0->SetIDark(m_IDark);
  }

  using UnsignedIntImageType = itk::Image<unsigned int, OutputImageDimension>;
  auto * nextInputUInt = dynamic_cast<UnsignedIntImageType *>(*nextInputBase);
  if (nextInputUInt != nullptr)
  {
    if (m_I0 == 0)
    {
      using I0EstimationType = rtk::I0EstimationProjectionFilter<UnsignedIntImageType, UnsignedIntImageType>;
      auto * i0est = dynamic_cast<I0EstimationType *>(m_I0EstimationFilter.GetPointer());
      assert(i0est != nullptr);
      i0est->SetInput(nextInputUInt);
      *nextInputBase = i0est->GetOutput();
    }
    using I0Type = rtk::VarianObiRawImageFilter<UnsignedIntImageType, OutputImageType>;
    auto * i0 = dynamic_cast<I0Type *>(m_RawToAttenuationFilter.GetPointer());
    i0->SetI0(m_I0);
    i0->SetIDark(m_IDark);
  }
  // Pipeline connection for m_RawToAttenuationFilter is done after the call to this function
}

} // namespace rtk

#endif
