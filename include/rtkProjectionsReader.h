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

#ifndef rtkProjectionsReader_h
#define rtkProjectionsReader_h

// ITK
#include <itkImageSource.h>
#include <itkImageIOFactory.h>
#include <itkStreamingImageFilter.h>

// RTK
#include "rtkWaterPrecorrectionImageFilter.h"
#include "rtkConditionalMedianImageFilter.h"

// Standard lib
#include <vector>
#include <string>

namespace rtk
{

/** \class ProjectionsReader
 *
 * This is the universal projections reader of rtk (raw data converted to
 * attenuation). Currently handles his (Elekta Synergy), hnd (Varian OBI),
 * edf (ESRF), XRad. For all other ITK file formats (mha, tif, ...), it is
 * assumed that the attenuation is directly passed if the pixel type is not
 * unsigned short and there is no processing. Optionnally, one can activate
 * cropping, binning, scatter correction, etc. The details of the mini-
 * pipeline is provided below, note that dashed filters are shunt if they
 * are not required according to parameters.
 *
 * \dot
 * digraph ProjectionsReader {
 *
 * Output [label="Output (Projections)", shape=Mdiamond];
 *
 * node [shape=box];
 * Raw [label="itk::ImageSeriesReader" URL="\ref itk::ImageSeriesReader"];
 * ElektaRaw [label="rtk::ElektaSynergyRawLookupTableImageFilter"
 *            URL="\ref rtk::ElektaSynergyRawLookupTableImageFilter"];
 * ChangeInformation [label="itk::ChangeInformationImageFilter"
 *                    URL="\ref itk::ChangeInformationImageFilter" style=dashed];
 * Crop [label="itk::CropImageFilter" URL="\ref itk::CropImageFilter" style=dashed];
 * Binning [label="itk::BinShrinkImageFilter" URL="\ref itk::BinShrinkImageFilter" style=dashed];
 * ConditionalMedian [label="rtk::ConditionalMedianImageFilter"
 *                    URL="\ref rtk::ConditionalMedianImageFilter" style=dashed];
 * Scatter [label="rtk::BoellaardScatterCorrectionImageFilter"
 *          URL="\ref rtk::BoellaardScatterCorrectionImageFilter" style=dashed];
 * I0est [label="rtk::I0EstimationProjectionFilter" URL="\ref rtk::I0EstimationProjectionFilter" style=dashed];
 * BeforeLUT [label="", fixedsize="false", width=0, height=0, shape=none];
 * LUT [label="rtk::LUTbasedVariableI0RawToAttenuationImageFilter"
 *      URL="\ref rtk::LUTbasedVariableI0RawToAttenuationImageFilter"];
 * BeforeVarian [label="", fixedsize="false", width=0, height=0, shape=none];
 * Varian [label="rtk::VarianObiRawImageFilter" URL="\ref rtk::VarianObiRawImageFilter"];
 * WPC [label="rtk::WaterPrecorrectionImageFilter" URL="\ref rtk::WaterPrecorrectionImageFilter" style=dashed];
 * Streaming [label="itk::StreamingImageFilter" URL="\ref itk::StreamingImageFilter"];
 * BeforeEDF [label="", fixedsize="false", width=0, height=0, shape=none];
 * EDF [label="rtk::EdfRawToAttenuationImageFilter"  URL="\ref rtk::EdfRawToAttenuationImageFilter"];
 * BeforeXRad [label="", fixedsize="false", width=0, height=0, shape=none];
 * XRad [label="rtk::XRadRawToAttenuationImageFilter"  URL="\ref rtk::XRadRawToAttenuationImageFilter"];
 * Cast [label="itk::CastImageFilter"  URL="\ref itk::CastImageFilter"];
 * OraRaw [label="rtk::OraLookupTableImageFilter" URL="\ref rtk::OraLookupTableImageFilter"];
 *
 * Raw->ChangeInformation [label="Default"]
 * ChangeInformation->Crop
 * Crop->ElektaRaw [label="Elekta"]
 * Binning->OraRaw [label="Ora && ushort"]
 * OraRaw->WPC
 * ElektaRaw->ConditionalMedian
 * Crop->ConditionalMedian[label="Default"]
 * ConditionalMedian->Binning
 * Binning->Scatter [label="Elekta, Varian, IBA, ushort"]
 * Scatter->I0est [label="Default"]
 * I0est->BeforeLUT
 * BeforeLUT->LUT [label="ComputeLineIntegral\n(default)"]
 * BeforeLUT->Cast
 * LUT->WPC
 * I0est->BeforeVarian [label="Varian"]
 * BeforeVarian->Varian [label="ComputeLineIntegral\n(default)"]
 * BeforeVarian->Cast
 * Varian->WPC
 * EDF->WPC
 * ChangeInformation->BeforeEDF [label="edf short"]
 * BeforeEDF->EDF [label="ComputeLineIntegral\n(default)"]
 * BeforeEDF->Cast
 * XRad->WPC
 * ChangeInformation->BeforeXRad [label="XRad"]
 * BeforeXRad->XRad [label="ComputeLineIntegral\n(default)"]
 * BeforeXRad->Cast
 * Cast->WPC
 * WPC->Streaming
 * Streaming->Output
 *
 * Binning->WPC [label="Default"]
 *
 * {rank=same; XRad EDF Varian LUT OraRaw}
 * }
 * \enddot
 *
 * \test rtkedftest.cxx, rtkelektatest.cxx, rtkimagxtest.cxx,
 * rtkdigisenstest.cxx, rtkxradtest.cxx, rtkvariantest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup RTK ImageSource
 */
template <class TOutputImage>
class ITK_TEMPLATE_EXPORT ProjectionsReader : public itk::ImageSource<TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(ProjectionsReader);

  /** Standard class type alias. */
  using Self = ProjectionsReader;
  using Superclass = itk::ImageSource<TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(ProjectionsReader);

  /** Some convenient type alias. */
  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;
  using OutputImageRegionType = typename OutputImageType::RegionType;
  using OutputImagePixelType = typename OutputImageType::PixelType;
  using OutputImageDirectionType = typename OutputImageType::DirectionType;
  using OutputImageSpacingType = typename OutputImageType::SpacingType;
  using OutputImagePointType = typename OutputImageType::PointType;
  using OutputImageSizeType = typename OutputImageType::SizeType;

  using FileNamesContainer = std::vector<std::string>;
  using ShrinkFactorsType = itk::FixedArray<unsigned int, TOutputImage::ImageDimension>;
  using MedianRadiusType = typename rtk::ConditionalMedianImageFilter<TOutputImage>::MedianRadiusType;
  using WaterPrecorrectionVectorType = std::vector<double>;

  /** Typdefs of filters of the mini-pipeline that do not depend on the raw
   * data type. */
  using WaterPrecorrectionType = rtk::WaterPrecorrectionImageFilter<TOutputImage, TOutputImage>;
  using StreamingType = itk::StreamingImageFilter<TOutputImage, TOutputImage>;

  /** ImageDimension constant */
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;

  /** Set the vector of strings that contains the file names. Files
   * are processed in sequential order. */
  void
  SetFileNames(const FileNamesContainer & name)
  {
    if (m_FileNames != name)
    {
      m_FileNames = name;
      this->Modified();
    }
  }
  const FileNamesContainer &
  GetFileNames() const
  {
    return m_FileNames;
  }

  /** Set/Get the new image information for the input projections before any pre-processing. */
  itkSetMacro(Origin, OutputImagePointType);
  itkGetConstMacro(Origin, OutputImagePointType);

  itkSetMacro(Spacing, OutputImageSpacingType);
  itkGetConstMacro(Spacing, OutputImageSpacingType);

  itkSetMacro(Direction, OutputImageDirectionType);
  itkGetConstMacro(Direction, OutputImageDirectionType);

  /** Set/Get the cropping sizes for the upper and lower boundaries. */
  itkSetMacro(UpperBoundaryCropSize, OutputImageSizeType);
  itkGetConstMacro(UpperBoundaryCropSize, OutputImageSizeType);
  itkSetMacro(LowerBoundaryCropSize, OutputImageSizeType);
  itkGetConstMacro(LowerBoundaryCropSize, OutputImageSizeType);

  /** Set/Get itk::BinShrinkImageFilter parameters */
  itkSetMacro(ShrinkFactors, ShrinkFactorsType);
  itkGetConstReferenceMacro(ShrinkFactors, ShrinkFactorsType);

  /** Set/Get itk::ConditionalMedianImageFilter parameters */
  itkSetMacro(MedianRadius, MedianRadiusType);
  itkGetConstReferenceMacro(MedianRadius, MedianRadiusType);
  itkGetMacro(ConditionalMedianThresholdMultiplier, double);
  itkSetMacro(ConditionalMedianThresholdMultiplier, double);

  /** Set/Get rtk::BoellaardScatterCorrectionImageFilter */
  itkGetMacro(AirThreshold, double);
  itkSetMacro(AirThreshold, double);

  itkGetMacro(ScatterToPrimaryRatio, double);
  itkSetMacro(ScatterToPrimaryRatio, double);

  itkGetMacro(NonNegativityConstraintThreshold, double);
  itkSetMacro(NonNegativityConstraintThreshold, double);

  /** Set/Get rtk::LUTbasedVariableI0RawToAttenuationImageFilter. Default is
   * used if not set which depends on the input image type max. If equals 0,
   * automated estimation is activated using rtk::I0EstimationProjectionFilter.
   */
  itkGetMacro(I0, double);
  itkSetMacro(I0, double);

  /** Set/Get the intensity with no photons for
   * rtk::LUTbasedVariableI0RawToAttenuationImageFilter.
   */
  itkGetMacro(IDark, double);
  itkSetMacro(IDark, double);

  /** Get / Set the water precorrection parameters. */
  itkGetMacro(WaterPrecorrectionCoefficients, WaterPrecorrectionVectorType);
  virtual void
  SetWaterPrecorrectionCoefficients(const WaterPrecorrectionVectorType _arg)
  {
    if (this->m_WaterPrecorrectionCoefficients != _arg)
    {
      this->m_WaterPrecorrectionCoefficients = _arg;
      this->Modified();
    }
  }

  /** Convert the projection data to line integrals after pre-processing.
  ** Default is on. */
  itkSetMacro(ComputeLineIntegral, bool);
  itkGetConstMacro(ComputeLineIntegral, bool);
  itkBooleanMacro(ComputeLineIntegral);

  /** Set/Get the index of the component to be extracted
   * if the projection data contains vectors instead of scalars. */
  itkGetMacro(VectorComponent, unsigned int);
  itkSetMacro(VectorComponent, unsigned int);

  /** Set/Get the ImageIO helper class. Often this is created via the object
   * factory mechanism that determines whether a particular ImageIO can
   * read a certain file. This method provides a way to get the ImageIO
   * instance that is created. Or you can directly specify the ImageIO
   * to use to read a particular file in case the factory mechanism will
   * not work properly (e.g., unknown or unusual extension). */
  itkSetObjectMacro(ImageIO, itk::ImageIOBase);
  itkGetConstObjectMacro(ImageIO, itk::ImageIOBase);

  /** Prepare the allocation of the output image during the first back
   * propagation of the pipeline. */
  void
  GenerateOutputInformation() override;

protected:
  ProjectionsReader();
  ~ProjectionsReader() override = default;
  void
  PrintSelf(std::ostream & os, itk::Indent indent) const override;

  /** Does the real work. */
  void
  GenerateData() override;

  /** A list of filenames to be processed. */
  FileNamesContainer m_FileNames;

private:
  /** Function that checks and propagates the parameters of the class to the
   * mini-pipeline. Due to concept checking, i0 propagation can only be done
   * with unsigned shorts and is left apart without template. */
  template <class TInputImage>
  void
  PropagateParametersToMiniPipeline();
  void
  ConnectElektaRawFilter(itk::ImageBase<OutputImageDimension> ** nextInputBase);
  void
  PropagateI0(itk::ImageBase<OutputImageDimension> ** nextInputBase);

  /** The projections reader which template depends on the scanner.
   * It is not typed because we want to keep the data as on disk.
   * The pointer is stored to reference the filter and avoid its destruction. */
  itk::ProcessObject::Pointer m_RawDataReader;

  /** Pointers for pre-processing filters that are created only when required. */
  itk::ProcessObject::Pointer m_VectorComponentSelectionFilter;
  itk::ProcessObject::Pointer m_ChangeInformationFilter;
  itk::ProcessObject::Pointer m_ElektaRawFilter;
  itk::ProcessObject::Pointer m_CropFilter;
  itk::ProcessObject::Pointer m_ConditionalMedianFilter;
  itk::ProcessObject::Pointer m_BinningFilter;
  itk::ProcessObject::Pointer m_ScatterFilter;
  itk::ProcessObject::Pointer m_I0EstimationFilter;

  /** Conversion from raw to attenuation. Depends on the input image type, set
   * to binning filter output by default. */
  typename itk::ImageSource<TOutputImage>::Pointer m_RawToAttenuationFilter;

  /** When m_ComputeLineIntegral is off, one just casts the value instead of
   * doing a line integral. */
  typename itk::ImageSource<TOutputImage>::Pointer m_RawCastFilter;

  /** Pointers for post-processing filters that are created only when required. */
  typename WaterPrecorrectionType::Pointer m_WaterPrecorrectionFilter;
  typename StreamingType::Pointer          m_StreamingFilter;

  /** Image IO object which is stored to create the pipe only when required */
  itk::ImageIOBase::Pointer m_ImageIO{ nullptr };

  /** Copy of parameters for the mini-pipeline. Parameters are checked and
   * propagated when required in the GenerateOutputInformation. Refer to the
   * documentation of the corresponding filter for more information. */
  OutputImagePointType         m_Origin;
  OutputImageSpacingType       m_Spacing;
  OutputImageDirectionType     m_Direction;
  OutputImageSizeType          m_LowerBoundaryCropSize;
  OutputImageSizeType          m_UpperBoundaryCropSize;
  ShrinkFactorsType            m_ShrinkFactors;
  MedianRadiusType             m_MedianRadius;
  double                       m_AirThreshold{ 32000 };
  double                       m_ScatterToPrimaryRatio{ 0. };
  double                       m_NonNegativityConstraintThreshold{ itk::NumericTraits<double>::NonpositiveMin() };
  double                       m_I0{ itk::NumericTraits<double>::NonpositiveMin() };
  double                       m_IDark{ 0. };
  double                       m_ConditionalMedianThresholdMultiplier{ 1. };
  WaterPrecorrectionVectorType m_WaterPrecorrectionCoefficients;
  bool                         m_ComputeLineIntegral{ true };
  unsigned int                 m_VectorComponent{ 0 };
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkProjectionsReader.hxx"
#endif

#endif // rtkProjectionsReader_h
