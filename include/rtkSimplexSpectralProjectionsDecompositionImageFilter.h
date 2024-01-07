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

#ifndef rtkSimplexSpectralProjectionsDecompositionImageFilter_h
#define rtkSimplexSpectralProjectionsDecompositionImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkAmoebaOptimizer.h>
#include "rtkSchlomka2008NegativeLogLikelihood.h"
#include "rtkDualEnergyNegativeLogLikelihood.h"

namespace rtk
{
/** \class SimplexSpectralProjectionsDecompositionImageFilter
 * \brief Decomposition of spectral projection images into material projections
 *
 * See the reference paper: "Experimental feasibility of multi-energy photon-counting
 * K-edge imaging in pre-clinical computed tomography", Schlomka et al, PMB 2008
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename DecomposedProjectionsType,
          typename MeasuredProjectionsType,
          typename IncidentSpectrumImageType = itk::VectorImage<float, 2>,
          typename DetectorResponseImageType = itk::Image<float, 2>,
          typename MaterialAttenuationsImageType = itk::Image<float, 2>>
class ITK_TEMPLATE_EXPORT SimplexSpectralProjectionsDecompositionImageFilter
  : public itk::ImageToImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(SimplexSpectralProjectionsDecompositionImageFilter);

  /** Standard class type alias. */
  using Self = SimplexSpectralProjectionsDecompositionImageFilter;
  using Superclass = itk::ImageToImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = DecomposedProjectionsType;
  using OutputImageType = DecomposedProjectionsType;

  /** Convenient information */
  using ThresholdsType = itk::VariableLengthVector<double>;
  using MeanAttenuationInBinType = itk::VariableSizeMatrix<double>;
  using DetectorResponseType = vnl_matrix<double>;
  using MaterialAttenuationsType = vnl_matrix<double>;
  using CostFunctionType = ProjectionsDecompositionNegativeLogLikelihood;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
#ifdef itkOverrideGetNameOfClassMacro
  itkOverrideGetNameOfClassMacro(SimplexSpectralProjectionsDecompositionImageFilter);
#else
  itkTypeMacro(SimplexSpectralProjectionsDecompositionImageFilter, ImageToImageFilter);
#endif

  /** Set/Get the input material-decomposed stack of projections (only used for initialization) */
  void
  SetInputDecomposedProjections(const DecomposedProjectionsType * DecomposedProjections);
  typename DecomposedProjectionsType::ConstPointer
  GetInputDecomposedProjections();

  /** Set/Get the input stack of measured projections (to be decomposed in materials) */
  void
  SetInputMeasuredProjections(const MeasuredProjectionsType * SpectralProjections);
  typename MeasuredProjectionsType::ConstPointer
  GetInputMeasuredProjections();

  /** Set/Get the detector response as an image */
  void
  SetDetectorResponse(const DetectorResponseImageType * DetectorResponse);
  typename DetectorResponseImageType::ConstPointer
  GetDetectorResponse();

  /** Set/Get the material attenuations as an image */
  void
  SetMaterialAttenuations(const MaterialAttenuationsImageType * MaterialAttenuations);
  typename MaterialAttenuationsImageType::ConstPointer
  GetMaterialAttenuations();

  /** Set/Get the incident spectrum input images */
  void
  SetInputIncidentSpectrum(const IncidentSpectrumImageType * IncidentSpectrum);
  void
  SetInputSecondIncidentSpectrum(const IncidentSpectrumImageType * SecondIncidentSpectrum);
  typename IncidentSpectrumImageType::ConstPointer
  GetInputIncidentSpectrum();
  typename IncidentSpectrumImageType::ConstPointer
  GetInputSecondIncidentSpectrum();

  /** Get / Set the number of iterations. Default is 300. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  itkSetMacro(NumberOfEnergies, unsigned int);
  itkGetMacro(NumberOfEnergies, unsigned int);

  itkSetMacro(NumberOfMaterials, unsigned int);
  itkGetMacro(NumberOfMaterials, unsigned int);

  itkSetMacro(OptimizeWithRestarts, bool);
  itkGetMacro(OptimizeWithRestarts, bool);

  itkSetMacro(Thresholds, ThresholdsType);
  itkGetMacro(Thresholds, ThresholdsType);

  itkSetMacro(NumberOfSpectralBins, unsigned int);
  itkGetMacro(NumberOfSpectralBins, unsigned int);

  itkSetMacro(OutputInverseCramerRaoLowerBound, bool);
  itkGetMacro(OutputInverseCramerRaoLowerBound, bool);

  itkSetMacro(OutputFischerMatrix, bool);
  itkGetMacro(OutputFischerMatrix, bool);

  itkSetMacro(LogTransformEachBin, bool);
  itkGetMacro(LogTransformEachBin, bool);

  itkSetMacro(GuessInitialization, bool);
  itkGetMacro(GuessInitialization, bool);

  itkSetMacro(IsSpectralCT, bool);
  itkGetMacro(IsSpectralCT, bool);

protected:
  SimplexSpectralProjectionsDecompositionImageFilter();
  ~SimplexSpectralProjectionsDecompositionImageFilter() override = default;

  void
  GenerateOutputInformation() override;

  void
  GenerateInputRequestedRegion() override;

  void
  BeforeThreadedGenerateData() override;
  void
  DynamicThreadedGenerateData(const typename DecomposedProjectionsType::RegionType & outputRegionForThread) override;

  /**  Create the Output */
  using DataObjectPointerArraySizeType = itk::ProcessObject::DataObjectPointerArraySizeType;
  using Superclass::MakeOutput;
  itk::DataObject::Pointer
  MakeOutput(DataObjectPointerArraySizeType idx) override;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void
  VerifyInputInformation() const override
  {}

  /** Parameters */
  MaterialAttenuationsType m_MaterialAttenuations;
  DetectorResponseType     m_DetectorResponse;
  ThresholdsType           m_Thresholds;
  MeanAttenuationInBinType m_MeanAttenuationInBin;
  bool                     m_OutputInverseCramerRaoLowerBound;
  bool                     m_OutputFischerMatrix;
  bool                     m_LogTransformEachBin;
  bool                     m_GuessInitialization;
  bool                     m_IsSpectralCT; // If not, it is dual energy CT
  bool                     m_OptimizeWithRestarts;
  unsigned int             m_NumberOfIterations;
  unsigned int             m_NumberOfMaterials;
  unsigned int             m_NumberOfEnergies;
  unsigned int             m_NumberOfSpectralBins;

}; // end of class

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkSimplexSpectralProjectionsDecompositionImageFilter.hxx"
#endif

#endif
