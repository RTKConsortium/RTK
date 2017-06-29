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

#ifndef rtkSimplexProjectionsDecompositionImageFilter_h
#define rtkSimplexProjectionsDecompositionImageFilter_h

#include "rtkProjectionsDecompositionNegativeLogLikelihood.h"

#include <itkImageToImageFilter.h>
#include <itkAmoebaOptimizer.h>
#include <itkVectorImage.h>
#include <itkVariableSizeMatrix.h>

namespace rtk
{
  /** \class SimplexProjectionsDecompositionImageFilter
   * \brief Base class for decomposition of projection images
   * into material projections for multi-energy CT (dual or spectral)
   *
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template<typename DecomposedProjectionsType,
         typename MeasuredProjectionsType,
         typename IncidentSpectrumImageType = itk::VectorImage<float, 2>,
         typename DetectorResponseImageType = itk::Image<float, 2>,
         typename MaterialAttenuationsImageType = itk::Image<float, 2> >
class ITK_EXPORT SimplexProjectionsDecompositionImageFilter :
  public itk::ImageToImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>
{
public:
  /** Standard class typedefs. */
  typedef SimplexProjectionsDecompositionImageFilter                                        Self;
  typedef itk::ImageToImageFilter<DecomposedProjectionsType, DecomposedProjectionsType>     Superclass;
  typedef itk::SmartPointer<Self>                                                           Pointer;
  typedef itk::SmartPointer<const Self>                                                     ConstPointer;

  /** Some convenient typedefs. */
  typedef DecomposedProjectionsType                 InputImageType;
  typedef DecomposedProjectionsType                 OutputImageType;

  /** Convenient information */
  typedef itk::VariableSizeMatrix<double>                   DetectorResponseType;
  typedef itk::VariableSizeMatrix<double>                   MaterialAttenuationsType;
  typedef itk::VariableLengthVector<double>                 RescalingFactorsType;
  typedef ProjectionsDecompositionNegativeLogLikelihood     CostFunctionType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(SimplexProjectionsDecompositionImageFilter, itk::ImageToImageFilter)

  /** Get / Set the number of iterations. Default is 300. */
  itkGetMacro(NumberOfIterations, unsigned int)
  itkSetMacro(NumberOfIterations, unsigned int)

  /** Set/Get the input material-decomposed stack of projections (only used for initialization) */
  void SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections);
  typename DecomposedProjectionsType::ConstPointer GetInputDecomposedProjections();

  /** Set/Get the input stack of measured projections (to be decomposed in materials) */
  void SetInputMeasuredProjections(const MeasuredProjectionsType* SpectralProjections);
  typename MeasuredProjectionsType::ConstPointer GetInputMeasuredProjections();

  /** Set/Get the detector response as an image */
  void SetDetectorResponse(const DetectorResponseImageType* DetectorResponse);
  typename DetectorResponseImageType::ConstPointer GetDetectorResponse();

  /** Set/Get the material attenuations as an image */
  void SetMaterialAttenuations(const MaterialAttenuationsImageType* MaterialAttenuations);
  typename MaterialAttenuationsImageType::ConstPointer GetMaterialAttenuations();

  itkSetMacro(NumberOfEnergies, unsigned int)
  itkGetMacro(NumberOfEnergies, unsigned int)

  itkSetMacro(NumberOfMaterials, unsigned int)
  itkGetMacro(NumberOfMaterials, unsigned int)

  itkSetMacro(OptimizeWithRestarts, bool)
  itkGetMacro(OptimizeWithRestarts, bool)

  itkSetMacro(RescaleAttenuations, bool)
  itkGetMacro(RescaleAttenuations, bool)

  itkGetMacro(RescalingFactors, RescalingFactorsType)

protected:
  SimplexProjectionsDecompositionImageFilter();
  ~SimplexProjectionsDecompositionImageFilter() {}

  void GenerateOutputInformation() ITK_OVERRIDE;

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  void RescaleMaterialAttenuations();

  /**  Create the Output */
  typedef itk::ProcessObject::DataObjectPointerArraySizeType DataObjectPointerArraySizeType;
  using Superclass::MakeOutput;
  itk::DataObject::Pointer MakeOutput(DataObjectPointerArraySizeType idx) ITK_OVERRIDE;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  MaterialAttenuationsType   m_MaterialAttenuations;
  MaterialAttenuationsType   m_RescaledMaterialAttenuations;
  DetectorResponseType       m_DetectorResponse;
  unsigned int               m_NumberOfEnergies;

  /** Parameters */
  unsigned int          m_NumberOfIterations;
  unsigned int          m_NumberOfMaterials;
  bool                  m_OptimizeWithRestarts;
  bool                  m_RescaleAttenuations;
  RescalingFactorsType  m_RescalingFactors;

private:
  //purposely not implemented
  SimplexProjectionsDecompositionImageFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSimplexProjectionsDecompositionImageFilter.hxx"
#endif

#endif
