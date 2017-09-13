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

#ifndef rtkDualEnergyForwardModelImageFilter_h
#define rtkDualEnergyForwardModelImageFilter_h

#include "rtkSimplexProjectionsDecompositionImageFilter.h"
#include "rtkDualEnergyNegativeLogLikelihood.h"

#include <itkInPlaceImageFilter.h>

namespace rtk
{
  /** \class DualEnergyForwardModelImageFilter
   * \brief Forward model for the decomposition of dual energy projection images into material projections.
   *
   * Computes both energy-integrating projections from the material-decomposed projections,
   * the combined incident spectrum and detector response, and the material attenuation curves.
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template<typename DecomposedProjectionsType,
         typename MeasuredProjectionsType,
         typename SpectrumAndDetectorResponseImageType = itk::VectorImage<float, 2>,
         typename MaterialAttenuationsImageType = itk::Image<float, 2> >
class ITK_EXPORT DualEnergyForwardModelImageFilter :
  public itk::InPlaceImageFilter<MeasuredProjectionsType, MeasuredProjectionsType>
{
public:
  /** Standard class typedefs. */
  typedef DualEnergyForwardModelImageFilter                                                Self;
  typedef itk::ImageToImageFilter<MeasuredProjectionsType, MeasuredProjectionsType>      Superclass;
  typedef itk::SmartPointer<Self>                                                        Pointer;
  typedef itk::SmartPointer<const Self>                                                  ConstPointer;

  /** Some convenient typedefs. */
  typedef MeasuredProjectionsType       InputImageType;
  typedef MeasuredProjectionsType       OutputImageType;

  /** Convenient information */
  typedef itk::VariableLengthVector<unsigned int>           ThresholdsType;
  typedef itk::VariableSizeMatrix<double>                   DetectorResponseType;
  typedef itk::VariableSizeMatrix<double>                   MaterialAttenuationsType;

  /** Typedefs of each subfilter of this composite filter */
  typedef DualEnergyNegativeLogLikelihood                             CostFunctionType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(DualEnergyForwardModelImageFilter, InPlaceImageFilter)

  /** Set/Get the incident spectrum input image */
  void SetInputSpectrumAndDetectorResponseHighEnergy(const SpectrumAndDetectorResponseImageType* SpectrumAndDetectorResponse);
  typename SpectrumAndDetectorResponseImageType::ConstPointer GetInputSpectrumAndDetectorResponseHighEnergy();
  void SetInputSpectrumAndDetectorResponseLowEnergy(const SpectrumAndDetectorResponseImageType* SpectrumAndDetectorResponse);
  typename SpectrumAndDetectorResponseImageType::ConstPointer GetInputSpectrumAndDetectorResponseLowEnergy();

  /** Set/Get the input material-decomposed stack of projections (only used for initialization) */
  void SetInputDecomposedProjections(const DecomposedProjectionsType* DecomposedProjections);
  typename DecomposedProjectionsType::ConstPointer GetInputDecomposedProjections();

  /** Set/Get the input stack of measured projections (to be decomposed in materials) */
  void SetInputDualEnergyProjections(const MeasuredProjectionsType* DualEnergyProjections);
  typename MeasuredProjectionsType::ConstPointer GetInputDualEnergyProjections();

  /** Set/Get the material attenuations as an image */
  void SetMaterialAttenuations(const MaterialAttenuationsImageType* MaterialAttenuations);
  typename MaterialAttenuationsImageType::ConstPointer GetMaterialAttenuations();

  /** Set/Get the number of discrete energy levels in the full spectrum (typically 1 per keV) */
  itkSetMacro(NumberOfEnergies, unsigned int)
  itkGetMacro(NumberOfEnergies, unsigned int)

protected:
  DualEnergyForwardModelImageFilter();
  ~DualEnergyForwardModelImageFilter() {}

  void GenerateInputRequestedRegion() ITK_OVERRIDE;

  void BeforeThreadedGenerateData() ITK_OVERRIDE;
  void ThreadedGenerateData(const typename OutputImageType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId)) ITK_OVERRIDE;

  /** The inputs should not be in the same space so there is nothing
   * to verify. */
  void VerifyInputInformation() ITK_OVERRIDE {}

  MaterialAttenuationsType   m_MaterialAttenuations;
  unsigned int               m_NumberOfEnergies;

  /** Parameters */
  unsigned int m_NumberOfIterations;
  bool         m_OptimizeWithRestarts;

private:
  //purposely not implemented
  DualEnergyForwardModelImageFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDualEnergyForwardModelImageFilter.hxx"
#endif

#endif
