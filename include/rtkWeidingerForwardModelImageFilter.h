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
#ifndef rtkWeidingerForwardModelImageFilter_h
#define rtkWeidingerForwardModelImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkMacro.h"

#ifdef RTK_USE_CUDA
#  include <itkCudaImage.h>
#endif

namespace rtk
{
/** \class WeidingerForwardModelImageFilter
 * \brief Performs intermediate computations in Weidinger2016
 *
 * This filter performs all computations between forward and
 * back projection in Weidinger2016
 *
 * \author Cyril Mory
 *
 * \ingroup RTK
 *
 */
template <class TDecomposedProjections,
          class TMeasuredProjections,
          class TIncidentSpectrum,
          class TProjections =
            itk::Image<typename TDecomposedProjections::PixelType::ValueType, TDecomposedProjections::ImageDimension>>
class ITK_TEMPLATE_EXPORT WeidingerForwardModelImageFilter
  : public itk::ImageToImageFilter<TDecomposedProjections, TDecomposedProjections>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(WeidingerForwardModelImageFilter);

  /** Standard class type alias. */
  using Self = WeidingerForwardModelImageFilter;
  using Superclass = itk::ImageToImageFilter<TDecomposedProjections, TDecomposedProjections>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkOverrideGetNameOfClassMacro(WeidingerForwardModelImageFilter);

  /** Convenient parameters extracted from template types */
  static constexpr unsigned int nBins = TMeasuredProjections::PixelType::Dimension;
  static constexpr unsigned int nMaterials = TDecomposedProjections::PixelType::Dimension;

  /** Convenient type alias */
  using dataType = typename TDecomposedProjections::PixelType::ValueType;

  /** Define types for output images:
   * - one n-vector per pixel
   * - one nxn-matrix per pixel, but stored as one nxn-vector
  to allow vector back projection */
  using TOutputImage1 = TDecomposedProjections;
  using TPixelOutput2 = itk::Vector<dataType, nMaterials * nMaterials>;
#ifdef RTK_USE_CUDA
  using TOutputImage2 = itk::CudaImage<TPixelOutput2, TDecomposedProjections::ImageDimension>;
#else
  using TOutputImage2 = itk::Image<TPixelOutput2, TDecomposedProjections::ImageDimension>;
#endif

  /** Define the getters for the outputs, with correct types */
  TOutputImage1 *
  GetOutput1();
  TOutputImage2 *
  GetOutput2();

  /** Set methods for all inputs, since they have different types */
  void
  SetInputDecomposedProjections(const TDecomposedProjections * decomposedProjections);
  void
  SetInputMeasuredProjections(const TMeasuredProjections * measuredProjections);
  void
  SetInputIncidentSpectrum(const TIncidentSpectrum * incidentSpectrum);
#ifndef ITK_FUTURE_LEGACY_REMOVE
  void
  SetInputMaterialProjections(const TDecomposedProjections * decomposedProjections);
  void
  SetInputPhotonCounts(const TMeasuredProjections * measuredProjections);
  void
  SetInputSpectrum(const TIncidentSpectrum * incidentSpectrum);
#endif
  void
  SetInputProjectionsOfOnes(const TProjections * projectionsOfOnes);

  /** Typedefs for additional input information */
  using BinnedDetectorResponseType = vnl_matrix<dataType>;
  using MaterialAttenuationsType = vnl_matrix<dataType>;

  /** Set and Get macros for the additional input information */
  itkGetConstReferenceMacro(BinnedDetectorResponse, BinnedDetectorResponseType);
  itkGetConstReferenceMacro(MaterialAttenuations, MaterialAttenuationsType);
  virtual void
  SetBinnedDetectorResponse(const BinnedDetectorResponseType & detResp);
  virtual void
  SetMaterialAttenuations(const MaterialAttenuationsType & matAtt);

protected:
  WeidingerForwardModelImageFilter();
  ~WeidingerForwardModelImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;
  void
  VerifyInputInformation() const override
  {}

  /** Does the real work. */
  void
  DynamicThreadedGenerateData(const typename TOutputImage1::RegionType & outputRegionForThread) override;

  /** Creates the Outputs */
  itk::ProcessObject::DataObjectPointer
  MakeOutput(itk::ProcessObject::DataObjectPointerArraySizeType idx) override;
  itk::ProcessObject::DataObjectPointer
  MakeOutput(const itk::ProcessObject::DataObjectIdentifierType &) override;

  /** Getters for the inputs */
  typename TDecomposedProjections::ConstPointer
  GetInputDecomposedProjections();
  typename TMeasuredProjections::ConstPointer
  GetInputMeasuredProjections();
  typename TIncidentSpectrum::ConstPointer
  GetInputIncidentSpectrum();
#ifndef ITK_FUTURE_LEGACY_REMOVE
  typename TDecomposedProjections::ConstPointer
  GetInputMaterialProjections();
  typename TMeasuredProjections::ConstPointer
  GetInputPhotonCounts();
  typename TIncidentSpectrum::ConstPointer
  GetInputSpectrum();
#endif
  typename TProjections::ConstPointer
  GetInputProjectionsOfOnes();

  /** Additional input parameters */
  BinnedDetectorResponseType m_BinnedDetectorResponse;
  MaterialAttenuationsType   m_MaterialAttenuations;
  itk::SizeValueType         m_NumberOfProjectionsInIncidentSpectrum;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkWeidingerForwardModelImageFilter.hxx"
#endif

#endif
