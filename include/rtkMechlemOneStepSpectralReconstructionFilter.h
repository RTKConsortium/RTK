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

#ifndef rtkMechlemOneStepSpectralReconstructionFilter_h
#define rtkMechlemOneStepSpectralReconstructionFilter_h

#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkWeidingerForwardModelImageFilter.h"
#include "rtkGetNewtonUpdateImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkNesterovUpdateImageFilter.h"
#include "rtkSeparableQuadraticSurrogateRegularizationImageFilter.h"
#include "rtkAddMatrixAndDiagonalImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

#ifdef RTK_USE_CUDA
#  include "rtkCudaWeidingerForwardModelImageFilter.h"
#endif

namespace rtk
{
/** \class MechlemOneStepSpectralReconstructionFilter
 * \brief Implements the one-step spectral CT inversion method described by Mechlem et al.
 *
 * This filter implements one-step spectral CT inversion method described by Mechlem et al.
 * in their paper "Joint statistical iterative material image reconstruction
 * for spectral computed tomography using a semi-empirical forward model", IEEE TMI 2017
 * It reconstructs a vector-valued volume (each component is a material) from vector-valued
 * projections (each component is the count of photons in an energy bin of the spectral detector).
 * It requires knowledge of the incident spectrum, of the detector's energy distribution and
 * of the materials' matrix of mass-attenuation coefficients as a function of the incident energy.
 *
 * \dot
 * digraph MechlemOneStepSpectralReconstructionFilter {
 *
 * Input0 [ label="Input 0 (Material volumes)"];
 * Input0 [shape=Mdiamond];
 * Input1 [label="Input 1 (Photon counts)"];
 * Input1 [shape=Mdiamond];
 * Input2 [label="Input 2 (Incident spectrum)"];
 * Input2 [shape=Mdiamond];
 * Input3 [label="Input 3 (Support mask)"];
 * Input3 [shape=Mdiamond];
 * Input4 [label="Input 4 (Spatial Regularization Weights)"];
 * Input4 [shape=Mdiamond];
 * Output [label="Output (Material volumes)"];
 * Output [shape=Mdiamond];
 *
 * node [shape=box];
 * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
 * VolumeSource [ label="rtk::ConstantImageSource (1 component volume, full of ones)"
 *                URL="\ref rtk::ConstantImageSource"];
 * SingleComponentProjectionsSource [ label="rtk::ConstantImageSource (1 component projections, full of zeros)"
 *                                    URL="\ref rtk::ConstantImageSource"];
 * VolumeSourceGradients [ label="rtk::ConstantImageSource (m components)" URL="\ref rtk::ConstantImageSource"];
 * VolumeSourceHessians [ label="rtk::ConstantImageSource (m x m components)" URL="\ref rtk::ConstantImageSource"];
 * ProjectionsSource [ label="rtk::ConstantImageSource (m components)" URL="\ref rtk::ConstantImageSource"];
 * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
 * SingleComponentForwardProjection [ label="rtk::ForwardProjectionImageFilter (1 component)"
 *                                    URL="\ref rtk::ForwardProjectionImageFilter"];
 * BackProjectionGradients [ label="rtk::BackProjectionImageFilter (gradients)"
 *                           URL="\ref rtk::BackProjectionImageFilter"];
 * BackProjectionHessians [ label="rtk::BackProjectionImageFilter (hessians)"
 *                          URL="\ref rtk::BackProjectionImageFilter"];
 * Weidinger [ label="rtk::WeidingerForwardModelImageFilter" URL="\ref rtk::WeidingerForwardModelImageFilter"];
 * SQSRegul [ label="rtk::SeparableQuadraticSurrogateRegularizationImageFilter"
 *            URL="\ref rtk::SeparableQuadraticSurrogateRegularizationImageFilter"];
 * MultiplyRegulGradient [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter" style=dashed];
 * MultiplyRegulHessian [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter" style=dashed];
 * AddGradients [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
 * AddHessians [ label="rtk::AddMatrixAndDiagonalImageFilter" URL="\ref rtk::AddMatrixAndDiagonalImageFilter"];
 * Newton [ label="rtk::GetNewtonUpdateImageFilter" URL="\ref rtk::GetNewtonUpdateImageFilter"];
 * Nesterov [ label="rtk::NesterovUpdateImageFilter" URL="\ref rtk::NesterovUpdateImageFilter"];
 * MultiplySupport [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter" style=dashed];
 * Alphak [ label="", fixedsize="false", width=0, height=0, shape=none];
 * NextAlphak [ label="", fixedsize="false", width=0, height=0, shape=none];
 *
 * Input0 -> Alphak [arrowhead=none];
 * Alphak -> ForwardProjection;
 * Alphak -> SQSRegul;
 * ProjectionsSource -> ForwardProjection;
 * Input1 -> Extract;
 * Extract -> Weidinger;
 * Input2 -> Weidinger;
 * ForwardProjection -> Weidinger;
 * VolumeSourceGradients -> BackProjectionGradients;
 * VolumeSourceHessians -> BackProjectionHessians;
 * VolumeSource -> SingleComponentForwardProjection;
 * SingleComponentProjectionsSource -> SingleComponentForwardProjection;
 * SingleComponentForwardProjection -> Weidinger;
 * Weidinger -> BackProjectionGradients;
 * Weidinger -> BackProjectionHessians;
 * Input4 -> MultiplyRegulGradient;
 * SQSRegul -> MultiplyRegulGradient;
 * MultiplyRegulGradient -> AddGradients;
 * BackProjectionGradients -> AddGradients;
 * AddGradients -> Newton;
 * Input4 -> MultiplyRegulHessian;
 * SQSRegul -> MultiplyRegulHessian;
 * MultiplyRegulHessian -> AddHessians;
 * BackProjectionHessians -> AddHessians;
 * AddHessians -> Newton;
 * Newton -> Nesterov;
 * Alphak -> Nesterov;
 * Nesterov -> MultiplySupport;
 * Input3 -> MultiplySupport;
 * MultiplySupport -> NextAlphak [arrowhead=none];
 * NextAlphak -> Output;
 * NextAlphak -> Alphak [style=dashed, constraint=false];
 * }
 * \enddot
 *
 * \test rtkmechlemtest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup RTK ReconstructionAlgorithm
 */

template <typename TOutputImage, typename TPhotonCounts, typename TSpectrum>
class MechlemOneStepSpectralReconstructionFilter
  : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_ASSIGN(MechlemOneStepSpectralReconstructionFilter);

  /** Standard class type alias. */
  using Self = MechlemOneStepSpectralReconstructionFilter;
  using Superclass = IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(MechlemOneStepSpectralReconstructionFilter, itk::ImageToImageFilter);

  /** Internal type alias and parameters */
  static constexpr unsigned int nBins = TPhotonCounts::PixelType::Dimension;
  static constexpr unsigned int nMaterials = TOutputImage::PixelType::Dimension;
  using dataType = typename TOutputImage::PixelType::ValueType;

  /** SFINAE type alias, depending on whether a CUDA image is used. */
  using CPUOutputImageType = typename itk::Image<typename TOutputImage::PixelType, TOutputImage::ImageDimension>;
#ifdef RTK_USE_CUDA
  typedef typename std::conditional<
    std::is_same<TOutputImage, CPUOutputImageType>::value,
    itk::Image<itk::Vector<dataType, nMaterials * nMaterials>, TOutputImage::ImageDimension>,
    itk::CudaImage<itk::Vector<dataType, nMaterials * nMaterials>, TOutputImage::ImageDimension>>::type THessiansImage;
  typedef
    typename std::conditional<std::is_same<TOutputImage, CPUOutputImageType>::value,
                              itk::Image<dataType, TOutputImage::ImageDimension>,
                              itk::CudaImage<dataType, TOutputImage::ImageDimension>>::type SingleComponentImageType;
#else
  using THessiansImage =
    typename itk::Image<itk::Vector<dataType, nMaterials * nMaterials>, TOutputImage::ImageDimension>;
  using SingleComponentImageType = typename itk::Image<dataType, TOutputImage::ImageDimension>;
#endif

#if !defined(ITK_WRAPPING_PARSER)
#  ifdef RTK_USE_CUDA
  typedef typename std::conditional<std::is_same<TOutputImage, CPUOutputImageType>::value,
                                    WeidingerForwardModelImageFilter<TOutputImage, TPhotonCounts, TSpectrum>,
                                    CudaWeidingerForwardModelImageFilter<TOutputImage, TPhotonCounts, TSpectrum>>::type
    WeidingerForwardModelType;
  typedef
    typename std::conditional<std::is_same<TOutputImage, CPUOutputImageType>::value,
                              JosephForwardProjectionImageFilter<SingleComponentImageType, SingleComponentImageType>,
                              CudaForwardProjectionImageFilter<SingleComponentImageType, SingleComponentImageType>>::
      type CudaSingleComponentForwardProjectionImageFilterType;
  typedef typename std::conditional<std::is_same<TOutputImage, CPUOutputImageType>::value,
                                    BackProjectionImageFilter<THessiansImage, THessiansImage>,
                                    CudaBackProjectionImageFilter<THessiansImage>>::type
    CudaHessiansBackProjectionImageFilterType;
#  else
  using WeidingerForwardModelType = WeidingerForwardModelImageFilter<TOutputImage, TPhotonCounts, TSpectrum>;
  using CudaSingleComponentForwardProjectionImageFilterType =
    JosephForwardProjectionImageFilter<SingleComponentImageType, SingleComponentImageType>;
  using CudaHessiansBackProjectionImageFilterType = BackProjectionImageFilter<THessiansImage, THessiansImage>;
#  endif
  using TGradientsImage = TOutputImage;
#endif

  using ForwardProjectionType = typename Superclass::ForwardProjectionType;
  using BackProjectionType = typename Superclass::BackProjectionType;

#if !defined(ITK_WRAPPING_PARSER)
  /** Filter type alias */
  using ExtractPhotonCountsFilterType = itk::ExtractImageFilter<TPhotonCounts, TPhotonCounts>;
  using AddFilterType = itk::AddImageFilter<TGradientsImage>;
  using SingleComponentForwardProjectionFilterType =
    rtk::ForwardProjectionImageFilter<SingleComponentImageType, SingleComponentImageType>;
  using ForwardProjectionFilterType = rtk::ForwardProjectionImageFilter<TOutputImage, TOutputImage>;
  using GradientsBackProjectionFilterType = rtk::BackProjectionImageFilter<TGradientsImage, TGradientsImage>;
  using HessiansBackProjectionFilterType = rtk::BackProjectionImageFilter<THessiansImage, THessiansImage>;
  using NesterovFilterType = rtk::NesterovUpdateImageFilter<TOutputImage>;
  using SingleComponentImageSourceType = rtk::ConstantImageSource<SingleComponentImageType>;
  using MaterialProjectionsSourceType = rtk::ConstantImageSource<TOutputImage>;
  using GradientsSourceType = rtk::ConstantImageSource<TGradientsImage>;
  using HessiansSourceType = rtk::ConstantImageSource<THessiansImage>;
  using SQSRegularizationType = rtk::SeparableQuadraticSurrogateRegularizationImageFilter<TGradientsImage>;
  using AddMatrixAndDiagonalFilterType = rtk::AddMatrixAndDiagonalImageFilter<TGradientsImage, THessiansImage>;
  using NewtonFilterType = rtk::GetNewtonUpdateImageFilter<TGradientsImage, THessiansImage>;
  using MultiplyFilterType = itk::MultiplyImageFilter<TOutputImage, SingleComponentImageType>;
  using MultiplyGradientFilterType = itk::MultiplyImageFilter<TGradientsImage, SingleComponentImageType>;
#endif

  /** Instantiate the forward projection filters */
  void
  SetForwardProjectionFilter(ForwardProjectionType _arg) override;

  /** Instantiate the back projection filters */
  void
  SetBackProjectionFilter(BackProjectionType _arg) override;

  /** Pass the geometry to all filters needing it */
  itkSetConstObjectMacro(Geometry, ThreeDCircularProjectionGeometry);

  itkSetMacro(NumberOfIterations, int);
  itkGetMacro(NumberOfIterations, int);

  /** Number of subsets per iteration. */
  itkSetMacro(NumberOfSubsets, int);
  itkGetMacro(NumberOfSubsets, int);

  /** Parameter to trigger Nesterov's reset. The value is a number of subsets
  ** which can be larger than the number of subsets per iteration. 1 means no
  ** Nesterov acceleration. */
  itkSetMacro(ResetNesterovEvery, int);
  itkGetMacro(ResetNesterovEvery, int);

  /** Set methods for all inputs, since they have different types */
  void
  SetInputMaterialVolumes(const TOutputImage * materialVolumes);
  void
  SetInputPhotonCounts(const TPhotonCounts * photonCounts);
  void
  SetInputSpectrum(const TSpectrum * spectrum);
  void
  SetSupportMask(const SingleComponentImageType * support);
  void
  SetSpatialRegularizationWeights(const SingleComponentImageType * regweights);

  /** Set/Get for the regularization weights */
  itkSetMacro(RegularizationWeights, typename TOutputImage::PixelType);
  itkGetMacro(RegularizationWeights, typename TOutputImage::PixelType);

  /** Set/Get for the radius */
  itkSetMacro(RegularizationRadius, typename TOutputImage::RegionType::SizeType);
  itkGetMacro(RegularizationRadius, typename TOutputImage::RegionType::SizeType);

  //    itkSetMacro(IterationCosts, bool);
  //    itkGetMacro(IterationCosts, bool);

  /** Set methods forwarding the detector response and material attenuation
   * matrices to the internal WeidingerForwardModel filter */
  using BinnedDetectorResponseType = vnl_matrix<dataType>;
  using MaterialAttenuationsType = vnl_matrix<dataType>;
  virtual void
  SetBinnedDetectorResponse(const BinnedDetectorResponseType & detResp);
  virtual void
  SetMaterialAttenuations(const MaterialAttenuationsType & matAtt);

protected:
  MechlemOneStepSpectralReconstructionFilter();
  ~MechlemOneStepSpectralReconstructionFilter() override = default;

  /** Does the real work. */
  void
  GenerateData() override;

#if !defined(ITK_WRAPPING_PARSER)
  /** Member pointers to the filters used internally (for convenience)*/
  typename ExtractPhotonCountsFilterType::Pointer              m_ExtractPhotonCountsFilter;
  typename AddFilterType::Pointer                              m_AddGradients;
  typename SingleComponentForwardProjectionFilterType::Pointer m_SingleComponentForwardProjectionFilter;
  typename MaterialProjectionsSourceType::Pointer              m_ProjectionsSource;
  typename SingleComponentImageSourceType::Pointer             m_SingleComponentProjectionsSource;
  typename SingleComponentImageSourceType::Pointer             m_SingleComponentVolumeSource;
  typename GradientsSourceType::Pointer                        m_GradientsSource;
  typename HessiansSourceType::Pointer                         m_HessiansSource;
  typename WeidingerForwardModelType::Pointer                  m_WeidingerForward;
  typename SQSRegularizationType::Pointer                      m_SQSRegul;
  typename AddMatrixAndDiagonalFilterType::Pointer             m_AddHessians;
  typename NewtonFilterType::Pointer                           m_NewtonFilter;
  typename NesterovFilterType::Pointer                         m_NesterovFilter;
  typename ForwardProjectionFilterType::Pointer                m_ForwardProjectionFilter;
  typename GradientsBackProjectionFilterType::Pointer          m_GradientsBackProjectionFilter;
  typename HessiansBackProjectionFilterType::Pointer           m_HessiansBackProjectionFilter;
  typename MultiplyFilterType::Pointer                         m_MultiplySupportFilter;
  typename MultiplyGradientFilterType::Pointer                 m_MultiplyRegulGradientsFilter;
  typename MultiplyGradientFilterType::Pointer                 m_MultiplyRegulHessiansFilter;
#endif

  /** The inputs of this filter have the same type but not the same meaning
   * It is normal that they do not occupy the same physical space. Therefore this check
   * must be removed */
  void
  VerifyInputInformation() const override
  {}

  /** The volume and the projections must have different requested regions
   */
  void
  GenerateInputRequestedRegion() override;
  void
  GenerateOutputInformation() override;

  /** Getters for the inputs */
  typename TOutputImage::ConstPointer
  GetInputMaterialVolumes();
  typename TPhotonCounts::ConstPointer
  GetInputPhotonCounts();
  typename TSpectrum::ConstPointer
  GetInputSpectrum();
  typename SingleComponentImageType::ConstPointer
  GetSupportMask();
  typename SingleComponentImageType::ConstPointer
  GetSpatialRegularizationWeights();

#if !defined(ITK_WRAPPING_PARSER)
  /** Functions to instantiate forward and back projection filters with a different
   * number of components than the ones provided by the IterativeConeBeamReconstructionFilter class */
  typename SingleComponentForwardProjectionFilterType::Pointer
  InstantiateSingleComponentForwardProjectionFilter(int fwtype);
  typename HessiansBackProjectionFilterType::Pointer
  InstantiateHessiansBackProjectionFilter(int bptype);
#endif

  ThreeDCircularProjectionGeometry::ConstPointer m_Geometry;

  int              m_NumberOfIterations;
  int              m_NumberOfProjectionsPerSubset;
  int              m_NumberOfSubsets;
  std::vector<int> m_NumberOfProjectionsInSubset;
  int              m_NumberOfProjections;
  int              m_ResetNesterovEvery;

  typename TOutputImage::PixelType            m_RegularizationWeights;
  typename TOutputImage::RegionType::SizeType m_RegularizationRadius;

  // private:
  //    bool                         m_IterationCosts;
};
} // namespace rtk


#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkMechlemOneStepSpectralReconstructionFilter.hxx"
#endif

#endif
