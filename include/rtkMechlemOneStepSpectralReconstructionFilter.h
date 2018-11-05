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

#ifdef RTK_USE_CUDA
  #include "rtkCudaWeidingerForwardModelImageFilter.h"
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
   * Output [label="Output (Material volumes)"];
   * Output [shape=Mdiamond];
   * 
   * node [shape=box];
   * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
   * VolumeSource [ label="rtk::ConstantImageSource (1 component volume, full of ones)" URL="\ref rtk::ConstantImageSource"];
   * SingleComponentProjectionsSource [ label="rtk::ConstantImageSource (1 component projections, full of zeros)" URL="\ref rtk::ConstantImageSource"];
   * VolumeSourceGradients [ label="rtk::ConstantImageSource (m components)" URL="\ref rtk::ConstantImageSource"];
   * VolumeSourceHessians [ label="rtk::ConstantImageSource (m x m components)" URL="\ref rtk::ConstantImageSource"];
   * ProjectionsSource [ label="rtk::ConstantImageSource (m components)" URL="\ref rtk::ConstantImageSource"];
   * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * SingleComponentForwardProjection [ label="rtk::ForwardProjectionImageFilter (1 component)" URL="\ref rtk::ForwardProjectionImageFilter"];
   * BackProjectionGradients [ label="rtk::BackProjectionImageFilter (gradients)" URL="\ref rtk::BackProjectionImageFilter"];
   * BackProjectionHessians [ label="rtk::BackProjectionImageFilter (hessians)" URL="\ref rtk::BackProjectionImageFilter"];
   * Weidinger [ label="rtk::WeidingerForwardModelImageFilter" URL="\ref rtk::WeidingerForwardModelImageFilter"];
   * SQSRegul [ label="rtk::SeparableQuadraticSurrogateRegularizationImageFilter" URL="\ref rtk::SeparableQuadraticSurrogateRegularizationImageFilter"];
   * AddGradients [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   * AddHessians [ label="rtk::AddMatrixAndDiagonalImageFilter" URL="\ref rtk::AddMatrixAndDiagonalImageFilter"];
   * Newton [ label="rtk::GetNewtonUpdateImageFilter" URL="\ref rtk::GetNewtonUpdateImageFilter"];
   * Nesterov [ label="rtk::NesterovUpdateImageFilter" URL="\ref rtk::NesterovUpdateImageFilter"];
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
   * SQSRegul -> AddGradients;
   * BackProjectionGradients -> AddGradients;
   * AddGradients -> Newton;
   * SQSRegul -> AddHessians;
   * BackProjectionHessians -> AddHessians;
   * AddHessians -> Newton;
   * Newton -> Nesterov;
   * Input0 -> Nesterov;
   * Nesterov -> NextAlphak [arrowhead=none];
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

template< typename TOutputImage, typename TPhotonCounts, typename TSpectrum >
class MechlemOneStepSpectralReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef MechlemOneStepSpectralReconstructionFilter                          Self;
    typedef IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>   Superclass;
    typedef itk::SmartPointer< Self >                                           Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(MechlemOneStepSpectralReconstructionFilter, itk::ImageToImageFilter)

    /** Internal typedefs and parameters */
    itkStaticConstMacro(nBins, unsigned int, TPhotonCounts::PixelType::Dimension);
    itkStaticConstMacro(nMaterials, unsigned int, TOutputImage::PixelType::Dimension);
    typedef typename TOutputImage::PixelType::ValueType dataType;
#if !defined( ITK_WRAPPING_PARSER )
#ifdef RTK_USE_CUDA
    typedef itk::CudaImage< itk::Vector<dataType, nMaterials * nMaterials>, TOutputImage::ImageDimension > THessiansImage;
    typedef itk::CudaImage<dataType, TOutputImage::ImageDimension> TSingleComponentImage;
#else
    typedef itk::Image< itk::Vector<dataType, nMaterials * nMaterials>, TOutputImage::ImageDimension > THessiansImage;
    typedef itk::Image<dataType, TOutputImage::ImageDimension> TSingleComponentImage;
#endif

    typedef TOutputImage TGradientsImage;
#endif

    typedef typename Superclass::ForwardProjectionType ForwardProjectionType;
    typedef typename Superclass::BackProjectionType    BackProjectionType;

#if !defined( ITK_WRAPPING_PARSER )
    /** Filter typedefs */
    typedef itk::ExtractImageFilter<TPhotonCounts, TPhotonCounts>                         ExtractPhotonCountsFilterType;
    typedef itk::AddImageFilter<TGradientsImage>                                          AddFilterType;
    typedef rtk::ForwardProjectionImageFilter< TSingleComponentImage, TSingleComponentImage > SingleComponentForwardProjectionFilterType;
    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >               ForwardProjectionFilterType;
    typedef rtk::BackProjectionImageFilter< TGradientsImage, TGradientsImage >            GradientsBackProjectionFilterType;
    typedef rtk::BackProjectionImageFilter< THessiansImage, THessiansImage >              HessiansBackProjectionFilterType;
    typedef rtk::NesterovUpdateImageFilter<TOutputImage>                                  NesterovFilterType;
    typedef rtk::ConstantImageSource<TSingleComponentImage>                               SingleComponentImageSourceType;
    typedef rtk::ConstantImageSource<TOutputImage>                                        MaterialProjectionsSourceType;
    typedef rtk::ConstantImageSource<TGradientsImage>                                     GradientsSourceType;
    typedef rtk::ConstantImageSource<THessiansImage>                                      HessiansSourceType;
    typedef rtk::SeparableQuadraticSurrogateRegularizationImageFilter<TGradientsImage>    SQSRegularizationType;
#ifdef RTK_USE_CUDA
    typedef rtk::CudaWeidingerForwardModelImageFilter<TOutputImage, TPhotonCounts, TSpectrum> WeidingerForwardModelType;
#else
    typedef rtk::WeidingerForwardModelImageFilter<TOutputImage, TPhotonCounts, TSpectrum> WeidingerForwardModelType;
#endif
    typedef rtk::AddMatrixAndDiagonalImageFilter<TGradientsImage, THessiansImage>         AddMatrixAndDiagonalFilterType;
    typedef rtk::GetNewtonUpdateImageFilter<TGradientsImage, THessiansImage>              NewtonFilterType;
#endif

    /** Instantiate the forward projection filters */
    void SetForwardProjectionFilter (ForwardProjectionType _arg) ITK_OVERRIDE;

    /** Instantiate the back projection filters */
    void SetBackProjectionFilter (BackProjectionType _arg) ITK_OVERRIDE;

    /** Pass the geometry to all filters needing it */
    itkSetConstObjectMacro(Geometry, ThreeDCircularProjectionGeometry)

    itkSetMacro(NumberOfIterations, int)
    itkGetMacro(NumberOfIterations, int)
    itkSetMacro(NumberOfSubsets, int)
    itkGetMacro(NumberOfSubsets, int)

    /** Set methods for all inputs, since they have different types */
    void SetInputMaterialVolumes(const TOutputImage* materialVolumes);
    void SetInputPhotonCounts(const TPhotonCounts* photonCounts);
    void SetInputSpectrum(const TSpectrum* spectrum);

    /** Set/Get for the regularization weights */
    itkSetMacro(RegularizationWeights, typename TOutputImage::PixelType)
    itkGetMacro(RegularizationWeights, typename TOutputImage::PixelType)

    /** Set/Get for the radius */
    itkSetMacro(RegularizationRadius, typename TOutputImage::RegionType::SizeType)
    itkGetMacro(RegularizationRadius, typename TOutputImage::RegionType::SizeType)

//    itkSetMacro(IterationCosts, bool)
//    itkGetMacro(IterationCosts, bool)

    /** Set methods forwarding the detector response and material attenuation
     * matrices to the internal WeidingerForwardModel filter */
    typedef vnl_matrix<dataType>  BinnedDetectorResponseType;
    typedef vnl_matrix<dataType>  MaterialAttenuationsType;
    virtual void SetBinnedDetectorResponse(const BinnedDetectorResponseType & detResp);
    virtual void SetMaterialAttenuations(const MaterialAttenuationsType & matAtt);

protected:
    MechlemOneStepSpectralReconstructionFilter();
    virtual ~MechlemOneStepSpectralReconstructionFilter() ITK_OVERRIDE {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

#if !defined( ITK_WRAPPING_PARSER )
    /** Member pointers to the filters used internally (for convenience)*/
    typename ExtractPhotonCountsFilterType::Pointer                          m_ExtractPhotonCountsFilter;
    typename AddFilterType::Pointer                                          m_AddGradients;
    typename SingleComponentForwardProjectionFilterType::Pointer             m_SingleComponentForwardProjectionFilter;
    typename MaterialProjectionsSourceType::Pointer                          m_ProjectionsSource;
    typename SingleComponentImageSourceType::Pointer                         m_SingleComponentProjectionsSource;
    typename SingleComponentImageSourceType::Pointer                         m_SingleComponentVolumeSource;
    typename GradientsSourceType::Pointer                                    m_GradientsSource;
    typename HessiansSourceType::Pointer                                     m_HessiansSource;
    typename WeidingerForwardModelType::Pointer                              m_WeidingerForward;
    typename SQSRegularizationType::Pointer                                  m_SQSRegul;
    typename AddMatrixAndDiagonalFilterType::Pointer                         m_AddHessians;
    typename NewtonFilterType::Pointer                                       m_NewtonFilter;
    typename NesterovFilterType::Pointer                                     m_NesterovFilter;
    typename ForwardProjectionFilterType::Pointer                            m_ForwardProjectionFilter;
    typename GradientsBackProjectionFilterType::Pointer                      m_GradientsBackProjectionFilter;
    typename HessiansBackProjectionFilterType::Pointer                       m_HessiansBackProjectionFilter;
#endif

    /** The inputs of this filter have the same type but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
#if ITK_VERSION_MAJOR<5
    void VerifyInputInformation() ITK_OVERRIDE {}
#else
    void VerifyInputInformation() const ITK_OVERRIDE {}
#endif

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion() ITK_OVERRIDE;
    void GenerateOutputInformation() ITK_OVERRIDE;

    /** Getters for the inputs */
    typename TOutputImage::ConstPointer GetInputMaterialVolumes();
    typename TPhotonCounts::ConstPointer GetInputPhotonCounts();
    typename TSpectrum::ConstPointer GetInputSpectrum();

#if !defined( ITK_WRAPPING_PARSER )
    /** Functions to instantiate forward and back projection filters with a different
     * number of components than the ones provided by the IterativeConeBeamReconstructionFilter class */
    typename SingleComponentForwardProjectionFilterType::Pointer InstantiateSingleComponentForwardProjectionFilter(int fwtype);
    typename HessiansBackProjectionFilterType::Pointer InstantiateHessiansBackProjectionFilter (int bptype);
#endif

    ThreeDCircularProjectionGeometry::ConstPointer m_Geometry;

    int                                            m_NumberOfIterations;
    int                                            m_NumberOfProjectionsPerSubset;
    int                                            m_NumberOfSubsets;
    std::vector<int>                               m_NumberOfProjectionsInSubset;
    int                                            m_NumberOfProjections;

    typename TOutputImage::PixelType               m_RegularizationWeights;
    typename TOutputImage::RegionType::SizeType    m_RegularizationRadius;

private:
    MechlemOneStepSpectralReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

//    bool                         m_IterationCosts;
};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkMechlemOneStepSpectralReconstructionFilter.hxx"
#endif

#endif
