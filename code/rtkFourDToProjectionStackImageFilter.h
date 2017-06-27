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
#ifndef rtkFourDToProjectionStackImageFilter_h
#define rtkFourDToProjectionStackImageFilter_h

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkConstantImageSource.h"
#include "rtkInterpolatorWithKnownWeightsImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

namespace rtk
{
  /** \class FourDToProjectionStackImageFilter
   * \brief Implements part of the 4D reconstruction by conjugate gradient
   *
   * See the reference paper: "Cardiac C-arm computed tomography using
   * a 3D + time ROI reconstruction method with spatial and temporal regularization"
   * by Mory et al.
   *
   * 4D conjugate gradient reconstruction consists in iteratively
   * minimizing the following cost function:
   *
   * Sum_over_theta || R_theta S_theta f - p_theta ||_2^2
   *
   * with
   * - f a 4D series of 3D volumes, each one being the reconstruction
   * at a given respiratory/cardiac phase
   * - p_theta is the projection measured at angle theta
   * - S_theta an interpolation operator which, from the 3D + time sequence f,
   * estimates the 3D volume through which projection p_theta has been acquired
   * - R_theta is the X-ray transform (the forward projection operator) for angle theta
   *
   * Computing the gradient of this cost function yields:
   *
   * S_theta^T R_theta^T R_theta S_theta f - S_theta^T R_theta^T p_theta
   *
   * where A^T means the adjoint of operator A.
   *
   * FourDToProjectionStackImageFilter implements R_theta S_theta.
   *
   * \dot
   * digraph FourDToProjectionStackImageFilter {
   *
   * Input1 [label="Input 1 (Input: 4D sequence of volumes)"];
   * Input1 [shape=Mdiamond];
   * Input0 [ label="Input 0 (Projections)"];
   * Input0 [shape=Mdiamond];
   * Output [label="Output (Output projections)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * FourDSource [ label="rtk::ConstantImageSource (4D volume)" URL="\ref rtk::ConstantImageSource"];
   * ProjectionSource [ label="rtk::ConstantImageSource (projections)" URL="\ref rtk::ConstantImageSource"];
   * ForwardProj [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Interpolation [ label="rtk::InterpolatorWithKnownWeightsImageFilter" URL="\ref rtk::InterpolatorWithKnownWeightsImageFilter"];
   * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
   * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
   * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * ProjectionSource -> ForwardProj;
   * BeforePaste -> Paste;
   * FourDSource -> Interpolation;
   * Input1 -> Interpolation;
   * Interpolation -> ForwardProj;
   * Input0 -> BeforePaste[arrowhead=none];
   * ForwardProj -> Paste;
   * Paste -> AfterPaste[arrowhead=none];
   * AfterPaste -> Output;
   * AfterPaste -> BeforePaste [style=dashed, constraint=false];
   * }
   * \enddot
   *
   * \test rtkfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename ProjectionStackType, typename VolumeSeriesType>
class FourDToProjectionStackImageFilter : public itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType>
{
public:
    /** Standard class typedefs. */
    typedef FourDToProjectionStackImageFilter             Self;
    typedef itk::ImageToImageFilter<ProjectionStackType, ProjectionStackType> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Convenient typedefs */
    typedef ProjectionStackType VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourDToProjectionStackImageFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The image that will be backprojected, then added, with coefficients, to each 3D volume of the 4D image.
    * It is 3D because the ForwardProjection filters need it, but the third dimension, which is the number of projections, is 1  */
    void SetInputProjectionStack(const ProjectionStackType* Projection);

    /** Typedefs for the sub filters */
    typedef rtk::ForwardProjectionImageFilter< ProjectionStackType, ProjectionStackType >   ForwardProjectionFilterType;
    typedef itk::PasteImageFilter<ProjectionStackType, ProjectionStackType>                 PasteFilterType;
    typedef rtk::InterpolatorWithKnownWeightsImageFilter< VolumeType, VolumeSeriesType>     InterpolatorFilterType;
    typedef rtk::ConstantImageSource<VolumeType>                                            ConstantVolumeSourceType;
    typedef rtk::ConstantImageSource<ProjectionStackType>                                   ConstantProjectionStackSourceType;
    typedef rtk::ThreeDCircularProjectionGeometry                                           GeometryType;

    /** Set the ForwardProjection filter */
    void SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg);

    /** Pass the geometry to SingleProjectionToFourDFilter */
    virtual void SetGeometry(GeometryType::Pointer _arg);

    /** Pass the interpolation weights to SingleProjectionToFourDFilter */
    void SetWeights(const itk::Array2D<float> _arg);

    /** Initializes the empty volume source, set it and update it */
    void InitializeConstantVolumeSource();

    /** Store the phase signal in a member variable */
    virtual void SetSignal(const std::vector<double> signal);

protected:
    FourDToProjectionStackImageFilter();
    ~FourDToProjectionStackImageFilter() {}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename ProjectionStackType::Pointer GetInputProjectionStack();

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    void GenerateOutputInformation() ITK_OVERRIDE;

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    /** Member pointers to the filters used internally (for convenience)*/
    typename PasteFilterType::Pointer                       m_PasteFilter;
    typename InterpolatorFilterType::Pointer                m_InterpolationFilter;
    typename ConstantVolumeSourceType::Pointer              m_ConstantVolumeSource;
    typename ConstantProjectionStackSourceType::Pointer     m_ConstantProjectionStackSource;
    typename ForwardProjectionFilterType::Pointer           m_ForwardProjectionFilter;

    /** Other member variables */
    itk::Array2D<float>                                                 m_Weights;
    GeometryType::Pointer                                               m_Geometry;
    typename ConstantProjectionStackSourceType::OutputImageRegionType   m_PasteRegion;
    std::vector<double>                                                 m_Signal;

private:
    FourDToProjectionStackImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDToProjectionStackImageFilter.hxx"
#endif

#endif
