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
#ifndef rtkProjectionStackToFourDImageFilter_h
#define rtkProjectionStackToFourDImageFilter_h

#include <itkExtractImageFilter.h>
#include <itkArray2D.h>

#include "rtkBackProjectionImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometry.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaSplatImageFilter.h"
  #include "rtkCudaConstantVolumeSource.h"
  #include "rtkCudaConstantVolumeSeriesSource.h"
#endif

namespace rtk
{
  /** \class ProjectionStackToFourDImageFilter
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
   * ProjectionStackToFourDImageFilter implements S_theta^T R_theta^T.
   *
   * \dot
   * digraph ProjectionStackToFourDImageFilter {
   *
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Input0 [ label="Input 0 (Input: 4D sequence of volumes)"];
   * Input0 [shape=Mdiamond];
   * Output [label="Output (Reconstruction: 4D sequence of volumes)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
   * VolumeSeriesSource [ label="rtk::ConstantImageSource (4D)" URL="\ref rtk::ConstantImageSource"];
   * AfterSource4D [label="", fixedsize="false", width=0, height=0, shape=none];
   * Source [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
   * Backproj [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * Splat [ label="rtk::SplatWithKnownWeightsImageFilter" URL="\ref rtk::SplatWithKnownWeightsImageFilter"];
   * AfterSplat [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input1 -> Extract;
   * Input0 -> VolumeSeriesSource [style=invis];
   * VolumeSeriesSource -> AfterSource4D[arrowhead=none];
   * AfterSource4D -> Splat;
   * Extract -> Backproj;
   * Source -> Backproj;
   * Backproj -> Splat;
   * Splat -> AfterSplat[arrowhead=none];
   * AfterSplat -> Output;
   * AfterSplat -> AfterSource4D[style=dashed, constraint=none];
   * }
   * \enddot
   *
   * \test rtkfourdconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision=double>
class ProjectionStackToFourDImageFilter : public itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType >
{
public:
    /** Standard class typedefs. */
    typedef ProjectionStackToFourDImageFilter                             Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >                                     Pointer;

    /** Convenient typedefs */
  typedef ProjectionStackType VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ProjectionStackToFourDImageFilter, itk::ImageToImageFilter)

    /** Set/Get the 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);
    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();

    /** Set/Get the stack of projections */
    void SetInputProjectionStack(const ProjectionStackType* Projections);
    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    typedef rtk::BackProjectionImageFilter< VolumeType, VolumeType >              BackProjectionFilterType;
    typedef itk::ExtractImageFilter< ProjectionStackType, ProjectionStackType >   ExtractFilterType;
    typedef rtk::ConstantImageSource< VolumeType >                                ConstantVolumeSourceType;
    typedef rtk::ConstantImageSource< VolumeSeriesType >                          ConstantVolumeSeriesSourceType;
    typedef rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>   SplatFilterType;

    typedef rtk::ThreeDCircularProjectionGeometry                                 GeometryType;

    /** Pass the backprojection filter to SingleProjectionToFourDFilter */
    void SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg);

    /** Pass the geometry to SingleProjectionToFourDFilter */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Use CUDA splat / sources */
    itkSetMacro(UseCudaSplat, bool)
    itkGetMacro(UseCudaSplat, bool)
    itkSetMacro(UseCudaSources, bool)
    itkGetMacro(UseCudaSources, bool)

    /** Macros that take care of implementing the Get and Set methods for Weights */
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    /** Store the phase signal in a member variable */
    virtual void SetSignal(const std::vector<double> signal);

protected:
    ProjectionStackToFourDImageFilter();
    ~ProjectionStackToFourDImageFilter() {}

    /** Does the real work. */
    void GenerateData() ITK_OVERRIDE;

    void GenerateOutputInformation() ITK_OVERRIDE;

    void GenerateInputRequestedRegion() ITK_OVERRIDE;

    void InitializeConstantSource();

    /** Member pointers to the filters used internally (for convenience)*/
    typename SplatFilterType::Pointer                       m_SplatFilter;
    typename BackProjectionFilterType::Pointer              m_BackProjectionFilter;
    typename ExtractFilterType::Pointer                     m_ExtractFilter;
    typename ConstantVolumeSourceType::Pointer              m_ConstantVolumeSource;
    typename ConstantVolumeSeriesSourceType::Pointer        m_ConstantVolumeSeriesSource;

    /** Other member variables */
    itk::Array2D<float>                                     m_Weights;
    GeometryType::Pointer                                   m_Geometry;
    bool                                                    m_UseCudaSplat;
    bool                                                    m_UseCudaSources;
    std::vector<double>                                     m_Signal;

private:
    ProjectionStackToFourDImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectionStackToFourDImageFilter.hxx"
#endif

#endif
