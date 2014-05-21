#ifndef __rtkFourDToProjectionStackImageFilter_h
#define __rtkFourDToProjectionStackImageFilter_h

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
   * Input0 [ label="Input 0 (Projections)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Input: 4D sequence of volumes)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Output projections)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * Extract [ label="itk::ExtractImageFilter" URL="\ref itk::ExtractImageFilter"];
   * Multiply [ label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * BeforePaste [label="", fixedsize="false", width=0, height=0, shape=none];
   * Source [ label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
   * ForwardProj [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Interpolation [ label="rtk::InterpolationImageFilter" URL="\ref rtk::InterpolationImageFilter"];
   * Paste [ label="itk::PasteImageFilter" URL="\ref itk::PasteImageFilter"];
   * AfterPaste [label="", fixedsize="false", width=0, height=0, shape=none];
   *
   * Input0 -> Multiply;
   * Multiply -> Extract;
   * Extract -> ForwardProj;
   * Input0 -> BeforePaste[arrowhead=None];
   * BeforePaste -> Paste;
   * Source -> Interpolation;
   * Input1 -> Interpolation;
   * Interpolation -> ForwardProj;
   * ForwardProj -> Paste;
   * Paste -> AfterPaste[arrowhead=None];
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
    typedef itk::ExtractImageFilter<ProjectionStackType, ProjectionStackType>               ExtractFilterType;
    typedef itk::PasteImageFilter<ProjectionStackType, ProjectionStackType>                 PasteFilterType;
    typedef rtk::InterpolatorWithKnownWeightsImageFilter< VolumeType, VolumeSeriesType>     InterpolatorFilterType;
    typedef rtk::ConstantImageSource<VolumeType>                                            ConstantSourceType;
    typedef itk::MultiplyImageFilter<ProjectionStackType>                                   MultiplyFilterType;
    typedef rtk::ThreeDCircularProjectionGeometry                                           GeometryType;

    /** Set the ForwardProjection filter */
    void SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg);

    /** Pass the geometry to SingleProjectionToFourDFilter */
    virtual void SetGeometry(GeometryType::Pointer _arg);

    /** Pass the interpolation weights to SingleProjectionToFourDFilter */
    void SetWeights(const itk::Array2D<float> _arg);

    /** Initializes the empty volume source, set it and update it */
    void InitializeConstantSource();

protected:
    FourDToProjectionStackImageFilter();
    ~FourDToProjectionStackImageFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename ProjectionStackType::Pointer GetInputProjectionStack();

    /** Does the real work. */
    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    virtual void GenerateInputRequestedRegion();

    /** Member pointers to the filters used internally (for convenience)*/
    typename ExtractFilterType::Pointer                     m_ExtractFilter;
    typename PasteFilterType::Pointer                       m_PasteFilter;
    typename InterpolatorFilterType::Pointer                m_InterpolationFilter;
    typename ConstantSourceType::Pointer                    m_ConstantSource;
    typename ForwardProjectionFilterType::Pointer           m_ForwardProjectionFilter;
    typename MultiplyFilterType::Pointer                    m_ZeroMultiplyFilter;

    /** Other member variables */
    itk::Array2D<float>                                     m_Weights;
    GeometryType::Pointer                                   m_Geometry;
    int                                                     m_ProjectionNumber;

private:
    FourDToProjectionStackImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDToProjectionStackImageFilter.txx"
#endif

#endif
