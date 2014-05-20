#ifndef __rtkFourDReconstructionConjugateGradientOperator_h
#define __rtkFourDReconstructionConjugateGradientOperator_h

#include "rtkConjugateGradientOperator.h"

#include <itkMultiplyImageFilter.h>
#include <itkExtractImageFilter.h>
#include <itkArray2D.h>

#include "rtkConstantImageSource.h"
#include "rtkInterpolatorWithKnownWeightsImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{
template< typename VolumeSeriesType, typename ProjectionStackType>
class FourDReconstructionConjugateGradientOperator : public ConjugateGradientOperator< VolumeSeriesType>
{
public:
    /** Standard class typedefs. */
    typedef FourDReconstructionConjugateGradientOperator        Self;
    typedef ConjugateGradientOperator< VolumeSeriesType>        Superclass;
    typedef itk::SmartPointer< Self >                           Pointer;

  /** Convenient typedef */
    typedef ProjectionStackType                                 VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(FourDReconstructionConjugateGradientOperator, ConjugateGradientOperator)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The image that will be backprojected, then added, with coefficients, to each 3D volume of the 4D image.
    * It is 3D because the backprojection filters need it, but the third dimension, which is the number of projections, is 1  */
    void SetInputProjectionStack(const ProjectionStackType* Projection);

    typedef rtk::BackProjectionImageFilter< ProjectionStackType, ProjectionStackType >          BackProjectionFilterType;
    typedef rtk::ForwardProjectionImageFilter< ProjectionStackType, ProjectionStackType >       ForwardProjectionFilterType;
    typedef rtk::InterpolatorWithKnownWeightsImageFilter<VolumeType, VolumeSeriesType>          InterpolationFilterType;
    typedef rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>                 SplatFilterType;
    typedef rtk::ConstantImageSource<VolumeType>                                                ConstantVolumeSourceType;
    typedef itk::ExtractImageFilter<ProjectionStackType, ProjectionStackType>                   ExtractFilterType;
    typedef itk::MultiplyImageFilter<VolumeSeriesType>                                          MultiplyVolumeSeriesType;
    typedef itk::MultiplyImageFilter<ProjectionStackType>                                       MultiplyProjectionStackType;

    /** Pass the backprojection filter to ProjectionStackToFourD*/
    void SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg);

    /** Pass the forward projection filter to FourDToProjectionStack */
    void SetForwardProjectionFilter (const typename ForwardProjectionFilterType::Pointer _arg);

    /** Pass the geometry to both ProjectionStackToFourD and FourDToProjectionStack */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Pass the interpolation weights to both ProjectionStackToFourD and FourDToProjectionStack */
    void SetWeights(const itk::Array2D<float> _arg);


protected:
    FourDReconstructionConjugateGradientOperator();
    ~FourDReconstructionConjugateGradientOperator(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    /** Builds the pipeline and computes output information */
    virtual void GenerateOutputInformation();

    /** Does the real work. */
    virtual void GenerateData();

    /** Initialize the ConstantImageSourceFilter */
    void InitializeConstantSource();

    /** Member pointers to the filters used internally (for convenience)*/
    typename BackProjectionFilterType::Pointer       m_BackProjectionFilter;
    typename ForwardProjectionFilterType::Pointer    m_ForwardProjectionFilter;
    typename InterpolationFilterType::Pointer        m_InterpolationFilter;
    typename SplatFilterType::Pointer                m_SplatFilter;
    typename ConstantVolumeSourceType::Pointer       m_ConstantVolumeSource1;
    typename ConstantVolumeSourceType::Pointer       m_ConstantVolumeSource2;
    typename ExtractFilterType::Pointer              m_ExtractFilter;
    typename MultiplyVolumeSeriesType::Pointer       m_ZeroMultiplyVolumeSeriesFilter;
    typename MultiplyProjectionStackType::Pointer    m_ZeroMultiplyProjectionStackFilter;

private:
    FourDReconstructionConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFourDReconstructionConjugateGradientOperator.txx"
#endif

#endif
