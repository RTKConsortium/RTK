#ifndef __rtkFourDToProjectionStackImageFilter_h
#define __rtkFourDToProjectionStackImageFilter_h

#include <itkExtractImageFilter.h>
#include <itkPasteImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkConstantImageSource.h"
#include "rtkInterpolatorWithKnownWeightsImageFilter.h"

namespace rtk
{
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
    typename MultiplyFilterType::Pointer                    m_ZeroMultiplyFilter2;

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
