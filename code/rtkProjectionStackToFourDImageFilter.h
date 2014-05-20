#ifndef __rtkProjectionStackToFourDImageFilter_h
#define __rtkProjectionStackToFourDImageFilter_h

#include <itkExtractImageFilter.h>
#include <itkMultiplyImageFilter.h>
#include <itkArray2D.h>

#include "rtkBackProjectionImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkConstantImageSource.h"

#ifdef RTK_USE_CUDA
#include "rtkCudaSplatImageFilter.h"
#endif

namespace rtk
{
template< typename VolumeSeriesType, typename ProjectionStackType, typename TFFTPrecision=double>
class ProjectionStackToFourDImageFilter : public itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType >
{
public:
    /** Standard class typedefs. */
    typedef ProjectionStackToFourDImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Convenient typedefs */
  typedef ProjectionStackType VolumeType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ProjectionStackToFourDImageFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The image that will be backprojected, then added, with coefficients, to each 3D volume of the 4D image.
    * It is 3D because the backprojection filters need it, but the third dimension, which is the number of projections, is 1  */
    void SetInputProjectionStack(const ProjectionStackType* Projection);

    typedef rtk::BackProjectionImageFilter< VolumeType, VolumeType >              BackProjectionFilterType;
    typedef itk::ExtractImageFilter< ProjectionStackType, ProjectionStackType >   ExtractFilterType;
    typedef rtk::ConstantImageSource< VolumeType >                                ConstantImageSourceType;
    typedef itk::MultiplyImageFilter< VolumeSeriesType >                          MultiplyFilterType;
    typedef rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>   SplatFilterType;

    typedef rtk::ThreeDCircularProjectionGeometry                                 GeometryType;

    /** Pass the backprojection filter to SingleProjectionToFourDFilter */
    void SetBackProjectionFilter (const typename BackProjectionFilterType::Pointer _arg);

    /** Pass the geometry to SingleProjectionToFourDFilter */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

//    /** Pass the interpolation weights to SingleProjectionToFourDFilter */
//    void SetWeights(const itk::Array2D<float> _arg);

    /** Use CUDA interpolation/splat filters */
    itkSetMacro(UseCudaSplat, bool)
    itkGetMacro(UseCudaSplat, bool)

    /** Macros that take care of implementing the Get and Set methods for Weights */
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

protected:
    ProjectionStackToFourDImageFilter();
    ~ProjectionStackToFourDImageFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    /** Does the real work. */
    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    virtual void GenerateInputRequestedRegion();

    void InitializeConstantSource();

    /** Member pointers to the filters used internally (for convenience)*/
    typename MultiplyFilterType::Pointer                    m_ZeroMultiplyFilter;
    typename SplatFilterType::Pointer                       m_SplatFilter;
    typename BackProjectionFilterType::Pointer              m_BackProjectionFilter;
    typename ExtractFilterType::Pointer                     m_ExtractFilter;
    typename ConstantImageSourceType::Pointer               m_ConstantImageSource;

    /** Other member variables */
    itk::Array2D<float>                                     m_Weights;
    GeometryType::Pointer                                   m_Geometry;
    int                                                     m_ProjectionNumber;
    bool                                                    m_UseCudaSplat;

private:
    ProjectionStackToFourDImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectionStackToFourDImageFilter.txx"
#endif

#endif
