#ifndef __rtkProjectionStackToFourDImageFilter_h
#define __rtkProjectionStackToFourDImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
#include "rtkConstantImageSource.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkSingleProjectionToFourDImageFilter.h"
#include <itkExtractImageFilter.h>
#include "rtkConstantImageSource.h"
#include "rtkFFTRampImageFilter.h"

#include "itkArray2D.h"

namespace rtk
{
template< typename VolumeSeriesType, typename VolumeType, typename TFFTPrecision=double>
class ProjectionStackToFourDImageFilter : public itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType >
{
public:
    /** Standard class typedefs. */
    typedef ProjectionStackToFourDImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ProjectionStackToFourDImageFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The image that will be backprojected, then added, with coefficients, to each 3D volume of the 4D image.
    * It is 3D because the backprojection filters need it, but the third dimension, which is the number of projections, is 1  */
    void SetInputProjectionStack(const VolumeType* Projection);

    typedef rtk::BackProjectionImageFilter< VolumeType, VolumeType >              BackProjectionFilterType;
    typedef typename BackProjectionFilterType::Pointer                                      BackProjectionFilterPointer;
    typedef rtk::SingleProjectionToFourDImageFilter<VolumeSeriesType, VolumeType>       SingleProjectionToFourDFilterType;
    typedef itk::ExtractImageFilter< VolumeType, VolumeType >                     ExtractFilterType;
    typedef rtk::ConstantImageSource< VolumeType >                                    ConstantImageSourceType;
    typedef rtk::FFTRampImageFilter<VolumeType, VolumeType, TFFTPrecision>          RampFilterType;
    typedef rtk::ThreeDCircularProjectionGeometry                                       GeometryType;

    /** Pass the backprojection filter to SingleProjectionToFourDFilter */
    void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

    /** Pass the geometry to SingleProjectionToFourDFilter */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Pass the interpolation weights to SingleProjectionToFourDFilter */
    void SetWeights(const itk::Array2D<float> _arg);

    /** Add a ramp filter if needed */
    void SetUseRampFilter(bool arg);

    /** Configure the FourDToSingleProjection to use Cuda for splat, or not*/
    void SetUseCuda(const bool _arg);


protected:
    ProjectionStackToFourDImageFilter();
    ~ProjectionStackToFourDImageFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename VolumeType::ConstPointer GetInputProjectionStack();

    /** Does the real work. */
    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    /** Member pointers to the filters used internally (for convenience)*/
    typename SingleProjectionToFourDFilterType::Pointer     m_SingleProjToFourDFilter;
    typename ExtractFilterType::Pointer                     m_ExtractFilter;
    typename ConstantImageSourceType::Pointer               m_constantImageSource;
    bool                                                    m_UseRampFilter;
    itk::Array2D<float>                                     m_Weights;
    GeometryType::Pointer                                   m_Geometry;

private:
    ProjectionStackToFourDImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented


};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkProjectionStackToFourDImageFilter.txx"
#endif

#endif
