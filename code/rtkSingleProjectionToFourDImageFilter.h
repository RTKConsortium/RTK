#ifndef __rtkSingleProjectionToFourDImageFilter_h
#define __rtkSingleProjectionToFourDImageFilter_h

#include "itkImageToImageFilter.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkSplatWithKnownWeightsImageFilter.h"
//#if RTK_USE_CUDA
//    #include "rtkCudaSplatImageFilter.h"
//#endif
#include "rtkThreeDCircularProjectionGeometry.h"
#include "itkArray2D.h"
#include "itkTimeProbe.h"

namespace rtk
{
template< typename VolumeSeriesType, typename VolumeType>
class SingleProjectionToFourDImageFilter : public itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType >
{
public:
    /** Standard class typedefs. */
    typedef SingleProjectionToFourDImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SingleProjectionToFourDImageFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The image that will be backprojected, then added, with coefficients, to each 3D volume of the 4D image.
    * It is 3D because the backprojection filters need it, but the third dimension, which is the number of projections, is 1  */
    void SetInputProjectionStack(const VolumeType* Projection);

    /** The constant image source input for the bakprojector (instead of creating one for every iteration)  */
    void SetInputEmptyVolume(const VolumeType* Volume);

    /** Macros that take care of implementing the Get and Set methods for Weights and projectionNumber.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    typedef rtk::BackProjectionImageFilter< VolumeType, VolumeType >              BackProjectionFilterType;
    typedef typename BackProjectionFilterType::Pointer                                      BackProjectionFilterPointer;
    typedef rtk::SplatWithKnownWeightsImageFilter<VolumeSeriesType, VolumeType>             SplatFilterType;

    /** Set and init the backprojection filter. Default is voxel based backprojection. */
    virtual void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

    /** Get / Set the object pointer to projection geometry */
    itkGetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)
    itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

    /** Use a cuda implementation of the splat filter, or not */
    itkGetMacro(UseCuda, bool)
    itkSetMacro(UseCuda, bool)

protected:
    SingleProjectionToFourDImageFilter();
    ~SingleProjectionToFourDImageFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename VolumeType::ConstPointer GetInputProjectionStack();
    typename VolumeType::ConstPointer GetInputEmptyVolume();

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename SplatFilterType::Pointer               m_SplatFilter;
    typename BackProjectionFilterType::Pointer      m_BackProjectionFilter;
    bool m_UseCuda;

private:
    SingleProjectionToFourDImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
    itk::Array2D<float> m_Weights;
    int m_ProjectionNumber;

    /** Geometry objects */
    ThreeDCircularProjectionGeometry::Pointer m_Geometry;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSingleProjectionToFourDImageFilter.txx"
#endif

#endif
