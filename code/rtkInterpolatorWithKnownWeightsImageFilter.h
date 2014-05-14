#ifndef __rtkInterpolatorWithKnownWeightsImageFilter_h
#define __rtkInterpolatorWithKnownWeightsImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "itkArray2D.h"
#include "rtkConfiguration.h"

namespace rtk
{
template< typename VolumeType, typename VolumeSeriesType>
class InterpolatorWithKnownWeightsImageFilter : public itk::InPlaceImageFilter< VolumeType, VolumeType >
{
public:
    /** Standard class typedefs. */
    typedef InterpolatorWithKnownWeightsImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeType, VolumeType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(InterpolatorWithKnownWeightsImageFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated.*/
    void SetInputVolume(const VolumeType* Volume);

    /** The 4D image that will be interpolated, with coefficients, generate a 3D volume.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** Macros that take care of implementing the Get and Set methods for Weights and ProjectionNumber.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    itkGetMacro(ProjectionNumber, int)
    itkSetMacro(ProjectionNumber, int)

protected:
    InterpolatorWithKnownWeightsImageFilter();
    ~InterpolatorWithKnownWeightsImageFilter(){}

    typename VolumeType::ConstPointer GetInputVolume();
    typename VolumeSeriesType::Pointer GetInputVolumeSeries();

    /** Does the real work. */
    virtual void ThreadedGenerateData( const typename VolumeType::RegionType& outputRegionForThread, ThreadIdType threadId );

    itk::Array2D<float> m_Weights;
    int m_ProjectionNumber;

private:
    InterpolatorWithKnownWeightsImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented


};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkInterpolatorWithKnownWeightsImageFilter.txx"
#endif

#endif
