#ifndef __rtkSplatWithKnownWeightsImageFilter_h
#define __rtkSplatWithKnownWeightsImageFilter_h

#include "itkInPlaceImageFilter.h"

#include "itkArray2D.h"

namespace rtk
{
template< typename VolumeSeriesType, typename VolumeType>
class SplatWithKnownWeightsImageFilter : public itk::InPlaceImageFilter< VolumeSeriesType, VolumeSeriesType >
{
public:
    /** Standard class typedefs. */
    typedef SplatWithKnownWeightsImageFilter             Self;
    typedef itk::ImageToImageFilter< VolumeSeriesType, VolumeSeriesType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef typename VolumeSeriesType::RegionType    OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SplatWithKnownWeightsImageFilter, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    void SetInputVolumeSeries(const VolumeSeriesType* VolumeSeries);

    /** The 3D image that will be added, with coefficients, to each 3D volume of the 4D image.*/
    void SetInputVolume(const VolumeType* Volume);

    /** Macros that take care of implementing the Get and Set methods for Weights and projectionNumber.*/
    itkGetMacro(Weights, itk::Array2D<float>)
    itkSetMacro(Weights, itk::Array2D<float>)

    itkGetMacro(ProjectionNumber, int)
    itkSetMacro(ProjectionNumber, int)

protected:
    SplatWithKnownWeightsImageFilter();
    ~SplatWithKnownWeightsImageFilter(){}

    typename VolumeSeriesType::ConstPointer GetInputVolumeSeries();
    typename VolumeType::Pointer GetInputVolume();

    /** Does the real work. */
    virtual void ThreadedGenerateData(const typename VolumeSeriesType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));

    /** Splats the OutputRequestedRegion along the first direction, not the last*/
    unsigned int SplitRequestedRegion(unsigned int i, unsigned int num, typename VolumeSeriesType::RegionType &splatRegion);

    itk::Array2D<float> m_Weights;
    int m_ProjectionNumber;
private:
    SplatWithKnownWeightsImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented



};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSplatWithKnownWeightsImageFilter.txx"
#endif

#endif
