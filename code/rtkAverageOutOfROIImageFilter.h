#ifndef __rtkAverageOutOfROIImageFilter_h
#define __rtkAverageOutOfROIImageFilter_h

//#include "itkImageToImageFilter.h"
#include "itkAccumulateImageFilter.h"
//#include "itkAddImageFilter.h"
//#include "itkMirrorPadImageFilter.h"
//#include "itkMultiplyImageFilter.h"
//#include "itkSubtractImageFilter.h"

namespace rtk
{

template< class TInputImage,
          class TROI = itk::Image< typename TInputImage::PixelType, TInputImage::ImageDimension -1 > >

class AverageOutOfROIImageFilter : public itk::ImageToImageFilter<TInputImage, TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef AverageOutOfROIImageFilter             Self;
    typedef itk::ImageToImageFilter<TInputImage, TInputImage> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension - 1>       LowerDimImage;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(AverageOutOfROIImageFilter, itk::ImageToImageFilter)

    /** The image containing the weights applied to the temporal components */
    void SetROI(const TROI* Map);

    typedef itk::AccumulateImageFilter<TInputImage, TInputImage> AccumulateFilterType;
//    typedef itk::AddImageFilter<TInputImage, TInputImage> AddFilterType;
//    typedef itk::MultiplyImageFilter<TInputImage, TInputImage> MultiplyFilterType;
//    typedef itk::SubtractImageFilter<TInputImage, TInputImage> SubtractFilterType;
//    typedef itk::MirrorPadImageFilter<LowerDimImage, TInputImage> PadFromLowerDimFilterType;
//    typedef itk::MirrorPadImageFilter<TInputImage, TInputImage> PadFilterType;


protected:
    AverageOutOfROIImageFilter();
    ~AverageOutOfROIImageFilter(){}

    typename TROI::Pointer GetROI();

    /** Does the real work. */
    virtual void GenerateData();
//    virtual void ThreadedGenerateData(const typename TInputImage::RegionType& outputRegionForThread, ThreadIdType itkNotUsed(threadId));

private:
    AverageOutOfROIImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkAverageOutOfROIImageFilter.txx"
#endif

#endif
