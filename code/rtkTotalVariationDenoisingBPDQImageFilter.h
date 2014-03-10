#ifndef __rtkTotalVariationDenoisingBPDQImageFilter_h
#define __rtkTotalVariationDenoisingBPDQImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCastImageFilter.h>
#include <itkImage.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "rtkMagnitudeThresholdImageFilter.h"

namespace rtk
{
/** \class TotalVariationDenoisingBPDQImageFilter
 * \brief Applies a total variation denoising, only along the dimensions specified, on an image.
 *
 * This filter finds the minimum of lambda * || f - f_0 ||_2^2 + TV(f)
 * using basis pursuit dequantaization, where f is the current image, f_0 the
 * input image, and TV the total variation calculated with only the gradients
 * along the dimensions specified. This filter can be used, for example, to
 * perform 3D total variation denoising on a 4D dataset
 * (by calling SetDimensionsProcessed([true true true false]).
 * More information on the algorithm can be found at
 * http://wiki.epfl.ch/bpdq#download
 *
 * \author Cyril Mory
 *
 * \ingroup IntensityImageFilters
 */
template <class TInputImage>
class TotalVariationDenoisingBPDQImageFilter :
        public itk::ImageToImageFilter< TInputImage, TInputImage >
{
public:
    /** Extract dimension from input and output image. */
    itkStaticConstMacro(InputImageDimension, unsigned int,
                        TInputImage::ImageDimension);

    /** Convenient typedefs for simplifying declarations. */
    typedef TInputImage InputImageType;

    /** Standard class typedefs. */
    typedef TotalVariationDenoisingBPDQImageFilter Self;
    typedef itk::ImageToImageFilter< InputImageType, InputImageType> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(TotalVariationDenoisingBPDQImageFilter, ImageToImageFilter)

    /** Image typedef support. */
    typedef typename InputImageType::PixelType InputPixelType;
    typedef typename InputImageType::RegionType InputImageRegionType;
    typedef typename InputImageType::SizeType InputSizeType;

    /** Sub filter type definitions */
    typedef ForwardDifferenceGradientImageFilter<TInputImage> GradientFilterType;
    typedef typename GradientFilterType::OutputImageType GradientImageType;
    typedef itk::MultiplyImageFilter<GradientImageType> MultiplyFilterType;
    typedef itk::SubtractImageFilter<TInputImage> SubtractImageFilterType;
    typedef itk::SubtractImageFilter<GradientImageType> SubtractGradientFilterType;
    typedef MagnitudeThresholdImageFilter<GradientImageType> MagnitudeThresholdFilterType;
    typedef BackwardDifferenceDivergenceImageFilter<GradientImageType> DivergenceFilterType;

    itkGetMacro(NumberOfIterations, int)
    itkSetMacro(NumberOfIterations, int)

    itkSetMacro(Lambda, double)
    itkGetMacro(Lambda, double)

    void SetDimensionsProcessed(bool* arg);

    protected:
        TotalVariationDenoisingBPDQImageFilter();
    virtual ~TotalVariationDenoisingBPDQImageFilter() {}

    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    /** Sub filter pointers */
    typename GradientFilterType::Pointer             m_GradientFilter;
    typename MultiplyFilterType::Pointer             m_MultiplyFilter;
    typename SubtractImageFilterType::Pointer        m_SubtractImageFilter;
    typename SubtractGradientFilterType::Pointer     m_SubtractGradientFilter;
    typename MagnitudeThresholdFilterType::Pointer   m_MagnitudeThresholdFilter;
    typename DivergenceFilterType::Pointer           m_DivergenceFilter;

    double m_Lambda;
    int m_NumberOfIterations;
    bool* m_DimensionsProcessed;

private:
    TotalVariationDenoisingBPDQImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    double m_beta;
    double m_gamma;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkTotalVariationDenoisingBPDQImageFilter.txx"
#endif

#endif //__rtkTotalVariationDenoisingBPDQImageFilter__
