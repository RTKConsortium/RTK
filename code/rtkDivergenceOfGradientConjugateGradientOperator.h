#ifndef __rtkDivergenceOfGradientConjugateGradientOperator_h
#define __rtkDivergenceOfGradientConjugateGradientOperator_h

#include "rtkConjugateGradientOperator.h"

#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

namespace rtk
{
/** \class DivergenceOfGradientConjugateGradientOperator
 * \brief Computes the divergence of the gradient of an image. To be used
 * with the ConjugateGradientImageFilter
 *
 * \author Cyril Mory
 *
 * \ingroup IntensityImageFilters
 */
template <class TInputImage>
class DivergenceOfGradientConjugateGradientOperator :
        public ConjugateGradientOperator< TInputImage >
{
public:
    /** Extract dimension from input and output image. */
    itkStaticConstMacro(InputImageDimension, unsigned int,
                        TInputImage::ImageDimension);

    /** Convenient typedefs for simplifying declarations. */
    typedef TInputImage InputImageType;

    /** Standard class typedefs. */
    typedef DivergenceOfGradientConjugateGradientOperator Self;
    typedef itk::ImageToImageFilter< InputImageType, InputImageType> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(DivergenceOfGradientConjugateGradientOperator, ImageToImageFilter)

    /** Image typedef support. */
    typedef typename InputImageType::PixelType InputPixelType;
    typedef typename InputImageType::RegionType InputImageRegionType;
    typedef typename InputImageType::SizeType InputSizeType;

    /** Sub filter type definitions */
    typedef ForwardDifferenceGradientImageFilter<TInputImage> GradientFilterType;
    typedef typename GradientFilterType::OutputImageType GradientImageType;
    typedef BackwardDifferenceDivergenceImageFilter<GradientImageType> DivergenceFilterType;

    void SetDimensionsProcessed(bool* arg);

    protected:
        DivergenceOfGradientConjugateGradientOperator();
    virtual ~DivergenceOfGradientConjugateGradientOperator() {}

    virtual void GenerateData();

    virtual void GenerateOutputInformation();

    /** Sub filter pointers */
    typename GradientFilterType::Pointer             m_GradientFilter;
    typename DivergenceFilterType::Pointer           m_DivergenceFilter;

    bool* m_DimensionsProcessed;

private:
    DivergenceOfGradientConjugateGradientOperator(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkDivergenceOfGradientConjugateGradientOperator.txx"
#endif

#endif //__rtkDivergenceOfGradientConjugateGradientOperator__
