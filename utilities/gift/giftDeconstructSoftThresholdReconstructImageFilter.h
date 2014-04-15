#ifndef __giftDeconstructSoftThresholdReconstructImageFilter_H
#define __giftDeconstructSoftThresholdReconstructImageFilter_H

//ITK includes
#include "itkMacro.h"
#include "itkProgressReporter.h"

//GIFT includes
#include "giftDaubechiesWaveletImageFilter.h"
#include "softThresholdImageFilter.h"


namespace gift {

/**
 * \class DeconstructSoftThresholdReconstructImageFilter
 * \brief Deconstructs an image, soft thresholds it wavelets coefficients,
 * then reconstructs
 *
 * \ingroup Wavelet Image Filters
 */
template <class TImage>
class DeconstructSoftThresholdReconstructImageFilter
    : public itk::ImageToImageFilter<TImage,TImage>
{
public:
    /** Standard class typedefs. */
    typedef DeconstructSoftThresholdReconstructImageFilter                   Self;
    typedef itk::ImageToImageFilter<TImage,TImage>  Superclass;
    typedef itk::SmartPointer<Self>                 Pointer;
    typedef itk::SmartPointer<const Self>           ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(DeconstructSoftThresholdReconstructImageFilter, ImageToImageFilter)

    /** ImageDimension enumeration. */
    itkStaticConstMacro(ImageDimension, unsigned int, TImage::ImageDimension);

    /** Inherit types from Superclass. */
    typedef typename Superclass::InputImageType         InputImageType;
    typedef typename Superclass::OutputImageType        OutputImageType;
    typedef typename Superclass::InputImagePointer      InputImagePointer;
    typedef typename Superclass::OutputImagePointer     OutputImagePointer;
    typedef typename Superclass::InputImageConstPointer InputImageConstPointer;
    typedef typename TImage::PixelType                  PixelType;
    typedef typename TImage::InternalPixelType          InternalPixelType;

    /** Define the types of subfilters */
    typedef gift::DaubechiesWaveletOperator<PixelType, ImageDimension>      DbWaveletType;
    typedef gift::DbWaveletImageFilter<InputImageType, DbWaveletType>       DbWaveletFilterType;
    typedef itk::SoftThresholdImageFilter<InputImageType, InputImageType>   SoftThresholdFilterType;

    /** Set the number of levels of the deconstruction and reconstruction */
    void SetNumberOfLevels(unsigned int levels);

    /** Sets the order of the Daubechies wavelet used to deconstruct/reconstruct the image pyramid */
    itkGetMacro(WaveletsOrder, unsigned int)
    itkSetMacro(WaveletsOrder, unsigned int)

    /** Sets the threshold used in soft thresholding */
    itkGetMacro(Threshold, float)
    itkSetMacro(Threshold, float)


protected:
    DeconstructSoftThresholdReconstructImageFilter();
    ~DeconstructSoftThresholdReconstructImageFilter(){}
    void PrintSelf(std::ostream&os, itk::Indent indent) const;

    /** Generate the output data. */
    void GenerateData();

private:
    DeconstructSoftThresholdReconstructImageFilter(const Self&);     //purposely not implemented
    void operator=(const Self&);            //purposely not implemented

    typename DbWaveletFilterType::Pointer    m_DeconstructionFilter, m_ReconstructionFilter;
    unsigned int    m_WaveletsOrder;
    float m_Threshold;

};

}// namespace gift

//Include CXX
#ifndef GIFT_MANUAL_INSTANTIATION
#include "giftDeconstructSoftThresholdReconstructImageFilter.txx"
#endif

#endif
