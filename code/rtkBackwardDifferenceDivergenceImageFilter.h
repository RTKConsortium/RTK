#ifndef __rtkBackwardDifferenceDivergenceImageFilter_h
#define __rtkBackwardDifferenceDivergenceImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkCastImageFilter.h"

namespace rtk
{
/** \class BackwardDifferenceDivergenceImageFilter
 * \brief Computes the backward differences divergence
 * (adjoint of the forward differences gradient) of the input image
 *
 * The exact definition of the desired divergence filter can
 * be found in Chambolle, Antonin. “An Algorithm for Total
 * Variation Minimization and Applications.” J. Math. Imaging Vis. 20,
 * no. 1–2 (January 2004): 89–97.
 *
 * \ingroup IntensityImageFilters
 */
template <class TInputImage, typename TOutputImage = itk::Image< float, //I'd like to be able to put here the type of the data held in the input covariant vectors
                                                                 TInputImage::ImageDimension > >
class BackwardDifferenceDivergenceImageFilter :
        public itk::ImageToImageFilter< TInputImage, TOutputImage >
{
public:
    /** Extract dimension from input and output image. */
    itkStaticConstMacro(InputImageDimension, unsigned int,
                        TInputImage::ImageDimension);

    /** Convenient typedefs for simplifying declarations. */
    typedef TInputImage InputImageType;

    /** Standard class typedefs. */
    typedef BackwardDifferenceDivergenceImageFilter Self;
    typedef itk::ImageToImageFilter< InputImageType, TOutputImage> Superclass;
    typedef itk::SmartPointer<Self> Pointer;
    typedef itk::SmartPointer<const Self>  ConstPointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(BackwardDifferenceDivergenceImageFilter, ImageToImageFilter)

    /** Use the image spacing information in calculations. Use this option if you
     *  want derivatives in physical space. Default is UseImageSpacingOn. */
    void SetUseImageSpacingOn()
    { this->SetUseImageSpacing(true); }

    /** Ignore the image spacing. Use this option if you want derivatives in
        isotropic pixel space.  Default is UseImageSpacingOn. */
    void SetUseImageSpacingOff()
    { this->SetUseImageSpacing(false); }

    /** Set/Get whether or not the filter will use the spacing of the input
        image in its calculations */
    itkSetMacro(UseImageSpacing, bool);
    itkGetConstMacro(UseImageSpacing, bool);

    /** Set along which dimensions the gradient computation should be
        performed. The vector components at unprocessed dimensions are ignored */
    void SetDimensionsProcessed(bool* DimensionsProcessed);

    /** Image typedef support. */
    typedef typename InputImageType::PixelType InputPixelType;
    typedef typename InputImageType::RegionType InputImageRegionType;
    typedef typename InputImageType::SizeType InputSizeType;
    typedef itk::CovariantVector< InputPixelType, InputImageDimension > CovariantVectorType;

    protected:
        BackwardDifferenceDivergenceImageFilter();
    virtual ~BackwardDifferenceDivergenceImageFilter() {}

    virtual void GenerateInputRequestedRegion();

    virtual void BeforeThreadedGenerateData();

    virtual void ThreadedGenerateData(const typename InputImageType::RegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));

    virtual void AfterThreadedGenerateData();

private:
    BackwardDifferenceDivergenceImageFilter(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented

    bool m_UseImageSpacing;
    typename TInputImage::SpacingType m_SpacingCoeffs;

    // list of the dimensions along which the divergence has
    // to be computed. The components on other dimensions
    // are ignored for performance, but the gradient filter
    // sets them to zero anyway
    bool* m_DimensionsProcessed;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkBackwardDifferenceDivergenceImageFilter.txx"
#endif

#endif //__rtkBackwardDifferenceDivergenceImageFilter__
