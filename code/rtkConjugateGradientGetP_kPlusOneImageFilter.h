#ifndef __rtkConjugateGradientGetP_kPlusOneImageFilter_h
#define __rtkConjugateGradientGetP_kPlusOneImageFilter_h

#include "itkImageToImageFilter.h"

namespace rtk
{
template< typename TInputImage>
class ConjugateGradientGetP_kPlusOneImageFilter : public itk::ImageToImageFilter< TInputImage, TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientGetP_kPlusOneImageFilter             Self;
    typedef itk::ImageToImageFilter< TInputImage, TInputImage> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef typename TInputImage::RegionType    OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientGetP_kPlusOneImageFilter, itk::ImageToImageFilter)

    /** Functions to set the inputs */
    void SetR_kPlusOne(const TInputImage* R_kPlusOne);
    void SetRk(const TInputImage* Rk);
    void SetPk(const TInputImage* Pk);

protected:
    ConjugateGradientGetP_kPlusOneImageFilter();
    ~ConjugateGradientGetP_kPlusOneImageFilter(){}

    typename TInputImage::Pointer GetR_kPlusOne();
    typename TInputImage::Pointer GetRk();
    typename TInputImage::Pointer GetPk();

    /** Does the real woPk. */
    virtual void GenerateData();

private:
    ConjugateGradientGetP_kPlusOneImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    //    virtual void GenerateInputRequestedRegion();

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientGetP_kPlusOneImageFilter.txx"
#endif

#endif
