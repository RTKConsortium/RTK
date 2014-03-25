#ifndef __rtkConjugateGradientGetR_kPlusOneImageFilter_h
#define __rtkConjugateGradientGetR_kPlusOneImageFilter_h

#include "itkImageToImageFilter.h"

namespace rtk
{
template< typename TInputImage>
class ConjugateGradientGetR_kPlusOneImageFilter : public itk::ImageToImageFilter< TInputImage, TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientGetR_kPlusOneImageFilter             Self;
    typedef itk::ImageToImageFilter< TInputImage, TInputImage> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef typename TInputImage::RegionType    OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientGetR_kPlusOneImageFilter, itk::ImageToImageFilter)

    /** Functions to set the inputs */
    void SetRk(const TInputImage* Rk);
    void SetPk(const TInputImage* Pk);
    void SetAPk(const TInputImage* APk);

    itkGetMacro(alphak, float)
    itkSetMacro(alphak, float)

protected:
    ConjugateGradientGetR_kPlusOneImageFilter();
    ~ConjugateGradientGetR_kPlusOneImageFilter(){}

    typename TInputImage::Pointer GetRk();
    typename TInputImage::Pointer GetPk();
    typename TInputImage::Pointer GetAPk();

    /** Does the real work. */
    virtual void GenerateData();

private:
    ConjugateGradientGetR_kPlusOneImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
    float m_alphak;

    //    virtual void GenerateInputRequestedRegion();

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientGetR_kPlusOneImageFilter.txx"
#endif

#endif
