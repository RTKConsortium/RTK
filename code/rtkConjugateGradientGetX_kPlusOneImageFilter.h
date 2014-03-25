#ifndef __rtkConjugateGradientGetX_kPlusOneImageFilter_h
#define __rtkConjugateGradientGetX_kPlusOneImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkMultiplyImageFilter.h>

namespace rtk
{
template< typename TInputImage>
class ConjugateGradientGetX_kPlusOneImageFilter : public itk::ImageToImageFilter< TInputImage, TInputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientGetX_kPlusOneImageFilter             Self;
    typedef itk::ImageToImageFilter< TInputImage, TInputImage> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;
    typedef typename TInputImage::RegionType    OutputImageRegionType;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientGetX_kPlusOneImageFilter, itk::ImageToImageFilter)

    /** Functions to set the inputs */
    void SetXk(const TInputImage* Xk);
    void SetPk(const TInputImage* Pk);

    itkGetMacro(alphak, float)
    itkSetMacro(alphak, float)

    /** Typedefs for sub filters */
    typedef itk::AddImageFilter<TInputImage>      AddFilterType;
    typedef itk::MultiplyImageFilter<TInputImage> MultiplyFilterType;

protected:
    ConjugateGradientGetX_kPlusOneImageFilter();
    ~ConjugateGradientGetX_kPlusOneImageFilter(){}

    typename TInputImage::Pointer GetXk();
    typename TInputImage::Pointer GetPk();

    /** Does the real work. */
    virtual void GenerateData();

    void GenerateOutputInformation();

private:
    ConjugateGradientGetX_kPlusOneImageFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented
    float m_alphak;

    /** Pointers to sub filters */
    typename AddFilterType::Pointer       m_AddFilter;
    typename MultiplyFilterType::Pointer  m_MultiplyFilter;

    //    virtual void GenerateInputRequestedRegion();

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientGetX_kPlusOneImageFilter.txx"
#endif

#endif
