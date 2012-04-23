#ifndef __rtkVarianObiRawImageFilter_h
#define __rtkVarianObiRawImageFilter_h

#include <itkUnaryFunctorImageFilter.h>
#include <itkConceptChecking.h>
#include <itkNumericTraits.h>

#define HND_INTENSITY_MAX (139000)

namespace rtk
{
  
/** \class VarianObiRawImageFilter
 * \brief Interprets the raw Varian OBI projection data to values.
 */
namespace Function {  
  
template< class TInput, class TOutput>
class ObiAttenuation
{
public:
  ObiAttenuation() {}
  ~ObiAttenuation() {}
  bool operator!=( const ObiAttenuation & ) const
    {
    return false;
    }
  bool operator==( const ObiAttenuation & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A ) const
    {
    TOutput output = A;
    if (A != itk::NumericTraits<TInput>::ZeroValue())
      {
      output *= -1.0/HND_INTENSITY_MAX;
      output += 1.0;
      if(output<itk::NumericTraits<TOutput>::ZeroValue())
        output = itk::NumericTraits<TOutput>::ZeroValue();
      }
    return output;
    }
}; 
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT VarianObiRawImageFilter :
    public
itk::UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                             Function::ObiAttenuation<
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef VarianObiRawImageFilter  Self;
  typedef itk::UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                       Function::ObiAttenuation< typename TInputImage::PixelType,
                                                      typename TOutputImage::PixelType> >  Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(VarianObiRawImageFilter,
               itk::UnaryFunctorImageFilter);

protected:
  VarianObiRawImageFilter() {}
  virtual ~VarianObiRawImageFilter() {}

private:
  VarianObiRawImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace rtk


#endif
