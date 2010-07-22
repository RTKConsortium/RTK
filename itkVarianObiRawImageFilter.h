#ifndef __itkVarianObiRawImageFilter_h
#define __itkVarianObiRawImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "itkConceptChecking.h"
#include <itkNumericTraits.h>

#define HND_INTENSITY_MAX (139000)

namespace itk
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
    if (A != NumericTraits<TInput>::ZeroValue())
      {
      output *= -1.0/HND_INTENSITY_MAX;
      output += 1.0;
      if(output<NumericTraits<TOutput>::ZeroValue())
        output = NumericTraits<TOutput>::ZeroValue();
      }
    return output;
    }
}; 
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT VarianObiRawImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Function::ObiAttenuation<
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef VarianObiRawImageFilter  Self;
  typedef UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                  Function::ObiAttenuation< typename TInputImage::PixelType,
                                                 typename TOutputImage::PixelType> >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(VarianObiRawImageFilter,
               UnaryFunctorImageFilter);

protected:
  VarianObiRawImageFilter() {}
  virtual ~VarianObiRawImageFilter() {}

private:
  VarianObiRawImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
