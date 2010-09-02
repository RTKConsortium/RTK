#ifndef __itkElektaSynergyRawImageFilter_h
#define __itkElektaSynergyRawImageFilter_h

#include "itkUnaryFunctorImageFilter.h"
#include "itkConceptChecking.h"
#include <itkNumericTraits.h>

#define HND_INTENSITY_MAX (139000)

namespace itk
{
  
/** \class ElektaSynergyRawImageFilter
 * \brief Interprets the raw Elekta Synergy projection data to values.
 */
namespace Function {  
  
template< class TInput, class TOutput>
class SynergyAttenuation
{
public:
  SynergyAttenuation() {
    logRef = log(TOutput(NumericTraits<TInput>::max()-NumericTraits<TInput>::min()+1));
  }
  ~SynergyAttenuation() {}
  bool operator!=( const SynergyAttenuation & ) const
    {
    return false;
    }
  bool operator==( const SynergyAttenuation & other ) const
    {
    return !(*this != other);
    }
  inline TOutput operator()( const TInput & A ) const
    {
    return log( TOutput(A+1) ) - logRef;
    }

private:
  TOutput logRef;
}; 
}

template <class TInputImage, class TOutputImage>
class ITK_EXPORT ElektaSynergyRawImageFilter :
    public
UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                        Function::SynergyAttenuation<
  typename TInputImage::PixelType, 
  typename TOutputImage::PixelType>   >
{
public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawImageFilter  Self;
  typedef UnaryFunctorImageFilter<TInputImage,TOutputImage, 
                                  Function::SynergyAttenuation< typename TInputImage::PixelType,
                                                 typename TOutputImage::PixelType> >  Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawImageFilter,
               UnaryFunctorImageFilter);

protected:
  ElektaSynergyRawImageFilter() {}
  virtual ~ElektaSynergyRawImageFilter() {}

private:
  ElektaSynergyRawImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

};

} // end namespace itk


#endif
