#ifndef __itkLutImageFilter_h
#define __itkLutImageFilter_h

#include <itkUnaryFunctorImageFilter.h>

namespace itk
{

namespace Functor
{
template< class TInput, class TOutput >
class LUT
{
public:
  typedef itk::Image<TOutput,1> LutType;
  typedef typename LutType::PixelType* LutDataPointerType;

  LUT() {};
  ~LUT() {};

  LutDataPointerType GetLutDataPointer () {
    return m_LutDataPointer;
  }
  void SetLutDataPointer (LutDataPointerType lut) {
    m_LutDataPointer = lut;
  }

  bool operator!=( const LUT & lut ) const {
    return m_LutDataPointer != lut->GetLutDataPointer;
  }
  bool operator==( const LUT & lut ) const {
    return m_LutDataPointer == lut->GetLutDataPointer;
  }

  inline TOutput operator()( const TInput & val ) const {
    return m_LutDataPointer[val];
  }

private:
  LutDataPointerType m_LutDataPointer;
};
} // end namespace Functor


template <class TInputImage, class TOutputImage>
class ITK_EXPORT LutImageFilter: public
  UnaryFunctorImageFilter< TInputImage,
  TOutputImage,
  Functor::LUT< typename TInputImage::PixelType,
  typename TOutputImage::PixelType> >
{

public:
  /** Lookup table type definition. */
  typedef Functor::LUT< typename TInputImage::PixelType, typename TOutputImage::PixelType > FunctorType;
  typedef typename FunctorType::LutType LutType;

  /** Standard class typedefs. */
  typedef LutImageFilter Self;
  typedef UnaryFunctorImageFilter<TInputImage, TOutputImage, FunctorType > Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(LutImageFilter, UnaryFunctorImageFilter);

  /** Set lookup table. */
  virtual void SetLut (LutType* _arg) { //Idem as itkSetObjectMacro + call to functor SetLutDataPointer
    itkDebugMacro("setting " << "Lut" " to " << _arg );
    if (this->m_Lut != _arg) {
      this->m_Lut = _arg;
      this->Modified();
      this->GetFunctor().SetLutDataPointer(_arg->GetBufferPointer());
    }
  }

  /** Get lookup table. */
  itkGetObjectMacro(Lut, LutType);

protected:
  LutImageFilter() {}
  virtual ~LutImageFilter() {}
  typename LutType::Pointer m_Lut;

private:
  LutImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
};

} // end namespace itk


#endif
