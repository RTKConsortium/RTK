#ifndef __rtkConjugateGradientOperator_h
#define __rtkConjugateGradientOperator_h

#include "itkImageToImageFilter.h"

namespace rtk
{
template< typename OutputImageType>
class ConjugateGradientOperator : public itk::ImageToImageFilter< OutputImageType, OutputImageType>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientOperator             Self;
    typedef itk::ImageToImageFilter< OutputImageType, OutputImageType > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientOperator, itk::ImageToImageFilter)

    /** The 4D image to be updated.*/
    virtual void SetX(const OutputImageType* OutputImage);

protected:
    ConjugateGradientOperator();
    ~ConjugateGradientOperator(){}

private:
    ConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientOperator.txx"
#endif

#endif
