#ifndef __itkCudaFFTRampImageFilter_h
#define __itkCudaFFTRampImageFilter_h

#include "itkFFTRampImageFilter.h"

/** \class CudaFFTRampImageFilter
 * \brief Implements the ramp image filter of the filtered backprojection algorithm.
 * uses CUFFT for the projection fft and ifft.
 *
 * \author Simon Rit
 */
namespace itk
{

class ITK_EXPORT CudaFFTRampImageFilter :
  public FFTRampImageFilter< itk::Image<float,3>, itk::Image<float,3>, float >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<float,3> ImageType;
  typedef CudaFFTRampImageFilter Self;
  typedef FFTRampImageFilter< ImageType, ImageType, double > Superclass;
  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(CudaFFTRampImageFilter, ImageToImageFilter);

protected:
  CudaFFTRampImageFilter();
  ~CudaFFTRampImageFilter(){}

  virtual void GenerateData( );

private:
  CudaFFTRampImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented
}; // end of class

} // end namespace itk

#endif
