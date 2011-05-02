#ifndef __itkAmsterdamShroudImageFilter_h
#define __itkAmsterdamShroudImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkRecursiveGaussianImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkSumProjectionImageFilter.h>
#include <itkConvolutionImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkPermuteAxesImageFilter.h>

/** \class AmsterdamShroudImageFilter
 * \brief TODO
 *
 * TODO
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=
           itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension-1> >
class ITK_EXPORT AmsterdamShroudImageFilter :
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef AmsterdamShroudImageFilter Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(AmsterdamShroudImageFilter, ImageToImageFilter);
protected:
  AmsterdamShroudImageFilter();
  ~AmsterdamShroudImageFilter(){}

  void GenerateOutputInformation();
  void GenerateInputRequestedRegion();

  /** Single-threaded version of GenerateData.  This filter delegates
   * to other filters. */
  void GenerateData();

private:
  AmsterdamShroudImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);             //purposely not implemented

  typedef RecursiveGaussianImageFilter< TInputImage, TInputImage >          DerivativeType;
  typedef MultiplyByConstantImageFilter< TInputImage, double, TInputImage > NegativeType;
  typedef ThresholdImageFilter< TInputImage >                               ThresholdType;
  typedef SumProjectionImageFilter< TInputImage, TOutputImage >             SumType;
  typedef ConvolutionImageFilter< TOutputImage, TOutputImage >              ConvolutionType;
  typedef SubtractImageFilter< TOutputImage, TOutputImage >                 SubtractType;
  typedef PermuteAxesImageFilter< TOutputImage >                            PermuteType;

  typename DerivativeType::Pointer m_DerivativeFilter;
  typename NegativeType::Pointer m_NegativeFilter;
  typename ThresholdType::Pointer m_ThresholdFilter;
  typename SumType::Pointer m_SumFilter;
  typename ConvolutionType::Pointer m_ConvolutionFilter;
  typename SubtractType::Pointer m_SubtractFilter;
  typename PermuteType::Pointer m_PermuteFilter;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAmsterdamShroudImageFilter.txx"
#endif

#endif
