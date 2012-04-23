#ifndef __rtkAmsterdamShroudImageFilter_h
#define __rtkAmsterdamShroudImageFilter_h

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
namespace rtk
{

template<class TInputImage, class TOutputImage=
           itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension-1> >
class ITK_EXPORT AmsterdamShroudImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef AmsterdamShroudImageFilter                         Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

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
  itkTypeMacro(AmsterdamShroudImageFilter, itk::ImageToImageFilter);
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

  typedef itk::RecursiveGaussianImageFilter< TInputImage, TInputImage >          DerivativeType;
  typedef itk::MultiplyByConstantImageFilter< TInputImage, double, TInputImage > NegativeType;
  typedef itk::ThresholdImageFilter< TInputImage >                               ThresholdType;
  typedef itk::SumProjectionImageFilter< TInputImage, TOutputImage >             SumType;
  typedef itk::ConvolutionImageFilter< TOutputImage, TOutputImage >              ConvolutionType;
  typedef itk::SubtractImageFilter< TOutputImage, TOutputImage >                 SubtractType;
  typedef itk::PermuteAxesImageFilter< TOutputImage >                            PermuteType;

  typename DerivativeType::Pointer m_DerivativeFilter;
  typename NegativeType::Pointer m_NegativeFilter;
  typename ThresholdType::Pointer m_ThresholdFilter;
  typename SumType::Pointer m_SumFilter;
  typename ConvolutionType::Pointer m_ConvolutionFilter;
  typename SubtractType::Pointer m_SubtractFilter;
  typename PermuteType::Pointer m_PermuteFilter;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkAmsterdamShroudImageFilter.txx"
#endif

#endif
