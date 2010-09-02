#ifndef __itkAmsterdamShroudImageFilter_h
#define __itkAmsterdamShroudImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkRecursiveGaussianImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <itkThresholdImageFilter.h>
#include <itkSumProjectionImageFilter.h>
#include <itkSubtractImageFilter.h>

/** \class AmsterdamShroudImageFilter
 * \brief TODO
 *
 * TODO
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension-1> >
class ITK_EXPORT AmsterdamShroudImageFilter:
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef AmsterdamShroudImageFilter Self;

  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                              InputImageType;
  typedef TOutputImage                             OutputImageType;
  typedef typename InputImageType::Pointer         InputImagePointer;
  typedef typename InputImageType::ConstPointer    InputImageConstPointer;
  typedef typename InputImageType::PixelType       InputImagePixelType;
  typedef typename OutputImageType::Pointer        OutputImagePointer;
  typedef typename OutputImageType::ConstPointer   OutputImageConstPointer;
  typedef typename OutputImageType::PixelType      OutputImagePixelType;
  typedef typename InputImageType::RegionType      RegionType;
  typedef typename InputImageType::IndexType       IndexType;
  typedef typename InputImageType::SizeType        SizeType;

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
  void operator=(const Self&); //purposely not implemented

  typedef RecursiveGaussianImageFilter< TInputImage, TInputImage > DerivativeType;
  typedef MultiplyByConstantImageFilter< TInputImage, double, TInputImage > NegativeType;
  typedef ThresholdImageFilter< TInputImage > ThresholdType;
  typedef SumProjectionImageFilter< TInputImage, TOutputImage > SumType;
  typedef RecursiveGaussianImageFilter< TOutputImage, TOutputImage > SmoothType;
  typedef SubtractImageFilter< TOutputImage, TOutputImage > SubtractType;

  typename DerivativeType::Pointer m_DerivativeFilter;
  typename NegativeType::Pointer m_NegativeFilter;
  typename ThresholdType::Pointer m_ThresholdFilter;
  typename SumType::Pointer m_SumFilter;
  typename SmoothType::Pointer m_SmoothFilter;
  typename SubtractType::Pointer m_SubtractFilter;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkAmsterdamShroudImageFilter.txx"
#endif

#endif
