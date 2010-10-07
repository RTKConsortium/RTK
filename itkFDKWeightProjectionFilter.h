#ifndef __itkFDKWeightProjectionFilter_h
#define __itkFDKWeightProjectionFilter_h

#include "itkImageToImageFilter.h"

/** \class FDKWeightProjectionFilter
 * \brief Weighting of projections to correct for the divergence in
 * filtered backprojection reconstruction algorithms.
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT FDKWeightProjectionFilter :
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKWeightProjectionFilter Self;

  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  typedef SmartPointer<Self>        Pointer;
  typedef SmartPointer<const Self>  ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                                         InputImageType;
  typedef TOutputImage                                                        OutputImageType;
  typedef itk::Image<typename TOutputImage::PixelType, TOutputImage::ImageDimension-1> WeightImageType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FDKWeightProjectionFilter, ImageToImageFilter);

  /** Get/ Set geometry parameters */
  itkSetMacro(SourceToDetectorDistance, double);
  itkGetMacro(SourceToDetectorDistance, double);

protected:
  FDKWeightProjectionFilter():m_WeightsImage(NULL), m_CurrentSDD(-1){}
  ~FDKWeightProjectionFilter(){}

  void BeforeThreadedGenerateData();
  void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId);

private:
  FDKWeightProjectionFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typename WeightImageType::Pointer m_WeightsImage;
  double m_SourceToDetectorDistance;
  double m_CurrentSDD;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFDKWeightProjectionFilter.txx"
#endif

#endif
