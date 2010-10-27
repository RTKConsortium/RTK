#ifndef __itkFDKBackProjectionImageFilter_h
#define __itkFDKBackProjectionImageFilter_h

#include "itkBackProjectionImageFilter.h"
#include "rtkThreeDCircularGeometry.h"

namespace itk
{
  
template <class TInputImage, class TOutputImage>
class ITK_EXPORT FDKBackProjectionImageFilter :
  public BackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKBackProjectionImageFilter                  Self;
  typedef ImageToImageFilter<TInputImage,TOutputImage>  Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  typedef rtk::ThreeDCircularGeometry                   GeometryType;
  typedef GeometryType::Pointer                         GeometryPointer;
  typedef GeometryType::MatrixType                      ProjectionMatrixType;
  typedef typename TOutputImage::RegionType             OutputImageRegionType;
  typedef itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension-1> ProjectionImageType;
  typedef typename ProjectionImageType::Pointer                     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FDKBackProjectionImageFilter, ImageToImageFilter);

protected:
  FDKBackProjectionImageFilter() {};
  virtual ~FDKBackProjectionImageFilter() {};

  virtual void BeforeThreadedGenerateData();
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, int threadId );

  /** Given the set of unordered projections, this functions computes the angular
      weights of FDK for each projection */
  virtual void UpdateAngularWeights();

private:
  FDKBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  /** Angular weights for each projection */
  std::vector<double> m_AngularWeights;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFDKBackProjectionImageFilter.txx"
#endif

#endif
