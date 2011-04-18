#ifndef __itkFDKBackProjectionImageFilter_h
#define __itkFDKBackProjectionImageFilter_h

#include "itkBackProjectionImageFilter.h"

namespace itk
{

template <class TInputImage, class TOutputImage>
class ITK_EXPORT FDKBackProjectionImageFilter :
  public BackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKBackProjectionImageFilter                        Self;
  typedef BackProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                                  Pointer;
  typedef SmartPointer<const Self>                            ConstPointer;

  typedef typename Superclass::ProjectionMatrixType                                  ProjectionMatrixType;                                                                                                      typedef typename TOutputImage::RegionType                                          OutputImageRegionType;
  typedef itk::Image<typename TInputImage::PixelType, TInputImage::ImageDimension-1> ProjectionImageType;
  typedef typename ProjectionImageType::Pointer                                      ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FDKBackProjectionImageFilter, ImageToImageFilter);

protected:
  FDKBackProjectionImageFilter() {};
  virtual ~FDKBackProjectionImageFilter() {};

  virtual void BeforeThreadedGenerateData();

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, int threadId );

  /** Optimized version when the rotation is parallel to X, i.e. matrix[1][0]
    and matrix[2][0] are zeros. */
  virtual void OptimizedBackprojectionX(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                                        const ProjectionImagePointer projection);

  /** Optimized version when the rotation is parallel to Y, i.e. matrix[1][1]
    and matrix[2][1] are zeros. */
  virtual void OptimizedBackprojectionY(const OutputImageRegionType& region, const ProjectionMatrixType& matrix,
                                        const ProjectionImagePointer projection);

private:
  FDKBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFDKBackProjectionImageFilter.txx"
#endif

#endif
