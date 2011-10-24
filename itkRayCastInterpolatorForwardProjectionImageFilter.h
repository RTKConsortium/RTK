#ifndef __itkRayCastInterpolatorForwardProjectionImageFilter_h
#define __itkRayCastInterpolatorForwardProjectionImageFilter_h

#include "itkForwardProjectionImageFilter.h"

namespace itk
{

/** \class RayCastInterpolatorForwardProjectionImageFilter
 * Forward projection using itk RayCastInterpolateFunction.
 * RayCastInterpolateFunction does not handle ITK geometry correctly but this
 * is accounted for this class.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayCastInterpolatorForwardProjectionImageFilter :
  public ForwardProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayCastInterpolatorForwardProjectionImageFilter        Self;
  typedef ForwardProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;

  /** Useful typedefs. */
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayCastInterpolatorForwardProjectionImageFilter, ForwardProjectionImageFilter);

protected:
  RayCastInterpolatorForwardProjectionImageFilter() {}
  virtual ~RayCastInterpolatorForwardProjectionImageFilter() {}

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

private:
  RayCastInterpolatorForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRayCastInterpolatorForwardProjectionImageFilter.txx"
#endif

#endif
