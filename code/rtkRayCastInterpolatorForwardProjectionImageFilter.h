#ifndef __rtkRayCastInterpolatorForwardProjectionImageFilter_h
#define __rtkRayCastInterpolatorForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"

namespace rtk
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
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;

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

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayCastInterpolatorForwardProjectionImageFilter.txx"
#endif

#endif
