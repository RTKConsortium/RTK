#ifndef __itkJosephForwardProjectionImageFilter_h
#define __itkJosephForwardProjectionImageFilter_h

#include "itkForwardProjectionImageFilter.h"

namespace itk
{

/** \class JosephForwardProjectionImageFilter
 * \brief Joseph forward projection.
 * Performs a forward projection, i.e. accumulation along x-ray lines,
 * using [Joseph, IEEE TMI, 1982].
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT JosephForwardProjectionImageFilter :
  public ForwardProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef JosephForwardProjectionImageFilter                     Self;
  typedef ForwardProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                                     Pointer;
  typedef SmartPointer<const Self>                               ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JosephForwardProjectionImageFilter, ForwardProjectionImageFilter);

protected:
  JosephForwardProjectionImageFilter() {}
  virtual ~JosephForwardProjectionImageFilter() {}

  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  inline OutputPixelType BilinearInterpolation(const InputPixelType *p1,
                                               const InputPixelType *p2,
                                               const InputPixelType *p3,
                                               const InputPixelType *p4,
                                               const double x,
                                               const double y,
                                               const unsigned int ox,
                                               const unsigned int oy) const;

private:
  JosephForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkJosephForwardProjectionImageFilter.txx"
#endif

#endif
