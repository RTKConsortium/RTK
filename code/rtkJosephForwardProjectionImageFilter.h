#ifndef __rtkJosephForwardProjectionImageFilter_h
#define __rtkJosephForwardProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkForwardProjectionImageFilter.h"

namespace rtk
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
  typedef itk::SmartPointer<Self>                                     Pointer;
  typedef itk::SmartPointer<const Self>                               ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef double                                                 CoordRepType;
  typedef itk::Vector<CoordRepType, TInputImage::ImageDimension> VectorType;

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

  inline OutputPixelType BilinearInterpolation(const InputPixelType *pxiyi,
                                               const InputPixelType *pxsyi,
                                               const InputPixelType *pxiys,
                                               const InputPixelType *pxsys,
                                               const double x,
                                               const double y,
                                               const unsigned int ox,
                                               const unsigned int oy) const;

private:
  JosephForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkJosephForwardProjectionImageFilter.txx"
#endif

#endif
