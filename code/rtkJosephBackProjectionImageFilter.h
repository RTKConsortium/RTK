#ifndef __rtkJosephBackProjectionImageFilter_h
#define __rtkJosephBackProjectionImageFilter_h

#include "rtkConfiguration.h"
#include "rtkBackProjectionImageFilter.h"

namespace rtk
{

/** \class JosephBackProjectionImageFilter
 * \brief Transpose of JosephForwardProjectionImageFilter.
 * This is expected to be slow compared to VoxelBasedBackProjectionImageFilter
 * because anti-aliasing strategy is required.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT JosephBackProjectionImageFilter :
  public BackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef JosephBackProjectionImageFilter                        Self;
  typedef BackProjectionImageFilter<TInputImage,TOutputImage>    Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::PixelType                       OutputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;
  typedef double                                                 CoordRepType;
  typedef itk::Vector<CoordRepType, TInputImage::ImageDimension> VectorType;

  typedef rtk::ThreeDCircularProjectionGeometry        GeometryType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(JosephBackProjectionImageFilter, BackProjectionImageFilter);

protected:
  JosephBackProjectionImageFilter() {this->SetInPlace(false);}
  virtual ~JosephBackProjectionImageFilter() {}

  virtual void GenerateData();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  void BilinearSplit(const InputPixelType ip,
                     const CoordRepType stepLengthInMM,
                     OutputPixelType *pxiyi,
                     OutputPixelType *pxsyi,
                     OutputPixelType *pxiys,
                     OutputPixelType *pxsys,
                     OutputPixelType *pxiyiw,
                     OutputPixelType *pxsyiw,
                     OutputPixelType *pxiysw,
                     OutputPixelType *pxsysw,
                     const CoordRepType x,
                     const CoordRepType y,
                     const unsigned int ox,
                     const unsigned int oy);

private:
  JosephBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                  //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkJosephBackProjectionImageFilter.txx"
#endif

#endif
