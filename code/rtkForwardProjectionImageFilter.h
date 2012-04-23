#ifndef __rtkForwardProjectionImageFilter_h
#define __rtkForwardProjectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

/** \class ForwardProjectionImageFilter
 * \brief Base class for forward projection, i.e. accumulation along x-ray lines.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT ForwardProjectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ForwardProjectionImageFilter                      Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef rtk::ThreeDCircularProjectionGeometry        GeometryType;
  typedef typename GeometryType::Pointer               GeometryPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardProjectionImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

protected:
  ForwardProjectionImageFilter() : m_Geometry(NULL) {
    this->SetNumberOfRequiredInputs(2); this->SetInPlace( true );
  };
  virtual ~ForwardProjectionImageFilter() {
  };

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

private:
  ForwardProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** RTK geometry object */
  GeometryPointer m_Geometry;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkForwardProjectionImageFilter.txx"
#endif

#endif
