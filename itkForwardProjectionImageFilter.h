#ifndef __itkForwardProjectionImageFilter_h
#define __itkForwardProjectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"

namespace itk
{

/** \class ForwardProjectionImageFilter
 * \brief Base class for forward projection, i.e. accumulation along x-ray lines.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT ForwardProjectionImageFilter :
  public InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ForwardProjectionImageFilter                 Self;
  typedef InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  typedef itk::ThreeDCircularProjectionGeometry        GeometryType;
  typedef typename GeometryType::Pointer               GeometryPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro(ForwardProjectionImageFilter, ImageToImageFilter);

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

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkForwardProjectionImageFilter.txx"
#endif

#endif
