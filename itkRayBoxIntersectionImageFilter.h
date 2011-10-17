#ifndef __itkRayBoxIntersectionImageFilter_h
#define __itkRayBoxIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"
#include "itkRayBoxIntersectionFunction.h"

namespace itk
{

/** \class RayBoxIntersectionImageFilter
 * \brief Computes intersection of projection rays with image box.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayBoxIntersectionImageFilter :
  public InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayBoxIntersectionImageFilter                Self;
  typedef InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  typedef typename TOutputImage::RegionType            OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer            OutputImageBaseConstPointer;
  typedef itk::ThreeDCircularProjectionGeometry        GeometryType;
  typedef typename GeometryType::Pointer               GeometryPointer;
  typedef RayBoxIntersectionFunction<double, 3>        RBIFunctionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayBoxIntersectionImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Set the box from an image */
  void SetBoxFromImage(OutputImageBaseConstPointer _arg);

protected:
  RayBoxIntersectionImageFilter() : m_RBIFunctor( RBIFunctionType::New() ), m_Geometry(NULL) { this->SetNumberOfThreads(1);}
  virtual ~RayBoxIntersectionImageFilter() {};

  /** Apply changes to the input image requested region. */
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

private:
  RayBoxIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Functor object to compute the intersection */
  RBIFunctionType::Pointer m_RBIFunctor;

  /** RTK geometry object */
  GeometryPointer m_Geometry;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRayBoxIntersectionImageFilter.txx"
#endif

#endif
