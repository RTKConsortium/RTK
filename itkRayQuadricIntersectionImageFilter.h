#ifndef __itkRayQuadricIntersectionImageFilter_h
#define __itkRayQuadricIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"
#include "itkRayQuadricIntersectionFunction.h"

namespace itk
{

/** \class RayQuadricIntersectionImageFilter
 * \brief Computes intersection of projection rays with quadric objects.
 * (ellipsoid, cone, cylinder...). See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for more information.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayQuadricIntersectionImageFilter :
  public InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayQuadricIntersectionImageFilter            Self;
  typedef InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef SmartPointer<Self>                           Pointer;
  typedef SmartPointer<const Self>                     ConstPointer;

  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer OutputImageBaseConstPointer;
  typedef itk::ThreeDCircularProjectionGeometry           GeometryType;
  typedef typename GeometryType::Pointer                  GeometryPointer;
  typedef RayQuadricIntersectionFunction<double, 3>       RQIFunctionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayQuadricIntersectionImageFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  /** Get the RayQuadricIntersectionFunction to set its parameters.
    * A call to this function will assume modification of the function.*/
  RQIFunctionType::Pointer GetRQIFunctor();

protected:
  RayQuadricIntersectionImageFilter() : m_RQIFunctor( RQIFunctionType::New() ), m_Geometry(NULL) { this->SetNumberOfThreads(1);}
  virtual ~RayQuadricIntersectionImageFilter() {};

  /** Apply changes to the input image requested region. */
  virtual void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

private:
  RayQuadricIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  /** Functor object to compute the intersection */
  RQIFunctionType::Pointer m_RQIFunctor;

  /** RTK geometry object */
  GeometryPointer m_Geometry;
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRayQuadricIntersectionImageFilter.txx"
#endif

#endif
