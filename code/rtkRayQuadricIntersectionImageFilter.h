#ifndef __rtkRayQuadricIntersectionImageFilter_h
#define __rtkRayQuadricIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionFunction.h"

namespace rtk
{

/** \class RayQuadricIntersectionImageFilter
 * \brief Computes intersection of projection rays with quadric objects.
 * (ellipsoid, cone, cylinder...). See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for more information.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayQuadricIntersectionImageFilter :
  public itk::InPlaceImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayQuadricIntersectionImageFilter                 Self;
  typedef itk::InPlaceImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                           Pointer;
  typedef itk::SmartPointer<const Self>                     ConstPointer;

  typedef typename TOutputImage::RegionType               OutputImageRegionType;
  typedef typename TOutputImage::Superclass::ConstPointer OutputImageBaseConstPointer;
  typedef rtk::ThreeDCircularProjectionGeometry           GeometryType;
  typedef typename GeometryType::Pointer                  GeometryPointer;
  typedef RayQuadricIntersectionFunction<double, 3>       RQIFunctionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayQuadricIntersectionImageFilter, itk::ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  itkGetMacro(Geometry, GeometryPointer);
  itkSetMacro(Geometry, GeometryPointer);

  itkGetMacro(MultiplicativeConstant, double);
  itkSetMacro(MultiplicativeConstant, double);

  /** Get the RayQuadricIntersectionFunction to set its parameters.
    * A call to this function will assume modification of the function.*/
  RQIFunctionType::Pointer GetRQIFunctor();

protected:
  RayQuadricIntersectionImageFilter();
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

  /** Multiplicative factor of intersection length */
  double m_MultiplicativeConstant;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayQuadricIntersectionImageFilter.txx"
#endif

#endif
