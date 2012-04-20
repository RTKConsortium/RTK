#ifndef __rtkRayEllipsoidIntersectionImageFilter_h
#define __rtkRayEllipsoidIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkRayQuadricIntersectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "rtkRayEllipsoidIntersectionImageFilter.h"
#include "rtkSetQuadricParamFromRegularParamFunction.h"

namespace rtk
{

/** \class RayEllipsoidIntersectionImageFilter
 * \brief Computes intersection of projection rays with ellipsoids.
 * See http://en.wikipedia.org/wiki/Ellipsoid
 * for more information.
 */

template <class TInputImage, class TOutputImage>
class ITK_EXPORT RayEllipsoidIntersectionImageFilter :
  public RayQuadricIntersectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef RayEllipsoidIntersectionImageFilter                         Self;
  typedef RayQuadricIntersectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                          Pointer;
  typedef itk::SmartPointer<const Self>                                    ConstPointer;

  typedef SetQuadricParamFromRegularParamFunction                     SQPFunctionType;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(RayEllipsoidIntersectionImageFilter, RayQuadricIntersectionImageFilter);

  /** Get/Set the semi-principal axes of the ellipsoid.*/
  itkGetMacro(SemiPrincipalAxisX, double);
  itkSetMacro(SemiPrincipalAxisX, double);
  itkGetMacro(SemiPrincipalAxisY, double);
  itkSetMacro(SemiPrincipalAxisY, double);
  itkGetMacro(SemiPrincipalAxisZ, double);
  itkSetMacro(SemiPrincipalAxisZ, double);
  itkGetMacro(CenterX, double);
  itkSetMacro(CenterX, double);
  itkGetMacro(CenterY, double);
  itkSetMacro(CenterY, double);
  itkGetMacro(CenterZ, double);
  itkSetMacro(CenterZ, double);

  itkGetMacro(RotationAngle, double);
  itkSetMacro(RotationAngle, double);

protected:
  RayEllipsoidIntersectionImageFilter();
  virtual ~RayEllipsoidIntersectionImageFilter() {};

  virtual void BeforeThreadedGenerateData();

private:
  RayEllipsoidIntersectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);            //purposely not implemented

  double m_SemiPrincipalAxisX;
  double m_SemiPrincipalAxisY;
  double m_SemiPrincipalAxisZ;
  double m_CenterX;
  double m_CenterY;
  double m_CenterZ;
  double m_RotationAngle;
  SQPFunctionType::Pointer m_SQPFunctor;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkRayEllipsoidIntersectionImageFilter.txx"
#endif

#endif
