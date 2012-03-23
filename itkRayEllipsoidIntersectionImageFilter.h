#ifndef __itkRayEllipsoidIntersectionImageFilter_h
#define __itkRayEllipsoidIntersectionImageFilter_h

#include <itkInPlaceImageFilter.h>
#include "itkThreeDCircularProjectionGeometry.h"
#include "itkRayQuadricIntersectionImageFilter.h"

#include "itkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkRayEllipsoidIntersectionImageFilter.h"

#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace itk
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
  typedef SmartPointer<Self>                                          Pointer;
  typedef SmartPointer<const Self>                                    ConstPointer;

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

  /** Translate user parameteres to quadric parameters.
   * A call to this function will assume modification of the function.*/
  void Translate();
  void Rotate();

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
};

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRayEllipsoidIntersectionImageFilter.txx"
#endif

#endif
