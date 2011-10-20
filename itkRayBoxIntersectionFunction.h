#ifndef __itkRayBoxIntersectionFunction_h
#define __itkRayBoxIntersectionFunction_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>

namespace itk
{

/** \class RayBoxIntersectionFunction
 * \brief Test if a ray intersects with a box.
 *
 * \ingroup Functions
 */
template <
class TCoordRep = double,
unsigned int VBoxDimension=3
>
class ITK_EXPORT RayBoxIntersectionFunction :
    public Object
{
public:
  /** Standard class typedefs. */
  typedef RayBoxIntersectionFunction                            Self;
  typedef Object                                                Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;

  typedef typename ImageBase<VBoxDimension>::ConstPointer       ImageBaseConstPointer;


  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Useful defines. */
  typedef Vector<TCoordRep, VBoxDimension> VectorType;

  /** Evaluate the intersection points */
  bool Evaluate( const VectorType& input );

  /** Set the box information (Min/Max corners) from an itk image.
   * \warning this method caches image information.
   * If the image information has changed, user must call
   * SetBoxFromImage again to update cached values. */
  void SetBoxFromImage( ImageBaseConstPointer img );

  /** Get / Set the box inferior corner. Every coordinate must be inferior to
   * those of the superior corner. */
  itkGetMacro(BoxMin, VectorType);
  itkSetMacro(BoxMin, VectorType);

  /** Get / Set the box superior corner. Every coordinate must be superior to
   * those of the inferior corner. */
  itkGetMacro(BoxMax, VectorType);
  itkSetMacro(BoxMax, VectorType);

  /** Get / Set the ray origin. */
  itkGetMacro(RayOrigin, VectorType);
  itkSetMacro(RayOrigin, VectorType);

  /** Get the distance with the nearest intersection.
    * \warning Only relevant if called after Evaluate. */
  itkGetMacro(NearestDistance, TCoordRep);

  /** Get the distance with the farthest intersection.
    * \warning Only relevant if called after Evaluate. */
  itkGetMacro(FarthestDistance, TCoordRep);

  /** Get the nearest point coordinates.
    * \warning Only relevant if called after Evaluate. */
  virtual VectorType GetNearestPoint()
  {
    return m_RayOrigin + m_NearestDistance * m_RayDirection;
  }

  /** Get the farthest point coordinates.
    * \warning Only relevant if called after Evaluate. */
  virtual VectorType GetFarthestPoint()
  {
    return m_RayOrigin + m_FarthestDistance * m_RayDirection;
  }

protected:

  /// Constructor
  RayBoxIntersectionFunction(){};

  /// Destructor
  ~RayBoxIntersectionFunction(){};

  /// The focal point or position of the ray source
  VectorType m_FocalPoint;

  /** Corners of the image box */
  VectorType m_BoxMin;
  VectorType m_BoxMax;
  VectorType m_RayOrigin;
  VectorType m_RayDirection;
  TCoordRep  m_NearestDistance;
  TCoordRep  m_FarthestDistance;

private:
  RayBoxIntersectionFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkRayBoxIntersectionFunction.txx"
#endif

#endif
