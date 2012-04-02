#ifndef __itkSetQuadricParamFromRegularParamFunction_h
#define __itkSetQuadricParamFromRegularParamFunction_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>
#include "itkRayQuadricIntersectionFunction.h"

namespace itk
{

/** \class SetQuadricParamFromRegularParamFunction
 * \brief Test if a ray intersects with a Quadric.
 * Return intersection points if there are some. See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for information on how this is implemented.
 * \ingroup Functions
 */
class ITK_EXPORT SetQuadricParamFromRegularParamFunction :
    public Object
{
public:
  /** Standard class typedefs. */
  typedef SetQuadricParamFromRegularParamFunction               Self;
  typedef Object                                                Superclass;
  typedef SmartPointer<Self>                                    Pointer;
  typedef SmartPointer<const Self>                              ConstPointer;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Useful defines. */
  typedef std::vector<double> VectorType;

  bool Translate( const VectorType& input );
  bool Rotate( const double input1, const VectorType& input2 );
  bool Config( const std::string input);

  /** Get / Set the quadric parameters. */
  itkGetMacro(A, double);
  itkSetMacro(A, double);
  itkGetMacro(B, double);
  itkSetMacro(B, double);
  itkGetMacro(C, double);
  itkSetMacro(C, double);
  itkGetMacro(D, double);
  itkSetMacro(D, double);
  itkGetMacro(E, double);
  itkSetMacro(E, double);
  itkGetMacro(F, double);
  itkSetMacro(F, double);
  itkGetMacro(G, double);
  itkSetMacro(G, double);
  itkGetMacro(H, double);
  itkSetMacro(H, double);
  itkGetMacro(I, double);
  itkSetMacro(I, double);
  itkGetMacro(J, double);
  itkSetMacro(J, double);

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

  itkSetMacro(Fig, std::vector< std::vector<double> >);
  itkGetMacro(Fig, std::vector< std::vector<double> >);

protected:

  /// Constructor
  SetQuadricParamFromRegularParamFunction();

  /// Destructor
  ~SetQuadricParamFromRegularParamFunction(){};

  /// The focal point or position of the ray source
  VectorType m_FocalPoint;

  /** Corners of the image Quadric */
  double m_SemiPrincipalAxisX;
  double m_SemiPrincipalAxisY;
  double m_SemiPrincipalAxisZ;
  double m_CenterX;
  double m_CenterY;
  double m_CenterZ;
  double m_RotationAngle;
  double m_A, m_B, m_C, m_D, m_E, m_F, m_G, m_H, m_I, m_J;
  std::vector< std::vector<double> > m_Fig;

private:
  SetQuadricParamFromRegularParamFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSetQuadricParamFromRegularParamFunction.txx"
#endif

#endif
