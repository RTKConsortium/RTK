/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __rtkConvertEllipsoidToQuadricParametersFunction_h
#define __rtkConvertEllipsoidToQuadricParametersFunction_h

#include <itkNumericTraits.h>
#include <itkVector.h>

namespace rtk
{

/** \class RegularToQuadricFunction
 * \brief Translates regular geometric expression to quadric expression.
 *
 * Translates regular geometric parameters to quadric
 * expressions.
 *
 *   http://fr.wikipedia.org/wiki/Quadrique
 *
 * It also rotates and translates a
 * quadric expression with a certain given angle
 * and center of the object, in order to create the
 * desired figure.
 *
 * \author Marc Vila
 *
 * \ingroup Object
 */
class ConvertEllipsoidToQuadricParametersFunction :
    public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef ConvertEllipsoidToQuadricParametersFunction  Self;
  typedef itk::Object                                  Superclass;
  typedef itk::SmartPointer<Self>                      Pointer;
  typedef itk::SmartPointer<const Self>                ConstPointer;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(ConvertEllipsoidToQuadricParametersFunction, itk::Object);

  /** Useful defines. */
  typedef std::vector<double>                VectorType;
  typedef std::vector< std::vector<double> > VectorOfVectorType;

  bool Translate( const VectorType& input );
  bool Rotate( const double input1, const VectorType& input2 );

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

protected:

  /// Constructor
  ConvertEllipsoidToQuadricParametersFunction();

  /// Destructor
  ~ConvertEllipsoidToQuadricParametersFunction() {};

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
  double m_A;
  double m_B;
  double m_C;
  double m_D;
  double m_E;
  double m_F;
  double m_G;
  double m_H;
  double m_I;
  double m_J;

private:
  ConvertEllipsoidToQuadricParametersFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConvertEllipsoidToQuadricParametersFunction.txx"
#endif

#endif
