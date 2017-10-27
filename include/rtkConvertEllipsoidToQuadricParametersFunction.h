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

#ifndef rtkConvertEllipsoidToQuadricParametersFunction_h
#define rtkConvertEllipsoidToQuadricParametersFunction_h

#include <itkNumericTraits.h>
#include <itkVector.h>
#include <itkObjectFactory.h>
#include "rtkWin32Header.h"
#include "rtkMacro.h"

namespace rtk
{

/** \class ConvertEllipsoidToQuadricParametersFunction
 * \brief Converts ellipsoid parameters to quadric parameters.
 *
 * Converts ellipsoid parameters, i.e., semi-principal axes, center and
 * rotation angle to quadric parameters.
 *
 * \author Marc Vila
 *
 * \ingroup Geometry
 */

class RTK_EXPORT ConvertEllipsoidToQuadricParametersFunction :
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
  typedef itk::Vector<double,3>              VectorType;
  typedef std::vector< std::vector<double> > VectorOfVectorType;
  typedef std::string                        StringType;

  bool Translate( const VectorType& input );
  bool Rotate( const double input1, const VectorType& input2 );

  /** Get / Set the quadric parameters. The convention for the names of the
   * quadric parameters is explained in
   * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm*/
  itkGetMacro(A, double);
  itkGetMacro(B, double);
  itkGetMacro(C, double);
  itkGetMacro(D, double);
  itkGetMacro(E, double);
  itkGetMacro(F, double);
  itkGetMacro(G, double);
  itkGetMacro(H, double);
  itkGetMacro(I, double);
  itkGetMacro(J, double);

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

  itkSetMacro(Figure, StringType);
  itkGetMacro(Figure, StringType);

protected:

  /// Constructor
  ConvertEllipsoidToQuadricParametersFunction();

  /// Destructor
  ~ConvertEllipsoidToQuadricParametersFunction() {}

  /** Corners of the image Quadric */
  double     m_SemiPrincipalAxisX;
  double     m_SemiPrincipalAxisY;
  double     m_SemiPrincipalAxisZ;
  double     m_CenterX;
  double     m_CenterY;
  double     m_CenterZ;
  double     m_RotationAngle;
  double     m_A;
  double     m_B;
  double     m_C;
  double     m_D;
  double     m_E;
  double     m_F;
  double     m_G;
  double     m_H;
  double     m_I;
  double     m_J;
  StringType m_Figure;

private:
  ConvertEllipsoidToQuadricParametersFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // end namespace rtk

#endif
