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

#ifndef __rtkSetQuadricParamFromRegularParamFunction_h
#define __rtkSetQuadricParamFromRegularParamFunction_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>
#include "rtkRayQuadricIntersectionFunction.h"

namespace rtk
{

/** \class SetQuadricParamFromRegularParamFunction
 * \brief Test if a ray intersects with a Quadric.
 * Return intersection points if there are some. See
 * http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter4.htm
 * for information on how this is implemented.
 * \ingroup Functions
 */
class ITK_EXPORT SetQuadricParamFromRegularParamFunction :
    public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef SetQuadricParamFromRegularParamFunction  Self;
  typedef itk::Object                              Superclass;
  typedef itk::SmartPointer<Self>                  Pointer;
  typedef itk::SmartPointer<const Self>            ConstPointer;
  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Useful defines. */
  typedef std::vector<double> VectorType;
  typedef std::vector< std::vector<double> > VectorOfVectorType;

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

  rtkSetMacro(Fig, VectorOfVectorType);
  rtkGetMacro(Fig, VectorOfVectorType);




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
  VectorOfVectorType m_Fig;

private:
  SetQuadricParamFromRegularParamFunction( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSetQuadricParamFromRegularParamFunction.txx"
#endif

#endif
