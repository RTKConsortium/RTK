/*=========================================================================
*
*  Copyright Insight Software Consortium
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
// CSharp wrapping definitions

#if SWIGCSHARP
%include "enumtypesafe.swg" // csharp/enums.swg can not parse the templated code in PixelIDValueEnum
%include "arrays_csharp.i"
%include "std_string.i"
%include "CSharpTypemapHelper.i"


// Enable CSharp classes derived from Command Execute method to be
// called from C++
%feature("director") rtk::simple::Command;

%CSharpPointerTypemapHelper( itk::DataObject*, System.IntPtr )
%CSharpPointerTypemapHelper( itk::Object::Pointer, System.IntPtr )
%CSharpPointerTypemapHelper( itk::Optimizer::Pointer, System.IntPtr )
%CSharpPointerTypemapHelper( itk::SingleValuedCostFunction::Pointer, System.IntPtr )
%CSharpPointerTypemapHelper( itk::TransformBase::Pointer, System.IntPtr )

%CSharpTypemapHelper( int8_t*, System.IntPtr )
%CSharpTypemapHelper( uint8_t*, System.IntPtr )
%CSharpTypemapHelper( int16_t*, System.IntPtr )
%CSharpTypemapHelper( uint16_t*, System.IntPtr )
%CSharpTypemapHelper( int32_t*, System.IntPtr )
%CSharpTypemapHelper( uint32_t*, System.IntPtr )
%CSharpTypemapHelper( float*, System.IntPtr )
%CSharpTypemapHelper( double*, System.IntPtr )

// Add override to ToString method
%csmethodmodifiers ToString "public override";

// Handle PermuteAxes DefaultOrder
// TODO:
//%apply unsigned int INOUT[] {unsigned int DefaultOrder[3]}

// Extend Image class
%typemap(cscode) rtk::simple::Image %{

  #region Unary operators

  ///<summary>Unary negation operator calls SimpleRTK.UnaryMinus.</summary>
  public static Image operator -(Image img1) {
    return SimpleRTK.UnaryMinus(img1);
  }

  ///<summary>Unary addition operator returns self.</summary>
  public static Image operator +(Image img1) {
    return img1;
  }

  ///<summary>Logical negation operator calls SimpleRTK.Not.</summary>
  public static Image operator !(Image img1) {
    return SimpleRTK.Not(img1);
  }

  ///<summary>Bitwise complement operator calls SimpleRTK.BitwiseNot.</summary>
  public static Image operator ~(Image img1) {
    return SimpleRTK.BitwiseNot(img1);
  }

  ///<summary>True operator provided to keep compiler happy. Always returns true.</summary>
  public static bool operator true(Image img1) {
    return true;
  }

  ///<summary>False operator provided to keep compiler happy. Always returns false.</summary>
  public static bool operator false(Image img1) {
    return false;
  }

  #endregion

  #region Binary mathematical operators

  ///<summary>Binary addition operator calls SimpleRTK.Add.</summary>
  public static Image operator +(Image img1, Image img2) {
    return SimpleRTK.Add(img1, img2);
  }

  ///<summary>Binary addition operator calls SimpleRTK.Add.</summary>
  public static Image operator +(Image img1, double constant) {
    return SimpleRTK.Add(img1, constant);
  }

  ///<summary>Binary addition operator calls SimpleRTK.Add.</summary>
  public static Image operator +(double constant, Image img1) {
    return SimpleRTK.Add(constant, img1);
  }

  ///<summary>Binary subtraction operator calls SimpleRTK.Subtract.</summary>
  public static Image operator -(Image img1, Image img2) {
    return SimpleRTK.Subtract(img1, img2);
  }

  ///<summary>Binary subtraction operator calls SimpleRTK.Subtract.</summary>
  public static Image operator -(Image img1, double constant) {
    return SimpleRTK.Subtract(img1, constant);
  }

  ///<summary>Binary subtraction operator calls SimpleRTK.Subtract.</summary>
  public static Image operator -(double constant, Image img1) {
    return SimpleRTK.Subtract(constant, img1);
  }

  ///<summary>Binary multiply operator calls SimpleRTK.Multiply.</summary>
  public static Image operator *(Image img1, Image img2) {
    return SimpleRTK.Multiply(img1, img2);
  }

  ///<summary>Binary multiply operator calls SimpleRTK.Multiply.</summary>
  public static Image operator *(Image img1, double constant) {
    return SimpleRTK.Multiply(img1, constant);
  }

  ///<summary>Binary multiply operator calls SimpleRTK.Multiply.</summary>
  public static Image operator *(double constant, Image img1) {
    return SimpleRTK.Multiply(constant, img1);
  }

  ///<summary>Binary division operator calls SimpleRTK.Divide.</summary>
  public static Image operator /(Image img1, Image img2) {
    return SimpleRTK.Divide(img1, img2);
  }

  ///<summary>Binary division operator calls SimpleRTK.Divide.</summary>
  public static Image operator /(Image img1, double constant) {
    return SimpleRTK.Divide(img1, constant);
  }

  ///<summary>Binary division operator calls SimpleRTK.Divide.</summary>
  public static Image operator /(double constant, Image img1) {
    return SimpleRTK.Divide(constant, img1);
  }

  #endregion

  #region Binary bitwise operators

  ///<summary>Binary bitwise AND operator calls SimpleRTK.And.</summary>
  public static Image operator &(Image img1, Image img2) {
    return SimpleRTK.And(img1, img2);
  }

  ///<summary>Binary bitwise AND operator calls SimpleRTK.And.</summary>
  public static Image operator &(Image img1, int constant) {
    return SimpleRTK.And(img1, constant);
  }

  ///<summary>Binary bitwise AND operator calls SimpleRTK.And.</summary>
  public static Image operator &(int constant, Image img1) {
    return SimpleRTK.And(constant, img1);
  }

  ///<summary>Binary bitwise OR operator calls SimpleRTK.Or.</summary>
  public static Image operator |(Image img1, Image img2) {
    return SimpleRTK.Or(img1, img2);
  }

  ///<summary>Binary bitwise OR operator calls SimpleRTK.Or.</summary>
  public static Image operator |(Image img1, int constant) {
    return SimpleRTK.Or(img1, constant);
  }

  ///<summary>Binary bitwise OR operator calls SimpleRTK.Or.</summary>
  public static Image operator |(int constant,Image img1) {
    return SimpleRTK.Or(constant, img1);
  }

  ///<summary>Binary bitwise XOR operator calls SimpleRTK.Xor.</summary>
  public static Image operator ^(Image img1, Image img2) {
    return SimpleRTK.Xor(img1, img2);
  }

  ///<summary>Binary bitwise XOR operator calls SimpleRTK.Xor.</summary>
  public static Image operator ^(Image img1, int constant) {
    return SimpleRTK.Xor(img1, constant);
  }

  ///<summary>Binary bitwise XOR operator calls SimpleRTK.Xor.</summary>
  public static Image operator ^(int constant, Image img1) {
    return SimpleRTK.Xor(constant, img1);
  }

  #endregion

  #region Comparison operators

  ///<summary>Less than operator calls SimpleRTK.Less.</summary>
  public static Image operator <(Image img1, Image img2) {
    return SimpleRTK.Less(img1, img2);
  }

  ///<summary>Less than operator calls SimpleRTK.Less.</summary>
  public static Image operator <(Image img1, double constant) {
    return SimpleRTK.Less(img1, constant);
  }

  ///<summary>Less than operator calls SimpleRTK.Less.</summary>
  public static Image operator <(double constant, Image img1) {
    return SimpleRTK.Less(constant, img1);
  }

  ///<summary>Greater than operator calls SimpleRTK.Greater.</summary>
  public static Image operator >(Image img1, Image img2) {
    return SimpleRTK.Greater(img1, img2);
  }

  ///<summary>Greater than operator calls SimpleRTK.Greater.</summary>
  public static Image operator >(Image img1, double constant) {
    return SimpleRTK.Greater(img1, constant);
  }

  ///<summary>Greater than operator calls SimpleRTK.Greater.</summary>
  public static Image operator >(double constant, Image img1) {
    return SimpleRTK.Greater(constant, img1);
  }

  ///<summary>Less than or equal to operator calls SimpleRTK.LessEqual.</summary>
  public static Image operator <=(Image img1, Image img2) {
    return SimpleRTK.LessEqual(img1, img2);
  }

  ///<summary>Less than or equal to operator calls SimpleRTK.LessEqual.</summary>
  public static Image operator <=(Image img1, double constant) {
    return SimpleRTK.LessEqual(img1, constant);
  }

  ///<summary>Less than or equal to operator calls SimpleRTK.LessEqual.</summary>
  public static Image operator <=(double constant,Image img1) {
    return SimpleRTK.LessEqual(constant, img1);
  }

  ///<summary>Greater than or equal to operator calls SimpleRTK.GreaterEqual.</summary>
  public static Image operator >=(Image img1, Image img2) {
    return SimpleRTK.GreaterEqual(img1, img2);
  }

  ///<summary>Greater than or equal to operator calls SimpleRTK.GreaterEqual.</summary>
  public static Image operator >=(Image img1, double constant) {
    return SimpleRTK.GreaterEqual(img1, constant);
  }

  ///<summary>Greater than or equal to operator calls SimpleRTK.GreaterEqual.</summary>
  public static Image operator >=(double constant, Image img1) {
    return SimpleRTK.GreaterEqual(constant, img1);
  }

  #endregion
%}

#endif // End of C# specific sections
