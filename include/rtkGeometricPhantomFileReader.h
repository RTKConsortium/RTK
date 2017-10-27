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

#ifndef rtkGeometricPhantomFileReader_h
#define rtkGeometricPhantomFileReader_h

#include <itkNumericTraits.h>
#include <vector>
#include <itkImageBase.h>
#include "rtkRayQuadricIntersectionFunction.h"
#include "rtkMacro.h"
#include "rtkWin32Header.h"

namespace rtk
{

/** \class GeometricPhantomFileReader
 * \brief Reads configuration file containing specifications of a geometric
 * phantom.
 *
 * \test rtkprojectgeometricphantomtest.cxx, rtkdrawgeometricphantomtest.cxx
 *
 * \author Marc Vila
 *
 * \ingroup Functions
 */
class RTK_EXPORT GeometricPhantomFileReader :
    public itk::Object
{
public:
  /** Standard class typedefs. */
  typedef GeometricPhantomFileReader               Self;
  typedef itk::Object                              Superclass;
  typedef itk::SmartPointer<Self>                  Pointer;
  typedef itk::SmartPointer<const Self>            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GeometricPhantomFileReader, itk::Object);

  /** Useful defines. */
  typedef std::vector<double>                VectorType;
  typedef std::vector< std::vector<double> > VectorOfVectorType;
  typedef std::string                        StringType;

//FIXME: this struct should be used, but error with Get/Set Macros
//  struct FigureType
//  {
//    //FigureType():angle(0.),density(0.){};
//    VectorOfVectorType       parameters;
//    std::vector<std::string> figure;
//  };

  bool Config( const std::string input);

  virtual VectorOfVectorType GetFig ();
  virtual void SetFig (const VectorOfVectorType _arg);

  /** Get/Set Number of Figures.*/
  rtkSetStdVectorMacro(FigureTypes, std::vector<StringType>);
  rtkGetStdVectorMacro(FigureTypes, std::vector<StringType>);

protected:

  /// Constructor
  GeometricPhantomFileReader() {};

  /// Destructor
  ~GeometricPhantomFileReader() {}

  /** Corners of the image Quadric */
  VectorOfVectorType      m_Fig;
  std::vector<StringType> m_FigureTypes;

private:
  GeometricPhantomFileReader( const Self& ); //purposely not implemented
  void operator=( const Self& ); //purposely not implemented
};

} // end namespace rtk

#endif
