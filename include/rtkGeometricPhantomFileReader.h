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

#include <itkLightProcessObject.h>
#include "rtkGeometricPhantom.h"

namespace rtk
{

/** \class GeometricPhantomFileReader
 * \brief Reads configuration file in a user-defined file format
 *
 * \test rtkprojectgeometricphantomtest.cxx, rtkdrawgeometricphantomtest.cxx,
 *       rtkforbildtest.cxx
 *
 * \author Marc Vila, Simon Rit
 *
 * \ingroup Functions
 */
class RTK_EXPORT GeometricPhantomFileReader :
    public itk::LightProcessObject
{
public:
  /** Standard class typedefs. */
  typedef GeometricPhantomFileReader     Self;
  typedef itk::Object                    Superclass;
  typedef itk::SmartPointer<Self>        Pointer;
  typedef itk::SmartPointer<const Self>  ConstPointer;

  /** Convenient typedefs. */
  typedef GeometricPhantom::Pointer      GeometricPhantomPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(GeometricPhantomFileReader, itk::LightProcessObject);

  /** Get / Set the object pointer to geometric phantom. */
  itkGetObjectMacro(GeometricPhantom, GeometricPhantom);
  itkSetObjectMacro(GeometricPhantom, GeometricPhantom);

  /** Get/Set the filename to read. */
  itkGetStringMacro(Filename);
  itkSetStringMacro(Filename);

  /** do the actual parsing of the input file */
  virtual void GenerateOutputInformation();

protected:

  /// Constructor
  GeometricPhantomFileReader() {};

  /// Destructor
  virtual ~GeometricPhantomFileReader() ITK_OVERRIDE {}

private:
  GeometricPhantomFileReader( const Self& ); //purposely not implemented
  void operator=( const Self& );             //purposely not implemented

  GeometricPhantomPointer m_GeometricPhantom;
  std::string             m_Filename;
};

} // end namespace rtk

#endif
