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
#ifndef __srtkThreeDCircularProjectionGeometryXMLFileReader_h
#define __srtkThreeDCircularProjectionGeometryXMLFileReader_h

#include "rtkMacro.h"
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProcessObject.h"

#include <memory>

namespace rtk {
  namespace simple {

    /** \class ThreeDCircularProjectionGeometryXMLFileReader
     * \brief Reads in an RTK 3D circular geometry as XML
     */
    class RTK_EXPORT ThreeDCircularProjectionGeometryXMLFileReader :
      public itk::ProcessObject
    {
    public:
      typedef ThreeDCircularProjectionGeometryXMLFileReader Self;
      typedef itk::ProcessObject                            Superclass;
      typedef itk::SmartPointer< Self >                     Pointer;
      typedef itk::SmartPointer< const Self >               ConstPointer;

      typedef rtk::ThreeDCircularProjectionGeometry         GeometryType;
      typedef GeometryType::Pointer                         GeometryPointer;

      /** Method for creation through the object factory. */
      itkNewMacro(Self);

      /** Run-time type information ( and related methods ). */
      itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileReader, itk::ProcessObject);

      /** Set the filename to read */
      void SetFilename(const std::string & _arg);
      /** Get the filename to read */
      const char* GetFilename() const;

      /** determine whether a file can be opened and read */
      int CanReadFile(const char *name);

      /** do the actual parsing of the input file */
      void Update();

      /** return the output geometry */
      GeometryPointer GetOutput();

    protected:
      ThreeDCircularProjectionGeometryXMLFileReader(void);
      ~ThreeDCircularProjectionGeometryXMLFileReader() {}
      //void PrintSelf(std::ostream & os, itk::Indent indent) const;

    private:
      ThreeDCircularProjectionGeometryXMLFileReader(const Self &);
      void operator=(const Self &);

      rtk::ThreeDCircularProjectionGeometryXMLFileReader::Pointer m_Reader;
    };

  }
}

#ifndef ITK_MANUAL_INSTANTIATION
  #include "srtkThreeDCircularProjectionGeometryXMLFileReader.hxx"
#endif

#endif
