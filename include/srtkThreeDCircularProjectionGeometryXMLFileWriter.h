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
#ifndef __srtkThreeDCircularProjectionGeometryXMLFileWriter_h
#define __srtkThreeDCircularProjectionGeometryXMLFileWriter_h

#include "rtkMacro.h"
#include "rtkThreeDCircularProjectionGeometryXMLFileWriter.h"
#include "rtkWin32Header.h"

#include <itkProcessObject.h>

#include <memory>

namespace rtk {
  namespace simple {

    /** \class ThreeDCircularProjectionGeometryXMLFileWriter
     * \brief Writes an RTK 3D circular geometry to an XML file
	 * This class shadows rtk::ThreeDCircularProjectionGeometryXMLFileWriter
	 * defined in rtkThreeDCircularProjectionGeometryXMLFile.h to expose the
	 * writer when wrapping RTK for python.
	 * \sa rtkThreeDCircularProjectionGeometryXMLFile.h
     */
    class RTK_EXPORT ThreeDCircularProjectionGeometryXMLFileWriter :
      public itk::ProcessObject
    {
    public:
      typedef ThreeDCircularProjectionGeometryXMLFileWriter Self;
      typedef itk::ProcessObject                            Superclass;
      typedef itk::SmartPointer< Self >                     Pointer;
      typedef itk::SmartPointer< const Self >               ConstPointer;

      typedef rtk::ThreeDCircularProjectionGeometry         GeometryType;
      typedef GeometryType::Pointer                         GeometryPointer;

      /** Method for creation through the object factory. */
      itkNewMacro(Self);

      /** Run-time type information ( and related methods ). */
      itkTypeMacro(ThreeDCircularProjectionGeometryXMLFileWriter, itk::ProcessObject);

      /** Set the filename to write */
      void SetFilename(const std::string & _arg);

      /** determine whether a file is writable */
      int CanWriteFile(const char *name);

      /** do the actual writing of the file */
      void Update();

      /** Set the input geometry */
      void SetInput(GeometryType* geometry);

    protected:
      ThreeDCircularProjectionGeometryXMLFileWriter();
      ~ThreeDCircularProjectionGeometryXMLFileWriter() {}
      //void PrintSelf(std::ostream & os, itk::Indent indent) const;

    private:
      ThreeDCircularProjectionGeometryXMLFileWriter(const Self &);
      void operator=(const Self &);

      rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer m_Writer;
    };

  }
}

#endif
