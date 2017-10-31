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
#include "rtkThreeDCircularProjectionGeometryXMLFile.h"
#include "itkProcessObject.h"

#include <memory>

namespace rtk {
  namespace simple {

    /** \class ThreeDCircularProjectionGeometryXMLFileWriter
     * \brief Reads in an RTK 3D circular geometry as XML
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
      ThreeDCircularProjectionGeometryXMLFileWriter(void);
      ~ThreeDCircularProjectionGeometryXMLFileWriter() {}
      //void PrintSelf(std::ostream & os, itk::Indent indent) const;

    private:
      ThreeDCircularProjectionGeometryXMLFileWriter(const Self &);
      void operator=(const Self &);

      rtk::ThreeDCircularProjectionGeometryXMLFileWriter::Pointer m_Writer;
    };

  }
}

#ifndef ITK_MANUAL_INSTANTIATION
#include "srtkThreeDCircularProjectionGeometryXMLFileWriter.hxx"
#endif

#endif
