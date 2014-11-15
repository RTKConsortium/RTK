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
#ifndef __srtkThreeDimCircularProjectionGeometryXMLFileWriter_h
#define __srtkThreeDimCircularProjectionGeometryXMLFileWriter_h

#include "srtkMacro.h"
#include "srtkThreeDimCircularProjectionGeometry.h"
#include "srtkMemberFunctionFactory.h"
#include "srtkIO.h"
#include "srtkProcessObject.h"

#include <memory>

namespace rtk {
  namespace simple {

    /** \class ThreeDimCircularProjectionGeometryXMLFileWriter
     * \brief Write out an RTK 3D circular geometry as XML
     */
    class SRTKIO_EXPORT ThreeDimCircularProjectionGeometryXMLFileWriter  :
      public ProcessObject
    {
    public:
      typedef ThreeDimCircularProjectionGeometryXMLFileWriter Self;

      // function pointer type
      typedef Self& (Self::*MemberFunctionType)( const ThreeDimCircularProjectionGeometry& );

      ThreeDimCircularProjectionGeometryXMLFileWriter( void );

      /** Print ourselves to string */
      virtual std::string ToString() const;

      /** return user readable name fo the filter */
      virtual std::string GetName() const { return std::string("ThreeDimCircularProjectionGeometryXMLFileWriter"); }

      Self& SetFileName ( std::string fileName );
      std::string GetFileName() const;

      Self& Execute ( const ThreeDimCircularProjectionGeometry& );
      Self& Execute ( const ThreeDimCircularProjectionGeometry& , const std::string &inFileName );

    private:

      std::string m_FileName;

    };

  SRTKIO_EXPORT void WriteXML ( const ThreeDimCircularProjectionGeometry& geometry, const std::string &fileName );
  }
}

#endif
