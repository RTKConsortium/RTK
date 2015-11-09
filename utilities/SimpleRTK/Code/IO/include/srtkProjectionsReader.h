/*=========================================================================
 *
 *  Copyright Insight Software Consortium & RTK Consortium
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
#ifndef __srtkProjectionsReader_h
#define __srtkProjectionsReader_h

#include "srtkMacro.h"
#include "srtkImage.h"
#include "srtkImageReaderBase.h"
#include "srtkMemberFunctionFactory.h"

namespace rtk {
  namespace simple {

    /** \class ProjectionsReader
     * \brief Read and convert to attenuation serie of projections into a SimpleRTK image
     *
     * \sa rtk::simple::ReadProjections for the procedural interface
     **/
    class SRTKIO_EXPORT ProjectionsReader
      : public ImageReaderBase
    {
    public:
      typedef ProjectionsReader Self;

      ProjectionsReader();

      /** Print ourselves to string */
      virtual std::string ToString() const;

      /** return user readable name fo the filter */
      virtual std::string GetName() const { return std::string("ProjectionsReader"); }

      Self& SetFileNames ( const std::vector<std::string> &fileNames );
      const std::vector<std::string> &GetFileNames() const;

      Image Execute();

    protected:

      template <class TImageType> Image ExecuteInternal ( void );

    private:

      // function pointer type
      typedef Image (Self::*MemberFunctionType)( void );

      // friend to get access to executeInternal member
      friend struct detail::MemberFunctionAddressor<MemberFunctionType>;
      std::auto_ptr<detail::MemberFunctionFactory<MemberFunctionType> > m_MemberFactory;

      std::vector<std::string> m_FileNames;
    };
  SRTKIO_EXPORT Image ReadProjections ( const std::vector<std::string> &fileNames );
  }
}

#endif
