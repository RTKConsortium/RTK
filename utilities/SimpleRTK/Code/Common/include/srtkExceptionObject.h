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
#ifndef __srtkExceptionObject_h
#define __srtkExceptionObject_h

#include "srtkMacro.h"
#include "srtkCommon.h"

#ifndef srtkMacro_h
#error "srtkMacro.h must be included before srtkExceptionObject.h"
#endif //__srtkMacro_h
#ifndef __srtkCommon_h
#error "srtkCommon.h must be included before srtkExceptionObject.h"
#endif //__srtkCommon_h


namespace itk
{
// forward declaration for encapsulation
class ExceptionObject;
}

namespace rtk
{
namespace simple
{

/** \class GenericException
 * \brief The base SimpleRTK exception class
 */
class SRTKCommon_EXPORT GenericException :
    public std::exception
{
public:
  /** Default constructor.  Needed to ensure the exception object can be
   * copied. */
  GenericException()  throw();
  GenericException( const GenericException &e )  throw();

  /** Constructor. Needed to ensure the exception object can be copied. */
  GenericException(const char *file, unsigned int lineNumber) throw();

  /** Constructor. Needed to ensure the exception object can be copied. */
  GenericException(const std::string & file, unsigned int lineNumber) throw();

  /** Constructor. Needed to ensure the exception object can be copied. */
  GenericException(const std::string & file,
                   unsigned int lineNumber,
                   const std::string & desc) throw();

  /** Virtual destructor needed for subclasses. Has to have empty throw(). */
  virtual ~GenericException() throw( );

  /** Assignment operator. */
  GenericException & operator=(const GenericException & orig);

  /** Equivalence operator. */
  virtual bool operator==(const GenericException & orig) const;


  /** Return a description of the error */
  std::string ToString() const;

  const char * what() const throw();

  virtual const char * GetNameOfClass() const;

  virtual const char * GetLocation()    const;

  virtual const char * GetDescription() const;

  /** What file did the exception occur in? */
  virtual const char * GetFile()    const;

  /** What line did the exception occur in? */
  virtual unsigned int GetLine() const;

private:
  const itk::ExceptionObject *m_PimpleException;
};

}
}

#endif // __srtkExceptionObject_h
