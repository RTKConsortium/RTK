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
#ifndef __srtkIO_h
#define __srtkIO_h

#include "srtkMacro.h"

#if defined( SRTKDLL )
  #ifdef SimpleRTKIO_EXPORTS
    #define SRTKIO_EXPORT SRTK_ABI_EXPORT
  #else
    #define SRTKIO_EXPORT SRTK_ABI_IMPORT
  #endif  /* SimpleRTKIO_EXPORTS */
#else
  // Don't hide symbols in the static SimpleRTKIO library in case
  // -fvisibility=hidden is used
  #define SRTKIO_EXPORT

#endif

#define SRTKIO_HIDDEN SRTK_ABI_HIDDEN

#endif // __srtkIO_h
