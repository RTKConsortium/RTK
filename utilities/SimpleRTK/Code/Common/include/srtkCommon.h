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
#ifndef __srtkCommon_h
#define __srtkCommon_h

#include "srtkMacro.h"

#ifndef srtkMacro_h
#error "srtkMacro.h must be included before srtkCommon.h"
#endif

#if defined( SRTKDLL )
  #ifdef SimpleRTKCommon_EXPORTS
    #define SRTKCommon_EXPORT SRTK_ABI_EXPORT
  #else
    #define SRTKCommon_EXPORT SRTK_ABI_IMPORT
  #endif  /* SimpleRTKCommon_EXPORTS */
#else
  // Don't hide symbols in the static SimpleRTKCommon library in case
  // -fvisibility=hidden is used
  #define SRTKCommon_EXPORT
#endif

#define SRTKCommon_HIDDEN SRTK_ABI_HIDDEN

#endif // __srtkCommon_h
