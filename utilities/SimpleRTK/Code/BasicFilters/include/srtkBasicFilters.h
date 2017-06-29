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
#ifndef __srtkBasicFilters_h
#define __srtkBasicFilters_h

#include "srtkMacro.h"
#include "srtkInterpolator.h"

// todo this should be moved to a more local place
#include "nsstd/auto_ptr.h"
#include "srtkTransform.h"
#include "srtkThreeDCircularProjectionGeometry.h"

#if defined( SRTKDLL )
  #ifdef SimpleRTKBasicFilters0_EXPORTS
    #define SRTKBasicFilters0_EXPORT SRTK_ABI_EXPORT
  #else
    #define SRTKBasicFilters0_EXPORT SRTK_ABI_IMPORT
  #endif  /* SimpleRTKBasicFilters0_EXPORTS */
#else
  // Don't hide symbols in the static SimpleRTKBasicFilters library in case
  // -fvisibility=hidden is used
  #define SRTKBasicFilters0_EXPORT
#endif

#if defined( SRTKDLL )
#ifdef SimpleRTKBasicFilters1_EXPORTS
#define SRTKBasicFilters1_EXPORT SRTK_ABI_EXPORT
#else
#define SRTKBasicFilters1_EXPORT SRTK_ABI_IMPORT
#endif  /* SimpleRTKBasicFilters1_EXPORTS */
#else
// Don't hide symbols in the static SimpleRTKBasicFilters library in case
// -fvisibility=hidden is used
#define SRTKBasicFilters1_EXPORT
#endif

#if defined( SRTKDLL )
  #ifdef SimpleRTKBasicFilters_EXPORTS
    #define SRTKBasicFilters_EXPORT SRTK_ABI_EXPORT
  #else
    #define SRTKBasicFilters_EXPORT SRTK_ABI_IMPORT
  #endif  /* SimpleRTKBasicFilters_EXPORTS */
#else
  // Don't hide symbols in the static SimpleRTKBasicFilters library in case
  // -fvisibility=hidden is used
  #define SRTKBasicFilters_EXPORT
#endif

#define SRTKBasicFilters0_HIDDEN SRTK_ABI_HIDDEN
#define SRTKBasicFilters_HIDDEN SRTK_ABI_HIDDEN

#endif // __srtkBasicFilters_h
