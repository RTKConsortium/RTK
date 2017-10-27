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
#ifndef rtkWin32Header_h
#define rtkWin32Header_h

#include "rtkConfiguration.h"

#if (defined(_WIN32) || defined(WIN32)) && defined(RTK_BUILD_SHARED_LIBS) 
# ifdef RTK_EXPORTS
#  define RTK_EXPORT __declspec(dllexport)
# else
#  define RTK_EXPORT __declspec(dllimport)
# endif  /* RTK_EXPORT */
#else
/* unix needs nothing */
#define RTK_EXPORT 
#endif


#if (defined(_WIN32) || defined(WIN32)) && defined(RTK_BUILD_SHARED_LIBS) 
# ifdef rtkcuda_EXPORTS
#  define rtkcuda_EXPORT __declspec(dllexport)
# else
#  define rtkcuda_EXPORT __declspec(dllimport)
# endif  /* RTK_EXPORT */
#else
/* unix needs nothing */
#define rtkcuda_EXPORT 
#endif

#if (defined(_WIN32) || defined(WIN32)) && defined(RTK_BUILD_SHARED_LIBS) 
# ifdef ITKCudaCommon_EXPORTS
#  define ITKCudaCommon_EXPORT __declspec(dllexport)
# else
#  define ITKCudaCommon_EXPORT __declspec(dllimport)
# endif  /* RTK_EXPORT */
#else
/* unix needs nothing */
#define ITKCudaCommon_EXPORT 
#endif

#endif
