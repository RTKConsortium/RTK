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
#ifndef __srtkMacro_h
#define __srtkMacro_h

#include <stdint.h>
#include <cassert>
#include <sstream>
#include <limits>

#include "srtkConfigure.h"


// Setup symbol exports
//
#if defined _WIN32 || defined __CYGWIN__
   #ifdef __GNUC__
    #define SRTK_ABI_EXPORT __attribute__ ((dllexport))
    #define SRTK_ABI_IMPORT __attribute__ ((dllimport))
    #define SRTK_ABI_HIDDEN
  #else
    #define SRTK_ABI_EXPORT __declspec(dllexport) // Note: actually gcc seems to also supports this syntax.
    #define SRTK_ABI_IMPORT __declspec(dllimport) // Note: actually gcc seems to also supports this syntax.
    #define SRTK_ABI_HIDDEN
  #endif
#else
  #if __GNUC__ >= 4
    #define SRTK_ABI_EXPORT __attribute__ ((visibility ("default")))
    #define SRTK_ABI_IMPORT __attribute__ ((visibility ("default")))
    #define SRTK_ABI_HIDDEN  __attribute__ ((visibility ("hidden")))
  #else
    #define SRTK_ABI_EXPORT
    #define SRTK_ABI_IMPORT
    #define SRTK_ABI_HIDDEN
  #endif
#endif

namespace rtk {

namespace simple {

class GenericException;

#define srtkExceptionMacro(x)                                           \
  {                                                                     \
    std::ostringstream message;                                         \
    message << "srtk::ERROR: " x;                                       \
    throw ::rtk::simple::GenericException(__FILE__, __LINE__, message.str().c_str()); \
  }


#ifdef SRTK_HAS_CXX11_STATIC_ASSERT
// utilize the c++11 static_assert if available
#define srtkStaticAssert( expr, str) static_assert( expr, str )
#else

template<bool> struct StaticAssertFailure;
template<> struct StaticAssertFailure<true>{ enum { Value = 1 }; };

#define BOOST_JOIN( X, Y ) BOOST_DO_JOIN( X, Y )
#define BOOST_DO_JOIN( X, Y ) BOOST_DO_JOIN2(X,Y)
#define BOOST_DO_JOIN2( X, Y ) X##Y

#define srtkStaticAssert( expr, str ) enum { BOOST_JOIN( static_assert_typedef, __LINE__) = sizeof( rtk::simple::StaticAssertFailure<((expr) == 0 ? false : true )> ) };


#endif


}
}

#endif
