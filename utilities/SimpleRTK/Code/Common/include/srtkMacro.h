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
#ifndef srtkMacro_h
#define srtkMacro_h

#include <stdint.h>
#include <stddef.h>
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


#if __cplusplus >= 201103L
// In c++11 the override keyword allows you to explicity define that a function
// is intended to override the base-class version.  This makes the code more
// managable and fixes a set of common hard-to-find bugs.
#define SRTK_OVERRIDE override
// In C++11 the throw-list specification has been deprecated,
// replaced with the noexcept specifier. Using this function
// specification adds the run-time check that the method does not
// throw. If it does throw then std::terminate will be called.
// Use cautiously.
#define SRTK_NOEXCEPT noexcept
#else
#define SRTK_OVERRIDE
#define SRTK_NOEXCEPT throw()
#endif


#if  !defined(SRTK_RETURN_SELF_TYPE_HEADER)
#define SRTK_RETURN_SELF_TYPE_HEADER Self &
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

#if defined(SRTK_HAS_CXX11_NULLPTR) && !defined(SRTK_HAS_TR1_SUB_INCLUDE)
#define SRTK_NULLPTR nullptr
#else
#define SRTK_NULLPTR NULL
#endif


#define srtkMacroJoin( X, Y ) srtkDoMacroJoin( X, Y )
#define srtkDoMacroJoin( X, Y ) srtkDoMacroJoin2(X,Y)
#define srtkDoMacroJoin2( X, Y ) X##Y

#ifdef SRTK_HAS_CXX11_STATIC_ASSERT
// utilize the c++11 static_assert if available
#define srtkStaticAssert( expr, str) static_assert( expr, str )
#else

template<bool> struct StaticAssertFailure;
template<> struct StaticAssertFailure<true>{ enum { Value = 1 }; };

#define srtkStaticAssert( expr, str ) enum { srtkMacroJoin( static_assert_typedef, __LINE__) = sizeof( rtk::simple::StaticAssertFailure<((expr) == 0 ? false : true )> ) };


#endif
}
}

#define srtkPragma(x) _Pragma (#x)

#if defined(__clang__) && defined(__has_warning)
#define srtkClangDiagnosticPush()       srtkPragma( clang diagnostic push )
#define srtkClangDiagnosticPop()        srtkPragma( clang diagnostic pop )
#define srtkClangWarningIgnore_0(x)
#define srtkClangWarningIgnore_1(x)  srtkPragma( clang diagnostic ignored x)
#define srtkClangWarningIgnore(x)    srtkMacroJoin( srtkClangWarningIgnore_, __has_warning(x) )(x)
#else
#define srtkClangDiagnosticPush()
#define srtkClangDiagnosticPop()
#define srtkClangWarningIgnore(x)
#endif


#endif
