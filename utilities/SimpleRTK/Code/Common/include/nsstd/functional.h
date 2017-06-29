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
#ifndef srtk_nsstd_functional_h
#define srtk_nsstd_functional_h

#include "srtkConfigure.h"

#if !defined SRTK_HAS_TR1_FUNCTIONAL && !defined SRTK_HAS_CXX11_FUNCTIONAL
#error "No system (tr1/c++11) functional header available!"
#endif

#if defined SRTK_HAS_CXX11_FUNCTIONAL || !defined SRTK_HAS_TR1_SUB_INCLUDE
#include <functional>
#else
#include <tr1/functional>
#endif

namespace rtk
{
namespace simple
{
namespace nsstd
{
#if defined SRTK_HAS_TR1_FUNCTIONAL && !defined SRTK_HAS_CXX11_FUNCTIONAL
using std::tr1::function;
using std::tr1::bind;
namespace placeholders =  std::tr1::placeholders;
#else
using std::function;
using std::bind;
namespace placeholders =  std::placeholders;
#endif
}
}
}

#endif // srtk_nsstd_functional_h
