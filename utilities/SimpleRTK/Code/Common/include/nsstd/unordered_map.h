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
#ifndef srtk_nsstd_unordered_map_h
#define srtk_nsstd_unordered_map_h

#include "srtkConfigure.h"

#if !defined SRTK_HAS_TR1_UNORDERED_MAP && !defined SRTK_HAS_CXX11_UNORDERED_MAP
#error "No system (tr1/c++11) unordered_map header available!"
#endif


#if defined SRTK_HAS_CXX11_UNORDERED_MAP && !defined SRTK_HAS_TR1_SUB_INCLUDE
#include <unordered_map>
#elif
#include <tr1/unordered_map>
#endif

namespace rtk
{
namespace simple
{
namespace nsstd
{
#if defined SRTK_HAS_TR1_UNORDERED_MAP && !defined SRTK_HAS_CXX11_UNORDERED_MAP
using std::tr1::unordered_map;
#else
using std::unordered_map;
#endif
}
}
}


#endif // srtk_nsstd_unordered_map_h
