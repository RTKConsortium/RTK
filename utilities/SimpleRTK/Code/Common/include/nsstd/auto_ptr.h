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
#ifndef srtk_nsstd_auto_ptr_h
#define srtk_nsstd_auto_ptr_h

#include "srtkConfigure.h"

#include <memory>

namespace rtk
{
namespace simple
{
namespace nsstd
{
#if defined SRTK_HAS_CXX11_UNIQUE_PTR && defined SRTK_HAS_CXX11_ALIAS_TEMPLATE
template <typename T>
using auto_ptr = std::unique_ptr<T>;
#else
using std::auto_ptr;
#endif
}
}
}

#endif // srtk_nsstd_auto_ptr_h
