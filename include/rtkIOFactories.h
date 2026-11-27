/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkIOFactories_h
#define rtkIOFactories_h

#include "RTKExport.h"

namespace rtk
{

/** Register all rtk IO factories. Only the first call registers them. The
 * registration modifies the list of ITK factories, which is not thread safe.
 * Applications which create ITK objects in several threads must call it
 * before starting them. */
void RTK_EXPORT
RegisterIOFactories();

} // namespace rtk

#endif
