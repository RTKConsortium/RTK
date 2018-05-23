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
#ifndef rtkGgoArgsInfoManager_h
#define rtkGgoArgsInfoManager_h

#include "rtkConfiguration.h"
#ifdef RTK_TIME_EACH_FILTER
# include "rtkGlobalTimer.h"
#endif

namespace rtk
{
template < class TArgsInfo, class TCleanupFunction = void (*)( TArgsInfo* ) >
class args_info_manager
{
  public:
    args_info_manager( TArgsInfo & args_info, TCleanupFunction cf)
      {
      this->args_info_pointer = &args_info;
      this->cleanup_function = cf;
#ifdef RTK_TIME_EACH_FILTER
      rtk::GlobalTimer::GetInstance()->SetVerbose(args_info.verbose_flag);
#endif
      }
    ~args_info_manager()
      {
      this->cleanup_function( this->args_info_pointer );
      }
  private:
    TArgsInfo * args_info_pointer;
    TCleanupFunction cleanup_function;
};
}
#endif
