/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_util_h_
#define _cuda_util_h_

#include "rtkConfiguration.h"
#include <cuda.h>

// NOTE: This file provides utility functions that do not use CUDA specific
//       types as parameters; and can, therefore, be used from standard C or
//       C++ files.
//
//       Helper functions that require parameters that use CUDA specific
//       types (dim3, float4, etc) should be placed in 'cuda_kernel_util.h'


#define GRID_LIMIT_X 65535
#define GRID_LIMIT_Y 65535


#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT void
CUDA_check_error (const char *msg);

gpuit_EXPORT int
CUDA_detect_error ();

gpuit_EXPORT void
CUDA_listgpu ();

gpuit_EXPORT void
CUDA_selectgpu (int gpuid);

gpuit_EXPORT int
CUDA_getarch (int gpuid);


#if defined __cplusplus
}
#endif

#endif
