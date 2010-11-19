/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include "cuda_util.h"

// NOTE: This file provides utility functions that do not use CUDA specific
//       types as parameters; and can, therefore, be used from standard C or
//       C++ files.
//
//       Helper functions that require parameters that use CUDA specific
//       types (dim3, float4, etc) should be placed in 'cuda_kernel_util.cu'



void
CUDA_check_error (const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf (stderr, "CUDA ERROR: %s (%s).\n", 
	    msg, cudaGetErrorString(err));
        exit (EXIT_FAILURE);
    }                         
}

int
CUDA_detect_error ()
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        return 1;
    }                         
    return 0;
}

void
CUDA_listgpu ()
{
    int num_gpus, i;
    int cores_per_sm;
    cudaDeviceProp props;

    cudaGetDeviceCount(&num_gpus);

    for (i = 0; i < num_gpus; i++) {
        cudaGetDeviceProperties(&props, i);
        if (props.major == 1) {
            cores_per_sm = 8;
        } else if (props.major == 2) {
            cores_per_sm = 32;
        } else {
            printf ("GPU Compute Capability: Unknown to Platimatch!\n");
            return;
        }

        printf ("GPU ID %i:\n", i);
        printf ("              Name: %s (%.2f GB)\n", props.name, props.totalGlobalMem / (float)(1024 * 1024 * 1024));
        printf ("Compute Capability: %d.%d\n", props.major, props.minor);
        printf ("     Shared Memory: %.1f MB\n", props.sharedMemPerBlock / (float)1024);
        printf ("         Registers: %i\n", props.regsPerBlock);
        printf ("        Clock Rate: %.2f MHz\n", props.clockRate / (float)(1024));
        printf ("           # Cores: %d\n", props.multiProcessorCount * cores_per_sm);
        printf ("\n");
    }
}

// Selects the best GPU or the user specified
// GPU as defiend on command line
void
CUDA_selectgpu (int gpuid)
{
    int num_gpus;
    int cores_per_sm;
    cudaDeviceProp props;

    cudaGetDeviceCount(&num_gpus);

    if (gpuid < num_gpus) {
        cudaGetDeviceProperties(&props, gpuid);
        if (props.major == 1) {
            cores_per_sm = 8;
        } else if (props.major == 2) {
            cores_per_sm = 32;
        } else {
            printf ("Compute Capability: Unknown to Platimatch!\n");
            return;
        }

        printf ("Using %s (%.2f GB)\n", props.name, props.totalGlobalMem / (float)(1024 * 1024 * 1024));
        printf ("  - Compute Capability: %d.%d\n", props.major, props.minor);
        printf ("  - # Multi-Processors: %d\n", props.multiProcessorCount);
        printf ("  -    Number of Cores: %d\n", props.multiProcessorCount * cores_per_sm);
        cudaSetDevice (gpuid);
    } else {
        printf ("\nInvalid GPU ID specified.  Choices are:\n\n");
        CUDA_listgpu ();
        exit (0);
    }
}


// Returns the value held in __CUDA_ARCH__
// __CUDA_ARCH__ is only accessable as a #define
// *inside* of CUDA kernels.  This allows us to
// use the compute capability in CPU code.
int
CUDA_getarch (int gpuid)
{
    int num_gpus;
    cudaDeviceProp props;

    cudaGetDeviceCount(&num_gpus);

    if (gpuid < num_gpus) {
        cudaGetDeviceProperties(&props, gpuid);

        return 100*props.major + 10*props.minor;

    } else {
        /* Invalid GPU ID specified */
        return -1;
    }
}

