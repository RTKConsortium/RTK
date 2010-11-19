/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plm_config.h"

#include <stdio.h>
#include <cuda.h>
#include "cuda_mem.h"
#include "cuda_util.h"

void
CUDA_alloc_copy (
    void** gpu_addr,
    void** cpu_addr,
    size_t mem_size,
    cuda_alloc_copy_mode mode
)
{
    // If zero copying, this will hold the CPU memory address of
    // the new pinned memory address in the CPU memory map.
    // After CPU memory contents is relocated to this new pinned
    // memory, this pointer will overwrite the original CPU
    // pointer (*cpu_addr).
    void* pinned_host_mem;

    if (mode == cudaZeroCopy) {
        // Allocate some pinned CPU memory for zero paging
        cudaHostAlloc ((void **)&pinned_host_mem, mem_size, cudaHostAllocMapped);
        CUDA_check_error ("Failed to allocate pinned memory.");

        // Relocate data to pinned memory
        memcpy (pinned_host_mem, *cpu_addr, mem_size);
        free (*cpu_addr);
        *cpu_addr = pinned_host_mem;

        // Get the address of the pinned page in the GPU memory map.
        cudaHostGetDevicePointer ((void **)gpu_addr, (void *)pinned_host_mem, 0);
        CUDA_check_error ("Failed to map CPU memory to GPU.");
    } else {
        // Allcoated some global memory on the GPU
        cudaMalloc ((void**)gpu_addr, mem_size);
        CUDA_check_error ("Out of GPU memory.");

        // Populate the allocated global GPU memory
        cudaMemcpy (*gpu_addr, *cpu_addr, mem_size, cudaMemcpyHostToDevice);
        CUDA_check_error ("Failed to copy data to GPU");
    }
}

// If you plan on using CUDA_alloc_vmem() to extend
// the GPU memory, then you must first call this.
void
CUDA_init_vmem (Vmem_Entry** head)
{
    *head = NULL;
}

// This function should only be used to supplement the GPU's
// available "Global Memory" with pinned CPU memory.  Currently,
// the GPU address bus is 32-bit, so using this function we are
// only able to supplement the GPU global memory *up to* 4GB.
// Cards already equiped with 4GB of global memory have a full
// memory map and can therefore be extended no further!
void
CUDA_alloc_vmem (
    void** gpu_addr,
    size_t mem_size,
    Vmem_Entry** head
)
{
    void* pinned_host_mem;
    Vmem_Entry* new_entry;

    // Allocate some pinned CPU memory for zero paging
    cudaHostAlloc ((void **)&pinned_host_mem, mem_size, cudaHostAllocMapped);
    CUDA_check_error ("Failed to allocate pinned memory.");

    // Clear out new pinned CPU memory
    memset (pinned_host_mem, 0, mem_size);

    // Get the address of the pinned page in the GPU memory map.
    cudaHostGetDevicePointer ((void **)gpu_addr, (void *)pinned_host_mem, 0);
    CUDA_check_error ("Failed to map CPU memory to GPU.");

    // Now we will register this allocation with my gpu "virtual memory"
    // system.  CUDA requires that we free pinned CPU memory with the CPU
    // pointer; NOT the GPU pointer.  This can be troublesome if you are only
    // tracking GPU pointers and have no need to access the CPU side memory
    // with the CPU.  So, every time we pin CPU memory, we register the pair of
    // pointers (CPU & GPU) in a linked list.  This allows us to only track
    // track one and look up the other.  It also allows us to free all pinned
    // memory without knowing the pointers by simply cycling through the linked
    // list and freeing everything.

    // create a new vmem entry
    new_entry = (Vmem_Entry*) malloc (sizeof(Vmem_Entry));

    // initialize the new entry
    new_entry->gpu_pointer = *gpu_addr;
    new_entry->cpu_pointer = pinned_host_mem;
    new_entry->size = mem_size;

    // insert new entry @ the head
    new_entry->next = *head;
    *head = new_entry;
}

// Returns the total amount of "virtual global"
// (i.e. pinned CPU) memory. Perhaps useful.
size_t
CUDA_tally_vmem (Vmem_Entry** head)
{
    size_t total_vmem = 0;
    Vmem_Entry* curr = *head;

    while (curr != NULL)
    {
        total_vmem += curr->size;
        curr = curr->next;
    }

    return total_vmem;
}

// For debugging.  Just prints out the virtual
// memory pointer association list.
void
CUDA_print_vmem (Vmem_Entry** head)
{
    int i = 0;
    Vmem_Entry* curr = *head;

    while (curr != NULL)
    {
        printf ("Entry #%i:\n", i);
        printf ("  gpu_pointer: %p\n", curr->gpu_pointer);
        printf ("  cpu_pointer: %p\n\n", curr->cpu_pointer);

        curr = curr->next;
        i++;
    }
}

// Free GPU "virtual memory" via GPU mapped address.
int
CUDA_free_vmem (
    void* gpu_pointer,
    Vmem_Entry** head
)
{
    Vmem_Entry* curr = *head;
    Vmem_Entry* prev = NULL;

    while (curr != NULL)
    {
        if (curr->gpu_pointer == gpu_pointer) {
            cudaFreeHost (curr->cpu_pointer);
            CUDA_check_error ("Failed to free virtual GPU memory.");

            if (prev == NULL) {
                // we are removing the head
                *head = curr->next;
                free (curr);
                return 0;
            } else {
                // removing past the head
                prev->next = curr->next;
                free (curr);
                return 0;
            }
        }
        prev = curr;
        curr = curr->next;
    }

    // Failed to free virtual GPU memory.
    return 1;
}

// Frees *ALL* GPU "virtual memory"
// Returns number of freed entries
int
CUDA_freeall_vmem (
    Vmem_Entry** head
)
{
    int i = 0;
    Vmem_Entry* curr = *head;

    while (curr != NULL)
    {
        cudaFreeHost (curr->cpu_pointer);
        CUDA_check_error ("Failed to free virtual GPU memory.");

        *head = curr->next;
        free (curr);

        curr = *head;
        i++;
    }

    return i;
}

int
CUDA_alloc_zero (
    void** gpu_addr,
    size_t mem_size,
    cuda_alloc_fail_mode fail_mode
)
{
    // Allcoated some global memory on the GPU
    cudaMalloc ((void**)gpu_addr, mem_size);
    if (fail_mode == cudaAllocStern) {
        CUDA_check_error ("Out of GPU memory.");
    } else {
        if (CUDA_detect_error()) {
            return 1;
        }
    }

    // Zero out the allocated global GPU memory
    cudaMemset (*gpu_addr, 0, mem_size);
    if (fail_mode == cudaAllocStern) {
        CUDA_check_error ("Failed to zero out GPU memory.");
    } else {
        if (CUDA_detect_error()) {
            return 1;
        }
    }

    // Success
    return 0;
}

int
CUDA_zero_copy_check (int gpuid)
{
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, gpuid);
    if (props.canMapHostMemory) {
        // GPU supports zero copy
        return 1;
    } else {
        // GPU doest not support zero copy
        return 0;
    }
}

