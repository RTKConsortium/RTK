/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _cuda_mem_h_
#define _cuda_mem_h_

typedef struct vmem_entry Vmem_Entry;
struct vmem_entry
{
    void* gpu_pointer;
    void* cpu_pointer;

    size_t size;

    Vmem_Entry* next;
};

enum cuda_alloc_copy_mode {
    cudaGlobalMem,
    cudaZeroCopy
};

enum cuda_alloc_fail_mode {
    cudaAllocStern,
    cudaAllocCasual
};



#if defined __cplusplus
extern "C" {
#endif

gpuit_EXPORT void
CUDA_alloc_copy (
    void** gpu_addr,
    void** cpu_addr,
    size_t mem_size,
    enum cuda_alloc_copy_mode mode
);

gpuit_EXPORT void
CUDA_init_vmem (Vmem_Entry** head);

gpuit_EXPORT void
CUDA_alloc_vmem (
    void** gpu_addr,
    size_t mem_size,
    Vmem_Entry** head
);

gpuit_EXPORT size_t
CUDA_tally_vmem (Vmem_Entry** head);

gpuit_EXPORT void
CUDA_print_vmem (Vmem_Entry** head);

gpuit_EXPORT int
CUDA_free_vmem (
    void* gpu_pointer,
    Vmem_Entry** head
);

gpuit_EXPORT int
CUDA_freeall_vmem (Vmem_Entry** head);

gpuit_EXPORT int
CUDA_alloc_zero (
    void** gpu_addr,
    size_t mem_size,
    enum cuda_alloc_fail_mode fail_mode
);

gpuit_EXPORT int
CUDA_zero_copy_check (int gpuid);


#if defined __cplusplus
}
#endif
#endif
