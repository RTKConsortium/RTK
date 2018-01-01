#include "lp_types.h"

#if defined INLINE
# define MYINLINE INLINE
#else
# define MYINLINE static
#endif

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#endif
MYINLINE void set_biton(MYBOOL *bitarray, int item)
{
  bitarray[item / 8] |= (1 << (item % 8));
}

/*
MYINLINE void set_bitoff(MYBOOL *bitarray, int item)
{
  bitarray[item / 8] &= ~(1 << (item % 8));
}
*/

MYINLINE MYBOOL is_biton(MYBOOL *bitarray, int item)
{
  return( (MYBOOL) ((bitarray[item / 8] & (1 << (item % 8))) != 0) );
}
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
