/* config file for windows environments */

/* Name of package */
#define PACKAGE "xbase"

/* Version number of package */
#define VERSION "2.1.1"

/* Define if you have the ANSI C header files.  */
#define STDC_HEADERS 1

/* Define if you have io.h */
#define HAVE_IO_H 1

/* Define if the C++ compiler supports BOOL */
#define HAVE_BOOL 1

/* Define if you need to have .ndx indexes */
#define XB_INDEX_NDX 1

/* Define if you need to have .ntx indexes */
#define XB_INDEX_NTX 1

/* Define if you need to support memo fields */
#define XB_MEMO_FIELDS 1

/* Define if you need expressions */
#define XB_EXPRESSIONS 1

/* Define if you need html support */
#define XB_HTML 1

/* Define if you need locking support */
#undef XB_LOCKING_ON 

/* Define if you need to turn on XBase specific debug */
#define XBASE_DEBUG 1

/* Define if your compiler support exceptions */
/* #undef HAVE_EXCEPTIONS */

/* Define if you want Castellano (Spanish) Dates */
/* #undef XB_CASTELLANO */

/* Define if using real deletes */
#define XB_REAL_DELETE 1

/* Define if need filters */
#define XB_FILTERS 1

/* Define if you have the fcntl function.  */
#define HAVE_FCNTL 1

/* Define if you have the flock function.  */
#define HAVE_FLOCK 1

/* Define if you have the vsnprintf function.  */
//#define HAVE_VSNPRINTF 1

/* Define if you have the vsprintf function.  */
#define HAVE_VSPRINTF 1

/* Define if you have the <ctype.h> header file.  */
#define HAVE_CTYPE_H 1

/* Define if you have the <exception> header file.  */
#define HAVE_EXCEPTION 1

/* Define if you have the <fcntl.h> header file.  */
#define HAVE_FCNTL_H 1

/* Define if you have the <string.h> header file.  */
#define HAVE_STRING_H 1

/* Define if you have the <sys/types.h> header file.  */
#define HAVE_SYS_TYPES_H 1

/* Define if you have the <tvision/tv.h> header file.  */
/* #undef HAVE_TVISION_TV_H */

/* Should we include generic index support? */
#if defined(XB_INDEX_NDX) || defined(XB_INDEX_NTX)
#define  XB_INDEX_ANY 1
#endif

/* expressions required for indexes */
#if defined(XB_INDEX_ANY) && !defined(XB_EXPRESSIONS)
#define XB_EXPRESSIONS 1
#endif

/* default memo block size */
#define XB_DBT_BLOCK_SIZE  512

/* filename path separator */
#define PATH_SEPARATOR '/'

#ifndef HAVE_BOOL
#define HAVE_BOOL 1
typedef int bool;
const bool false = 0;
const bool true = 1;
#endif
