
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton implementation for Bison's Yacc-like parsers in C
   
      Copyright (C) 1984, 1989, 1990, 2000, 2001, 2002, 2003, 2004, 2005, 2006
   Free Software Foundation, Inc.
   
   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.
   
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
   
   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.
   
   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* C LALR(1) parser skeleton written by Richard Stallman, by
   simplifying the original so-called "semantic" parser.  */

/* All symbols defined below should begin with yy or YY, to avoid
   infringing on user name space.  This should be done even for local
   variables, as they might otherwise be expanded by user macros.
   There are some unavoidable exceptions within include files to
   define necessary library symbols; they are noted "INFRINGES ON
   USER NAME SPACE" below.  */

/* Identify Bison output.  */
#define YYBISON 1

/* Bison version.  */
#define YYBISON_VERSION "2.4.1"

/* Skeleton name.  */
#define YYSKELETON_NAME "yacc.c"

/* Pure parsers.  */
#define YYPURE 0

/* Push parsers.  */
#define YYPUSH 0

/* Pull parsers.  */
#define YYPULL 1

/* Using locations.  */
#define YYLSP_NEEDED 1



/* Copy the first part of user declarations.  */

/* Line 189 of yacc.c  */
#line 22 "../../src/parser.yy"


#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <string>

#include "my_sstream.h"

#include "acceptedvalues.h"

#include "argsdef.h"

#include "gengetopt.h"
#include "errorcodes.h"
#include "ggos.h"
#include "yyerror.h"

extern int gengetopt_count_line;
extern char * gengetopt_input_filename;

static int gengetopt_package_given = 0;
static int gengetopt_version_given = 0;
static int gengetopt_purpose_given = 0;
static int gengetopt_usage_given = 0;
static int gengetopt_description_given = 0;

/// the last option parsed
static gengetopt_option *current_option = 0;

extern int yylex (void) ;

//#define YYERROR_VERBOSE 1

void check_result(int o, gengetopt_option *opt)
{
  if (o)
    {
        ostringstream err;

    switch (o)
    {
    case NOT_ENOUGH_MEMORY:
        yyerror (opt, "not enough memory");
    	break;
    case REQ_LONG_OPTION:
        err << "long option redefined \'" << opt->long_opt << "\'";
        yyerror (opt, err.str().c_str());
		break;
    case REQ_SHORT_OPTION:
        err << "short option redefined \'" << opt->short_opt << "\'";
        yyerror (opt, err.str().c_str());
        break;
    case FOUND_BUG:
        yyerror (opt, "bug found!!");
        break;
    case GROUP_UNDEFINED:
        yyerror (opt, "group undefined");
        break;
    case MODE_UNDEFINED:
        yyerror (opt, "mode undefined");
        break;
    case INVALID_DEFAULT_VALUE:
        yyerror (opt, "invalid default value");
        break;
    case NOT_REQUESTED_TYPE:
        yyerror (opt, "type specification not requested");
        break;
    case NOT_VALID_SPECIFICATION:
      yyerror (opt, "invalid specification for this kind of option");
      break;
    case SPECIFY_FLAG_STAT:
      yyerror (opt, "you must specify the default flag status");
      break;
    case NOT_GROUP_OPTION:
      yyerror (opt, "group specification for a non group option");
      break;
    case NOT_MODE_OPTION:
      yyerror (opt, "mode specification for an option not belonging to a mode");
      break;
    case SPECIFY_GROUP:
      yyerror (opt, "missing group specification");
      break;
    case SPECIFY_MODE:
      yyerror (opt, "missing mode specification");
      break;
    case INVALID_NUMERIC_VALUE:
        yyerror (opt, "invalid numeric value");
        break;
    case INVALID_ENUM_TYPE_USE:
    	yyerror (opt, "enum type can only be specified for options with values");
        break;
    case HELP_REDEFINED:
    	yyerror (opt, "if you want to redefine --help, please use option --no-help");
        break;
    case VERSION_REDEFINED:
    	yyerror (opt, "if you want to redefine --version, please use option --no-version");
        break;
    }
  }
}

/* the number of allowed occurrences of a multiple option */
struct multiple_size
{
    /* these strings are allocated dynamically and NOT
      automatically freed upon destruction */
    char *min;
    char *max;

    /* if no limit is specified then initialized to 0.
       if the same size is specified for min and max, it means that an exact
       number of occurrences is required*/
    multiple_size(const char *m = "0", const char *M = "0") :
        min(strdup(m)), max(strdup(M))
    {}
};

#define check_error if (o) YYERROR;



/* Line 189 of yacc.c  */
#line 200 "../../src/parser.cc"

/* Enabling traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif

/* Enabling verbose error messages.  */
#ifdef YYERROR_VERBOSE
# undef YYERROR_VERBOSE
# define YYERROR_VERBOSE 1
#else
# define YYERROR_VERBOSE 0
#endif

/* Enabling the token table.  */
#ifndef YYTOKEN_TABLE
# define YYTOKEN_TABLE 0
#endif


/* Tokens.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
   /* Put the tokens into the symbol table, so that GDB and other debuggers
      know about them.  */
   enum yytokentype {
     TOK_PACKAGE = 258,
     TOK_VERSION = 259,
     TOK_OPTION = 260,
     TOK_DEFGROUP = 261,
     TOK_GROUPOPTION = 262,
     TOK_DEFMODE = 263,
     TOK_MODEOPTION = 264,
     TOK_YES = 265,
     TOK_NO = 266,
     TOK_ON = 267,
     TOK_OFF = 268,
     TOK_FLAG = 269,
     TOK_PURPOSE = 270,
     TOK_DESCRIPTION = 271,
     TOK_USAGE = 272,
     TOK_DEFAULT = 273,
     TOK_GROUP = 274,
     TOK_GROUPDESC = 275,
     TOK_MODE = 276,
     TOK_MODEDESC = 277,
     TOK_MULTIPLE = 278,
     TOK_ARGOPTIONAL = 279,
     TOK_TYPESTR = 280,
     TOK_SECTION = 281,
     TOK_DETAILS = 282,
     TOK_SECTIONDESC = 283,
     TOK_TEXT = 284,
     TOK_ARGS = 285,
     TOK_VALUES = 286,
     TOK_HIDDEN = 287,
     TOK_DEPENDON = 288,
     TOK_STRING = 289,
     TOK_CHAR = 290,
     TOK_ARGTYPE = 291,
     TOK_SIZE = 292
   };
#endif
/* Tokens.  */
#define TOK_PACKAGE 258
#define TOK_VERSION 259
#define TOK_OPTION 260
#define TOK_DEFGROUP 261
#define TOK_GROUPOPTION 262
#define TOK_DEFMODE 263
#define TOK_MODEOPTION 264
#define TOK_YES 265
#define TOK_NO 266
#define TOK_ON 267
#define TOK_OFF 268
#define TOK_FLAG 269
#define TOK_PURPOSE 270
#define TOK_DESCRIPTION 271
#define TOK_USAGE 272
#define TOK_DEFAULT 273
#define TOK_GROUP 274
#define TOK_GROUPDESC 275
#define TOK_MODE 276
#define TOK_MODEDESC 277
#define TOK_MULTIPLE 278
#define TOK_ARGOPTIONAL 279
#define TOK_TYPESTR 280
#define TOK_SECTION 281
#define TOK_DETAILS 282
#define TOK_SECTIONDESC 283
#define TOK_TEXT 284
#define TOK_ARGS 285
#define TOK_VALUES 286
#define TOK_HIDDEN 287
#define TOK_DEPENDON 288
#define TOK_STRING 289
#define TOK_CHAR 290
#define TOK_ARGTYPE 291
#define TOK_SIZE 292




#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
typedef union YYSTYPE
{

/* Line 214 of yacc.c  */
#line 148 "../../src/parser.yy"

    char   *str;
    char    chr;
    int	    argtype;
    int	    boolean;
    class AcceptedValues *ValueList;
    struct gengetopt_option *gengetopt_option;
    struct multiple_size *multiple_size;



/* Line 214 of yacc.c  */
#line 322 "../../src/parser.cc"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

#if ! defined YYLTYPE && ! defined YYLTYPE_IS_DECLARED
typedef struct YYLTYPE
{
  int first_line;
  int first_column;
  int last_line;
  int last_column;
} YYLTYPE;
# define yyltype YYLTYPE /* obsolescent; will be withdrawn */
# define YYLTYPE_IS_DECLARED 1
# define YYLTYPE_IS_TRIVIAL 1
#endif


/* Copy the second part of user declarations.  */


/* Line 264 of yacc.c  */
#line 347 "../../src/parser.cc"

#ifdef short
# undef short
#endif

#ifdef YYTYPE_UINT8
typedef YYTYPE_UINT8 yytype_uint8;
#else
typedef unsigned char yytype_uint8;
#endif

#ifdef YYTYPE_INT8
typedef YYTYPE_INT8 yytype_int8;
#elif (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
typedef signed char yytype_int8;
#else
typedef short int yytype_int8;
#endif

#ifdef YYTYPE_UINT16
typedef YYTYPE_UINT16 yytype_uint16;
#else
typedef unsigned short int yytype_uint16;
#endif

#ifdef YYTYPE_INT16
typedef YYTYPE_INT16 yytype_int16;
#else
typedef short int yytype_int16;
#endif

#ifndef YYSIZE_T
# ifdef __SIZE_TYPE__
#  define YYSIZE_T __SIZE_TYPE__
# elif defined size_t
#  define YYSIZE_T size_t
# elif ! defined YYSIZE_T && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#  include <stddef.h> /* INFRINGES ON USER NAME SPACE */
#  define YYSIZE_T size_t
# else
#  define YYSIZE_T unsigned int
# endif
#endif

#define YYSIZE_MAXIMUM ((YYSIZE_T) -1)

#ifndef YY_
# if YYENABLE_NLS
#  if ENABLE_NLS
#   include <libintl.h> /* INFRINGES ON USER NAME SPACE */
#   define YY_(msgid) dgettext ("bison-runtime", msgid)
#  endif
# endif
# ifndef YY_
#  define YY_(msgid) msgid
# endif
#endif

/* Suppress unused-variable warnings by "using" E.  */
#if ! defined lint || defined __GNUC__
# define YYUSE(e) ((void) (e))
#else
# define YYUSE(e) /* empty */
#endif

/* Identity function, used to suppress warnings about constant conditions.  */
#ifndef lint
# define YYID(n) (n)
#else
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static int
YYID (int yyi)
#else
static int
YYID (yyi)
    int yyi;
#endif
{
  return yyi;
}
#endif

#if ! defined yyoverflow || YYERROR_VERBOSE

/* The parser invokes alloca or malloc; define the necessary symbols.  */

# ifdef YYSTACK_USE_ALLOCA
#  if YYSTACK_USE_ALLOCA
#   ifdef __GNUC__
#    define YYSTACK_ALLOC __builtin_alloca
#   elif defined __BUILTIN_VA_ARG_INCR
#    include <alloca.h> /* INFRINGES ON USER NAME SPACE */
#   elif defined _AIX
#    define YYSTACK_ALLOC __alloca
#   elif defined _MSC_VER
#    include <malloc.h> /* INFRINGES ON USER NAME SPACE */
#    define alloca _alloca
#   else
#    define YYSTACK_ALLOC alloca
#    if ! defined _ALLOCA_H && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
#     include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#     ifndef _STDLIB_H
#      define _STDLIB_H 1
#     endif
#    endif
#   endif
#  endif
# endif

# ifdef YYSTACK_ALLOC
   /* Pacify GCC's `empty if-body' warning.  */
#  define YYSTACK_FREE(Ptr) do { /* empty */; } while (YYID (0))
#  ifndef YYSTACK_ALLOC_MAXIMUM
    /* The OS might guarantee only one guard page at the bottom of the stack,
       and a page size can be as small as 4096 bytes.  So we cannot safely
       invoke alloca (N) if N exceeds 4096.  Use a slightly smaller number
       to allow for a few compiler-allocated temporary stack slots.  */
#   define YYSTACK_ALLOC_MAXIMUM 4032 /* reasonable circa 2006 */
#  endif
# else
#  define YYSTACK_ALLOC YYMALLOC
#  define YYSTACK_FREE YYFREE
#  ifndef YYSTACK_ALLOC_MAXIMUM
#   define YYSTACK_ALLOC_MAXIMUM YYSIZE_MAXIMUM
#  endif
#  if (defined __cplusplus && ! defined _STDLIB_H \
       && ! ((defined YYMALLOC || defined malloc) \
	     && (defined YYFREE || defined free)))
#   include <stdlib.h> /* INFRINGES ON USER NAME SPACE */
#   ifndef _STDLIB_H
#    define _STDLIB_H 1
#   endif
#  endif
#  ifndef YYMALLOC
#   define YYMALLOC malloc
#   if ! defined malloc && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void *malloc (YYSIZE_T); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
#  ifndef YYFREE
#   define YYFREE free
#   if ! defined free && ! defined _STDLIB_H && (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
void free (void *); /* INFRINGES ON USER NAME SPACE */
#   endif
#  endif
# endif
#endif /* ! defined yyoverflow || YYERROR_VERBOSE */


#if (! defined yyoverflow \
     && (! defined __cplusplus \
	 || (defined YYLTYPE_IS_TRIVIAL && YYLTYPE_IS_TRIVIAL \
	     && defined YYSTYPE_IS_TRIVIAL && YYSTYPE_IS_TRIVIAL)))

/* A type that is properly aligned for any stack member.  */
union yyalloc
{
  yytype_int16 yyss_alloc;
  YYSTYPE yyvs_alloc;
  YYLTYPE yyls_alloc;
};

/* The size of the maximum gap between one aligned stack and the next.  */
# define YYSTACK_GAP_MAXIMUM (sizeof (union yyalloc) - 1)

/* The size of an array large to enough to hold all stacks, each with
   N elements.  */
# define YYSTACK_BYTES(N) \
     ((N) * (sizeof (yytype_int16) + sizeof (YYSTYPE) + sizeof (YYLTYPE)) \
      + 2 * YYSTACK_GAP_MAXIMUM)

/* Copy COUNT objects from FROM to TO.  The source and destination do
   not overlap.  */
# ifndef YYCOPY
#  if defined __GNUC__ && 1 < __GNUC__
#   define YYCOPY(To, From, Count) \
      __builtin_memcpy (To, From, (Count) * sizeof (*(From)))
#  else
#   define YYCOPY(To, From, Count)		\
      do					\
	{					\
	  YYSIZE_T yyi;				\
	  for (yyi = 0; yyi < (Count); yyi++)	\
	    (To)[yyi] = (From)[yyi];		\
	}					\
      while (YYID (0))
#  endif
# endif

/* Relocate STACK from its old location to the new one.  The
   local variables YYSIZE and YYSTACKSIZE give the old and new number of
   elements in the stack, and YYPTR gives the new location of the
   stack.  Advance YYPTR to a properly aligned location for the next
   stack.  */
# define YYSTACK_RELOCATE(Stack_alloc, Stack)				\
    do									\
      {									\
	YYSIZE_T yynewbytes;						\
	YYCOPY (&yyptr->Stack_alloc, Stack, yysize);			\
	Stack = &yyptr->Stack_alloc;					\
	yynewbytes = yystacksize * sizeof (*Stack) + YYSTACK_GAP_MAXIMUM; \
	yyptr += yynewbytes / sizeof (*yyptr);				\
      }									\
    while (YYID (0))

#endif

/* YYFINAL -- State number of the termination state.  */
#define YYFINAL  43
/* YYLAST -- Last index in YYTABLE.  */
#define YYLAST   92

/* YYNTOKENS -- Number of terminals.  */
#define YYNTOKENS  43
/* YYNNTS -- Number of nonterminals.  */
#define YYNNTS  27
/* YYNRULES -- Number of rules.  */
#define YYNRULES  66
/* YYNRULES -- Number of states.  */
#define YYNSTATES  116

/* YYTRANSLATE(YYLEX) -- Bison symbol number corresponding to YYLEX.  */
#define YYUNDEFTOK  2
#define YYMAXUTOK   292

#define YYTRANSLATE(YYX)						\
  ((unsigned int) (YYX) <= YYMAXUTOK ? yytranslate[YYX] : YYUNDEFTOK)

/* YYTRANSLATE[YYLEX] -- Bison symbol number corresponding to YYLEX.  */
static const yytype_uint8 yytranslate[] =
{
       0,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
      40,    41,     2,     2,    39,    42,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,    38,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     2,     2,     2,     2,
       2,     2,     2,     2,     2,     2,     1,     2,     3,     4,
       5,     6,     7,     8,     9,    10,    11,    12,    13,    14,
      15,    16,    17,    18,    19,    20,    21,    22,    23,    24,
      25,    26,    27,    28,    29,    30,    31,    32,    33,    34,
      35,    36,    37
};

#if YYDEBUG
/* YYPRHS[YYN] -- Index of the first RHS symbol of rule number YYN in
   YYRHS.  */
static const yytype_uint8 yyprhs[] =
{
       0,     0,     3,     4,     7,     9,    11,    13,    15,    17,
      19,    21,    23,    25,    27,    29,    31,    33,    36,    39,
      42,    45,    48,    52,    55,    58,    63,    67,    73,    79,
      85,    87,    90,    93,    98,   103,   108,   113,   118,   123,
     128,   131,   135,   138,   141,   144,   145,   147,   149,   150,
     152,   154,   156,   158,   159,   163,   164,   168,   169,   173,
     175,   179,   181,   182,   186,   191,   196
};

/* YYRHS -- A `-1'-separated list of the rules' RHS.  */
static const yytype_int8 yyrhs[] =
{
      44,     0,    -1,    -1,    45,    44,    -1,    46,    -1,    47,
      -1,    53,    -1,    48,    -1,    49,    -1,    50,    -1,    51,
      -1,    56,    -1,    52,    -1,    57,    -1,    54,    -1,    58,
      -1,    55,    -1,     3,    34,    -1,     4,    34,    -1,    15,
      59,    -1,    16,    59,    -1,    17,    59,    -1,    26,    59,
      66,    -1,    29,    59,    -1,    30,    34,    -1,     6,    34,
      64,    62,    -1,     8,    34,    65,    -1,     5,    34,    35,
      59,    60,    -1,     7,    34,    35,    59,    60,    -1,     9,
      34,    35,    59,    60,    -1,    34,    -1,    60,    63,    -1,
      60,    36,    -1,    60,    25,    38,    34,    -1,    60,    27,
      38,    59,    -1,    60,    31,    38,    67,    -1,    60,    18,
      38,    34,    -1,    60,    19,    38,    34,    -1,    60,    21,
      38,    34,    -1,    60,    33,    38,    34,    -1,    60,    24,
      -1,    60,    23,    69,    -1,    60,    14,    -1,    60,    32,
      -1,    60,    61,    -1,    -1,    12,    -1,    13,    -1,    -1,
      10,    -1,    11,    -1,    10,    -1,    11,    -1,    -1,    20,
      38,    34,    -1,    -1,    22,    38,    34,    -1,    -1,    28,
      38,    34,    -1,    68,    -1,    67,    39,    68,    -1,    34,
      -1,    -1,    40,    37,    41,    -1,    40,    37,    42,    41,
      -1,    40,    42,    37,    41,    -1,    40,    37,    42,    37,
      41,    -1
};

/* YYRLINE[YYN] -- source line where rule number YYN was defined.  */
static const yytype_uint16 yyrline[] =
{
       0,   209,   209,   211,   216,   217,   218,   219,   220,   221,
     222,   223,   224,   225,   226,   227,   228,   233,   253,   273,
     293,   313,   334,   341,   358,   365,   376,   387,   407,   426,
     448,   451,   457,   462,   467,   472,   477,   482,   487,   492,
     497,   502,   510,   515,   520,   525,   529,   530,   534,   535,
     536,   540,   541,   545,   546,   550,   551,   555,   556,   560,
     561,   565,   569,   570,   571,   572,   573
};
#endif

#if YYDEBUG || YYERROR_VERBOSE || YYTOKEN_TABLE
/* YYTNAME[SYMBOL-NUM] -- String name of the symbol SYMBOL-NUM.
   First, the terminals, then, starting at YYNTOKENS, nonterminals.  */
static const char *const yytname[] =
{
  "$end", "error", "$undefined", "\"package\"", "\"version\"",
  "\"option\"", "\"defgroup\"", "\"groupoption\"", "\"defmode\"",
  "\"modeoption\"", "\"yes\"", "\"no\"", "\"on\"", "\"off\"", "\"flag\"",
  "\"purpose\"", "\"description\"", "\"usage\"", "\"default\"",
  "\"group\"", "\"groupdesc\"", "\"mode\"", "\"modedesc\"", "\"multiple\"",
  "\"argoptional\"", "\"typestr\"", "\"section\"", "\"details\"",
  "\"sectiondesc\"", "\"text\"", "\"args\"", "\"values\"", "\"hidden\"",
  "\"dependon\"", "TOK_STRING", "TOK_CHAR", "TOK_ARGTYPE", "TOK_SIZE",
  "'='", "','", "'('", "')'", "'-'", "$accept", "input", "statement",
  "package", "version", "purpose", "description", "usage", "sectiondef",
  "text", "args", "groupdef", "modedef", "option", "groupoption",
  "modeoption", "quoted_string", "option_parts", "req_onoff",
  "optional_yesno", "opt_yesno", "opt_groupdesc", "opt_modedesc",
  "opt_sectiondesc", "listofvalues", "acceptedvalue", "multiple_size", 0
};
#endif

# ifdef YYPRINT
/* YYTOKNUM[YYLEX-NUM] -- Internal token number corresponding to
   token YYLEX-NUM.  */
static const yytype_uint16 yytoknum[] =
{
       0,   256,   257,   258,   259,   260,   261,   262,   263,   264,
     265,   266,   267,   268,   269,   270,   271,   272,   273,   274,
     275,   276,   277,   278,   279,   280,   281,   282,   283,   284,
     285,   286,   287,   288,   289,   290,   291,   292,    61,    44,
      40,    41,    45
};
# endif

/* YYR1[YYN] -- Symbol number of symbol that rule YYN derives.  */
static const yytype_uint8 yyr1[] =
{
       0,    43,    44,    44,    45,    45,    45,    45,    45,    45,
      45,    45,    45,    45,    45,    45,    45,    46,    47,    48,
      49,    50,    51,    52,    53,    54,    55,    56,    57,    58,
      59,    60,    60,    60,    60,    60,    60,    60,    60,    60,
      60,    60,    60,    60,    60,    60,    61,    61,    62,    62,
      62,    63,    63,    64,    64,    65,    65,    66,    66,    67,
      67,    68,    69,    69,    69,    69,    69
};

/* YYR2[YYN] -- Number of symbols composing right hand side of rule YYN.  */
static const yytype_uint8 yyr2[] =
{
       0,     2,     0,     2,     1,     1,     1,     1,     1,     1,
       1,     1,     1,     1,     1,     1,     1,     2,     2,     2,
       2,     2,     3,     2,     2,     4,     3,     5,     5,     5,
       1,     2,     2,     4,     4,     4,     4,     4,     4,     4,
       2,     3,     2,     2,     2,     0,     1,     1,     0,     1,
       1,     1,     1,     0,     3,     0,     3,     0,     3,     1,
       3,     1,     0,     3,     4,     4,     5
};

/* YYDEFACT[STATE-NAME] -- Default rule to reduce with in state
   STATE-NUM when YYTABLE doesn't specify something else to do.  Zero
   means the default is an error.  */
static const yytype_uint8 yydefact[] =
{
       2,     0,     0,     0,     0,     0,     0,     0,     0,     0,
       0,     0,     0,     0,     0,     2,     4,     5,     7,     8,
       9,    10,    12,     6,    14,    16,    11,    13,    15,    17,
      18,     0,    53,     0,    55,     0,    30,    19,    20,    21,
      57,    23,    24,     1,     3,     0,     0,    48,     0,     0,
      26,     0,     0,    22,    45,     0,    49,    50,    25,    45,
       0,    45,     0,    27,    54,    28,    56,    29,    58,    51,
      52,    46,    47,    42,     0,     0,     0,    62,    40,     0,
       0,     0,    43,     0,    32,    44,    31,     0,     0,     0,
       0,    41,     0,     0,     0,     0,    36,    37,    38,     0,
       0,    33,    34,    61,    35,    59,    39,    63,     0,     0,
       0,     0,    64,    65,    60,    66
};

/* YYDEFGOTO[NTERM-NUM].  */
static const yytype_int8 yydefgoto[] =
{
      -1,    14,    15,    16,    17,    18,    19,    20,    21,    22,
      23,    24,    25,    26,    27,    28,    37,    63,    85,    58,
      86,    47,    50,    53,   104,   105,    91
};

/* YYPACT[STATE-NUM] -- Index in YYTABLE of the portion describing
   STATE-NUM.  */
#define YYPACT_NINF -40
static const yytype_int8 yypact[] =
{
       1,   -13,    -8,    -6,    -5,    -2,    -1,     0,     3,     3,
       3,     3,     3,     4,    25,     1,   -40,   -40,   -40,   -40,
     -40,   -40,   -40,   -40,   -40,   -40,   -40,   -40,   -40,   -40,
     -40,     5,    15,     6,    26,    14,   -40,   -40,   -40,   -40,
      22,   -40,   -40,   -40,   -40,     3,    17,     2,     3,    21,
     -40,     3,    23,   -40,   -40,    19,   -40,   -40,   -40,   -40,
      28,   -40,    29,    33,   -40,    33,   -40,    33,   -40,   -40,
     -40,   -40,   -40,   -40,    30,    32,    34,    27,   -40,    35,
      36,    37,   -40,    38,   -40,   -40,   -40,    43,    44,    45,
     -23,   -40,    46,     3,    47,    48,   -40,   -40,   -40,   -18,
      49,   -40,   -40,   -40,    50,   -40,   -40,   -40,   -26,    42,
      47,    51,   -40,   -40,   -40,   -40
};

/* YYPGOTO[NTERM-NUM].  */
static const yytype_int8 yypgoto[] =
{
     -40,    56,   -40,   -40,   -40,   -40,   -40,   -40,   -40,   -40,
     -40,   -40,   -40,   -40,   -40,   -40,    -9,   -39,   -40,   -40,
     -40,   -40,   -40,   -40,   -40,   -25,   -40
};

/* YYTABLE[YYPACT[STATE-NUM]].  What to do in state STATE-NUM.  If
   positive, shift that token.  If negative, reduce the rule which
   number is the opposite.  If zero, do what YYDEFACT says.
   If YYTABLE_NINF, syntax error.  */
#define YYTABLE_NINF -1
static const yytype_uint8 yytable[] =
{
      38,    39,    40,    41,     1,     2,     3,     4,     5,     6,
       7,   111,    56,    57,    99,   112,     8,     9,    10,   100,
      65,    29,    67,   107,   108,    43,    30,    11,    31,    32,
      12,    13,    33,    34,    35,    46,    54,    36,    42,    59,
      45,    48,    61,    69,    70,    71,    72,    73,    49,    51,
      52,    74,    75,    64,    76,    55,    77,    78,    79,    60,
      80,    62,    66,    68,    81,    82,    83,    90,    87,    84,
      88,    44,    89,    92,    93,    94,    95,    96,    97,    98,
     101,   103,   106,   113,   102,   114,   109,     0,     0,   110,
       0,     0,   115
};

static const yytype_int8 yycheck[] =
{
       9,    10,    11,    12,     3,     4,     5,     6,     7,     8,
       9,    37,    10,    11,    37,    41,    15,    16,    17,    42,
      59,    34,    61,    41,    42,     0,    34,    26,    34,    34,
      29,    30,    34,    34,    34,    20,    45,    34,    34,    48,
      35,    35,    51,    10,    11,    12,    13,    14,    22,    35,
      28,    18,    19,    34,    21,    38,    23,    24,    25,    38,
      27,    38,    34,    34,    31,    32,    33,    40,    38,    36,
      38,    15,    38,    38,    38,    38,    38,    34,    34,    34,
      34,    34,    34,    41,    93,   110,    37,    -1,    -1,    39,
      -1,    -1,    41
};

/* YYSTOS[STATE-NUM] -- The (internal number of the) accessing
   symbol of state STATE-NUM.  */
static const yytype_uint8 yystos[] =
{
       0,     3,     4,     5,     6,     7,     8,     9,    15,    16,
      17,    26,    29,    30,    44,    45,    46,    47,    48,    49,
      50,    51,    52,    53,    54,    55,    56,    57,    58,    34,
      34,    34,    34,    34,    34,    34,    34,    59,    59,    59,
      59,    59,    34,     0,    44,    35,    20,    64,    35,    22,
      65,    35,    28,    66,    59,    38,    10,    11,    62,    59,
      38,    59,    38,    60,    34,    60,    34,    60,    34,    10,
      11,    12,    13,    14,    18,    19,    21,    23,    24,    25,
      27,    31,    32,    33,    36,    61,    63,    38,    38,    38,
      40,    69,    38,    38,    38,    38,    34,    34,    34,    37,
      42,    34,    59,    34,    67,    68,    34,    41,    42,    37,
      39,    37,    41,    41,    68,    41
};

#define yyerrok		(yyerrstatus = 0)
#define yyclearin	(yychar = YYEMPTY)
#define YYEMPTY		(-2)
#define YYEOF		0

#define YYACCEPT	goto yyacceptlab
#define YYABORT		goto yyabortlab
#define YYERROR		goto yyerrorlab


/* Like YYERROR except do call yyerror.  This remains here temporarily
   to ease the transition to the new meaning of YYERROR, for GCC.
   Once GCC version 2 has supplanted version 1, this can go.  */

#define YYFAIL		goto yyerrlab

#define YYRECOVERING()  (!!yyerrstatus)

#define YYBACKUP(Token, Value)					\
do								\
  if (yychar == YYEMPTY && yylen == 1)				\
    {								\
      yychar = (Token);						\
      yylval = (Value);						\
      yytoken = YYTRANSLATE (yychar);				\
      YYPOPSTACK (1);						\
      goto yybackup;						\
    }								\
  else								\
    {								\
      yyerror (YY_("syntax error: cannot back up")); \
      YYERROR;							\
    }								\
while (YYID (0))


#define YYTERROR	1
#define YYERRCODE	256


/* YYLLOC_DEFAULT -- Set CURRENT to span from RHS[1] to RHS[N].
   If N is 0, then set CURRENT to the empty location which ends
   the previous symbol: RHS[0] (always defined).  */

#define YYRHSLOC(Rhs, K) ((Rhs)[K])
#ifndef YYLLOC_DEFAULT
# define YYLLOC_DEFAULT(Current, Rhs, N)				\
    do									\
      if (YYID (N))                                                    \
	{								\
	  (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;	\
	  (Current).first_column = YYRHSLOC (Rhs, 1).first_column;	\
	  (Current).last_line    = YYRHSLOC (Rhs, N).last_line;		\
	  (Current).last_column  = YYRHSLOC (Rhs, N).last_column;	\
	}								\
      else								\
	{								\
	  (Current).first_line   = (Current).last_line   =		\
	    YYRHSLOC (Rhs, 0).last_line;				\
	  (Current).first_column = (Current).last_column =		\
	    YYRHSLOC (Rhs, 0).last_column;				\
	}								\
    while (YYID (0))
#endif


/* YY_LOCATION_PRINT -- Print the location on the stream.
   This macro was not mandated originally: define only if we know
   we won't break user code: when these are the locations we know.  */

#ifndef YY_LOCATION_PRINT
# if YYLTYPE_IS_TRIVIAL
#  define YY_LOCATION_PRINT(File, Loc)			\
     fprintf (File, "%d.%d-%d.%d",			\
	      (Loc).first_line, (Loc).first_column,	\
	      (Loc).last_line,  (Loc).last_column)
# else
#  define YY_LOCATION_PRINT(File, Loc) ((void) 0)
# endif
#endif


/* YYLEX -- calling `yylex' with the right arguments.  */

#ifdef YYLEX_PARAM
# define YYLEX yylex (YYLEX_PARAM)
#else
# define YYLEX yylex ()
#endif

/* Enable debugging if requested.  */
#if YYDEBUG

# ifndef YYFPRINTF
#  include <stdio.h> /* INFRINGES ON USER NAME SPACE */
#  define YYFPRINTF fprintf
# endif

# define YYDPRINTF(Args)			\
do {						\
  if (yydebug)					\
    YYFPRINTF Args;				\
} while (YYID (0))

# define YY_SYMBOL_PRINT(Title, Type, Value, Location)			  \
do {									  \
  if (yydebug)								  \
    {									  \
      YYFPRINTF (stderr, "%s ", Title);					  \
      yy_symbol_print (stderr,						  \
		  Type, Value, Location); \
      YYFPRINTF (stderr, "\n");						  \
    }									  \
} while (YYID (0))


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_value_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (!yyvaluep)
    return;
  YYUSE (yylocationp);
# ifdef YYPRINT
  if (yytype < YYNTOKENS)
    YYPRINT (yyoutput, yytoknum[yytype], *yyvaluep);
# else
  YYUSE (yyoutput);
# endif
  switch (yytype)
    {
      default:
	break;
    }
}


/*--------------------------------.
| Print this symbol on YYOUTPUT.  |
`--------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_symbol_print (FILE *yyoutput, int yytype, YYSTYPE const * const yyvaluep, YYLTYPE const * const yylocationp)
#else
static void
yy_symbol_print (yyoutput, yytype, yyvaluep, yylocationp)
    FILE *yyoutput;
    int yytype;
    YYSTYPE const * const yyvaluep;
    YYLTYPE const * const yylocationp;
#endif
{
  if (yytype < YYNTOKENS)
    YYFPRINTF (yyoutput, "token %s (", yytname[yytype]);
  else
    YYFPRINTF (yyoutput, "nterm %s (", yytname[yytype]);

  YY_LOCATION_PRINT (yyoutput, *yylocationp);
  YYFPRINTF (yyoutput, ": ");
  yy_symbol_value_print (yyoutput, yytype, yyvaluep, yylocationp);
  YYFPRINTF (yyoutput, ")");
}

/*------------------------------------------------------------------.
| yy_stack_print -- Print the state stack from its BOTTOM up to its |
| TOP (included).                                                   |
`------------------------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_stack_print (yytype_int16 *yybottom, yytype_int16 *yytop)
#else
static void
yy_stack_print (yybottom, yytop)
    yytype_int16 *yybottom;
    yytype_int16 *yytop;
#endif
{
  YYFPRINTF (stderr, "Stack now");
  for (; yybottom <= yytop; yybottom++)
    {
      int yybot = *yybottom;
      YYFPRINTF (stderr, " %d", yybot);
    }
  YYFPRINTF (stderr, "\n");
}

# define YY_STACK_PRINT(Bottom, Top)				\
do {								\
  if (yydebug)							\
    yy_stack_print ((Bottom), (Top));				\
} while (YYID (0))


/*------------------------------------------------.
| Report that the YYRULE is going to be reduced.  |
`------------------------------------------------*/

#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yy_reduce_print (YYSTYPE *yyvsp, YYLTYPE *yylsp, int yyrule)
#else
static void
yy_reduce_print (yyvsp, yylsp, yyrule)
    YYSTYPE *yyvsp;
    YYLTYPE *yylsp;
    int yyrule;
#endif
{
  int yynrhs = yyr2[yyrule];
  int yyi;
  unsigned long int yylno = yyrline[yyrule];
  YYFPRINTF (stderr, "Reducing stack by rule %d (line %lu):\n",
	     yyrule - 1, yylno);
  /* The symbols being reduced.  */
  for (yyi = 0; yyi < yynrhs; yyi++)
    {
      YYFPRINTF (stderr, "   $%d = ", yyi + 1);
      yy_symbol_print (stderr, yyrhs[yyprhs[yyrule] + yyi],
		       &(yyvsp[(yyi + 1) - (yynrhs)])
		       , &(yylsp[(yyi + 1) - (yynrhs)])		       );
      YYFPRINTF (stderr, "\n");
    }
}

# define YY_REDUCE_PRINT(Rule)		\
do {					\
  if (yydebug)				\
    yy_reduce_print (yyvsp, yylsp, Rule); \
} while (YYID (0))

/* Nonzero means print parse trace.  It is left uninitialized so that
   multiple parsers can coexist.  */
int yydebug;
#else /* !YYDEBUG */
# define YYDPRINTF(Args)
# define YY_SYMBOL_PRINT(Title, Type, Value, Location)
# define YY_STACK_PRINT(Bottom, Top)
# define YY_REDUCE_PRINT(Rule)
#endif /* !YYDEBUG */


/* YYINITDEPTH -- initial size of the parser's stacks.  */
#ifndef	YYINITDEPTH
# define YYINITDEPTH 200
#endif

/* YYMAXDEPTH -- maximum size the stacks can grow to (effective only
   if the built-in stack extension method is used).

   Do not make this value too large; the results are undefined if
   YYSTACK_ALLOC_MAXIMUM < YYSTACK_BYTES (YYMAXDEPTH)
   evaluated with infinite-precision integer arithmetic.  */

#ifndef YYMAXDEPTH
# define YYMAXDEPTH 10000
#endif



#if YYERROR_VERBOSE

# ifndef yystrlen
#  if defined __GLIBC__ && defined _STRING_H
#   define yystrlen strlen
#  else
/* Return the length of YYSTR.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static YYSIZE_T
yystrlen (const char *yystr)
#else
static YYSIZE_T
yystrlen (yystr)
    const char *yystr;
#endif
{
  YYSIZE_T yylen;
  for (yylen = 0; yystr[yylen]; yylen++)
    continue;
  return yylen;
}
#  endif
# endif

# ifndef yystpcpy
#  if defined __GLIBC__ && defined _STRING_H && defined _GNU_SOURCE
#   define yystpcpy stpcpy
#  else
/* Copy YYSRC to YYDEST, returning the address of the terminating '\0' in
   YYDEST.  */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static char *
yystpcpy (char *yydest, const char *yysrc)
#else
static char *
yystpcpy (yydest, yysrc)
    char *yydest;
    const char *yysrc;
#endif
{
  char *yyd = yydest;
  const char *yys = yysrc;

  while ((*yyd++ = *yys++) != '\0')
    continue;

  return yyd - 1;
}
#  endif
# endif

# ifndef yytnamerr
/* Copy to YYRES the contents of YYSTR after stripping away unnecessary
   quotes and backslashes, so that it's suitable for yyerror.  The
   heuristic is that double-quoting is unnecessary unless the string
   contains an apostrophe, a comma, or backslash (other than
   backslash-backslash).  YYSTR is taken from yytname.  If YYRES is
   null, do not copy; instead, return the length of what the result
   would have been.  */
static YYSIZE_T
yytnamerr (char *yyres, const char *yystr)
{
  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
	switch (*++yyp)
	  {
	  case '\'':
	  case ',':
	    goto do_not_strip_quotes;

	  case '\\':
	    if (*++yyp != '\\')
	      goto do_not_strip_quotes;
	    /* Fall through.  */
	  default:
	    if (yyres)
	      yyres[yyn] = *yyp;
	    yyn++;
	    break;

	  case '"':
	    if (yyres)
	      yyres[yyn] = '\0';
	    return yyn;
	  }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}
# endif

/* Copy into YYRESULT an error message about the unexpected token
   YYCHAR while in state YYSTATE.  Return the number of bytes copied,
   including the terminating null byte.  If YYRESULT is null, do not
   copy anything; just return the number of bytes that would be
   copied.  As a special case, return 0 if an ordinary "syntax error"
   message will do.  Return YYSIZE_MAXIMUM if overflow occurs during
   size calculation.  */
static YYSIZE_T
yysyntax_error (char *yyresult, int yystate, int yychar)
{
  int yyn = yypact[yystate];

  if (! (YYPACT_NINF < yyn && yyn <= YYLAST))
    return 0;
  else
    {
      int yytype = YYTRANSLATE (yychar);
      YYSIZE_T yysize0 = yytnamerr (0, yytname[yytype]);
      YYSIZE_T yysize = yysize0;
      YYSIZE_T yysize1;
      int yysize_overflow = 0;
      enum { YYERROR_VERBOSE_ARGS_MAXIMUM = 5 };
      char const *yyarg[YYERROR_VERBOSE_ARGS_MAXIMUM];
      int yyx;

# if 0
      /* This is so xgettext sees the translatable formats that are
	 constructed on the fly.  */
      YY_("syntax error, unexpected %s");
      YY_("syntax error, unexpected %s, expecting %s");
      YY_("syntax error, unexpected %s, expecting %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s");
      YY_("syntax error, unexpected %s, expecting %s or %s or %s or %s");
# endif
      char *yyfmt;
      char const *yyf;
      static char const yyunexpected[] = "syntax error, unexpected %s";
      static char const yyexpecting[] = ", expecting %s";
      static char const yyor[] = " or %s";
      char yyformat[sizeof yyunexpected
		    + sizeof yyexpecting - 1
		    + ((YYERROR_VERBOSE_ARGS_MAXIMUM - 2)
		       * (sizeof yyor - 1))];
      char const *yyprefix = yyexpecting;

      /* Start YYX at -YYN if negative to avoid negative indexes in
	 YYCHECK.  */
      int yyxbegin = yyn < 0 ? -yyn : 0;

      /* Stay within bounds of both yycheck and yytname.  */
      int yychecklim = YYLAST - yyn + 1;
      int yyxend = yychecklim < YYNTOKENS ? yychecklim : YYNTOKENS;
      int yycount = 1;

      yyarg[0] = yytname[yytype];
      yyfmt = yystpcpy (yyformat, yyunexpected);

      for (yyx = yyxbegin; yyx < yyxend; ++yyx)
	if (yycheck[yyx + yyn] == yyx && yyx != YYTERROR)
	  {
	    if (yycount == YYERROR_VERBOSE_ARGS_MAXIMUM)
	      {
		yycount = 1;
		yysize = yysize0;
		yyformat[sizeof yyunexpected - 1] = '\0';
		break;
	      }
	    yyarg[yycount++] = yytname[yyx];
	    yysize1 = yysize + yytnamerr (0, yytname[yyx]);
	    yysize_overflow |= (yysize1 < yysize);
	    yysize = yysize1;
	    yyfmt = yystpcpy (yyfmt, yyprefix);
	    yyprefix = yyor;
	  }

      yyf = YY_(yyformat);
      yysize1 = yysize + yystrlen (yyf);
      yysize_overflow |= (yysize1 < yysize);
      yysize = yysize1;

      if (yysize_overflow)
	return YYSIZE_MAXIMUM;

      if (yyresult)
	{
	  /* Avoid sprintf, as that infringes on the user's name space.
	     Don't have undefined behavior even if the translation
	     produced a string with the wrong number of "%s"s.  */
	  char *yyp = yyresult;
	  int yyi = 0;
	  while ((*yyp = *yyf) != '\0')
	    {
	      if (*yyp == '%' && yyf[1] == 's' && yyi < yycount)
		{
		  yyp += yytnamerr (yyp, yyarg[yyi++]);
		  yyf += 2;
		}
	      else
		{
		  yyp++;
		  yyf++;
		}
	    }
	}
      return yysize;
    }
}
#endif /* YYERROR_VERBOSE */


/*-----------------------------------------------.
| Release the memory associated to this symbol.  |
`-----------------------------------------------*/

/*ARGSUSED*/
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
static void
yydestruct (const char *yymsg, int yytype, YYSTYPE *yyvaluep, YYLTYPE *yylocationp)
#else
static void
yydestruct (yymsg, yytype, yyvaluep, yylocationp)
    const char *yymsg;
    int yytype;
    YYSTYPE *yyvaluep;
    YYLTYPE *yylocationp;
#endif
{
  YYUSE (yyvaluep);
  YYUSE (yylocationp);

  if (!yymsg)
    yymsg = "Deleting";
  YY_SYMBOL_PRINT (yymsg, yytype, yyvaluep, yylocationp);

  switch (yytype)
    {

      default:
	break;
    }
}

/* Prevent warnings from -Wmissing-prototypes.  */
#ifdef YYPARSE_PARAM
#if defined __STDC__ || defined __cplusplus
int yyparse (void *YYPARSE_PARAM);
#else
int yyparse ();
#endif
#else /* ! YYPARSE_PARAM */
#if defined __STDC__ || defined __cplusplus
int yyparse (void);
#else
int yyparse ();
#endif
#endif /* ! YYPARSE_PARAM */


/* The lookahead symbol.  */
int yychar;

/* The semantic value of the lookahead symbol.  */
YYSTYPE yylval;

/* Location data for the lookahead symbol.  */
YYLTYPE yylloc;

/* Number of syntax errors so far.  */
int yynerrs;



/*-------------------------.
| yyparse or yypush_parse.  |
`-------------------------*/

#ifdef YYPARSE_PARAM
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void *YYPARSE_PARAM)
#else
int
yyparse (YYPARSE_PARAM)
    void *YYPARSE_PARAM;
#endif
#else /* ! YYPARSE_PARAM */
#if (defined __STDC__ || defined __C99__FUNC__ \
     || defined __cplusplus || defined _MSC_VER)
int
yyparse (void)
#else
int
yyparse ()

#endif
#endif
{


    int yystate;
    /* Number of tokens to shift before error messages enabled.  */
    int yyerrstatus;

    /* The stacks and their tools:
       `yyss': related to states.
       `yyvs': related to semantic values.
       `yyls': related to locations.

       Refer to the stacks thru separate pointers, to allow yyoverflow
       to reallocate them elsewhere.  */

    /* The state stack.  */
    yytype_int16 yyssa[YYINITDEPTH];
    yytype_int16 *yyss;
    yytype_int16 *yyssp;

    /* The semantic value stack.  */
    YYSTYPE yyvsa[YYINITDEPTH];
    YYSTYPE *yyvs;
    YYSTYPE *yyvsp;

    /* The location stack.  */
    YYLTYPE yylsa[YYINITDEPTH];
    YYLTYPE *yyls;
    YYLTYPE *yylsp;

    /* The locations where the error started and ended.  */
    YYLTYPE yyerror_range[2];

    YYSIZE_T yystacksize;

  int yyn;
  int yyresult;
  /* Lookahead token as an internal (translated) token number.  */
  int yytoken;
  /* The variables used to return semantic value and location from the
     action routines.  */
  YYSTYPE yyval;
  YYLTYPE yyloc;

#if YYERROR_VERBOSE
  /* Buffer for error messages, and its allocated size.  */
  char yymsgbuf[128];
  char *yymsg = yymsgbuf;
  YYSIZE_T yymsg_alloc = sizeof yymsgbuf;
#endif

#define YYPOPSTACK(N)   (yyvsp -= (N), yyssp -= (N), yylsp -= (N))

  /* The number of symbols on the RHS of the reduced rule.
     Keep to zero when no symbol should be popped.  */
  int yylen = 0;

  yytoken = 0;
  yyss = yyssa;
  yyvs = yyvsa;
  yyls = yylsa;
  yystacksize = YYINITDEPTH;

  YYDPRINTF ((stderr, "Starting parse\n"));

  yystate = 0;
  yyerrstatus = 0;
  yynerrs = 0;
  yychar = YYEMPTY; /* Cause a token to be read.  */

  /* Initialize stack pointers.
     Waste one element of value and location stack
     so that they stay on the same level as the state stack.
     The wasted elements are never initialized.  */
  yyssp = yyss;
  yyvsp = yyvs;
  yylsp = yyls;

#if YYLTYPE_IS_TRIVIAL
  /* Initialize the default location before parsing starts.  */
  yylloc.first_line   = yylloc.last_line   = 1;
  yylloc.first_column = yylloc.last_column = 1;
#endif

  goto yysetstate;

/*------------------------------------------------------------.
| yynewstate -- Push a new state, which is found in yystate.  |
`------------------------------------------------------------*/
 yynewstate:
  /* In all cases, when you get here, the value and location stacks
     have just been pushed.  So pushing a state here evens the stacks.  */
  yyssp++;

 yysetstate:
  *yyssp = yystate;

  if (yyss + yystacksize - 1 <= yyssp)
    {
      /* Get the current used size of the three stacks, in elements.  */
      YYSIZE_T yysize = yyssp - yyss + 1;

#ifdef yyoverflow
      {
	/* Give user a chance to reallocate the stack.  Use copies of
	   these so that the &'s don't force the real ones into
	   memory.  */
	YYSTYPE *yyvs1 = yyvs;
	yytype_int16 *yyss1 = yyss;
	YYLTYPE *yyls1 = yyls;

	/* Each stack pointer address is followed by the size of the
	   data in use in that stack, in bytes.  This used to be a
	   conditional around just the two extra args, but that might
	   be undefined if yyoverflow is a macro.  */
	yyoverflow (YY_("memory exhausted"),
		    &yyss1, yysize * sizeof (*yyssp),
		    &yyvs1, yysize * sizeof (*yyvsp),
		    &yyls1, yysize * sizeof (*yylsp),
		    &yystacksize);

	yyls = yyls1;
	yyss = yyss1;
	yyvs = yyvs1;
      }
#else /* no yyoverflow */
# ifndef YYSTACK_RELOCATE
      goto yyexhaustedlab;
# else
      /* Extend the stack our own way.  */
      if (YYMAXDEPTH <= yystacksize)
	goto yyexhaustedlab;
      yystacksize *= 2;
      if (YYMAXDEPTH < yystacksize)
	yystacksize = YYMAXDEPTH;

      {
	yytype_int16 *yyss1 = yyss;
	union yyalloc *yyptr =
	  (union yyalloc *) YYSTACK_ALLOC (YYSTACK_BYTES (yystacksize));
	if (! yyptr)
	  goto yyexhaustedlab;
	YYSTACK_RELOCATE (yyss_alloc, yyss);
	YYSTACK_RELOCATE (yyvs_alloc, yyvs);
	YYSTACK_RELOCATE (yyls_alloc, yyls);
#  undef YYSTACK_RELOCATE
	if (yyss1 != yyssa)
	  YYSTACK_FREE (yyss1);
      }
# endif
#endif /* no yyoverflow */

      yyssp = yyss + yysize - 1;
      yyvsp = yyvs + yysize - 1;
      yylsp = yyls + yysize - 1;

      YYDPRINTF ((stderr, "Stack size increased to %lu\n",
		  (unsigned long int) yystacksize));

      if (yyss + yystacksize - 1 <= yyssp)
	YYABORT;
    }

  YYDPRINTF ((stderr, "Entering state %d\n", yystate));

  if (yystate == YYFINAL)
    YYACCEPT;

  goto yybackup;

/*-----------.
| yybackup.  |
`-----------*/
yybackup:

  /* Do appropriate processing given the current state.  Read a
     lookahead token if we need one and don't already have one.  */

  /* First try to decide what to do without reference to lookahead token.  */
  yyn = yypact[yystate];
  if (yyn == YYPACT_NINF)
    goto yydefault;

  /* Not known => get a lookahead token if don't already have one.  */

  /* YYCHAR is either YYEMPTY or YYEOF or a valid lookahead symbol.  */
  if (yychar == YYEMPTY)
    {
      YYDPRINTF ((stderr, "Reading a token: "));
      yychar = YYLEX;
    }

  if (yychar <= YYEOF)
    {
      yychar = yytoken = YYEOF;
      YYDPRINTF ((stderr, "Now at end of input.\n"));
    }
  else
    {
      yytoken = YYTRANSLATE (yychar);
      YY_SYMBOL_PRINT ("Next token is", yytoken, &yylval, &yylloc);
    }

  /* If the proper action on seeing token YYTOKEN is to reduce or to
     detect an error, take that action.  */
  yyn += yytoken;
  if (yyn < 0 || YYLAST < yyn || yycheck[yyn] != yytoken)
    goto yydefault;
  yyn = yytable[yyn];
  if (yyn <= 0)
    {
      if (yyn == 0 || yyn == YYTABLE_NINF)
	goto yyerrlab;
      yyn = -yyn;
      goto yyreduce;
    }

  /* Count tokens shifted since error; after three, turn off error
     status.  */
  if (yyerrstatus)
    yyerrstatus--;

  /* Shift the lookahead token.  */
  YY_SYMBOL_PRINT ("Shifting", yytoken, &yylval, &yylloc);

  /* Discard the shifted token.  */
  yychar = YYEMPTY;

  yystate = yyn;
  *++yyvsp = yylval;
  *++yylsp = yylloc;
  goto yynewstate;


/*-----------------------------------------------------------.
| yydefault -- do the default action for the current state.  |
`-----------------------------------------------------------*/
yydefault:
  yyn = yydefact[yystate];
  if (yyn == 0)
    goto yyerrlab;
  goto yyreduce;


/*-----------------------------.
| yyreduce -- Do a reduction.  |
`-----------------------------*/
yyreduce:
  /* yyn is the number of a rule to reduce with.  */
  yylen = yyr2[yyn];

  /* If YYLEN is nonzero, implement the default value of the action:
     `$$ = $1'.

     Otherwise, the following line sets YYVAL to garbage.
     This behavior is undocumented and Bison
     users should not rely upon it.  Assigning to YYVAL
     unconditionally makes the parser a bit smaller, and it avoids a
     GCC warning that YYVAL may be used uninitialized.  */
  yyval = yyvsp[1-yylen];

  /* Default location.  */
  YYLLOC_DEFAULT (yyloc, (yylsp - yylen), yylen);
  YY_REDUCE_PRINT (yyn);
  switch (yyn)
    {
        case 17:

/* Line 1455 of yacc.c  */
#line 234 "../../src/parser.yy"
    {
	      if (gengetopt_package_given)
		{
		  yyerror ("package redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_package_given = 1;
		  if (gengetopt_define_package ((yyvsp[(2) - (2)].str)))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
    break;

  case 18:

/* Line 1455 of yacc.c  */
#line 254 "../../src/parser.yy"
    {
	      if (gengetopt_version_given)
		{
		  yyerror ("version redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_version_given = 1;
		  if (gengetopt_define_version ((yyvsp[(2) - (2)].str)))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
    break;

  case 19:

/* Line 1455 of yacc.c  */
#line 274 "../../src/parser.yy"
    {
	      if (gengetopt_purpose_given)
		{
		  yyerror ("purpose redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_purpose_given = 1;
		  if (gengetopt_define_purpose ((yyvsp[(2) - (2)].str)))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
    break;

  case 20:

/* Line 1455 of yacc.c  */
#line 294 "../../src/parser.yy"
    {
	      if (gengetopt_description_given)
		{
		  yyerror ("description redefined");
		  YYERROR;
		}
	      else
		{
		  gengetopt_description_given = 1;
		  if (gengetopt_define_description ((yyvsp[(2) - (2)].str)))
		    {
		      yyerror ("not enough memory");
		      YYERROR;
		    }
		}
	    }
    break;

  case 21:

/* Line 1455 of yacc.c  */
#line 314 "../../src/parser.yy"
    {
      if (gengetopt_usage_given)
      {
	  yyerror ("usage redefined");
	  YYERROR;
      }
      else
      {
	  gengetopt_usage_given = 1;
	  if (gengetopt_define_usage ((yyvsp[(2) - (2)].str)))
          {
	      yyerror ("not enough memory");
	      YYERROR;
          }
      }
  }
    break;

  case 22:

/* Line 1455 of yacc.c  */
#line 335 "../../src/parser.yy"
    {
                gengetopt_set_section ((yyvsp[(2) - (3)].str), (yyvsp[(3) - (3)].str));
              }
    break;

  case 23:

/* Line 1455 of yacc.c  */
#line 342 "../../src/parser.yy"
    {
            	if (current_option) {
            		std::string current_option_text;
            		if (current_option->text_after) {
            			current_option_text = std::string(current_option->text_after) + (yyvsp[(2) - (2)].str);
            			current_option->text_after = strdup(current_option_text.c_str()); 
            		} else {
	            		current_option->text_after = strdup((yyvsp[(2) - (2)].str));
	            	}
            	} else {
					gengetopt_set_text((yyvsp[(2) - (2)].str));
  				}
            }
    break;

  case 24:

/* Line 1455 of yacc.c  */
#line 359 "../../src/parser.yy"
    {
  gengetopt_set_args((yyvsp[(2) - (2)].str));
            }
    break;

  case 25:

/* Line 1455 of yacc.c  */
#line 366 "../../src/parser.yy"
    {
              if (gengetopt_add_group ((yyvsp[(2) - (4)].str), (yyvsp[(3) - (4)].str), (yyvsp[(4) - (4)].boolean)))
                {
		  	yyerror ("group redefined");
		  	YYERROR;
		  }
	    }
    break;

  case 26:

/* Line 1455 of yacc.c  */
#line 377 "../../src/parser.yy"
    {
              if (gengetopt_add_mode ((yyvsp[(2) - (3)].str), (yyvsp[(3) - (3)].str)))
                {
		  	yyerror ("mode redefined");
		  	YYERROR;
		  }
	    }
    break;

  case 27:

/* Line 1455 of yacc.c  */
#line 389 "../../src/parser.yy"
    {
          (yyvsp[(5) - (5)].gengetopt_option)->filename = gengetopt_input_filename;
          (yyvsp[(5) - (5)].gengetopt_option)->linenum = (yylsp[(1) - (5)]).first_line;
	      (yyvsp[(5) - (5)].gengetopt_option)->long_opt = strdup((yyvsp[(2) - (5)].str));
	      if ((yyvsp[(3) - (5)].chr) != '-')
	      	(yyvsp[(5) - (5)].gengetopt_option)->short_opt = (yyvsp[(3) - (5)].chr);
	      (yyvsp[(5) - (5)].gengetopt_option)->desc = strdup((yyvsp[(4) - (5)].str));
	      int o = gengetopt_check_option ((yyvsp[(5) - (5)].gengetopt_option), false);
	      check_result(o, (yyvsp[(5) - (5)].gengetopt_option));
          check_error;
	      o = gengetopt_add_option ((yyvsp[(5) - (5)].gengetopt_option));
	      check_result(o, (yyvsp[(5) - (5)].gengetopt_option));
	      check_error;
	      current_option = (yyvsp[(5) - (5)].gengetopt_option);
	    }
    break;

  case 28:

/* Line 1455 of yacc.c  */
#line 409 "../../src/parser.yy"
    {
          (yyvsp[(5) - (5)].gengetopt_option)->filename = gengetopt_input_filename;
          (yyvsp[(5) - (5)].gengetopt_option)->linenum = (yylsp[(1) - (5)]).first_line;
	      (yyvsp[(5) - (5)].gengetopt_option)->long_opt = strdup((yyvsp[(2) - (5)].str));
          if ((yyvsp[(3) - (5)].chr) != '-')
            (yyvsp[(5) - (5)].gengetopt_option)->short_opt = (yyvsp[(3) - (5)].chr);
          (yyvsp[(5) - (5)].gengetopt_option)->desc = strdup((yyvsp[(4) - (5)].str));
          int o = gengetopt_check_option ((yyvsp[(5) - (5)].gengetopt_option), true);
          check_result(o, (yyvsp[(5) - (5)].gengetopt_option));
          check_error;
          o = gengetopt_add_option ((yyvsp[(5) - (5)].gengetopt_option));
          check_result(o, (yyvsp[(5) - (5)].gengetopt_option));
          check_error;
	    }
    break;

  case 29:

/* Line 1455 of yacc.c  */
#line 428 "../../src/parser.yy"
    {
          (yyvsp[(5) - (5)].gengetopt_option)->filename = gengetopt_input_filename;
          (yyvsp[(5) - (5)].gengetopt_option)->linenum = (yylsp[(1) - (5)]).first_line;
	      (yyvsp[(5) - (5)].gengetopt_option)->long_opt = strdup((yyvsp[(2) - (5)].str));
          if ((yyvsp[(3) - (5)].chr) != '-')
            (yyvsp[(5) - (5)].gengetopt_option)->short_opt = (yyvsp[(3) - (5)].chr);
          (yyvsp[(5) - (5)].gengetopt_option)->desc = strdup((yyvsp[(4) - (5)].str));
          int o = gengetopt_check_option ((yyvsp[(5) - (5)].gengetopt_option), false, true);
          check_result(o, (yyvsp[(5) - (5)].gengetopt_option));
          check_error;
          o = gengetopt_add_option ((yyvsp[(5) - (5)].gengetopt_option));
          check_result(o, (yyvsp[(5) - (5)].gengetopt_option));
          check_error;
	    }
    break;

  case 31:

/* Line 1455 of yacc.c  */
#line 452 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (2)].gengetopt_option);
			  	(yyval.gengetopt_option)->required = (yyvsp[(2) - (2)].boolean);
			  	(yyval.gengetopt_option)->required_set = true;
			  }
    break;

  case 32:

/* Line 1455 of yacc.c  */
#line 458 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (2)].gengetopt_option);
			  	(yyval.gengetopt_option)->type = (yyvsp[(2) - (2)].argtype);
			  }
    break;

  case 33:

/* Line 1455 of yacc.c  */
#line 463 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
			  	(yyval.gengetopt_option)->type_str = strdup((yyvsp[(4) - (4)].str));
			  }
    break;

  case 34:

/* Line 1455 of yacc.c  */
#line 468 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
			  	(yyval.gengetopt_option)->details = strdup((yyvsp[(4) - (4)].str));
			  }
    break;

  case 35:

/* Line 1455 of yacc.c  */
#line 473 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
			  	(yyval.gengetopt_option)->acceptedvalues = (yyvsp[(4) - (4)].ValueList);
			  }
    break;

  case 36:

/* Line 1455 of yacc.c  */
#line 478 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
			  	(yyval.gengetopt_option)->default_string = strdup((yyvsp[(4) - (4)].str));
			  }
    break;

  case 37:

/* Line 1455 of yacc.c  */
#line 483 "../../src/parser.yy"
    {
                (yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
                (yyval.gengetopt_option)->group_value = strdup((yyvsp[(4) - (4)].str));
              }
    break;

  case 38:

/* Line 1455 of yacc.c  */
#line 488 "../../src/parser.yy"
    {
                (yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
                (yyval.gengetopt_option)->mode_value = strdup((yyvsp[(4) - (4)].str));
              }
    break;

  case 39:

/* Line 1455 of yacc.c  */
#line 493 "../../src/parser.yy"
    {
                (yyval.gengetopt_option) = (yyvsp[(1) - (4)].gengetopt_option);
                (yyval.gengetopt_option)->dependon = strdup((yyvsp[(4) - (4)].str));
              }
    break;

  case 40:

/* Line 1455 of yacc.c  */
#line 498 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (2)].gengetopt_option);
			  	(yyval.gengetopt_option)->arg_is_optional = true;
			  }
    break;

  case 41:

/* Line 1455 of yacc.c  */
#line 503 "../../src/parser.yy"
    {
			  	(yyval.gengetopt_option) = (yyvsp[(1) - (3)].gengetopt_option);
			  	(yyval.gengetopt_option)->multiple = true;
                (yyval.gengetopt_option)->multiple_min = (yyvsp[(3) - (3)].multiple_size)->min;
                (yyval.gengetopt_option)->multiple_max = (yyvsp[(3) - (3)].multiple_size)->max;
                delete (yyvsp[(3) - (3)].multiple_size);
			  }
    break;

  case 42:

/* Line 1455 of yacc.c  */
#line 511 "../../src/parser.yy"
    {
          (yyval.gengetopt_option) = (yyvsp[(1) - (2)].gengetopt_option);
          (yyval.gengetopt_option)->type = ARG_FLAG;
        }
    break;

  case 43:

/* Line 1455 of yacc.c  */
#line 516 "../../src/parser.yy"
    {
          (yyval.gengetopt_option) = (yyvsp[(1) - (2)].gengetopt_option);
          (yyval.gengetopt_option)->hidden = true;
        }
    break;

  case 44:

/* Line 1455 of yacc.c  */
#line 521 "../../src/parser.yy"
    {
          (yyval.gengetopt_option) = (yyvsp[(1) - (2)].gengetopt_option);
          (yyval.gengetopt_option)->flagstat = (yyvsp[(2) - (2)].boolean);
        }
    break;

  case 45:

/* Line 1455 of yacc.c  */
#line 525 "../../src/parser.yy"
    { (yyval.gengetopt_option) = new gengetopt_option; }
    break;

  case 46:

/* Line 1455 of yacc.c  */
#line 529 "../../src/parser.yy"
    { (yyval.boolean) = 1; }
    break;

  case 47:

/* Line 1455 of yacc.c  */
#line 530 "../../src/parser.yy"
    { (yyval.boolean) = 0; }
    break;

  case 48:

/* Line 1455 of yacc.c  */
#line 534 "../../src/parser.yy"
    { (yyval.boolean) = 0; }
    break;

  case 49:

/* Line 1455 of yacc.c  */
#line 535 "../../src/parser.yy"
    { (yyval.boolean) = 1; }
    break;

  case 50:

/* Line 1455 of yacc.c  */
#line 536 "../../src/parser.yy"
    { (yyval.boolean) = 0; }
    break;

  case 51:

/* Line 1455 of yacc.c  */
#line 540 "../../src/parser.yy"
    { (yyval.boolean) = 1; }
    break;

  case 52:

/* Line 1455 of yacc.c  */
#line 541 "../../src/parser.yy"
    { (yyval.boolean) = 0; }
    break;

  case 53:

/* Line 1455 of yacc.c  */
#line 545 "../../src/parser.yy"
    { (yyval.str) = 0; }
    break;

  case 54:

/* Line 1455 of yacc.c  */
#line 546 "../../src/parser.yy"
    { (yyval.str) = (yyvsp[(3) - (3)].str); }
    break;

  case 55:

/* Line 1455 of yacc.c  */
#line 550 "../../src/parser.yy"
    { (yyval.str) = 0; }
    break;

  case 56:

/* Line 1455 of yacc.c  */
#line 551 "../../src/parser.yy"
    { (yyval.str) = (yyvsp[(3) - (3)].str); }
    break;

  case 57:

/* Line 1455 of yacc.c  */
#line 555 "../../src/parser.yy"
    { (yyval.str) = 0; }
    break;

  case 58:

/* Line 1455 of yacc.c  */
#line 556 "../../src/parser.yy"
    { (yyval.str) = (yyvsp[(3) - (3)].str); }
    break;

  case 59:

/* Line 1455 of yacc.c  */
#line 560 "../../src/parser.yy"
    { (yyval.ValueList) = new AcceptedValues; (yyval.ValueList)->insert((yyvsp[(1) - (1)].str)); }
    break;

  case 60:

/* Line 1455 of yacc.c  */
#line 561 "../../src/parser.yy"
    { (yyvsp[(1) - (3)].ValueList)->insert((yyvsp[(3) - (3)].str)); (yyval.ValueList) = (yyvsp[(1) - (3)].ValueList); }
    break;

  case 61:

/* Line 1455 of yacc.c  */
#line 565 "../../src/parser.yy"
    { (yyval.str) = (yyvsp[(1) - (1)].str); }
    break;

  case 62:

/* Line 1455 of yacc.c  */
#line 569 "../../src/parser.yy"
    { (yyval.multiple_size) = new multiple_size; }
    break;

  case 63:

/* Line 1455 of yacc.c  */
#line 570 "../../src/parser.yy"
    { (yyval.multiple_size) = new multiple_size((yyvsp[(2) - (3)].str), (yyvsp[(2) - (3)].str)); }
    break;

  case 64:

/* Line 1455 of yacc.c  */
#line 571 "../../src/parser.yy"
    { (yyval.multiple_size) = new multiple_size((yyvsp[(2) - (4)].str), "0"); free((yyvsp[(2) - (4)].str)); }
    break;

  case 65:

/* Line 1455 of yacc.c  */
#line 572 "../../src/parser.yy"
    { (yyval.multiple_size) = new multiple_size("0", (yyvsp[(3) - (4)].str)); free((yyvsp[(3) - (4)].str)); }
    break;

  case 66:

/* Line 1455 of yacc.c  */
#line 573 "../../src/parser.yy"
    { (yyval.multiple_size) = new multiple_size((yyvsp[(2) - (5)].str), (yyvsp[(4) - (5)].str)); free((yyvsp[(2) - (5)].str)); free((yyvsp[(4) - (5)].str)); }
    break;



/* Line 1455 of yacc.c  */
#line 2212 "../../src/parser.cc"
      default: break;
    }
  YY_SYMBOL_PRINT ("-> $$ =", yyr1[yyn], &yyval, &yyloc);

  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);

  *++yyvsp = yyval;
  *++yylsp = yyloc;

  /* Now `shift' the result of the reduction.  Determine what state
     that goes to, based on the state we popped back to and the rule
     number reduced by.  */

  yyn = yyr1[yyn];

  yystate = yypgoto[yyn - YYNTOKENS] + *yyssp;
  if (0 <= yystate && yystate <= YYLAST && yycheck[yystate] == *yyssp)
    yystate = yytable[yystate];
  else
    yystate = yydefgoto[yyn - YYNTOKENS];

  goto yynewstate;


/*------------------------------------.
| yyerrlab -- here on detecting error |
`------------------------------------*/
yyerrlab:
  /* If not already recovering from an error, report this error.  */
  if (!yyerrstatus)
    {
      ++yynerrs;
#if ! YYERROR_VERBOSE
      yyerror (YY_("syntax error"));
#else
      {
	YYSIZE_T yysize = yysyntax_error (0, yystate, yychar);
	if (yymsg_alloc < yysize && yymsg_alloc < YYSTACK_ALLOC_MAXIMUM)
	  {
	    YYSIZE_T yyalloc = 2 * yysize;
	    if (! (yysize <= yyalloc && yyalloc <= YYSTACK_ALLOC_MAXIMUM))
	      yyalloc = YYSTACK_ALLOC_MAXIMUM;
	    if (yymsg != yymsgbuf)
	      YYSTACK_FREE (yymsg);
	    yymsg = (char *) YYSTACK_ALLOC (yyalloc);
	    if (yymsg)
	      yymsg_alloc = yyalloc;
	    else
	      {
		yymsg = yymsgbuf;
		yymsg_alloc = sizeof yymsgbuf;
	      }
	  }

	if (0 < yysize && yysize <= yymsg_alloc)
	  {
	    (void) yysyntax_error (yymsg, yystate, yychar);
	    yyerror (yymsg);
	  }
	else
	  {
	    yyerror (YY_("syntax error"));
	    if (yysize != 0)
	      goto yyexhaustedlab;
	  }
      }
#endif
    }

  yyerror_range[0] = yylloc;

  if (yyerrstatus == 3)
    {
      /* If just tried and failed to reuse lookahead token after an
	 error, discard it.  */

      if (yychar <= YYEOF)
	{
	  /* Return failure if at end of input.  */
	  if (yychar == YYEOF)
	    YYABORT;
	}
      else
	{
	  yydestruct ("Error: discarding",
		      yytoken, &yylval, &yylloc);
	  yychar = YYEMPTY;
	}
    }

  /* Else will try to reuse lookahead token after shifting the error
     token.  */
  goto yyerrlab1;


/*---------------------------------------------------.
| yyerrorlab -- error raised explicitly by YYERROR.  |
`---------------------------------------------------*/
yyerrorlab:

  /* Pacify compilers like GCC when the user code never invokes
     YYERROR and the label yyerrorlab therefore never appears in user
     code.  */
  if (/*CONSTCOND*/ 0)
     goto yyerrorlab;

  yyerror_range[0] = yylsp[1-yylen];
  /* Do not reclaim the symbols of the rule which action triggered
     this YYERROR.  */
  YYPOPSTACK (yylen);
  yylen = 0;
  YY_STACK_PRINT (yyss, yyssp);
  yystate = *yyssp;
  goto yyerrlab1;


/*-------------------------------------------------------------.
| yyerrlab1 -- common code for both syntax error and YYERROR.  |
`-------------------------------------------------------------*/
yyerrlab1:
  yyerrstatus = 3;	/* Each real token shifted decrements this.  */

  for (;;)
    {
      yyn = yypact[yystate];
      if (yyn != YYPACT_NINF)
	{
	  yyn += YYTERROR;
	  if (0 <= yyn && yyn <= YYLAST && yycheck[yyn] == YYTERROR)
	    {
	      yyn = yytable[yyn];
	      if (0 < yyn)
		break;
	    }
	}

      /* Pop the current state because it cannot handle the error token.  */
      if (yyssp == yyss)
	YYABORT;

      yyerror_range[0] = *yylsp;
      yydestruct ("Error: popping",
		  yystos[yystate], yyvsp, yylsp);
      YYPOPSTACK (1);
      yystate = *yyssp;
      YY_STACK_PRINT (yyss, yyssp);
    }

  *++yyvsp = yylval;

  yyerror_range[1] = yylloc;
  /* Using YYLLOC is tempting, but would change the location of
     the lookahead.  YYLOC is available though.  */
  YYLLOC_DEFAULT (yyloc, (yyerror_range - 1), 2);
  *++yylsp = yyloc;

  /* Shift the error token.  */
  YY_SYMBOL_PRINT ("Shifting", yystos[yyn], yyvsp, yylsp);

  yystate = yyn;
  goto yynewstate;


/*-------------------------------------.
| yyacceptlab -- YYACCEPT comes here.  |
`-------------------------------------*/
yyacceptlab:
  yyresult = 0;
  goto yyreturn;

/*-----------------------------------.
| yyabortlab -- YYABORT comes here.  |
`-----------------------------------*/
yyabortlab:
  yyresult = 1;
  goto yyreturn;

#if !defined(yyoverflow) || YYERROR_VERBOSE
/*-------------------------------------------------.
| yyexhaustedlab -- memory exhaustion comes here.  |
`-------------------------------------------------*/
yyexhaustedlab:
  yyerror (YY_("memory exhausted"));
  yyresult = 2;
  /* Fall through.  */
#endif

yyreturn:
  if (yychar != YYEMPTY)
     yydestruct ("Cleanup: discarding lookahead",
		 yytoken, &yylval, &yylloc);
  /* Do not reclaim the symbols of the rule which action triggered
     this YYABORT or YYACCEPT.  */
  YYPOPSTACK (yylen);
  YY_STACK_PRINT (yyss, yyssp);
  while (yyssp != yyss)
    {
      yydestruct ("Cleanup: popping",
		  yystos[*yyssp], yyvsp, yylsp);
      YYPOPSTACK (1);
    }
#ifndef yyoverflow
  if (yyss != yyssa)
    YYSTACK_FREE (yyss);
#endif
#if YYERROR_VERBOSE
  if (yymsg != yymsgbuf)
    YYSTACK_FREE (yymsg);
#endif
  /* Make sure YYID is used.  */
  return YYID (yyresult);
}



/* Line 1675 of yacc.c  */
#line 576 "../../src/parser.yy"


