
/* A Bison parser, made by GNU Bison 2.4.1.  */

/* Skeleton interface for Bison's Yacc-like parsers in C
   
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

/* Line 1676 of yacc.c  */
#line 148 "../../src/parser.yy"

    char   *str;
    char    chr;
    int	    argtype;
    int	    boolean;
    class AcceptedValues *ValueList;
    struct gengetopt_option *gengetopt_option;
    struct multiple_size *multiple_size;



/* Line 1676 of yacc.c  */
#line 138 "../../src/parser.h"
} YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define yystype YYSTYPE /* obsolescent; will be withdrawn */
# define YYSTYPE_IS_DECLARED 1
#endif

extern YYSTYPE yylval;

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

extern YYLTYPE yylloc;

