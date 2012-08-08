/**
 * Copyright (C) 1999-2007  Free Software Foundation, Inc.
 *
 * This file is part of GNU gengetopt 
 *
 * GNU gengetopt is free software; you can redistribute it and/or modify 
 * it under the terms of the GNU General Public License as published by 
 * the Free Software Foundation; either version 3, or (at your option) 
 * any later version. 
 *
 * GNU gengetopt is distributed in the hope that it will be useful, but 
 * WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
 * Public License for more details. 
 *
 * You should have received a copy of the GNU General Public License along 
 * with gengetopt; see the file COPYING. If not, write to the Free Software 
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA. 
 */

#ifndef GENGETOPT_ARGSDEF_H
#define GENGETOPT_ARGSDEF_H

#define ARG_NO		0
#define ARG_FLAG	1
#define ARG_STRING	2
#define ARG_INT		3
#define ARG_SHORT	4
#define ARG_LONG	5
#define ARG_FLOAT	6
#define ARG_DOUBLE	7
#define ARG_LONGDOUBLE	8
#define ARG_LONGLONG	9
#define ARG_ENUM    10

/** corresponding strings for above defines */
extern const char * arg_type_constants [];
/** symbolic names for argument types */
extern const char * arg_names [];
/** corresponding C types */
extern const char * arg_types [];
/** string representation of types */
extern const char * arg_types_names [];

#define ARGS_STRUCT "args_info"

#endif /* GENGETOPT_ARGSDEF_H */

