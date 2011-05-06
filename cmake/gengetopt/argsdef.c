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

#include "argsdef.h"

const char * arg_names [] = { (const char*)0, (const char*)0, "STRING", "INT",
        "SHORT", "LONG", "FLOAT", "DOUBLE", "LONGDOUBLE", "LONGLONG", "ENUM" };

const char * arg_type_constants [] = { "ARG_NO", "ARG_FLAG", "ARG_STRING",
        "ARG_INT", "ARG_SHORT", "ARG_LONG", "ARG_FLOAT", "ARG_DOUBLE",
        "ARG_LONGDOUBLE", "ARG_LONGLONG", "ARG_ENUM" };

const char * arg_types [] = { (const char*)0, "int", "char *", "int", "short",
        "long", "float", "double", "long double", "long long int", "enum" };

const char * arg_types_names [] = { (const char*)0, "int", "string", "int",
        "short", "long", "float", "double", "longdouble", "longlong", "int" };

