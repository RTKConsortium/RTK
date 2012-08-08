/**
 * Copyright (C) 1999, 2000, 2001, 2002, 2003  Free Software Foundation, Inc.
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

#ifndef GGO_GROUPS_H
#define GGO_GROUPS_H

#include "my_string.h"
#include "my_map.h"

/**
 * Represents a group of options
 */
struct Group
{
  string desc;
  bool required;

  Group(const string &s, bool r) : desc (s), required (r) {}
};

typedef map<string,Group> groups_collection_t;

/**
 * Represents a mode of options
 */
struct Mode
{
  string desc;

  Mode(const string &s) : desc (s) {}
};

typedef map<string,Mode> modes_collection_t;

#endif // GROUPS_H
