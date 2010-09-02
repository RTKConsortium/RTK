/*  $Id: xstack.h,v 1.7 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file conatains a header file for the xbStack object which
    is used for handling expressions.

    Copyright (C) 1997  Gary A. Kunkel

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public
    License along with this library; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

    Contact:

      Mail:

        Technology Associates, Inc.
        XBase Project
        1455 Deming Way #11
        Sparks, NV 89434
        USA

      Email:

        xbase@techass.com
        xdb-devel@lists.sourceforge.net
        xdb-users@lists.sourceforge.net

      See our website at:

        xdb.sourceforge.net

*/

/*! \file xstack.h
*/

#ifndef __XB_STACK_H__
#define __XB_STACK_H__

#ifdef __GNUG__
#pragma interface
#endif

#include <xbase/xtypes.h>

//! xbStackElement class
/*!
*/
class XBDLLEXPORT xbStackElement
{
protected:
  xbStackElement *Previous;
  xbStackElement *Next;
  void *UserPtr;

public:
  xbStackElement();
  ~xbStackElement();

  friend class xbStack;
};

//! xbStack class
/*!
*/
class XBDLLEXPORT xbStack
{
public:
  xbStack(void);
  virtual ~xbStack();

  void InitStack();
  void *Pop();
  xbShort Push(void *);
   //! Short description.
   /*!
   */
  xbShort GetStackDepth( void ) { return StackDepth; }
  void    DumpStack( void );

protected:
  xbShort StackDepth;
  xbStackElement *First;
  xbStackElement *Last;
};

#endif               // __XB_STACK_H__
