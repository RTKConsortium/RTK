/*  $Id: index.h,v 1.8 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains a header file for the NTX object, which is used
    for handling NTX type indices. NTX are the Clipper equivalant of xbNdx
    files.

    Copyright (C) 1998  SynXis Corp., Bob Cotton

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

      See our website at:

        xdb.sourceforge.net

*/

#ifndef __XB_INDEX_H__
#define __XB_INDEX_H__

#ifdef __GNUG__
#pragma interface
#endif

#include <xbase/xbase.h>
#include <string.h>

/*! \file index.h
*/

#define XB_UNIQUE     1
#define XB_NOT_UNIQUE 0

//! xbIndex class
/*!
*/

class XBDLLEXPORT xbIndex
{
 public:
    xbIndex *index;
    xbDbf *dbf;
    xbExpNode *ExpressionTree;

    xbString IndexName;
    FILE *indexfp;

    int IndexStatus;            /* 0 = closed, 1 = open */

    xbULong  CurDbfRec;     /* current Dbf record number */
    char  *KeyBuf;               /* work area key buffer */
    char  *KeyBuf2;              /* work area key buffer */

#ifdef XB_LOCKING_ON
protected:
    int CurLockCount;
    int CurLockType;
#endif

    xbShort NodeSize;

public:
    xbIndex() {}
    xbIndex(xbDbf *);

    virtual ~xbIndex() {}

    virtual xbShort  OpenIndex ( const char * ) = 0;
    virtual xbShort  CloseIndex() = 0;
#ifdef XBASE_DEBUG
    virtual void     DumpHdrNode() = 0;
    virtual void     DumpNodeRec( xbLong ) = 0;
    virtual void     DumpNodeChain() = 0;
    virtual xbShort  CheckIndexIntegrity( const xbShort ) = 0;
#endif
    virtual xbShort  CreateIndex( const char *, const char *, xbShort, xbShort ) = 0;
    virtual xbLong   GetTotalNodes() = 0;
    virtual xbULong   GetCurDbfRec() = 0;
    virtual xbShort  CreateKey( xbShort, xbShort ) = 0;
    virtual xbShort  GetCurrentKey(char *key) = 0;
    virtual xbShort  AddKey( xbLong ) = 0;
    virtual xbShort  UniqueIndex() = 0;
    virtual xbShort  DeleteKey( xbLong ) = 0;
    virtual xbShort  KeyWasChanged() = 0;
    virtual xbShort  FindKey( const char * ) = 0;
    virtual xbShort  FindKey() = 0;
    virtual xbShort  FindKey( xbDouble ) = 0;
    virtual xbShort  GetNextKey() = 0;
    virtual xbShort  GetLastKey() = 0;
    virtual xbShort  GetFirstKey() = 0;
    virtual xbShort  GetPrevKey() = 0;
    virtual xbShort  ReIndex(void (*statusFunc)(xbLong itemNum, xbLong numItems) = 0) = 0;
//   virtual xbShort  KeyExists( char * Key ) { return FindKey( Key, strlen( Key ), 0 ); }
    virtual xbShort  KeyExists( xbDouble ) = 0;

#ifdef XB_LOCKING_ON
    virtual xbShort  LockIndex( const xbShort, const xbShort );
#else
    virtual xbShort  LockIndex( const xbShort, const xbShort ) const { return XB_NO_ERROR; }
#endif

    virtual xbShort TouchIndex( void ) { return XB_NO_ERROR; }

    virtual void    SetNodeSize(xbShort size) {}
    virtual xbShort GetNodeSize(void) { return NodeSize; }

    virtual void    GetExpression(char *buf, int len) = 0;

    virtual void    Flush();
};


#endif /* __XB_INDEX_H__ */
