/*  $Id: ntx.h,v 1.9 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains a header file for the xbNdx object, which is used
    for handling xbNdx type indices.

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

#ifndef __XB_NTX_H__
#define __XB_NTX_H__

#ifdef __GNUG__
#pragma interface
#endif

#include <xbase/xbase.h>
#include <string.h>

/*! \file ntx.h
*/

#define XB_NTX_NODE_SIZE 1024

//! xbNtxHeadNode struct
/*!
*/

struct NtxHeadNode {       /* ntx header on disk */
    xbUShort Signature;           /* Clipper 5.x or Clipper 87 */
    xbUShort Version;             /* Compiler Version */
                                /* Also turns out to be a last modified counter */
    xbLong   StartNode;       /* Offset in file for first index */
    xbULong  UnusedOffset;        /* First free page offset */
    xbUShort KeySize;             /* Size of items (KeyLen + 8) */
    xbUShort KeyLen;              /* Size of the Key */
    xbUShort DecimalCount;        /* Number of decimal positions */
    xbUShort KeysPerNode;         /* Max number of keys per page */
    xbUShort HalfKeysPerNode;     /* Min number of keys per page */
    char KeyExpression[256];    /* Null terminated key expression */
    unsigned  Unique;              /* Unique Flag */
    char NotUsed[745];
};

//! xbNtxLeafNode struct
/*!
*/

struct NtxLeafNode {       /* ndx node on disk */
    xbUShort NoOfKeysThisNode;
    char     KeyRecs[XB_NTX_NODE_SIZE];
};


//! xbNtxItem struct
/*!
*/

struct NtxItem
{
    xbULong Node;
    xbULong RecordNumber;
    char Key[256];
};

//! xbNtxNodeLink struct
/*!
*/

struct xbNodeLink {        /* ndx node memory */
   xbNodeLink * PrevNode;
   xbNodeLink * NextNode;
   xbUShort       CurKeyNo;                 /* 0 - KeysPerNode-1 */
   xbLong       NodeNo;
   struct NtxLeafNode Leaf;
    xbUShort *offsets;
};

//! xbNtx class
/*!
*/

class XBDLLEXPORT xbNtx : public xbIndex
{
protected:
   NtxHeadNode HeadNode;
   NtxLeafNode LeafNode;
   xbLong NodeLinkCtr;
   xbLong ReusedNodeLinks;

   char  Node[XB_NTX_NODE_SIZE];

   xbNodeLink * NodeChain;        /* pointer to node chain of index nodes */
   xbNodeLink * FreeNodeChain;    /* pointer to chain of free index nodes */
   xbNodeLink * CurNode;          /* pointer to current node              */
   xbNodeLink * DeleteChain;      /* pointer to chain to delete           */
   xbNodeLink * CloneChain;       /* pointer to node chain copy (add dup) */

   NtxItem PushItem;

/* private functions */
   xbLong     GetLeftNodeNo( xbShort, xbNodeLink * );
   xbShort    CompareKey( const char *, const char *, xbShort );
   xbShort    CompareKey( const char *, const char * );
   xbLong     GetDbfNo( xbShort, xbNodeLink * );
   char *   GetKeyData( xbShort, xbNodeLink * );
   xbUShort   GetItemOffset ( xbShort, xbNodeLink *, xbShort );
   xbUShort   InsertKeyOffset ( xbShort, xbNodeLink * );
   xbUShort   GetKeysPerNode( void );
   xbShort    GetHeadNode( void );
   xbShort    GetLeafNode( xbLong, xbShort );
   xbNodeLink * GetNodeMemory( void );
   xbLong    GetNextNodeNo( void );
   void     ReleaseNodeMemory(xbNodeLink *n, bool doFree = false);
   xbULong     GetLeafFromInteriorNode( const char *, xbShort );
   xbShort    CalcKeyLen( void );
   xbShort    PutKeyData( xbShort, xbNodeLink * );
   xbShort    PutLeftNodeNo( xbShort, xbNodeLink *, xbLong );
   xbShort    PutLeafNode( xbLong, xbNodeLink * );
   xbShort    PutHeadNode( NtxHeadNode *, FILE *, xbShort );
   xbShort    TouchIndex( void );
   xbShort    PutDbfNo( xbShort, xbNodeLink *, xbLong );
   xbShort    PutKeyInNode( xbNodeLink *, xbShort, xbLong, xbLong, xbShort );
   xbShort    SplitLeafNode( xbNodeLink *, xbNodeLink *, xbShort, xbLong );
   xbShort    SplitINode( xbNodeLink *, xbNodeLink *, xbLong );
   xbShort    AddToIxList( void );
   xbShort    RemoveFromIxList( void );
   xbShort    RemoveKeyFromNode( xbShort, xbNodeLink * );
   xbShort    DeleteKeyFromNode( xbShort, xbNodeLink * );
   xbShort    JoinSiblings(xbNodeLink *, xbShort, xbNodeLink *, xbNodeLink *);
   xbUShort   DeleteKeyOffset( xbShort, xbNodeLink *);
   xbShort    FindKey( const char *, xbShort, xbShort );
   xbShort    UpdateParentKey( xbNodeLink * );
   xbShort    GetFirstKey( xbShort );
   xbShort    GetNextKey( xbShort );
   xbShort    GetLastKey( xbLong, xbShort );
   xbShort    GetPrevKey( xbShort );
   void     UpdateDeleteList( xbNodeLink * );
   void     ProcessDeleteList( void );
//    xbNodeLink * LeftSiblingHasSpace( xbNodeLink * );
//    xbNodeLink * RightSiblingHasSpace( xbNodeLink * );
//    xbShort    DeleteSibling( xbNodeLink * );
//    xbShort    MoveToLeftNode( xbNodeLink *, xbNodeLink * );
//    xbShort    MoveToRightNode( xbNodeLink *, xbNodeLink * );
   xbShort    FindKey( const char *, xbLong );         /* for a specific dbf no */

   xbShort    CloneNodeChain( void );          /* test */
   xbShort    UncloneNodeChain( void );        /* test */

public:
   xbNtx();
   xbNtx(xbDbf *);
   virtual ~xbNtx();

/* note to gak - don't uncomment next line - it causes seg faults */
//   ~NTX() { if( NtxStatus ) CloseIndex(); }

   xbShort  OpenIndex ( const char * );
   xbShort  CloseIndex( void );
   void   DumpHdrNode  ( void );
   void   DumpNodeRec  ( xbLong );
   xbShort  CreateIndex( const char *, const char *, xbShort, xbShort );
   xbLong   GetTotalNodes( void );
   xbULong  GetCurDbfRec( void ) { return CurDbfRec; }
   void   DumpNodeChain( void );
   xbShort  CreateKey( xbShort, xbShort );
   xbShort  GetCurrentKey(char *key);
   xbShort  AddKey( xbLong );
   xbShort  UniqueIndex( void ) { return HeadNode.Unique; }
   xbShort  DeleteKey( xbLong DbfRec );
   xbShort  KeyWasChanged( void );
   xbShort  FindKey( const char * );
   xbShort  FindKey( void );
   xbShort  FindKey( xbDouble );
#ifdef XBASE_DEBUG
   xbShort  CheckIndexIntegrity( const xbShort Option );
#endif
   xbShort  GetNextKey( void )  { return GetNextKey( 1 ); }
   xbShort  GetLastKey( void )  { return GetLastKey( 0, 1 ); }
   xbShort  GetFirstKey( void ) { return GetFirstKey( 1 ); }
   xbShort  GetPrevKey( void )  { return GetPrevKey( 1 ); }
   xbShort  ReIndex(void (*statusFunc)(xbLong itemNum, xbLong numItems) = 0) ;
   xbShort  KeyExists( char * Key ) { return FindKey( Key, strlen( Key ), 0 ); }
   xbShort  KeyExists( xbDouble );

   xbShort AllocKeyBufs(void);

   virtual void GetExpression(char *buf, int len);
};
#endif      /* __XB_NTX_H__ */
