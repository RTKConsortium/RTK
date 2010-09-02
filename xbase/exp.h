/*  $Id: exp.h,v 1.10 2003/08/20 01:53:27 gkunkel Exp $

    Xbase project source code 

    This file contains a header file for the EXP object, which is
    used for expression processing.

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

#ifndef __XB_EXP_H__
#define __XB_EXP_H__

#ifdef __GNUG__
#pragma interface
#endif

#include <xbase/xbase.h>

#ifdef XB_EXPRESSIONS             /* compile if expression logic on */

#define XB_EXPRESSION xbExpNode

#include <xbase/xtypes.h>
#include <xbase/xstack.h>

/*! \file exp.h
*/

#undef ABS
#undef MIN
#undef MAX

class XBDLLEXPORT xbDbf;

//! xbFuncDtl struct
/*!
*/

struct XBDLLEXPORT xbFuncDtl {
   const char * FuncName;     /* function name               */
   xbShort ParmCnt;                  /* no of parms it needs        */
   char    ReturnType;               /* return type of function     */
   void    (*ExpFuncPtr)();          /* pointer to function routine */
};

//! xbExpNode struct
/*!
*/

class XBDLLEXPORT xbExpNode {
public:
   char * NodeText;           /* expression text                 */
   char Type;                 /* same as TokenType below         */
   xbShort Len;                 /* length of expression text       */
   xbShort InTree;              /* this node in the tree? 1=yes    */
   xbExpNode * Node;            /* pointer to parent               */
   xbExpNode * Sibling1;        /* pointer to sibling 1            */
   xbExpNode * Sibling2;        /* pointer to sibling 2            */
   xbExpNode * Sibling3;        /* pointer to sibling 3            */

   xbShort  DataLen;            /* length of data in result buffer */
   xbShort  ResultLen;          /* length of result buffer         */
//   char * Result;             /* result buffer - ptr to result   */
   xbString StringResult;
   xbDouble DoubResult;         /* Numeric Result                  */
   xbShort  IntResult;          /* logical result                  */

   xbDbf *  dbf;                /* pointer to datafile             */
   xbShort  FieldNo;            /* field no if DBF field           */
   char   ExpressionType;     /* used in head node C,N,L or D    */


  public:
   xbExpNode();
   virtual ~xbExpNode();
};

//! xbExpn class
/*!
*/

/* Expression handler */

class XBDLLEXPORT xbExpn : public xbStack, public xbDate {
public:
   xbShort ProcessExpression( xbExpNode *, xbShort );
   xbExpNode * GetTree( void ) { return Tree; }
   void SetTreeToNull( void ) { Tree = NULL; }
   xbExpNode * GetFirstTreeNode( xbExpNode * );

   xbExpn( void );
   virtual ~xbExpn();

   xbShort  GetNextToken( const char *s, xbShort MaxLen);

   /* expression methods */
   xbDouble ABS( xbDouble );
   xbLong   ASC( const char * );
   xbLong   AT( const char *, const char * );
   char *   CDOW( const char * );
   char *   CHR( xbLong );
   char *   CMONTH( const char * );
   char *   DATE();
   xbLong   DAY( const char * );
   xbLong   DESCEND( const char * );
   xbLong   DOW( const char * );
   char *   DTOC( const char * );
   char *   DTOS( const char * );
   xbDouble EXP( xbDouble );
   xbLong   INT( xbDouble );
   xbLong   ISALPHA( const char * );
   xbLong   ISLOWER( const char * );
   xbLong   ISUPPER( const char * );
   char *   LEFT( const char *, xbShort );
   xbLong   LEN( const char * );
   xbDouble LOG( xbDouble );
   char *   LOWER( const char * );
   char *   LTRIM( const char * );
   xbDouble MAX( xbDouble, xbDouble );
   xbLong   MONTH( const char * );         /* MONTH() */
   xbDouble MIN( xbDouble, xbDouble );
   char *   RECNO( xbULong );
   xbLong   RECNO( xbDbf * );
   char *   REPLICATE( const char *, xbShort );
   char *   RIGHT( const char *, xbShort );
   char *   RTRIM( const char * );
   char *   SPACE( xbShort );   
   xbDouble SQRT( xbDouble );
   char *   STR( const char * );
   char *   STR( const char *, xbShort );
   char *   STR( const char *, xbShort, xbShort );
   char *   STR( xbDouble );
   char *   STR( xbDouble, xbShort );
   char *   STR(xbDouble, xbUShort length, xbShort numDecimals );
   char *   STRZERO( const char * );
   char *   STRZERO( const char *, xbShort );
   char *   STRZERO( const char *, xbShort, xbShort );
   char *   STRZERO( xbDouble );
   char *   STRZERO( xbDouble, xbShort );
   char *   STRZERO( xbDouble, xbShort, xbShort );
   char *   SUBSTR( const char *, xbShort, xbShort );
   char *   TRIM( const char * );
   char *   UPPER( const char * );
   xbLong   VAL( const char * );
   xbLong   YEAR( const char * );  
   //! Short description.
   /*!
     \param f
   */
   void     SetDefaultDateFormat(const xbString & f){ DefaultDateFormat = f; }

   xbString GetDefaultDateFormat() const { return DefaultDateFormat; }
   xbShort  ProcessExpression( const char *exp, xbDbf * d );
   xbShort  ParseExpression( const char *exp, xbDbf * d );
   XB_EXPRESSION * GetExpressionHandle();
   char     GetExpressionResultType(XB_EXPRESSION * );
   char *   GetCharResult();
   xbString & GetStringResult();
   xbDouble GetDoubleResult();
   xbLong   GetIntResult();
   xbShort  ProcessExpression( xbExpNode * );
   xbShort  BuildExpressionTree( const char * Expression, xbShort MaxTokenLen,
            xbDbf *d );

#ifdef XBASE_DEBUG
   void DumpExpressionTree( xbExpNode * );
   void DumpExpNode( xbExpNode * );
#endif

protected:
   xbFuncDtl *XbaseFuncList;    /* pointer to list of Xbase functions    */
//   xbExpNode *NextFreeExpNode;  /* pointer to chain of free nodes        */
   xbExpNode *Tree;
   xbShort LogicalType;         /* set to 1 for logical type nodes       */

   char TokenType;            /* E - Expression, not in simplest form  */
                              /* C - Constant                          */
                              /* N - Numeric Constant                  */
                              /* O - Operator                          */
                              /* F - Function                          */
                              /* D - Database Field                    */
                              /* s - character string result           */
                              /* l - logical or short int result       */
                              /* d - double result                     */

   char  PreviousType;         /* used to see if "-" follows operator     */
   char  *  Op1;               /* pointer to operand 1                    */
   char  *  Op2;               /* pointer to operand 2                    */
   xbDouble Opd1;              /* double result 1                         */
   xbDouble Opd2;              /* double result 2                         */
   xbShort OpLen1;             /* length of memory allocated to operand 1 */
   xbShort OpLen2;             /* length of memory allocated to operand 2 */
   xbShort OpDataLen1;         /* length of data in op1                   */
   xbShort OpDataLen2;         /* length of data in op2                   */

   char    OpType1;            /* type of operand 1                       */
   char    OpType2;            /* type of operand 2                       */
   xbShort TokenLen;           /* length of token                         */

   static xbString DefaultDateFormat;  /*default date format for DTOC func*/

   enum { WorkBufMaxLen = 200 };
   char  WorkBuf[WorkBufMaxLen+1];

   xbShort  IsWhiteSpace( char );
   char     IsSeparator( char );
   xbExpNode * LoadExpNode( const char * ENodeText, const char EType,
            const xbShort ELen, const xbShort BufLen );
   xbShort  OperatorWeight( const char *Oper, xbShort len );
   xbShort  ReduceComplexExpression( const char * NextToken, xbShort Len,
            xbExpNode * cn, xbDbf *d );
   xbShort  GetFunctionTokenLen( const char *s );
   xbShort  ReduceFunction( const char *NextToken, xbExpNode *cn, xbDbf *d );
   xbExpNode * GetNextTreeNode( xbExpNode * );
   xbShort  ProcessOperator( xbShort );
   xbShort  ProcessFunction( char * );
   xbShort  ValidOperation( char *, char, char );
   char     GetOperandType( xbExpNode * );
   xbShort  AlphaOperation( char * );
   xbShort  NumericOperation( char * );
   xbExpNode * GetExpNode( xbShort );
   xbShort  GetFuncInfo( const char *Function, xbShort Option );
   xbDouble GetDoub( xbExpNode * );
   xbLong   GetInt( xbExpNode * );
};
#endif               // XB_EXPRESSIONS
#endif               // __XB_EXP_H__
