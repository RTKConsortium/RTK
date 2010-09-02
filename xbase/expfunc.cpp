/*  $Id: expfunc.cpp,v 1.9 2003/08/20 01:53:27 gkunkel Exp $

    Xbase project source code

    This file contains logic for handling Xbase expressions.

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

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xbase.h>
#ifdef XB_EXPRESSIONS

#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include <xbase/exp.h>
#include <xbase/xbexcept.h>


/*! \file expfunc.cpp
*/

/*************************************************************************/
//! Short description.
/*!
  \param Func
*/
xbShort xbExpn::ProcessFunction( char * Func )
{
/* 1 - pop function from stack
   2 - verify function name and get no of parms needed 
   3 - verify no of parms >= remainder of stack
   4 - pop parms off stack
   5 - execute function
   6 - push result back on stack
*/


   char   *buf = 0;
   xbExpNode *p1, *p2, *p3, *WorkNode, *FuncNode;
   xbShort  ParmsNeeded,len;
   char   ptype = 0;  /* process type s=string, l=logical, d=double */
   xbDouble DoubResult = 0;
   xbLong   IntResult = 0;
   FuncNode = (xbExpNode *) Pop();

   ParmsNeeded = GetFuncInfo( Func, 1 );

   if( ParmsNeeded == -1 ) {
     xb_error(XB_INVALID_FUNCTION);
   }
   else {
     ParmsNeeded = 0;
     if( FuncNode->Sibling1 ) ParmsNeeded++;
     if( FuncNode->Sibling2 ) ParmsNeeded++;
     if( FuncNode->Sibling3 ) ParmsNeeded++;
   }

   if (ParmsNeeded > GetStackDepth())
     xb_error(XB_INSUFFICIENT_PARMS);

   p1 = p2 = p3 = NULL;
   if( ParmsNeeded > 2 ) p3 = (xbExpNode *) Pop(); 
   if( ParmsNeeded > 1 ) p2 = (xbExpNode *) Pop(); 
   if( ParmsNeeded > 0 ) p1 = (xbExpNode *) Pop(); 
   memset( WorkBuf, 0x00, WorkBufMaxLen+1);

   if( strncmp( Func, "ABS", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = ABS( GetDoub( p1 ));
   }
   else if( strncmp( Func, "ASC", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = ASC( p1->StringResult );
   }
   else if( strncmp( Func, "AT", 2 ) == 0 ) {  
      ptype = 'd';
      DoubResult = AT( p1->StringResult, p2->StringResult );
   }
   else if( strncmp( Func, "CDOW", 4 ) == 0 ) {  
      ptype = 's';
      buf = CDOW( p1->StringResult );
   }
   else if( strncmp( Func, "CHR", 3 ) == 0 ) {  
      ptype = 's';
      buf = CHR( GetInt( p1 ));
   }
   else if( strncmp( Func, "CMONTH", 6 ) == 0 ) {  
      ptype = 's';
      buf = CMONTH( p1->StringResult );
   }
   else if( strncmp( Func, "DATE", 4 ) == 0 ) {  
      ptype = 's';
      buf = DATE();
   }
   else if( strncmp( Func, "DAY", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = DAY( p1->StringResult );
   }
   else if( strncmp( Func, "DESCEND", 7 ) == 0 ) {  
      ptype = 'd';
      DoubResult = DESCEND( p1->StringResult );
   }
   else if( strncmp( Func, "DOW", 3 ) == 0 ) {
      ptype = 'd';
      DoubResult = DOW( p1->StringResult );
   }
   else if( strncmp( Func, "DTOC", 4 ) == 0 ) {  
      ptype = 's';
      buf = DTOC( p1->StringResult );
   }
   else if( strncmp( Func, "DTOS", 4 ) == 0 ) {  
      ptype = 's';
      buf = DTOS( p1->StringResult );
   }
   else if( strncmp( Func, "EXP", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = EXP( GetDoub( p1 ));
   }
   else if( strncmp( Func, "INT", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = INT( GetDoub( p1 ));
   }
   else if( strncmp( Func, "ISALPHA", 7 ) == 0 ) {  
      ptype = 'l';
      IntResult = ISALPHA( p1->StringResult );
   }
   else if( strncmp( Func, "ISLOWER", 7 ) == 0 ) {  
      ptype = 'l';
      IntResult = ISLOWER( p1->StringResult );
   }
   else if( strncmp( Func, "ISUPPER", 7 ) == 0 ) {  
      ptype = 'l';
      IntResult = ISUPPER( p1->StringResult );
   }
   else if( strncmp( Func, "LEN", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = LEN( p1->StringResult );
   }
   else if( strncmp( Func, "LEFT", 4 ) == 0 ) {  
      ptype = 's';
      buf = LEFT( p1->StringResult, INT( p2->DoubResult ));
   }
   else if( strncmp( Func, "LTRIM", 5 ) == 0 ) {  
      ptype = 's';
      buf = LTRIM( p1->StringResult );
   }  
   else if( strncmp( Func, "LOG", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = LOG( GetDoub( p1 ));
   }  
   else if( strncmp( Func, "LOWER", 5 ) == 0 ) {  
      ptype = 's';
      buf = LOWER( p1->StringResult );
   }  
   else if( strncmp( Func, "MAX", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = MAX( GetDoub( p1 ), GetDoub( p2 ));
   }  
   else if( strncmp( Func, "MIN", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = MIN( GetDoub( p1 ), GetDoub( p2 ));
   }  
   else if( strncmp( Func, "MONTH", 5 ) == 0 ) {  
      ptype = 'd';
      DoubResult = MONTH( p1->StringResult );
   } 
   else if( strncmp( Func, "RECNO", 5 ) == 0 ) {
      ptype = 'l';
      IntResult = RECNO( FuncNode->dbf );
   }

//  else if( strncmp( Func, "RECNO", 5 ) == 0 )
//  {
//      ptype = 'd';
//      DoubResult = FuncNode->dbf->GetCurRecNo();
//   }

   else if( strncmp( Func, "REPLICATE", 9 ) == 0 ) {
      ptype = 's';
      buf = REPLICATE( p1->StringResult, GetInt( p2 ));
   }
   else if( strncmp( Func, "RIGHT", 5 ) == 0 ) {
      ptype = 's';
      buf = RIGHT( p1->StringResult, GetInt( p2 ));
   }
   else if( strncmp( Func, "RTRIM", 5 ) == 0 ) {  
      ptype = 's';
      buf = RTRIM( p1->StringResult );
   }  
   else if( strncmp( Func, "SPACE", 5 ) == 0 ) {  
      ptype = 's';
      buf = SPACE( INT( GetDoub( p1 )));
   }
   else if( strncmp( Func, "SQRT", 4 ) == 0 ) {  
      ptype = 'd';
      DoubResult = SQRT( GetDoub( p1 ));
   }
   else if( strncmp( Func, "STRZERO", 7 ) == 0 && ParmsNeeded == 1 ) {
      ptype = 's';
      buf = STRZERO( p1->StringResult );
   }   
   else if( strncmp( Func, "STRZERO", 7 ) == 0 && ParmsNeeded == 2 ) {
      ptype = 's';
      buf = STRZERO( p1->StringResult, GetInt( p2 ));
   }   
   else if( strncmp( Func, "STRZERO", 7 ) == 0 && ParmsNeeded == 3 ) {
      ptype = 's';
      buf = STRZERO( p1->StringResult, GetInt( p2 ), GetInt( p3 ));
   }   
   else if( strncmp( Func, "STR", 3 ) == 0 && ParmsNeeded == 1) {
      ptype = 's';
      buf = STR( p1->StringResult );
   }   
   else if( strncmp( Func, "STR", 3 ) == 0 && ParmsNeeded == 2) {
      ptype = 's';
      buf = STR( p1->StringResult, GetInt( p2 ));
   }
   else if( strncmp( Func, "STR", 3 ) == 0 && ParmsNeeded == 3) {
      ptype = 's';
      if(p1->ExpressionType == 'N')
        buf = STR( p1->DoubResult, GetInt( p2 ), GetInt( p3 ));
      else
        buf = STR( p1->StringResult, GetInt( p2 ), GetInt( p3 ));
   }   
   else if( strncmp( Func, "SUBSTR", 6 ) == 0 ) {
      ptype = 's';
      buf = SUBSTR( p1->StringResult, GetInt( p2 ), GetInt( p3 )); 
   }
   else if( strncmp( Func, "TRIM", 4 ) == 0 ) {  
      ptype = 's';
      buf = TRIM( p1->StringResult );
   }  
   else if( strncmp( Func, "UPPER", 5 ) == 0 ) {  
      ptype = 's';
      buf = UPPER( p1->StringResult );
   }  
   else if( strncmp( Func, "VAL", 3 ) == 0 ) {  
      ptype = 'd';
      DoubResult = VAL( p1->StringResult );
   }  
   else if( strncmp( Func, "YEAR", 4 ) == 0 ) {  
      ptype = 'd';
      DoubResult = YEAR( p1->StringResult );
   }  
   if( p1 && !p1->InTree ) delete p1;
   if( p2 && !p2->InTree ) delete p2;
   if( p3 && !p3->InTree ) delete p3;
   if( !FuncNode->InTree ) delete FuncNode;
   if( buf ){
    len = strlen( buf );
    if(( WorkNode = GetExpNode( len+1 )) == NULL ) xb_memory_error;
   }
   else{    
    len = 0;
    if(( WorkNode = GetExpNode( 0 )) == NULL ) xb_memory_error;
   }

   switch( ptype ) {
      case 's':
      WorkNode->DataLen = len;
      WorkNode->ExpressionType = 'C';
      WorkNode->Type = 's';
      WorkNode->StringResult = buf;
      break;

      case 'd':
      WorkNode->DataLen = 0;
      WorkNode->ExpressionType = 'N';
      WorkNode->Type = 'd';
      WorkNode->DoubResult = DoubResult;
      break;

      case 'l':
      WorkNode->DataLen = 0;
      WorkNode->ExpressionType = 'L';
      WorkNode->Type = 'l';
      WorkNode->IntResult = IntResult;
      break;

      default:
      std::cout << "\nInternal error. " << ptype;
      break;
   }
   Push(WorkNode);
   return XB_NO_ERROR;
}
/*************************************************************************/
//! Short description.
/*!
*/
xbString & xbExpn::GetStringResult()
{
  xbString *s = 0;
  xbExpNode *e;
  if( GetStackDepth() < 1 ) return *s;
  e = (xbExpNode *) Pop();
  s = &e->StringResult;
  Push(e);
  return *s;
}
/*************************************************************************/
//! Short description.
/*!
*/
xbLong xbExpn::GetIntResult( void )
{
   xbLong l;
   xbExpNode * e;
   if( GetStackDepth() < 1 ) return 0L;
   e = (xbExpNode *) Pop();
   l = e->IntResult;
   Push(e);
   return l;
}
/*************************************************************************/
//! Short description.
/*!
*/
xbDouble xbExpn::GetDoubleResult( void )
{
   xbDouble d;
   xbExpNode * e;
   if( GetStackDepth() < 1 ) return (xbDouble) 0;
   e = (xbExpNode *) Pop();
   d = e->DoubResult;
   Push(e);
   return d;
}
/*************************************************************************/
//! Short description.
/*!
  \param p
*/
xbDouble xbExpn::GetDoub( xbExpNode * p )
{
   if( p->Type == 'd' )
      return p->DoubResult;
   else if( p->Type == 'N' || p->Type == 's' )
      return( strtod( p->StringResult, NULL ));
   else if( p->Type == 'D' )
      return( p->dbf->GetDoubleField( p->FieldNo ));
   else
      return 0;
}
/*************************************************************************/
//! Short description.
/*!
  \param p
*/
xbLong xbExpn::GetInt( xbExpNode *p )
{
   if( p->Type == 'l' || p->Type == 'i' )
      return p->IntResult;
   else if( p->Type == 'N' || p->Type == 's' )
      return atoi( p->StringResult );
   else if( p->Type == 'D' )
      return p->dbf->GetLongField( p->FieldNo );
   else
      return 0L;
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
xbDouble xbExpn::ABS( xbDouble d )
{
   if( d < (xbDouble) 0 )
      return d * -1;
   else
      return d;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
xbLong xbExpn::ASC( const char * String )
{
   return *String;
}
/*************************************************************************/
//! Short description.
/*!
  \param s1
  \param s2
*/
xbLong xbExpn::AT( const char * s1, const char *s2 )
{
   /* looks for s1 in s2 */
   xbLong cnt;
   const char *p;
   if( strlen( s1 ) > strlen( s2 )) return 0;
   if(( p = strstr( s2, s1 )) == NULL )
      return 0;
   cnt = 1;
   while( s2++ != p ) cnt++;
   return cnt;
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
char * xbExpn::CDOW( const char * Date8 )
{
   static char buf[10];
   xbShort len,i;
   strcpy( buf, FormatDate( "DDDD", Date8 )); 
   len = strlen( buf );
   for( i = len; i < 9; i++ ) buf[i] = 0x20;
   buf[9] = 0x00;
   return buf;
}
/*************************************************************************/
//! Short description.
/*!
  \param l
*/
char * xbExpn::CHR( xbLong l )
{
   static char buf[2];
   xbShort i;
   i = (xbShort) l;
   buf[0] = i;
   buf[1] = 0x00;
   return buf;
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
char * xbExpn::CMONTH( const char * Date8 )
{
   static char buf[10];
   xbShort len,i;
   strcpy( buf, FormatDate( "MMMM", Date8 ));
   len = strlen( buf );
   for( i = len; i < 9; i++ ) buf[i] = 0x20;
   buf[9] = 0x00;
   return buf;
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
xbLong xbExpn::DAY( const char * Date8 )
{
   return DayOf( XB_FMT_MONTH, Date8 );
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
xbLong xbExpn::DESCEND( const char * Date8 )
{

/* this function may be broken, went thru major code merge and couldn't
   figure out which code was correct - think it is ok */

  xbDate x;
  xbLong l;
  l = x.JulianDays( "29991231" ) - x.JulianDays( Date8 );
  return 2415021 + l;
 
// This is for a string - but it doesn't work yet
// int i,j;
// for(i = 7, j = 0; j < 8; i--, j++ ) 
//   WorkBuf[j] = Date8[i];
// WorkBuf[8] = 0x00;
//   return DayOf( XB_FMT_MONTH, Date8 );
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
xbLong xbExpn::DOW( const char * Date8 )
{
   return DayOf( XB_FMT_WEEK, Date8 );
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
char * xbExpn::DTOC( const char * Date8 )
{
   strcpy( WorkBuf, FormatDate( GetDefaultDateFormat(), Date8 ));
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
char * xbExpn::DTOS( const char * Date8 )
{
   strcpy( WorkBuf, FormatDate( "YYYYMMDD", Date8 ));
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
xbDouble xbExpn::EXP( xbDouble d )
{
   return exp( d );
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
xbLong xbExpn::INT( xbDouble d )
{
   return (xbLong) d;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
xbLong xbExpn::ISALPHA( const char * String )
{
   if( isalpha(*String) ) return 1;
   else return 0;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
xbLong xbExpn::ISLOWER( const char * String )
{
   if( islower(*String) ) return 1;
   else return 0;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
xbLong xbExpn::ISUPPER( const char * String )
{
   if( isupper(*String) ) return 1;
   else return 0;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
xbLong xbExpn::LEN( const char * String )
{
   xbLong len;
   len = strlen( String );
   len--;
   while( len >= 0 && String[len] == 0x20 ) len--;
   return ++len;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param Len
*/
char * xbExpn::LEFT( const char * String, xbShort Len )
{
   xbShort i;
   for( i = 0; i < Len && i < 100; i++ )
      WorkBuf[i] = String[i];
   WorkBuf[i] = 0x00;
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
/* This method removes any leading spaces from String */
char * xbExpn::LTRIM( const char *String) {
  WorkBuf[0] = 0x00;
  if (!String)
    return WorkBuf;

  xbShort i;
  i = 0;
  while( *String && *String == 0x20 ) String++;
  while( *String && i < WorkBufMaxLen ){
    WorkBuf[i++] = *String;
    String++;
  }
  WorkBuf[i] = 0x00;
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
xbDouble xbExpn::LOG( xbDouble d )
{
   return log( d );
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
char *xbExpn::LOWER( const char *String ) 
{
  WorkBuf[0] = 0x00;
  if (!String)
    return WorkBuf;
  xbShort i = 0;
  while( *String && i < WorkBufMaxLen) {
    WorkBuf[i++] = tolower( *String );
    String++;
  }
  WorkBuf[i] = 0x00;
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param d1
  \param d2
*/
xbDouble xbExpn::MAX( xbDouble d1, xbDouble d2 )
{
   if( d1 > d2 ) 
      return d1;
   else
      return d2;
}
/*************************************************************************/
//! Short description.
/*!
  \param d1
  \param d2
*/
xbDouble xbExpn::MIN( xbDouble d1, xbDouble d2 )
{
   if( d1 < d2 ) 
      return d1;
   else
      return d2;
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
xbLong xbExpn::MONTH( const char * Date8 )
{
   return MonthOf( Date8 );
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
xbLong xbExpn::RECNO( xbDbf * d ) {
   return d->GetCurRecNo();
}
/*************************************************************************/
//! Short description.
/*!
  \param CurRec
*/
char * xbExpn::RECNO( xbULong CurRec ) {
   sprintf( WorkBuf, "%ld%c", CurRec, 0 );
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param Cnt
*/
char * xbExpn::REPLICATE( const char * String, xbShort Cnt )
{
   xbShort len, i;
   len = strlen( String );
   if(( len * Cnt ) > 100 ) return NULL;
   memset( WorkBuf, 0x00, len+1 );
   for( i = 0; i < Cnt; i++ )
      strcat( WorkBuf, String );
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \paran cnt
*/
char * xbExpn::RIGHT( const char * String, xbShort cnt )
{
   xbShort len;
   strcpy( WorkBuf, String );
   len = strlen( String );
   if( len < cnt ) return WorkBuf;
   len = LEN( String );
   if( len < cnt ) return WorkBuf;
   strcpy( WorkBuf, String + len - cnt );
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
char * xbExpn::RTRIM( const char * String )
{
   return TRIM( String );
}
/*************************************************************************/
//! Short description.
/*!
  \param Cnt
*/
char * xbExpn::SPACE( xbShort Cnt )
{ 
   if( Cnt > 100 ) return NULL;
   memset( WorkBuf, 0x20, Cnt );
   WorkBuf[Cnt] = 0x00;
   return WorkBuf;
} 
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
xbDouble xbExpn::SQRT( xbDouble d )
{
   return sqrt( d );
}
/*************************************************************************/
//! Short description.
/*!
  \param d
  \param length
  \param numDecimals
*/
char * xbExpn::STR(xbDouble d, xbUShort length, xbShort numDecimals) {
  // sanity check for length arg
  if (length > WorkBufMaxLen) 
  {
  // maybe should generate an error here instead ?
    length = WorkBufMaxLen;
  }

  // check the length required
  sprintf(WorkBuf, "%.*f", numDecimals, d);

  if ((xbUShort) strlen(WorkBuf) > length) {
    memset(WorkBuf, '*', length);
    WorkBuf[length] = 0x00;
  } else
    sprintf( WorkBuf, "%*.*f", length, numDecimals, d );
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param d
  \param length
*/
char * xbExpn::STR( xbDouble d, xbShort length )
{
  return STR( d, length, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
char * xbExpn::STR( xbDouble d )
{
  return STR( d, 10, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param length
  \param
*/
char * xbExpn::STR( const char * String, xbShort length, xbShort )
{
  xbShort len, i;
  len = strlen( String );
  strcpy( WorkBuf, String );
  for( i = len; i < length; i++ )
    WorkBuf[i] = 0x20;
  WorkBuf[i] = 0x00;
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param length
*/
char * xbExpn::STR( const char *String, xbShort length )
{
  return STR( String, length, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
char * xbExpn::STR( const char * String )
{
  return STR( String, 10, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param d
  \param length
  \param 
*/
char * xbExpn::STRZERO( xbDouble d, xbShort length, xbShort )
{
  xbShort len,i;
  sprintf(WorkBuf, "%*.*g", length, length, d);
//  gcvt( d, length, WorkBuf );
  len = strlen( WorkBuf );
  if( len > length )
    strcpy( WorkBuf, "**********" );
  else if( len < length )
  {
    for( i = len; i < length; i++ )
      WorkBuf[i] = 0x30;
    WorkBuf[i] = 0x00;
  }
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param d
  \param length
*/
char * xbExpn::STRZERO( xbDouble d, xbShort length )
{
  return STRZERO( d, length, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param d
*/
char * xbExpn::STRZERO( xbDouble d )
{
  return STRZERO( d, 10, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param length
  \param
*/
char * xbExpn::STRZERO( const char * String, xbShort length, xbShort )
{
  xbShort i, len ;
  while( *String == ' ' ) String++;
  len = strlen(String);
  for( i = 0; i < abs( length-len); i++ )
    WorkBuf[i] = 0x30;
  WorkBuf[i] = 0x00;
  strcat( WorkBuf, String );
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param length
*/
char * xbExpn::STRZERO( const char * String, xbShort length )
{
  return STRZERO( String, length, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
char * xbExpn::STRZERO( const char * String )
{
  return STRZERO( String, 10, 0 );
}
/*************************************************************************/
//! Short description.
/*!
  \param String
  \param StartPos
  \param Len
*/
char * xbExpn::SUBSTR( const char * String, xbShort StartPos, xbShort Len )
{
   xbShort i;
   if( StartPos < 1 ) return NULL;
   String += (StartPos - 1);
   for( i = 0; i < Len; i++ )
      WorkBuf[i] = *String++;
   WorkBuf[i] = 0x00;
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
*/
char * xbExpn::DATE()
{
   xbDate d;
   strcpy( WorkBuf, d.Sysdate());
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
char * xbExpn::TRIM( const char * String )
{
   WorkBuf[0] = 0x00;
   if( !String ) 
     return WorkBuf;
   char *sp;
   xbShort len;
   len = strlen( String );
   if( len < WorkBufMaxLen ) {
     strcpy( WorkBuf, String );
   }
   else {
     strncpy( WorkBuf, String, WorkBufMaxLen );
     WorkBuf[ WorkBufMaxLen ] = 0x00;
     len = WorkBufMaxLen;
   }
   sp = WorkBuf + len - 1;
   while( *sp == 0x20 && sp >= WorkBuf ) {
     *sp = 0x00;
     sp--;
   }
   return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
char *xbExpn::UPPER( const char *String )
{
  WorkBuf[0] = 0x00;
  if (!String)
    return WorkBuf;
  xbShort i;
  i = 0;
  while(*String && i < WorkBufMaxLen) {
    WorkBuf[i++] = toupper(*String);
    String++;
  }
  WorkBuf[i] = 0x00; 
  return WorkBuf;
}
/*************************************************************************/
//! Short description.
/*!
  \param String
*/
xbLong xbExpn::VAL( const char * String )
{
  if( String )
   return (xbLong) *String;
  else
   return 0;
}
/*************************************************************************/
//! Short description.
/*!
  \param Date8
*/
xbLong xbExpn::YEAR( const char * Date8 ){
   return YearOf( Date8 );
}
/*************************************************************************/
#endif     // XB_EXPRESSIONS
