/*  $Id: html.h,v 1.10 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code
 
    This file contains a header file for the HTML object which is used
    for HTML generation.

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

#ifndef __XB_HTML_H__
#define __XB_HTML_H__

#ifdef __GNUG__
#pragma interface
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <stdlib.h>
#include <string.h>

#include <xbase/xtypes.h>

/*! \file html.h
*/

//! xbFieldList struct
/*!
*/

struct xbFieldList{
   char * Label;                     /* field label on form */
   char * FieldName;                 /* form field name     */
   xbShort FieldLen;                   /* form display length */
   xbShort FieldNo;                    /* database field no   */
   xbShort Option;                     /* field option        */
};

typedef char **xbArrayPtr;

//! xbHtml class
/*!
*/
class XBDLLEXPORT xbHtml {
public:
   xbHtml();
   virtual ~xbHtml();

   //! Short description.
   /*!
   */
   void   BoldOff( void ) { std::cout << "</b>\n"; };
   //! Short description.
   /*!
   */
   void   BoldOn( void ) { std::cout << "<b>"; };
   //! Short description.
   /*!
   */
   void   Bullet( void ) { std::cout << "<li>"; };
   void   DumpArray( void );
   //! Short description.
   /*!
   */
   void   EmphasizeOff( void ) { std::cout << "</em>\n"; };
   //! Short description.
   /*!
   */
   void   EmphasizeOn( void ) { std::cout << "<em>"; };
   //! Short description.
   /*!
   */
   void   EndHtmlPage( void ) { std::cout << "</BODY>\n</HTML>\n"; }
   xbShort  GenFormFields(xbDbf *d, xbShort Option,const char * Title,xbFieldList *fl);
   xbShort  GetArrayNo( const char * FieldName );
   const  char * GetCookie( const char *CookieName );
   char * GetData( xbShort );
   char * GetDataForField( const char *FieldName );
   char * GetEnv( char * s ){ return getenv( s ); }
   xbShort  GetMethod( void );
   //! Short description.
   /*!
   */
   void   HeaderOff( xbShort i ){ std::cout << "</h" << i << ">\n"; };
   //! Short description.
   /*!
   */
   void   HeaderOn( xbShort i ){ std::cout << "<h" << i << ">\n"; };
   //! Short description.
   /*!
   */
   void   ItalicOff( void ) { std::cout << "</i>\n"; };
   //! Short description.
   /*!
   */
   void   ItalicOn( void ) { std::cout << "<i>"; };
   //! Short description.
   /*!
   */
   void   NewLine( void ) { std::cout << "<br>"; }
   xbShort  PostMethod( void );
   void   PrintEncodedChar( char );
   void   PrintEncodedString( const char *s );
   //! Short description.
   /*!
   */
   void   PrintHtml( char * s ) { std::cout << s; };
   //! Short description.
   /*!
   */
   void   PrintHtml( xbLong l ) { std::cout << l; };
   //! Short description.
   /*!
   */
   void   PrintHtml( xbShort i ) { std::cout << i; };
   //! Short description.
   /*!
   */
   void   PrintHtml( int i ) { std::cout << i; };
   void   StartHtmlPage( const char *Title );
   //! Short description.
   /*!
   */
   void   StartTextPage( void ) { std::cout << "Content-type: text/plain\n\n"; }
   void   TextOut( const char *String );
   xbLong   Tally( const char *FileName );
   xbShort  SetCookie(const char *Name, const char *Value, const char *ExpDate,
           const char *ExpTime,const char *TimeZone, const char *Path,
           const char *Domain, xbShort Secure );
   void   SpaceToPlus( char * );
   void   PlusToSpace( char * );
   void   SendRedirect( char * ) const;

protected:
   xbArrayPtr FieldNameArray;
   xbArrayPtr DataValueArray;
   xbShort    NoOfDataFields;
   char     * HtmlWorkBuf;
   xbShort    HtmlBufLen;
   void     LoadArray( void );
   void     DeleteEscChars( char *String );
   void     InitVals( void );
};

#endif      // __XB_HTML_H__
