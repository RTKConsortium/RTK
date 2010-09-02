/*  $Id: xbase.cpp,v 1.12 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains logic for the basic Xbase class.

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

#ifdef __GNUG__
  #pragma implementation "xbase.h"
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xbase.h>
#include <ctype.h>
#include <string.h>

#include <xbase/xbexcept.h>

/*! \file xbase.cpp
*/

/*************************************************************************/
//! Constructor.
/*!
*/
xbXBase::xbXBase( void )
{
   xbShort e = 1;
   EndianType = *(char *) &e;
   if( EndianType )
      EndianType = 'L';
   else
      EndianType = 'B';
   DbfList = NULL;
   FreeDbfList = NULL;
}

/*************************************************************************/
//! Get pointer to named dbf.
/*!
  Looks up an open DBF file by Name.
  
  \param Name
  \returns A pointer to the xbDbf class instance if found or NULL if
    not found.
*/
xbDbf *xbXBase::GetDbfPtr(const char *Name) {
  xbDbList *t;

  t = DbfList;
  xbShort len = strlen(Name);

  /* check for -> embedded in the name */
  for( xbShort i = 0; i < (len-1); i++ )
    if( Name[i] == '-' && Name[i+1] == '>' )
      len = i-1;

  while (t) {
    if (strncmp(Name, t->DbfName, len) == 0 )
      return t->dbf;
    t = t->NextDbf;
  }
  return NULL;
}

/*************************************************************************/
//! Destructor.
/*!
*/
xbXBase::~xbXBase()
{
   xbDbList *i = FreeDbfList;
   while (i) {
      xbDbList *t = i->NextDbf;
      if (i->DbfName) {
         free(i->DbfName);
      }
      free(i);
      i = t;
   }
}

/*************************************************************************/
//! Add dbf to dbf list.
/*!
  Adds an xbDbf class instance to the list of dbf's.
  
  \param d the xbDbf instance to be added
  \param DatabaseName name of the database
  
  \returns One of the following return codes:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_NO_MEMORY</td><td>Out of memory</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No Error \\ \hline
        XB\_NO\_MEMORY & Out of memory \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbXBase::AddDbfToDbfList(xbDbf *d, const char *DatabaseName) {
   xbDbList *i, *s, *t;

   if(!FreeDbfList) {
      if((i = (xbDbList *)malloc(sizeof(xbDbList))) == NULL) {
         xb_memory_error;
      }
   } else {
      i = FreeDbfList;
      FreeDbfList = i->NextDbf;
   }
   memset(i, 0x00, sizeof(xbDbList));

   i->DbfName  = strdup(DatabaseName);
   i->dbf      = d;

   s = NULL;
   t = DbfList;
   while(t && strcmp(t->DbfName, DatabaseName) < 0) {
      s = t;
      t = t->NextDbf;
   }
   i->NextDbf = t;
   if (s == NULL)
      DbfList = i;
   else
      s->NextDbf = i;
      
   return 0;
}

/***********************************************************************/
//!  Remove dbf from dbf list.
/*!
  Removes the specified xbDbf class instance from the list of dbf's.
  
  \param d xbDbf to be removed
  
  \returns One of the following return codes:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No Error \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbXBase::RemoveDbfFromDbfList(xbDbf *d) {
   xbDbList *i, *s;

   i = DbfList;
   s = NULL;

   while (i) {
      if(i->dbf == d) {
         /* remove it from current chain */
         if(s)
            s->NextDbf = i->NextDbf;
         else
            DbfList = i->NextDbf;

         /* add i to the current free chain */
         i->NextDbf = FreeDbfList;
         FreeDbfList = i;
         free(FreeDbfList->DbfName);
         FreeDbfList->DbfName = NULL;
         break;
      } else {
         s = i;
         i = i->NextDbf;
      }
   }
   return XB_NO_ERROR;
} 

// FIXME: byte reverse methods are awful, compared to bitwise shifts  -- willy

/************************************************************************/
//! Get a portable short value.
/*!
  Converts a short (16 bit integer) value stored at p from a portable 
  format to the machine format.
  
  \param p pointer to memory containing the portable short value
  
  \returns the short value.
*/
/* This routine returns a short value from a 2 byte character stream */
xbShort xbXBase::GetShort(const char *p) {
   xbShort s, i;
   const char *sp;
   char *tp;

   s = 0;
   tp = (char *) &s;
   sp = p;
   if( EndianType == 'L' )
      for( i = 0; i < 2; i++ ) *tp++ = *sp++;
   else
   {
      sp++;
      for( i = 0; i < 2; i++ ) *tp++ = *sp--;
   }  
   return s;
}

//! Get a portable long value.
/*!
  Converts a long (32 bit integer) value stored at p from a portable 
  format to the machine format.
  
  \param p pointer to memory containing the portable long value
  
  \returns the long value.
*/
/* This routine returns a long value from a 4 byte character stream */
xbLong xbXBase::GetLong( const char *p )
{
   xbLong l;
   const char *sp;
   char *tp;
   xbShort i;

   tp = (char *) &l;
   sp = p;
   if( EndianType == 'L' )
      for( i = 0; i < 4; i++ ) *tp++ = *sp++;
   else
   {
      sp+=3;
      for( i = 0; i < 4; i++ )  *tp++ = *sp--;
   }  
   return l;
}

//! Get a portable unsigned long value.
/*!
  Converts an unsigned long (32 bit integer) value stored at p from a portable 
  format to the machine format.
  
  \param p pointer to memory containing the portable unsigned long value
  
  \returns the unsigned long value.
*/
/* This routine returns a long value from a 4 byte character stream */
xbULong xbXBase::GetULong( const char *p )
{
  xbULong l;
  char *tp;
  xbShort i;
  
  tp = (char *) &l;
  if( EndianType == 'L' )
    for( i = 0; i < 4; i++ ) *tp++ = *p++;
  else{
    p+=3;
    for( i = 0; i < 4; i++ ) *tp++ = *p--;
  }
  return l;
}

/************************************************************************/
//! Get a high byte first short value.
/*!
  Converts a short (16 bit integer) value stored at p from a high byte first
  format to the machine format.

  \param p pointer to memory containing the high byte first short value

  \returns the short value.
*/
/* This routine returns a short value from a 2 byte character stream */
xbShort xbXBase::GetHBFShort(const char *p) {
   xbShort s, i;
   const char *sp;
   char *tp;

   s = 0;
   tp = (char *) &s;
   sp = p;
   if( EndianType == 'B' )
      for( i = 0; i < 2; i++ ) *tp++ = *sp++;
   else
   {
      sp++;
      for( i = 0; i < 2; i++ ) *tp++ = *sp--;
   }
   return s;
}

//! Get a high byte first unsigned long value.
/*!
  Converts an unsigned long (32 bit integer) value stored at p from a high byte first
  format to the machine format.

  \param p pointer to memory containing the high byte first unsigned long value

  \returns the unsigned long value.
*/
/* This routine returns a long value from a 4 byte character stream */
xbULong xbXBase::GetHBFULong( const char *p )
{
  xbULong l;
  char *tp;
  xbShort i;

  tp = (char *) &l;
  if( EndianType == 'B' )
    for( i = 0; i < 4; i++ ) *tp++ = *p++;
  else{
    p+=3;
    for( i = 0; i < 4; i++ ) *tp++ = *p--;
  }
  return l;
}

//! Get a portable double value.
/*!
  Converts a double (64 bit floating point) value stored at p from a portable 
  format to the machine format.
  
  \param p pointer to memory containing the portable double value
  
  \returns the double value.
*/
/* This routine returns a double value from an 8 byte character stream */
xbDouble xbXBase::GetDouble( const char *p )
{
   xbDouble d;
   const char *sp;
   char *tp;
   xbShort i;

   tp = (char *) &d;
   sp = p;
   if( EndianType == 'L' )
      for( i = 0; i < 8; i++ ) *tp++ = *sp++;
   else
   {
      sp+=7;
      for( i = 0; i < 8; i++ )  *tp++ = *sp--;
   } 

   return d;
}

//! Put a portable short value.
/*!
  Converts a short (16 bit integer) value from machine format to a
  portable format and stores the converted value in the memory referenced
  by c.
  
  \param c pointer to memory to hold converted value
  \param s value to be converted
*/
/* This routine puts a short value to a 2 byte character stream */
void xbXBase::PutShort( char * c, const xbShort s )
{
   const char *sp;
   char *tp;
   xbShort i;

   tp = c;
   sp = (const char *) &s;

   if( EndianType == 'L' )
   {
      for( i = 0; i < 2; i++ ) *tp++ = *sp++;
   }
   else      /* big endian */
   {
      sp++;
      for( i = 0; i < 2; i++ ) *tp++ = *sp--;
   }
   return;
}

//! Put a portable long value.
/*!
  Converts a long (32 bit integer) value from machine format to a
  portable format and stores the converted value in the memory referenced
  by c.
  
  \param c pointer to memory to hold converted value
  \param s value to be converted
*/
/* This routine puts a long value to a 4 byte character stream */
void xbXBase::PutLong( char * c, const xbLong l )
{
   const char *sp;
   char *tp;
   xbShort i;

   tp = c;
   sp = (const char *) &l;
   if( EndianType == 'L' )
      for( i = 0; i < 4; i++ ) *tp++ = *sp++;
   else
   {
      sp+=3;
      for( i = 0; i < 4; i++ ) *tp++ = *sp--;
   }
   return;
}

//! Put a portable unsigned short value.
/*!
  Converts an unsigned long (16 bit integer) value from machine format to a
  portable format and stores the converted value in the memory referenced
  by c.
  
  \param c pointer to memory to hold converted value
  \param s value to be converted
*/
/* This routine puts a short value to a 2 byte character stream */
void xbXBase::PutUShort( char * c, const xbUShort s )
{
   const char *sp;
   char *tp;
   xbShort i;

   tp = c;
   sp = (const char *) &s;
   if( EndianType == 'L' )
      for( i = 0; i < 2; i++ ) *tp++ = *sp++;
   else
   {
      sp++;
      for( i = 0; i < 2; i++ ) *tp++ = *sp--;
   }
   return;
}

//! Put a portable unsigned long value.
/*!
  Converts an unsigned long (32 bit integer) value from machine format to a
  portable format and stores the converted value in the memory referenced
  by c.
  
  \param c pointer to memory to hold converted value
  \param s value to be converted
*/
/* This routine puts a long value to a 4 byte character stream */
void xbXBase::PutULong( char * c, const xbULong l )
{
   const char *sp;
   char *tp;
   xbShort i;

   tp = c;
   sp = (const char *) &l;
   if( EndianType == 'L' )
      for( i = 0; i < 4; i++ ) *tp++ = *sp++;
   else
   {
      sp+=3;
      for( i = 0; i < 4; i++ ) *tp++ = *sp--;
   }
   return;
}

//! Put a portable double value.
/*!
  Converts a double (64 floating point) value from machine format to a
  portable format and stores the converted value in the memory referenced
  by c.
  
  \param c pointer to memory to hold converted value
  \param s value to be converted
*/
/* This routine puts a double value to an 8 byte character stream */
void xbXBase::PutDouble( char * c, const xbDouble d )
{
   const char *sp;
   char *tp;
   xbShort i;

   tp = c;
   sp = (const char *) &d;
   if( EndianType == 'L' )
      for( i = 0; i < 8; i++ ) *tp++ = *sp++;
   else
   {
      sp+=7;
      for( i = 0; i < 8; i++ ) *tp++ = *sp--;
   }
   return;
}

/************************************************************************/
//! Get offset of last PATH_SEPARATOR in Name.
/*!
  Scans the specified Name for the last occurance of PATH_SEPARATOR.
  
  \param Name string to be scanned.
  
  \returns offset of last occurance of PATH_SEPARATOR
*/
xbShort xbXBase::DirectoryExistsInName( const char * Name )
{
   /* returns the offset in the string of the last directory slash */

   xbShort Count, Mark;
   char  Delim;
   const char  *p;

   Delim = PATH_SEPARATOR;

   Count = Mark = 0;
   p = Name;

   while( *p )
   {
      Count++;
      if( *p++ == Delim ) Mark = Count;
   }
   return Mark;
}

/************************************************************************/
//! Display description of error code.
/*!
  Displays a text description of an XBase error code.
  
  \param ErrorNo error to be displayed
*/
void xbXBase::DisplayError( const xbShort ErrorNo ) const
{
#if 0 // replaced following code to remove duplicate strings (9/27/2000 DTB)
  switch( ErrorNo ) {
    case    0: std::cout << "No Error" << std::endl;                    break;
    case -100: std::cout << "End Of File" << std::endl;                 break;
//  case -101: std::cout << "Beginning Of File" << std::endl;           break;
    case -102: std::cout << "No Memory" << std::endl;                   break;
    case -103: std::cout << "File Already Exists" << std::endl;         break;
    case -104: std::cout << "Database or Index Open Error" << std::endl;break;
    case -105: std::cout << "Error writing to disk drive" << std::endl; break;
    case -106: std::cout << "Unknown Field Type" << std::endl;          break;
    case -107: std::cout << "Database already open" << std::endl;       break;
    case -108: std::cout << "Not an Xbase type database" << std::endl;  break;
    case -109: std::cout << "Invalid Record Number" << std::endl;       break;
    case -110: std::cout << "Invalid Option" << std::endl;              break;
    case -111: std::cout << "Database not open" << std::endl;           break;
    case -112: std::cout << "Disk Drive Seek Error" << std::endl;       break;
    case -113: std::cout << "Disk Drive Read Error" << std::endl;       break;
    case -114: std::cout << "Search Key Not Found" << std::endl;        break;
    case -115: std::cout << "Search Key Found" << std::endl;            break;
    case -116: std::cout << "Invalid Key" << std::endl;                 break;
    case -117: std::cout << "Invalid Node Link" << std::endl;           break;
    case -118: std::cout << "Key Not Unique" << std::endl;              break;
    case -119: std::cout << "Invalid Key Expression" << std::endl;      break;
//  case -120: std::cout << "DBF File Not Open" << std::endl;           break;
    case -121: std::cout << "Invalid Key Type" << std::endl;            break;
    case -122: std::cout << "Invalid Node No" << std::endl;             break;
    case -123: std::cout << "Node Full" << std::endl;                   break;
    case -124: std::cout << "Invalid Field Number" << std::endl;        break;
    case -125: std::cout << "Invalid Data" << std::endl;                break;
    case -126: std::cout << "Not a leaf node" << std::endl;             break;
    case -127: std::cout << "Lock Failed" << std::endl;                 break;
    case -128: std::cout << "Close Error" << std::endl;                 break;
    case -129: std::cout << "Invalid Schema" << std::endl;              break;
    case -130: std::cout << "Invalid Name" << std::endl;                break;
    case -131: std::cout << "Invalid Block Size" << std::endl;          break;
    case -132: std::cout << "Invalid Block Number" << std::endl;        break;
    case -133: std::cout << "Not a Memo field" << std::endl;            break;
    case -134: std::cout << "No Memo Data" << std::endl;                break;
    case -135: std::cout << "Expression syntax error" << std::endl;     break;
    case -136: std::cout << "Parse Error" << std::endl;                 break;
    case -137: std::cout << "No Data" << std::endl;                     break;
//  case -138: std::cout << "Unknown Token Type" << std::endl;          break;

    case -140: std::cout << "Invalid Field" << std::endl;               break;
    case -141: std::cout << "Insufficient Parms" << std::endl;          break;
    case -142: std::cout << "Invalid Function" << std::endl;            break;
    case -143: std::cout << "Invalid Field Length" << std::endl;        break;
    case -144: std::cout << "Harvest Node Error" << std::endl;          break;
    case -145: std::cout << "Invalid Date" << std::endl;                break;
    default:   std::cout << "Unknown error code" << std::endl;          break;
  }
#else
  std::cout << GetErrorMessage(ErrorNo) << std::endl;
#endif
}

/************************************************************************/
//! Get description of error code.
/*!
  Returns a pointer to string containing a text description of an
  error code.
  
  \param ErrorNo error number of description to be returned
*/
const char* xbXBase::GetErrorMessage( const xbShort ErrorNo )
{
  switch( ErrorNo ) {
    case    0: return "No Error";
    case -100: return "End Of File";
    case -101: return "Beginning Of File";
    case -102: return "No Memory";
    case -103: return "File Already Exists";
    case -104: return "Database or Index Open Error";
    case -105: return "Error writing to disk drive";
    case -106: return "Unknown Field Type";
    case -107: return "Database already open";
    case -108: return "Not an Xbase type database";
    case -109: return "Invalid Record Number";
    case -110: return "Invalid Option";
    case -111: return "Database not open";
    case -112: return "Disk Drive Seek Error";
    case -113: return "Disk Drive Read Error";
    case -114: return "Search Key Not Found";
    case -115: return "Search Key Found";
    case -116: return "Invalid Key";
    case -117: return "Invalid Node Link";
    case -118: return "Key Not Unique";
    case -119: return "Invalid Key Expression";
    case -120: return "DBF File Not Open";
    case -121: return "Invalid Key Type";
    case -122: return "Invalid Node No";
    case -123: return "Node Full";
    case -124: return "Invalid Field Number";
    case -125: return "Invalid Data";
    case -126: return "Not a leaf node";
    case -127: return "Lock Failed";
    case -128: return "Close Error";
    case -129: return "Invalid Schema";
    case -130: return "Invalid Name";
    case -131: return "Invalid Block Size";
    case -132: return "Invalid Block Number";
    case -133: return "Not a Memo field";
    case -134: return "No Memo Data";
    case -135: return "Expression syntax error";
    case -136: return "Parse Error";
    case -137: return "No Data";
    case -138: return "Unknown Token Type";

    case -140: return "Invalid Field";
    case -141: return "Insufficient Parms";
    case -142: return "Invalid Function";
    case -143: return "Invalid Field Length";
    default:   return "Unknown error code";
  }
}
