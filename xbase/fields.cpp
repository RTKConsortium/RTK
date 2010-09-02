/*  $Id: fields.cpp,v 1.10 2003/08/20 01:53:27 gkunkel Exp $

    Xbase project source code

    This file contains the basic X-Base routines for reading and writing
    Xbase fields.

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

#include <stdlib.h>
#include <string.h>

#include <xbase/xbexcept.h>

/*! \file fields.cpp
*/

/************************************************************************/
/* This function returns true if the data is valid logical data         */
//! Determines if data is valid logical data.
/*! Determines if the data in buf is valid for a logical field value.
    
    \param buf data to be tested
    \returns TRUE (non-zero) if valid, FALSE (zero) if not.
*/
xbShort xbDbf::ValidLogicalData(const char * buf) {
   if( buf[0] )
     if( buf[0] == 'T' || buf[0] == 't' || buf[0] == 'F' || buf[0] == 'f' ||
         buf[0] == 'Y' || buf[0] == 'y' || buf[0] == 'N' || buf[0] == 'n' ||
         buf[0] == '?' )
       return 1;
   return 0; 
}
/************************************************************************/
/* This function returns true if the data is valid numeric data         */
//! Determines if data is valid numeric data.
/*! Determines if the data in buf is valid for a numeric field value.

    \param buf
    \returns TRUE (non-zero) if valid, FALSE (zero) if not.
*/
xbShort xbDbf::ValidNumericData(const char * buf) {
   const char *p;

   p = buf;
   while( *p )
   {
      if( *p != '+' && *p != '-' && *p != '.' && *p != '0' && *p != '1' &&
          *p != '2' && *p != '3' && *p != '4' && *p != '5' && *p != '6' &&
          *p != '7' && *p != '8' && *p != '9' )
         return 0;
      else
         p++;
   }
   return 1;
}
/************************************************************************/
/* This function returns a fields length */
//! Returns the length of the specified field.
/*! Returns the length of the field specified by FieldNo.

    \param FieldNo Number of field.
    \returns Length of the specified field in bytes.
*/
xbShort xbDbf::GetFieldLen( const xbShort FieldNo )
{
   if( FieldNo >= 0 && FieldNo < NoOfFields )
   {
     if( SchemaPtr[FieldNo].Type == 'C' && SchemaPtr[FieldNo].NoOfDecs > 0 )
       return SchemaPtr[FieldNo].LongFieldLen;
     else    
       return SchemaPtr[FieldNo].FieldLen;
   }
   else
      return 0;
}
/************************************************************************/
/* This function returns a fields decimal length */
//! Returns the number of decimals in the specified field.
/*! Returns the number decimals in the field specified by FieldNo.
  
    \param FieldNo Number of field.
    \returns Number of decimals in the specified field.
*/
xbShort xbDbf::GetFieldDecimal( const xbShort FieldNo )
{
   if( FieldNo >= 0 && FieldNo < NoOfFields )
      return SchemaPtr[FieldNo].NoOfDecs;
   else
      return 0;
}
/************************************************************************/
/* This function returns a fields type */
//! Returns the type of the specified field.
/*! Returns the type of the field specified by FieldNo.

    \param FieldNo Number of field.
    \returns Type of specified field.
*/
char xbDbf::GetFieldType( const xbShort FieldNo ) const
{
   if( FieldNo >= 0 && FieldNo < NoOfFields )
      return SchemaPtr[FieldNo].Type;
   else
      return 0;
}
/************************************************************************/
/* This function returns a fields name */
//! Returns the name of the specified field.
/*! Returns a pointer to the name for the field specified by FieldNo.

    \param FieldNo Number of field.
    \returns A pointer to the name of the field.
*/
char * xbDbf::GetFieldName( const xbShort FieldNo )
{
   if( FieldNo >= 0 && FieldNo < NoOfFields )
      return SchemaPtr[FieldNo].FieldName;
   else
      return 0;
}
/************************************************************************/
/* This function returns the field ID number for a given field
   or -1 if the field is not one of the fields of the database  */
//! Returns the field number of the specified field.
/*! Returns the field number for the named field.

    \param name Name of field.
    \returns Number of field named name.
*/
xbShort xbDbf::GetFieldNo( const char * name ) const
{
   int i, len1, len2;

   if(( len1 = strlen( name )) > 10 )
      return -1;

   for( i = 0; i < NoOfFields; i++ )
   {
     len2 = strlen( SchemaPtr[i].FieldName );
     if( len1 == len2 )
//        if( strstr( SchemaPtr[i].FieldName, name )) return i;
#ifndef __WIN32__
        if(!strcasecmp( SchemaPtr[i].FieldName, name )) 
#else
        if(!stricmp( SchemaPtr[i].FieldName, name )) 
#endif
           return i;
   }
   return -1;
}

/************************************************************************/
/*
   Helpers
*/

//! Get the value of the specified field.
/*! Get the value of the field referenced by Name and place its value
    in buf.
    
    \param Name Name of field.
    \param buf Buffer to hold field value.  Must be large enough to hold
               the entire field value.  Use GetFieldLen() to determine
               the length of the field, if necessary.
    \param RecBufSw
    \returns One of the following:
*/
xbShort xbDbf::GetField(const char *Name, char *buf,
         const xbShort RecBufSw ) const
{
   return GetField(GetFieldNo(Name), buf, RecBufSw);
}

//! Get the value of the specified field.
/*! Get the value of the field specified by Name and place its value
    in buf.
    
    \param Name Name of field.
    \param buf Buffer to hold field value.  Must be large enough to hold
               the entire field value.  Use GetFieldLen() to determine
               the length of the field, if necessary.
    \returns One of the following:
*/
xbShort xbDbf::GetField(const char *Name, char *buf) const
{
   return GetField(GetFieldNo(Name), buf);
}

//! Get the raw value of the specified field.
/*! Get the value of the field specified by Name and place its value
    in buf.
    
    \param Name Name of field.
    \param buf Buffer to hold field value.  Must be large enough to hold
               the entire field value.  Use GetFieldLen() to determine
               the length of the field, if necessary.
    \returns One of the following:
*/
xbShort xbDbf::GetRawField(const char *Name, char *buf) const
{
   return GetRawField(GetFieldNo(Name), buf);
}

static char __buf[1024];

static void trim(char *s) {
  int len = strlen(s)-1;
  if (len > 0) {
    while ((len != 0) && (s[len] == ' '))
      len--;
    s[len+1] = 0;
  }
}

//! Get the value of the specified field.
/*! Returns the value of the field specified by Name.

    \param Name Name of field.
    \returns Value of the specified field.
*/
const char *xbDbf::GetField(const char *Name) const {
   GetField(GetFieldNo(Name), __buf);
   trim(__buf);
   return __buf;
}

//! Get the value of the specified field.
/*! Returns the value of the field specified by FieldNo.

    \param FieldNo Number of field.
    \returns Value of the specified field.
*/
const char *xbDbf::GetField(xbShort FieldNo) const {
   GetField(FieldNo, __buf);
   trim(__buf);
   return __buf;
}
/************************************************************************/
/* This function fills a buffer with data from the record buffer
   for a particular field number. 

   Use GetFieldNo to get a number based on a field's name 

   If successful, this function returns the field size.
*/

//! Get the value of the specified field.
/*! Get the value of the field specified by FieldNo and place its value
    in buf.
    
    \param FieldNo Number of field.
    \param buf Buffer to hold field value.  Must be large enough to hold
               the entire field value.  Use GetFieldLen() to determine
               the length of the field, if necessary.
    \param RecBufSw
    \returns The length of the field.
*/
xbShort xbDbf::GetField(const xbShort FieldNo, char * buf, 
         const xbShort RecBufSw) const
{
     xbShort length;        
     if( FieldNo < 0 || FieldNo >= NoOfFields ) {
#ifdef HAVE_EXCEPTIONS
       xb_error(XB_INVALID_FIELDNO);
#else
       buf[0] = 0x00;
       return 0x00;
#endif
      }

   // Check for existence of a long field length
   if( SchemaPtr[FieldNo].Type == 'C' && SchemaPtr[FieldNo].NoOfDecs > 0 )
     length = SchemaPtr[FieldNo].LongFieldLen;
   else
     length = SchemaPtr[FieldNo].FieldLen;

   if( RecBufSw )
     memcpy( buf, SchemaPtr[FieldNo].Address2, length );
   else
     memcpy( buf, SchemaPtr[FieldNo].Address, length );
   buf[length] = 0x00;
   return( length ); 
}
/************************************************************************/

xbShort xbDbf::GetField(const xbShort FieldNo, xbString & sf, 
         const xbShort RecBufSw) const
{
     xbShort length;        
     if( FieldNo < 0 || FieldNo >= NoOfFields ) {
#ifdef HAVE_EXCEPTIONS
       sf = "";
       xb_error(XB_INVALID_FIELDNO);
#else 
       sf = "";
       return 0;
#endif
      }

   // Check for existence of a long field length
   if( SchemaPtr[FieldNo].Type == 'C' && SchemaPtr[FieldNo].NoOfDecs > 0 )
     length = SchemaPtr[FieldNo].LongFieldLen;
   else
     length = SchemaPtr[FieldNo].FieldLen;

   if( RecBufSw )
     sf.assign( xbString(SchemaPtr[FieldNo].Address2, length), 0, length );
//     sf.assign( SchemaPtr[FieldNo].Address2, length );
   else
     sf.assign( xbString(SchemaPtr[FieldNo].Address, length), 0, length );
//     sf.assign( SchemaPtr[FieldNo].Address, length );
   return( length ); 
}
/************************************************************************/
/* This function fills a field in the record buffer with data from
   a buffer for a particular field.

   Use GetFieldNo to get a number based on a field's name 

   Field type N or F is loaded as right justified, left blank filled.
   Other fields are loaded as left justified, right blank filled.

   This method does check the data's validity.

   If successful, this function returns 0, if invalid data, it returns -1
   or XB_INVALID_FIELDNO
*/

//! Put a value into the specified field.
/*!
*/
xbShort xbDbf::PutField(const char *Name, const char *buf) {
   return PutField(GetFieldNo(Name), buf);
}

//! Put a raw value into the specified field.
/*!
*/
xbShort xbDbf::PutRawField(const char *Name, const char *buf) {
   return PutRawField(GetFieldNo(Name), buf);
}

//! Put a value into the specified field.
/*!
*/
xbShort xbDbf::PutField(const xbShort FieldNo, const char *buf) {
   xbShort len, i;
   char * startpos;
   char * tp;           /*  target pointer */
   const char * sp;        /*  source pointer */

   if( FieldNo < 0 || FieldNo >= NoOfFields )
      xb_error(XB_INVALID_FIELDNO);

   if( DbfStatus != XB_UPDATED )
   {
      DbfStatus = XB_UPDATED;
      memcpy( RecBuf2, RecBuf, RecordLen );
   }
  
   if( SchemaPtr[FieldNo].Type == 'L' && !ValidLogicalData( buf ))
     xb_error(XB_INVALID_DATA)

   else if(( SchemaPtr[FieldNo].Type == 'F' || SchemaPtr[FieldNo].Type == 'N' )
        && !ValidNumericData( buf )) 
     xb_error(XB_INVALID_DATA)

   else if( SchemaPtr[FieldNo].Type == 'D' ){
    xbDate d;
    if( !d.DateIsValid( buf ))
      xb_error(XB_INVALID_DATA);
   }

   if( SchemaPtr[FieldNo].Type == 'C' && SchemaPtr[FieldNo].NoOfDecs > 0 )
     memset( SchemaPtr[FieldNo].Address, 0x20, SchemaPtr[FieldNo].LongFieldLen );
   else
     memset( SchemaPtr[FieldNo].Address, 0x20, SchemaPtr[FieldNo].FieldLen );

   len = strlen( buf );

   if(( SchemaPtr[FieldNo].Type == 'N' || SchemaPtr[FieldNo].Type == 'F')
         && len > SchemaPtr[FieldNo].FieldLen )
     xb_error(XB_INVALID_DATA)
   else if( len > SchemaPtr[FieldNo].FieldLen )
     len = SchemaPtr[FieldNo].FieldLen;

   if( SchemaPtr[FieldNo].Type == 'F' || SchemaPtr[FieldNo].Type == 'N' 
       || SchemaPtr[FieldNo].Type == 'M' 
     ) {
      const char *sdp = strchr( buf, '.' ); /*  source decimal point */
      len = 0;
      sp =buf;
      while( *sp && *sp != '.' ) { len++; sp++; }
      
      if( SchemaPtr[FieldNo].NoOfDecs > 0 )
      {
         /* do the right of decimal area */
         tp = SchemaPtr[FieldNo].Address;
         tp += SchemaPtr[FieldNo].FieldLen - SchemaPtr[FieldNo].NoOfDecs - 1;
         *tp++ = '.';
         sp = sdp;
         if( sp ) sp++;
         for( i = 0; i < SchemaPtr[FieldNo].NoOfDecs; i++ )
            if( sp && *sp ) *tp++ = *sp++; else *tp++ = '0'; 

         startpos= SchemaPtr[FieldNo].Address +
                   SchemaPtr[FieldNo].FieldLen -
                   SchemaPtr[FieldNo].NoOfDecs - len - 1; 
      }
      else
      {
         startpos=SchemaPtr[FieldNo].Address+SchemaPtr[FieldNo].FieldLen-len; 
      }
   }
   else
      startpos = SchemaPtr[FieldNo].Address;

   memcpy( startpos, buf, len );
   return 0;
}

//! Put a raw value into the specified field.
/*!
*/
xbShort xbDbf::PutRawField(const xbShort FieldNo, const char *buf) {
   xbShort len;
   char * startpos;

   if( FieldNo < 0 || FieldNo >= NoOfFields )
      xb_error(XB_INVALID_FIELDNO);

   if( DbfStatus != XB_UPDATED )
   {
      DbfStatus = XB_UPDATED;
      memcpy( RecBuf2, RecBuf, RecordLen );
   }
  
   startpos = SchemaPtr[FieldNo].Address;
   len = SchemaPtr[FieldNo].FieldLen;
   memcpy( startpos, buf, len );
   
   return 0;
}

/************************************************************************/
//! Get the value of the specified field.
/*!
*/
xbShort xbDbf::GetField(const xbShort FieldNo, char *buf) const {
   return GetField(FieldNo, buf, 0);
}

/************************************************************************/
//! Get the raw value of the specified field.
/*!
*/
xbShort xbDbf::GetRawField(const xbShort FieldNo, char *buf) const {
   return GetField(FieldNo, buf, 0);
}

/************************************************************************/
//! Get the long value of the specified field.
/*!
*/
xbLong xbDbf::GetLongField( const xbShort FieldNo ) const
{
   char buf[18];
   memset( buf, 0x00, 18 );
   GetField( FieldNo, buf );
   return atol( buf );
}
/************************************************************************/
//! Get the long value of the specified field.
/*!
*/
xbLong xbDbf::GetLongField( const char * FieldName ) const
{
  return( GetLongField( GetFieldNo( FieldName )));
}
/************************************************************************/
//! Put a long value into the specified field.
/*!
*/
xbShort xbDbf::PutLongField( const xbShort FieldNo, const xbLong Val )
{
   char buf[18];
   memset( buf, 0x00, 18 );
   sprintf( buf, "%ld", Val );
   return( PutField( FieldNo, buf ));
}
/************************************************************************/
//! Put a long value into the specified field.
/*!
*/
xbShort xbDbf::PutLongField(const char *FieldName, const xbLong Val) {
  return (PutLongField(GetFieldNo(FieldName), Val));
}
/************************************************************************/
//! Get the float value of the specified field.
/*!
*/
xbFloat xbDbf::GetFloatField(const xbShort FieldNo) {
   char buf[21];
   memset( buf, 0x00, 21 );
   if( GetField( FieldNo, buf ) != 0 )
      return atof( buf );
   else
      return 0;
}
/************************************************************************/
//! Get the float value of the specified field.
/*!
*/
xbFloat xbDbf::GetFloatField(const char * FieldName) {
xbShort fnum;
  if ((fnum = GetFieldNo(FieldName)) != -1)
    return GetFloatField(fnum);
  else
    return 0;
}
/************************************************************************/
//! Put a float value into the specified field.
/*!
*/
xbShort xbDbf::PutFloatField( const xbShort FldNo, const xbFloat f )
{
   char buf[25];
   char buf2[12];
   memset( buf, 0x00, 25 );
   memset( buf2, 0x00, 12 );
   sprintf( buf, "%d.%df", GetFieldLen( FldNo ), GetFieldDecimal( FldNo ));
   strcpy( buf2, "%-" );
   strcat( buf2, buf );
   sprintf( buf, buf2, f );

   /* remove trailing space */
   xbShort i = 0;
   while( i < 25 ) 
     if( buf[i] == 0x20 ){
       buf[i] = 0x00;
       break;
     } else
       i++;
   return PutField( FldNo, buf );
}
/************************************************************************/
//! Put a float value into the specified field.
/*!
*/
xbShort xbDbf::PutFloatField(const char *FieldName, const xbFloat f) {
  xbShort fnum;
  if ((fnum = GetFieldNo(FieldName)) != -1)
    return PutFloatField(fnum, f);
  else
    return 0;
}
/************************************************************************/
//! Get the double value of the specified field.
/*!
*/
xbDouble xbDbf::GetDoubleField( const xbShort FieldNo, xbShort RecBufSw )
{
   char buf[21];
   memset( buf, 0x00, 21 );
   if( GetField( FieldNo, buf, RecBufSw ) != 0 )
      return strtod( buf, NULL );
   else
      return 0;
}
/************************************************************************/
//! Get the double value of the specified field.
/*!
*/
xbDouble xbDbf::GetDoubleField(const char *FieldName) {
  xbShort fnum;
  if ((fnum = GetFieldNo(FieldName)) != -1)
    return GetDoubleField(fnum);
  else
    return 0;
}
/************************************************************************/
//! Put a double value into the specified field.
/*!
*/
xbShort xbDbf::PutDoubleField( const xbShort FieldNo, const xbDouble d) {
  return PutFloatField(FieldNo, (xbFloat)d);
}
/************************************************************************/
//! Put a double value into the specified field.
/*!
*/
xbShort xbDbf::PutDoubleField(const char *FieldName, const xbDouble d) {
  xbShort fnum;
  if ((fnum = GetFieldNo(FieldName)) != -1)
    return PutFloatField(fnum, (xbFloat)d);
  else
    return 0;
}
/************************************************************************/
//! Get the logical value of the specified field.
/*!
*/
xbShort xbDbf::GetLogicalField( const xbShort FieldNo )
{
   char buf[3];
   if( GetFieldType( FieldNo ) != 'L' ) return -1;
   memset( buf, 0x00, 3 );
   GetField( FieldNo, buf );
   if( buf[0] == 'Y' || buf[0] == 'y' || buf[0] == 'T' || buf[0] == 't' )
      return 1;
   else
      return 0;
}
/************************************************************************/
//! Get the logical value of the specified field.
/*!
*/
xbShort xbDbf::GetLogicalField( const char * FieldName )
{
   xbShort fnum;
   if(( fnum = GetFieldNo( FieldName )) != -1 )
      return GetLogicalField( fnum );
   else
      return -1;
}
/************************************************************************/
//! Get the string value of the specified field.
/*!
*/
char * xbDbf::GetStringField( const char * FieldName )
{
  return GetStringField(GetFieldNo(FieldName));
}
/************************************************************************/
//! Get the string value of the specified field.
/*!
*/
char * xbDbf::GetStringField( const xbShort FieldNo )
{
  /* allocate memory if needed */
  if( !SchemaPtr[FieldNo].fp )
    SchemaPtr[FieldNo].fp = new char[GetFieldLen(FieldNo)+1];

  if( !SchemaPtr[FieldNo].fp )
#ifdef HAVE_EXCEPTIONS
    throw xbOutOfMemoryException();
#else
    return 0;
#endif

  GetField( FieldNo, SchemaPtr[FieldNo].fp );
  return SchemaPtr[FieldNo].fp;
}
/************************************************************************/
