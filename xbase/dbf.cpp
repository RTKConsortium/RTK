/*  $Id: dbf.cpp,v 1.25 2003/08/20 01:53:27 gkunkel Exp $

    Xbase project source code
   
    This file contains the basic Xbase routines for reading and writing
    Xbase .DBF files.

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


  V 1.0 10/10/97 - Initial release of software

*/

#ifdef __GNUG__
  #pragma implementation "dbf.h"
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xbase.h>

#ifdef HAVE_IO_H
#include <io.h>
#endif
#include <errno.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <xbase/xbexcept.h>

/*! \file dbf.cpp
*/

/************************************************************************/
//! Constructor
/*!
  \param x pointer to the global xbXbase class
*/
xbDbf::xbDbf( xbXBase * x )
{
   xbase = x;
   InitVars();
}

/************************************************************************/
//! Destructor
/*!
  \param x pointer to the global xbXbase class
*/
xbDbf::~xbDbf()
{
  CloseDatabase(true);
}

/************************************************************************/
//! Initialize private data members.
/*!
  Internal use only.
*/
void xbDbf::InitVars( void )
{
   DatabaseName    = (char)0x00;
   NoOfFields      = 0;
   DbfStatus       = XB_CLOSED;
   fp              = NULL;
   CurRec          = 0L;
   SchemaPtr       = NULL;
   RecBuf          = NULL;
   RecBuf2         = NULL;
   Version         = 0x00;
   UpdateYY        = 0x00;
   UpdateMM        = 0x00;
   UpdateDD        = 0x00;
   NoOfRecs        = 0L;
   HeaderLen       = 0x00;
   RecordLen       = 0x00;
   MdxList         = NULL;
   NdxList         = NULL;
//   NtxList         = NULL;
   FreeIxList      = NULL;
   XFV             = 3;            /* Xbase file version */
//   strncpy(EofChar, "\x0D\x1A", 10);

#ifdef XB_LOCKING_ON
   AutoLock        = 1;

   CurLockType = -1;
   CurLockCount = 0;
   CurLockedRecNo = 0L;
   CurRecLockType = -1;
   CurRecLockCount = 0;
   CurMemoLockType = -1;
   CurMemoLockCount = 0;
#else
   AutoLock        = 0;
#endif

#ifdef XB_MEMO_FIELDS
   MemoHeader.BlockSize  = XB_DBT_BLOCK_SIZE;
   MemoHeader.Version    = 0x03;
   mfp                   = NULL;
   mbb                   = NULL;
   CurMemoBlockNo        = -1;
   mfield1               = 0;
   MStartPos             = 0;
   MFieldLen             = 0;
   NextFreeBlock         = 0L;
   FreeBlockCnt          = 0L;
   MNextBlockNo          = 0L;
   MNoOfFreeBlocks       = 0L;
#endif

//#ifdef XB_REAL_DELETE
   RealDelete = 0;
   FirstFreeRec = 0L;
   RealNumRecs = 0L;
//#endif
}

/************************************************************************/
//! Calculate checksum for the current record.
/*!
  Internal use only.
*/
xbLong xbDbf::CalcCheckSum()
{
   xbShort i;
   char *p;
   p = RecBuf;
   xbLong l = 0L;
   for( i = 0; i < RecordLen; i++ ) l+= *p++;
   return l;
}

/************************************************************************/
//! Set dbase version for the dbf file.
/*!
  Set dbase version.  Should only be used before creating a database with
  xbDbf::CreateDatabase().
  
  \param v version, either 3 or 4.
*/
xbShort xbDbf::SetVersion(xbShort v) {
  if (v == 0)
    return XFV;
  else
    if (v == 3) {
      XFV = 3;
#ifdef XB_MEMO_FIELDS
      MemoHeader.Version = 0x03;
#endif
      return XFV;
   } else
     if (v == 4) {
       XFV = 4;
#ifdef XB_MEMO_FIELDS
       MemoHeader.Version = 0x00;
#endif
       return XFV;
   }

  xb_error(XB_INVALID_OPTION);
}

/************************************************************************/
//! Write the dbf header
/*!
  Internal use only.

  \param PositionOption flag that indicates whether file postition should
  be moved.  non-zero if so, zero if not.
*/
xbShort xbDbf::WriteHeader( const xbShort PositionOption )
{
#if 0
   char buf[4];

   if (PositionOption)
      rewind( fp );
   if(fwrite(&Version, 4, 1, fp) != 1)
     xb_error(XB_WRITE_ERROR);

   memset( buf, 0x00, 4 );
   xbase->PutLong( buf, NoOfRecs );
   if (fwrite(buf, 4, 1, fp) != 1)
     xb_error(XB_WRITE_ERROR);

   if (PositionOption == 2)
      return XB_NO_ERROR;

   memset( buf, 0x00, 4 );
   xbase->PutShort( buf, HeaderLen );
   if (fwrite(buf, 2, 1, fp) != 1)
     xb_error(XB_WRITE_ERROR);

   memset( buf, 0x00, 4 );
   xbase->PutShort( buf, RecordLen );
   if (fwrite(buf, 2, 1, fp) != 1)
     xb_error(XB_WRITE_ERROR);

#ifdef XB_REAL_DELETE
   if(RealDelete)
   {
     memset(buf, 0x00, 4);
     xbase->PutULong(buf, FirstFreeRec);
     if (fwrite(buf, 4, 1, fp) != 1)
       xb_error(XB_WRITE_ERROR);

     memset(buf, 0x00, 4);
     xbase->PutULong(buf, RealNumRecs);
     if (fwrite(buf, 4, 1, fp) != 1)
       xb_error(XB_WRITE_ERROR);
   }
#endif
#else
   char buf[32];

   memset(buf, 0, 32);

   if(PositionOption)
     rewind(fp);

   memcpy(&buf[0], &Version, 4);
   xbase->PutLong(&buf[4], NoOfRecs);
   xbase->PutShort(&buf[8], HeaderLen );
   xbase->PutShort(&buf[10], RecordLen );

#ifdef XB_REAL_DELETE
   if(RealDelete)
   {
     xbase->PutULong(&buf[12], FirstFreeRec);
     xbase->PutULong(&buf[16], RealNumRecs);
   }
#endif
   if(fwrite(buf, 32, 1, fp) != 1)
     xb_error(XB_WRITE_ERROR);
#endif
   return XB_NO_ERROR;
}
/************************************************************************/
//! Read the dbf header.
/*!
  Internal use only.

  \param PositionOption
*/
xbShort xbDbf::ReadHeader( xbShort PositionOption )
{
#if 0
   char buf[4];
   if (PositionOption)
       rewind(fp);
   if (fread(&Version, 4, 1, fp) != 1)
     xb_error(XB_READ_ERROR);

   if (fread(buf, 4, 1, fp ) != 1)
     xb_error(XB_READ_ERROR);

   NoOfRecs = xbase->GetLong( buf );
   if(fread(buf, 2, 1, fp) != 1)
     xb_error(XB_READ_ERROR);

   HeaderLen = xbase->GetShort( buf );
   if(fread(buf, 2, 1, fp) != 1)
     xb_error(XB_READ_ERROR);

   RecordLen = xbase->GetShort(buf);

#ifdef XB_REAL_DELETE
   if(RealDelete)
   {
     if (fread(buf, 4, 1, fp ) != 1)
       xb_error(XB_READ_ERROR);
     FirstFreeRec = xbase->GetULong( buf );
     
     if (fread(buf, 4, 1, fp ) != 1)
       xb_error(XB_READ_ERROR);
     RealNumRecs = xbase->GetULong( buf );
   }
#endif
#else
   char buf[32];

   if(PositionOption)
     rewind(fp);

   if(fread(buf, 32, 1, fp) != 1)
     xb_error(XB_READ_ERROR);

   memcpy(&Version, buf, 4);
   NoOfRecs = xbase->GetLong(&buf[4]);
   HeaderLen = xbase->GetShort(&buf[8]);
   RecordLen = xbase->GetShort(&buf[10]);

#ifdef XB_REAL_DELETE
   if(RealDelete)
   {
     FirstFreeRec = xbase->GetULong(&buf[12]);
     RealNumRecs = xbase->GetULong(&buf[16]);
   }
#endif
#endif

   return XB_NO_ERROR;
}
/************************************************************************/
//! Determine if file name suffix is missing
/*!
  Internal use only.
*/
xbShort xbDbf::NameSuffixMissing( xbShort type, const char * name )
{
  /*  type 1 is DBF check
      type 2 is NDX check
      type 3 is MDX check
      type 4 is NTX check

      Returns 0 if suffix found
              1 if suffix not found, lower case
              2 is suffix not found, upper, case
*/

xbShort len;

   len = strlen( name );
   if( len <= 4 )
     if( name[len-1] >= 'A' && name[len-1] <= 'Z' )
       return 2;
     else
       return 1;

   if(  type == 1          && name[len-4] == '.' &&
      ( name[len-3] == 'd' || name[len-3] == 'D' ) && 
      ( name[len-2] == 'b' || name[len-2] == 'B' ) && 
      ( name[len-1] == 'f' || name[len-1] == 'F' )
     )
      return 0;

   if(  type == 2          && name[len-4] == '.' &&
      ( name[len-3] == 'n' || name[len-3] == 'N' ) && 
      ( name[len-2] == 'd' || name[len-2] == 'D' ) && 
      ( name[len-1] == 'x' || name[len-1] == 'X' )
     )
      return 0; 

   if(  type == 4          && name[len-4] == '.' &&
      ( name[len-3] == 'n' || name[len-3] == 'N' ) && 
      ( name[len-2] == 't' || name[len-2] == 'T' ) && 
      ( name[len-1] == 'x' || name[len-1] == 'X' )
     )
      return 0;

   if( name[len-5] >= 'A' && name[len-5] <= 'Z' )
       return 2;
   else
       return 1;
}

/************************************************************************/
//! Create the dbf file.
/*!
  This method attempts to create the XDB DBF file with the specified
  name (TableName) and schema (xbSchema s).  The OverLay switch is used to determine
  if an existing file should be overwritten or an error flagged if the
  file already exists.  The  record buffer is blanked (set to spaces).
  
  \param TableName name of the table
  \param s xbSchema
  \param Overlay One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>OverLay</th><th>Description</th></tr>
        <tr><td>XB_OVERLAY</td><td>Overwrite existing file if it exists</td></tr>
        <tr><td>XB_DONTOVERLAY</td><td>Report an error if file exists</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{OverLay} & \textbf{Description} \\ \hline \hline
        XB\_OVERLAY      & Overwrite existing file if it exists \\ \hline
        XB\_DONTOVERLAY  & Report an error if file exists \\ \hline
      \end{tabular}
    \endlatexonly
  \returns One of the following return codes:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_FILE_EXISTS</td><td>If the file exists and OverLay is XB_DONTOVERLAY</td></tr>
        <tr><td>XB_OPEN_ERROR</td><td>Couldn't open the file</td></tr>            <tr><td>XB_NO_MEMORY</td><td>Memory allocation error</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Couldn't write to disk</td><tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No Error \\ \hline
        XB\_FILE\_EXISTS & If the file exists and OverLay is XB\_DONTOVERAY \\ \hline
        XB\_OPEN\_ERROR & Couldn't open the file \\ \hline
        XB\_WRITE\_ERROR & Couldn't write to disk \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::CreateDatabase( const char * TableName, xbSchema * s,
    const xbShort Overlay )
{
/* future release - add logic to check number of fields and record length */

   xbShort    i, j, k, k2, rc;  /* , count; */

#ifdef XB_MEMO_FIELDS
   xbShort MemoSw = 0;
#endif

   DbfStatus = XB_CLOSED;                 

   /* Get the datafile name and store it in the class */
   rc = NameSuffixMissing( 1, TableName );
   DatabaseName = TableName;
   if( rc == 1)
      DatabaseName +=".dbf";
   else if( rc == 2 )
      DatabaseName += ".DBF";

   /* check if the file already exists */
   if((( fp = fopen( DatabaseName, "r" )) != NULL ) && !Overlay ){
     fclose( fp );
     xb_error(XB_FILE_EXISTS);
   }
   else if( fp ) fclose( fp );

   if(( fp = fopen( DatabaseName, "w+b" )) == NULL )
     xb_open_error(DatabaseName);

#ifdef XB_LOCKING_ON
   /* no buffering in multi user mode */
   setbuf( fp, NULL );
#endif

   /* count the number of fields and check paramaters */
   i = 0;
   while( s[i].Type != 0 )
   {
      NoOfFields++;

#ifdef XB_MEMO_FIELDS
      //  Make sure memo fields are 10 bytes long.  04/04/2002 dtb
      //
      if(s[i].Type == 'M')
        s[i].FieldLen = 10;
#endif /* XB_MEMO_FIELDS */

      RecordLen += s[i].FieldLen;

      if( s[i].Type != 'C' &&
          s[i].Type != 'N' &&
          s[i].Type != 'F' &&
          s[i].Type != 'D' &&
#ifdef XB_MEMO_FIELDS
          s[i].Type != 'M' &&
#endif /* XB_MEMO_FIELDS */
          s[i].Type != 'L' )
      {
        fclose( fp );
        InitVars();
        xb_error(XB_UNKNOWN_FIELD_TYPE);
      }

#ifdef XB_MEMO_FIELDS
//  8/18/03 types B and O dont exist - gkunkel
//      if( !MemoSw && ( s[i].Type=='M' || s[i].Type=='B' || s[i].Type=='O'))
      if( !MemoSw && ( s[i].Type=='M' ))
         MemoSw++;
#endif

// check for numeric fields which are too long
      if((s[i].Type == 'N' || s[i].Type == 'F') && s[i].FieldLen > 19 )
      {
        fclose( fp );
        InitVars();
        xb_error(XB_INVALID_FIELD_LEN);
      }

      i++;
   }
   RecordLen++;                  /* add one byte for 0x0D    */

   if(( RecBuf = (char *) malloc( RecordLen )) == NULL )
   {
      fclose( fp );
      InitVars();
      xb_memory_error;
   }

   if(( RecBuf2 = (char *) malloc( RecordLen )) == NULL ){
      free( RecBuf );
      fclose( fp );
      InitVars();
      xb_memory_error;
   }

   /* BlankRecord(); */
   memset( RecBuf, 0x20, RecordLen );
   memset( RecBuf2, 0x20, RecordLen );

   /* set class variables */

   Version = XFV & 0x7;            // file version - bit 0-2
#ifdef XB_MEMO_FIELDS
   if (MemoSw)
   {
     if(XFV & 0x7 == 3)
      Version |= 0x80;    // memo presence - bit 7
    else
      Version = 0x8b;
   }
#endif

   CurRec  = 0L;
   HeaderLen = 33 + NoOfFields * 32;
   xbDate d;
   UpdateYY = (d.YearOf() - 1900);
   if(XFV & 0x7 == 3)
     UpdateYY %= 100;   // dBASE III seems to do this, but IV does not.  DTB

   UpdateMM = d.MonthOf();
   UpdateDD = d.DayOf( XB_FMT_MONTH );

   /* write the header prolog */
   if(( rc = WriteHeader( 0 )) != XB_NO_ERROR ){
      free( RecBuf );
      free( RecBuf2 );
      fclose( fp );
      InitVars();
      xb_error(XB_WRITE_ERROR);
   }

/* the following logic was commented out 8/18/03 by gkunkel because 
   it does not appear to do anything except generate compiler warnings
   
   count = 20;
#ifdef XB_REAL_DELETE
   if(RealDelete)
     count = 12;
#endif
#if 0
   for( i = 0; i < count; i++ )
   {
      if(( fwrite( "\x00", 1, 1, fp )) != 1 )
      {
         free( RecBuf );
         free( RecBuf2 );
         fclose( fp );
         InitVars();
         xb_error(XB_WRITE_ERROR);
      }
   }
#endif
*/

   if((SchemaPtr=(xbSchemaRec *)malloc(NoOfFields*sizeof(xbSchemaRec)))==NULL){
      free( RecBuf );
      free( RecBuf2 );
      fclose( fp );
      InitVars();
      xb_memory_error;
   }
   memset( SchemaPtr, 0x00, ( NoOfFields * sizeof(xbSchemaRec)));

   /* write the field information into the header */
   for( i = 0, k = 1; i < NoOfFields; i++ ){
// next two lines updated 9/14/99 - reported by Piotr Klaban
// if field name too long ( > 10 bytes ) then  SIGSEGV errors
      memset( SchemaPtr[i].FieldName, 0x00, 11 );
      strncpy( SchemaPtr[i].FieldName, s[i].FieldName, 10 );

      SchemaPtr[i].Type = s[i].Type;
      if( s[i].Type == 'M' || s[i].Type == 'B' || s[i].Type == 'O' ) {
        /* memo fields are always 10 bytes */
        SchemaPtr[i].FieldLen = 10;
        SchemaPtr[i].NoOfDecs = 0;
      } else {
        SchemaPtr[i].FieldLen = s[i].FieldLen;
        SchemaPtr[i].NoOfDecs = s[i].NoOfDecs;
      }

      if( SchemaPtr[i].NoOfDecs > SchemaPtr[i].FieldLen ) {
         fclose( fp );
         free( SchemaPtr );
         free( RecBuf );
         free( RecBuf2 );
         InitVars(); 
         xb_error(XB_INVALID_SCHEMA);
      }

      k2 = k;
      k += SchemaPtr[i].FieldLen;

      if(( fwrite( &SchemaPtr[i], 1, 18, fp )) != 18 ) {
         fclose( fp );
         free( SchemaPtr );
         free( RecBuf );
         free( RecBuf2 );
         InitVars(); 
         xb_error(XB_WRITE_ERROR);
      }

      for( j = 0; j < 14; j++ ) {
         if(( fwrite( "\x00", 1, 1, fp )) != 1 ) {
            free( SchemaPtr );
            free( RecBuf );
            free( RecBuf2 );
            fclose( fp );
            InitVars(); 
            xb_error(XB_WRITE_ERROR);
         }
      }
      SchemaPtr[i].Address  = RecBuf  + k2;
      SchemaPtr[i].Address2 = RecBuf2 + k2;
   }

   /* write the header terminator */
// if(( fwrite( EofChar, 2, 1, fp )) != 1 ) {
   if(( fputc( XB_CHARHDR, fp )) != XB_CHARHDR ){
      fclose( fp );
      free( SchemaPtr );
      free( RecBuf );
      free( RecBuf2 );
      InitVars(); 
      xb_error(XB_WRITE_ERROR);
   }
#ifdef XB_MEMO_FIELDS
   if( MemoSw )
      if((rc = CreateMemoFile()) != XB_NO_ERROR)
      {
        fclose(fp);
        free(RecBuf);
        free(RecBuf2);
        InitVars();
        xb_error(rc);
      }
#endif

   DbfStatus = XB_OPEN;
   return xbase->AddDbfToDbfList(this, DatabaseName);
}
/************************************************************************/
//! Close the dbf file.
/*!
  This method attempts to close the XDB DBF file which was previously
  opened with either CreateDatabase() or OpenDatabase().  Deletes any
  memory allocated.  Automatically closes any open indexes associated
  with this data file.
  
  \param deleteIndexes if TRUE, the indexes (xbIndex instances) will also 
    be deleted (index files will not be deleted)
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File was not open</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No Error \\ \hline
        XB\_NOT\_OPEN\_ERROR & File was not open \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::CloseDatabase(bool deleteIndexes)
{
#if defined(XB_INDEX_ANY)
   xbIxList *i, *ti;
#endif

   if (DbfStatus == XB_CLOSED)
     return XB_NO_ERROR;
//     xb_error(XB_NOT_OPEN);

#if 0 //  shouldn't need any of this 05/08/2002 DTB
   if (DbfStatus == XB_UPDATED /*&& AutoUpdate*/ ) {
      xbDate d;
      UpdateYY = d.YearOf() - 1900;
     if(XFV == 3)
       UpdateYY %= 100; // dBASE III seems to do this, IV does not.  DTB
      UpdateMM = d.MonthOf();
      UpdateDD = d.DayOf( XB_FMT_MONTH );

      /* update the header */
      WriteHeader( 1 );

      /* write eof marker */
      fseek( fp, 0L, 2 );
//      fwrite( EofChar, 1, 1, fp );
      fputc( XB_CHAREOF, fp );
      PutRecord( CurRec );
   }
#endif

#if defined(XB_INDEX_ANY)
   i = NdxList;
   while (i)
   {
     i->index->CloseIndex();
     if(deleteIndexes)
       delete i->index;
     i = NdxList;
   }
/* free up unused nodes */
   i = FreeIxList;
   while( i ) {
     ti = i;
     i = i->NextIx;
     free(ti);
   }
#endif

   if (SchemaPtr){
     for( int j = 0; j < NoOfFields; j++ )
       if( SchemaPtr[j].fp ) delete SchemaPtr[j].fp;
     free( SchemaPtr );
   }
   if (RecBuf)
     free( RecBuf );
   if (RecBuf2)
     free( RecBuf2 );

#ifdef XB_MEMO_FIELDS
   if (mbb)
     free( mbb );         /* memo block buffer */
   if (mfp)
     fclose( mfp );       /* memo file pointer */
#endif

   xbase->RemoveDbfFromDbfList( this );
   if(fp)
     fclose( fp );
   InitVars();
   return XB_NO_ERROR;
}
/************************************************************************/
/* options  1 = Print header only
            2 = Field data only
            3 = Header and Field data */

//! Dump header information.
/*!
  \param Option One of the following:
     \htmlonly
       <p>
       <table border=2><tr><th>Option</th><th>Description</th></tr>
         <tr><td>1</td><td>Print header only</td></tr>
         <tr><td>2</td><td>Field data only</td></tr>
         <tr><td>3</td><td>Header and field data</td></tr>
       </table>
     \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Option} & \textbf{Description} \\ \hline \hline
        1 & Header only \\ \hline
        2 & Field data only \\ \hline
        3 & Header and field data \\ \hline
      \end{tabular}
    \endlatexonly
*/
#ifdef XBASE_DEBUG
xbShort xbDbf::DumpHeader( xbShort Option )
{
   int i;

   if( Option < 1 || Option > 3 )
     xb_error(XB_INVALID_OPTION);

   if( DbfStatus == XB_CLOSED )
     xb_error(XB_NOT_OPEN);

   std::cout << "\nDatabase file " << DatabaseName << std::endl << std::endl;

   if( Option != 2 )
   {
      std::cout << "File header data:" << std::endl;
      if( Version == 3 )
         std::cout << "Dbase III file" << std::endl;
      else if ( Version == 83 )
         std::cout << "Dbase III file with memo fields" << std::endl << std::endl;

      std::cout << "Last update date = " 
          << (int) UpdateMM << "/" << (int) UpdateDD << "/" << (int) UpdateYY % 100 << std::endl;  

      std::cout << "Header length    = " << HeaderLen << std::endl;
      std::cout << "Record length    = " << RecordLen << std::endl;
      std::cout << "Records in file  = " << NoOfRecs << std::endl << std::endl;
#ifdef XB_REAL_DELETE
      std::cout << "First Free Rec   = " << FirstFreeRec << std::endl << std::endl;
#endif      
   }
   if( Option != 1 )
   {

      std::cout << "Field Name   Type  Length  Decimals" << std::endl;
      std::cout << "----------   ----  ------  --------" << std::endl;
      for( i = 0; i <NoOfFields; i++ )
      {
         if( SchemaPtr[i].Type == 'C' && SchemaPtr[i].NoOfDecs > 0 )
           printf( "%10s    %1c     %4d    %4d\n", SchemaPtr[i].FieldName,
                  SchemaPtr[i].Type, SchemaPtr[i].FieldLen, 0 );
    else    
           printf( "%10s    %1c     %4d    %4d\n", SchemaPtr[i].FieldName,
                  SchemaPtr[i].Type, SchemaPtr[i].FieldLen, SchemaPtr[i].NoOfDecs );
      }
   }
   std::cout << std::endl;
   return XB_NO_ERROR;
}
#endif
/************************************************************************/
//! Open the DBF file.
/*!
  This method attempts to open the XDB DBF file with the specified
  name (TableName).  This method does not position to any particular
  record in the file.  The record buffer is blanked (set to spaces).

  \param TableName Name of table to open
  \returns One of the following:  
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_OPEN_ERROR</td><td>Couldn't open file</td></tr>
        <tr><td>XB_NO_MEMORY</td><td>Memory allocation error</td></tr>
        <tr><td>XB_NOT_XBASE</td><td>Not an XDB DBF file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
        XB\_OPEN\_ERROR & Couldn't open file \\ \hline
        XB\_NO\_MEMORY & Memory allocation error \\ \hline
        XB\_NOT\_XBASE & Not an XDB DBF file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::OpenDatabase( const char * TableName )
{
   xbShort i, j, NameLen, rc;
   char buf[33];
   char *p;

#ifdef XB_MEMO_FIELDS
   xbShort MemoSw = 0;
#endif

   /* verify the file is not already open */
   if ( DbfStatus != XB_CLOSED )
     xb_error(XB_ALREADY_OPEN);

   /* Get the datafile name and store it in the class */
   rc = NameSuffixMissing( 1, TableName );

   /* copy the file name to the class variable */
   DatabaseName = TableName;

   if( rc == 1)
      DatabaseName += ".dbf";
   else if( rc == 2 )
      DatabaseName += ".DBF";

   /* open the file */
   if(( fp = fopen(DatabaseName, "r+b")) == NULL )
   {
      //
      //  Try to open read only if failed to open read/write
      //
      if(( fp = fopen(DatabaseName, "rb")) == NULL )
         xb_open_error(DatabaseName);
   }

#ifdef XB_LOCKING_ON
   /* no buffering in multi user mode - may not see what others have updated */
   setbuf( fp, NULL );
#endif

#ifdef XB_LOCKING_ON
   if( AutoLock )
      if(( rc = LockDatabase( F_SETLKW, F_RDLCK, 0L )) != XB_NO_ERROR)
         return rc;
#endif

   /* copy the header into memory */
   if(( rc = ReadHeader( 1 )) != XB_NO_ERROR )
   {
      InitVars();
      return rc;
   }

   /* check the version */
   if( Version == 3 || Version == (char)0x83 )     /* dBASE III+ */
   {
      XFV = 3;
#ifdef XB_MEMO_FIELDS
      MemoHeader.Version = 0x03;
#endif
   }
   else if( Version == 4 || Version == (char)0x8B )    /* dBASE IV */
   {
      XFV = 4;
#ifdef XB_MEMO_FIELDS
      MemoHeader.Version = 0x00;
#endif
   }
   else if( Version == (char)0xf5 )    /* FoxPro */
   {
      XFV = 4;
#ifdef XB_MEMO_FIELDS
      MemoHeader.Version = 0x00;
#endif
   }
	else if( Version == (char)0x30 )  /* Visual Foxpro */
	{
		XFV = 4;
#ifdef XB_MEMO_FIELDS
      MemoHeader.Version = 0x00;
#endif
	}
   else
   {
     InitVars();
     xb_error(XB_NOT_XBASE);
   }

   // it would seem that dBASE III+ generates an UpdateYY value
   // of 0 for 2000 and dBASE IV uses 100, so I have removed the
   // check for UpdateYY being 0 (which might be valid).  DTB
   if (/*UpdateYY == 0 ||*/ UpdateMM == 0 || UpdateDD == 0 )
   {
     InitVars();
     xb_error(XB_NOT_XBASE);
   }

   /* calculate the number of fields */
    if( Version == (char)0x30 ) {
          NoOfFields = ( HeaderLen - 296 ) / 32 ;
	} else {
	   NoOfFields = ( HeaderLen - 33 ) / 32;
	}

   if(( RecBuf = (char *) malloc( RecordLen )) == NULL ) {
      fclose( fp );
      InitVars();
      xb_memory_error;
   }
   if(( RecBuf2 = (char *) malloc( RecordLen )) == NULL ) {
      fclose( fp );
      free( RecBuf );
      InitVars();
      xb_memory_error;
   }
   if((SchemaPtr=(xbSchemaRec *)malloc(NoOfFields*sizeof(xbSchemaRec)))==NULL){
      free( RecBuf );
      free( RecBuf2 );
      fclose( fp );
      InitVars();
      xb_memory_error;
   }
   memset( SchemaPtr, 0x00, ( NoOfFields * sizeof(xbSchemaRec)));

   /* copy field info into memory */
   for( i = 0, j = 1; i < NoOfFields; i++ ){
      fseek( fp, i*32+32, 0 );

//      fread( &SchemaPtr[i].FieldName, 1, 18, fp );
      fread( &buf, 1, 32, fp );
      p = buf;
      strncpy( SchemaPtr[i].FieldName, p, 10 );
      p += 11;
      SchemaPtr[i].Type = *p++;

      SchemaPtr[i].Address  = RecBuf + j;
      SchemaPtr[i].Address2 = RecBuf2 + j;

      SchemaPtr[i].FieldLen = *( p + 4 );
      SchemaPtr[i].NoOfDecs = *( p + 5 );

      if( SchemaPtr[i].Type == 'C' && SchemaPtr[i].NoOfDecs > 0 )
      {
        SchemaPtr[i].LongFieldLen = xbase->GetShort( p + 4 );
      j += SchemaPtr[i].LongFieldLen;
      }
      else
        j += SchemaPtr[i].FieldLen;
#ifdef XB_MEMO_FIELDS
      if( !MemoSw && (SchemaPtr[i].Type == 'M' || 
          SchemaPtr[i].Type == 'B' || SchemaPtr[i].Type == 'O' ))
         MemoSw++;
#endif
   } 
   CurRec = 0L;
   DbfStatus = XB_OPEN;
   BlankRecord();

#ifdef XB_MEMO_FIELDS
   if( MemoSw )   /* does this table have memo fields ? */
      if(( rc = OpenMemoFile()) != XB_NO_ERROR )
      {
         free( RecBuf );
         free( RecBuf2 );
         free( SchemaPtr );
         fclose( fp );
         InitVars(); 
         return rc;
      }
#endif

#ifdef XB_LOCKING_ON
   if( AutoLock )
      LockDatabase( F_SETLK, F_UNLCK, 0L );
#endif               /* XB_LOCKING_ON */

   return xbase->AddDbfToDbfList( this, DatabaseName );
}
/************************************************************************/
//! Blank the record buffer.
/*!
  Sets the record to spaces.
*/
xbShort xbDbf::BlankRecord( void )
{
   if( DbfStatus == XB_CLOSED ) 
     xb_error(XB_NOT_OPEN);

   if( DbfStatus != XB_UPDATED )
   {
      DbfStatus = XB_UPDATED;
      memcpy( RecBuf2, RecBuf, RecordLen );
   }
  
   memset( RecBuf, 0x20, RecordLen );
   return XB_NO_ERROR;
}
/************************************************************************/
//! Append the current record to the data file
/*!
  This method attempts to append the contents of the current record buffer
  to the end of the XDB DBF file and updates the file date and number of
  records in the file.  Also updates any open indexes associated with
  this data file.
  
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::AppendRecord( void )
{
   xbShort rc;      
   xbULong nextRecNo;

#if defined(XB_INDEX_ANY)
   xbIxList *i;
#endif

/* lock the database */
#ifdef XB_LOCKING_ON
   if( AutoLock )
      if(( rc = LockDatabase( F_SETLKW, F_WRLCK, 0L )) != XB_NO_ERROR) 
         return rc;
      
   if((rc = ReadHeader(1)) != XB_NO_ERROR)
   {
      if(AutoLock)
         LockDatabase( F_SETLK, F_UNLCK, 0L );
      return rc;
   }
#endif

/* lock any indexes */
#if defined(XB_INDEX_ANY)
#ifdef XB_LOCKING_ON
   i = NdxList;
   while( i && AutoLock )
   {
      if(( rc = i->index->LockIndex( F_SETLKW, F_WRLCK )) != XB_NO_ERROR ) 
         return rc;
      i = i->NextIx;
   }
#endif               /* XB_LOCKING_ON */
#endif

// if there are no duplicates, and no records set the CurRec to the
// last record + 1.  This is for EXP::RECNO()

/* check for any duplicate keys */
#if defined(XB_INDEX_ANY)
   i = NdxList;
   while( i )
   {
      if( i->index->UniqueIndex() )
      {
         i->index->CreateKey( 0, 0 );
         if( i->index->FindKey() == XB_FOUND )
           xb_error(XB_KEY_NOT_UNIQUE);
      }
      i = i->NextIx;
   }
#endif

#ifdef XB_REAL_DELETE
   if(RealDelete && FirstFreeRec)
     nextRecNo = FirstFreeRec;
   else
     nextRecNo = NoOfRecs + 1;
#else
   nextRecNo = NoOfRecs + 1;
#endif

   CurRec = NoOfRecs + 1;

#if defined(XB_INDEX_ANY)
/* update the indexes */
   i = NdxList;
   while( i )
   {
      if( !i->index->UniqueIndex() )          /* if we didn't prepare the key */
         if(( rc = i->index->CreateKey( 0, 0 )) != XB_NO_ERROR ) /* then do it before the add    */
            return rc;
      if(( rc =  i->index->AddKey(nextRecNo)) != XB_NO_ERROR )
   return rc;
      i->index->TouchIndex();
      i = i->NextIx;
   }
#endif              /* XB_INDEX_ANY */

#ifdef XB_REAL_DELETE
   char
     buf[4];
     
   if(RealDelete && FirstFreeRec)
   {
     /*
     **  Grab the next free rec no and put it in FirstFreeRec
     */
     if(fseek(fp, ((long) HeaderLen + ((FirstFreeRec - 1) * RecordLen) + 1), 0) != 0)
       xb_error(XB_SEEK_ERROR);
       
     if(fread(buf, 4, 1, fp) != 1)
       xb_error(XB_READ_ERROR);
       
     FirstFreeRec = xbase->GetULong(buf);
   }
   
   /*
   **  Okay, seek and write the record out
   */
   if(fseek(fp, ((long) HeaderLen + ((nextRecNo - 1) * RecordLen)), 0) != 0)
     xb_error(XB_SEEK_ERROR);
     
   if(fwrite( RecBuf, RecordLen, 1, fp) != 1)
     xb_error(XB_WRITE_ERROR);

   /*
   **  If we just appended the record to the file, then write the EOF char
   */
   if(nextRecNo == NoOfRecs + 1)
   {
//     if(fwrite( EofChar, 1, 1, fp ) != 1 )
     if( fputc( XB_CHAREOF, fp ) != XB_CHAREOF )
       xb_error(XB_WRITE_ERROR);
   }
#else
   /* write the last record */
   if( fseek( fp, ((long) HeaderLen + ( NoOfRecs * RecordLen )), 0 ) != 0 )
     xb_error(XB_SEEK_ERROR);

   if( fwrite( RecBuf, RecordLen, 1, fp ) != 1 )
     xb_error(XB_WRITE_ERROR);

   /* write the end of file marker */
//   if( fwrite( "\x0d", 1, 1, fp ) != 1 )
//   if( fwrite( EofChar, 1, 1, fp ) != 1 )
   if( fputc( XB_CHAREOF, fp ) != XB_CHAREOF )
     xb_error(XB_WRITE_ERROR);
#endif

   /* calculate the latest header information */
   xbDate d; 
   UpdateYY = d.YearOf() - 1900;
   if(XFV == 3)
      UpdateYY %= 100;  // dBASE III seems to do this, IV does not.  DTB
   UpdateMM = d.MonthOf();
   UpdateDD = d.DayOf( XB_FMT_MONTH );
#ifndef XB_REAL_DELETE
   NoOfRecs++;
#else
   if(RealDelete)
   {
     if(nextRecNo == NoOfRecs + 1)
       NoOfRecs++;
     RealNumRecs++;
   }
   else
     NoOfRecs++;
#endif
   CurRec = nextRecNo;
//   CurRec = NoOfRecs;

   /* rewrite the header record */
   if(( rc = WriteHeader( 1 )) != XB_NO_ERROR )
      return rc;

#ifdef XB_LOCKING_ON
   if( AutoLock )
      LockDatabase( F_SETLK, F_UNLCK, 0L );

#if defined(XB_INDEX_ANY)
   i = NdxList;
   while( i && AutoLock ){
      i->index->LockIndex( F_SETLK, F_UNLCK );
      i = i->NextIx;
   }
#endif               /* XB_INDEX_ANY  */
#endif               /* XB_LOCKING_ON */

   DbfStatus = XB_OPEN;
   return XB_NO_ERROR;
}
/************************************************************************/
//! Get a record from the data file
/*!
  This method attempts to retrieve the record specified by RecNo from the
  data file into the record buffer.  
  
  \param RecNo Record number to retrieve
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::GetRecord( xbULong RecNo )
{
/* 4/16/2000 - gk - made mods to record locking logic */

   int rc;

   if( DbfStatus == XB_CLOSED )
    xb_error(XB_NOT_OPEN);

#if 0 // This shouldn't be here 05/08/2002 DTB
   if( DbfStatus == XB_UPDATED /*&& AutoUpdate*/ )   /* update previous rec if necessary */
      if(( rc = PutRecord( CurRec )) != 0 )
         return rc;
#endif

#ifdef XB_LOCKING_ON
   if( AutoLock )
      if(( rc = LockDatabase( F_SETLKW, F_RDLCK, RecNo )) != 0 ) return rc;

   if((rc = ReadHeader(1)) != XB_NO_ERROR)
   {
      if(AutoLock)
         LockDatabase( F_SETLK, F_UNLCK, RecNo );
      return rc;
   }
#endif

   if( RecNo > NoOfRecs || RecNo == 0L )
     xb_error(XB_INVALID_RECORD);

   /* michael - modified code to avoid unecessary fseek work */
   /* commented out this code because it doesn't work in DOS/Win environments */
//   if( !CurRec || RecNo != CurRec+1 )
//   {
      if( fseek( fp, (long) HeaderLen+((RecNo-1L)*RecordLen), SEEK_SET ))
      {
#ifdef XB_LOCKING_ON
         LockDatabase( F_SETLK, F_UNLCK, RecNo );
#endif
         xb_error(XB_SEEK_ERROR);
      }
//   }

   if( fread( RecBuf, RecordLen, 1, fp ) != 1 )
   {
#ifdef XB_LOCKING_ON
      LockDatabase( F_SETLK, F_UNLCK, RecNo );
#endif
      xb_error(XB_READ_ERROR);
   }

#ifdef XB_LOCKING_ON
   if( AutoLock )
      LockDatabase( F_SETLKW, F_UNLCK, RecNo );
#endif

   DbfStatus = XB_OPEN;
   CurRec = RecNo;
   return XB_NO_ERROR;
}
/************************************************************************/
//! Get the first physical record in the data file
/*!
  Attempts to retrieve the first physical record from the data file into
  the record buffer.
  
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::GetFirstRecord( void )
{
   xbShort rc;
   if( NoOfRecs == 0 )
     xb_error(XB_INVALID_RECORD);

#if 0 // This shouldn't be here 05/08/2002 DTB
   if( DbfStatus == XB_UPDATED /*&& AutoUpdate*/ )  /* updatfe previous rec if necessary */
      if(( rc = PutRecord( CurRec )) != 0 )
         return rc;
#endif         

   rc = GetRecord( 1L );
#ifdef XB_REAL_DELETE
   if(!rc && RealDelete && RecordDeleted())
     rc = GetNextRecord();
#endif

   return rc;
}
/************************************************************************/
//! Get the last phyiscal record in the data file
/*!
  Attempts to retrieve the last physical record from the data file into
  the record buffer.
  
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_EOF</td><td>At end of file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_EOF & At end of file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::GetLastRecord( void )
{
   xbShort rc;
   if( NoOfRecs == 0 )
     xb_error(XB_INVALID_RECORD);

#if 0 // This shouldn't be here 05/08/2002 DTB
   if( DbfStatus == XB_UPDATED /*&& AutoUpdate*/ )  /* update previous rec if necessary */
      if(( rc = PutRecord( CurRec )) != 0 )
         return rc;
#endif

   rc = GetRecord( NoOfRecs );
#ifdef XB_REAL_DELETE
   if(!rc && RealDelete && RecordDeleted())
     rc = GetPrevRecord();
#endif

   return rc;
}
/************************************************************************/
//! Get the next physical record in the data file
/*!
  Attempts to retrieve the next physical record from the data file into
  the record buffer.

  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_EOF</td><td>At end of file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_EOF & At end of file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::GetNextRecord( void )
{
   xbShort rc;
   if( NoOfRecs == 0 ) {
     xb_error(XB_INVALID_RECORD);
   } else if( CurRec >= NoOfRecs ) {
     xb_eof_error;
   }

#if 0 // This shouldn't be here 05/08/2002 DTB
   if( DbfStatus == XB_UPDATED /*&& AutoUpdate*/ )  /* update previous rec if necessary */
      if(( rc = PutRecord( CurRec )) != 0 )
         return rc;
#endif         

   rc = GetRecord( ++CurRec );
#ifdef XB_REAL_DELETE
   while(!rc && RealDelete && RecordDeleted())
     rc = GetRecord(++CurRec);
#endif
  
   return rc;
}
/************************************************************************/
//! Get the previous physical record in the data file
/*!
  Attempts to retrieve the previous physical record from the data file into
  the record buffer.
  
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_BOF</td><td>At beginning of file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_BOF & At beginning of file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::GetPrevRecord( void )
{
   xbShort rc;
   if( NoOfRecs == 0 ) {
     xb_error(XB_INVALID_RECORD);
   } else if( CurRec <= 1L ) {
     xb_eof_error;
   }

#if 0 // This shouldn't be here 05/08/2002 DTB
   if( DbfStatus == XB_UPDATED /*&& AutoUpdate*/ )  /* update previous rec if necessary */
      if(( rc = PutRecord( CurRec )) != 0 )
         return rc;
#endif         

   rc = GetRecord( --CurRec );
#ifdef XB_REAL_DELETE
   while(!rc && RealDelete && RecordDeleted())
     rc = GetRecord(--CurRec);
#endif

   return rc;
}
/************************************************************************/
//! Dump record
/*!
  Dump the contents of the specified record to stdout.

  \param RecNo Record number of record to be dumped.
  \returns An error code (same as GetRecord()).
*/
xbShort xbDbf::DumpRecord( xbULong RecNo )
{
   int i, rc;
   char buf[1024];

   if( RecNo == 0 || RecNo > NoOfRecs )
     xb_error(XB_INVALID_RECORD);

   i = GetRecord( RecNo );
   if( i != XB_NO_ERROR )
      return i;

   std::cout << "\nREC NUMBER " << RecNo << "\n";

   if( RecordDeleted() )
      std::cout << "\nRecord deleted...\n";

   for( i = 0; i < NoOfFields; i++ )
   {
#ifdef XB_MEMO_FIELDS
      if(SchemaPtr[i].Type == 'M')
      {
        rc = GetMemoField(i, 1023, buf, F_SETLKW);
        if(rc != XB_NO_ERROR)
          std::cout << "Error " << rc << "\n";
      }
      else
        GetField( i, buf );
      std::cout << SchemaPtr[i].Type << " " << SchemaPtr[i].FieldName << " = '" << buf << "'\n";
#else
      GetField( i, buf );
      std::cout << SchemaPtr[i].FieldName << " = '" << buf << "'\n";
#endif
   }
   std::cout << "\n";
   return XB_NO_ERROR;
}
/************************************************************************/
//! Write the current record buffer to the current record in the data file.
/*!
  Attempts to write the contents of the record buffer to the current
  record in the data file.  Updates any open indexes.

  \sa PutRecord(xbULong RecNo)
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::PutRecord(void) {
   return PutRecord(CurRec);
}

//! Write the current record buffer to the specified record in the data file.
/*!
  Attempts to write the contents of the record buffer to the record specified
  by RecNo.  Updates any open indexes.
  
  \param RecNo Record number to which data should be written
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_LOCK_FAILED</td><td>Couldn't lock file</td></tr>
        <tr><td>XB_NOT_OPEN</td><td>File is not open</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
        <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Error writing to file</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_LOCK\_FAILED & Couldn't lock file \\ \hline
   XB\_NOT\_OPEN & File is not open \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_WRITE\_ERROR & Error writing to file \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::PutRecord(xbULong RecNo) 
{
   xbShort  
      rc;

//fprintf(stderr, "PutRecord Start\n");
#if defined(XB_INDEX_ANY)
   xbIxList *i;
#endif

   if( DbfStatus == XB_CLOSED )
     xb_error(XB_NOT_OPEN);

/* lock the database */
#ifdef XB_LOCKING_ON
   if( AutoLock )
   {
      if(( rc = LockDatabase( F_SETLKW, F_WRLCK, RecNo )) != XB_NO_ERROR ) 
      {
fprintf(stderr, "%s", DatabaseName.getData());
perror("failed record lock");
        return rc;
      }
      if(( rc = LockDatabase( F_SETLKW, F_WRLCK, 0L )) != XB_NO_ERROR )
      {
fprintf(stderr, "%s", DatabaseName.getData());
perror("failed file lock");       
         LockDatabase( F_SETLK, F_UNLCK, RecNo );
         return rc;
      }
      
      if((rc = ReadHeader(1)) != XB_NO_ERROR)
      {
         if(AutoLock)
         {
            LockDatabase( F_SETLK, F_UNLCK, RecNo );
            LockDatabase( F_SETLK, F_UNLCK, 0L );
         }
         return rc;
      }
   }
#endif

   if( RecNo > NoOfRecs || RecNo == 0L )
     xb_error(XB_INVALID_RECORD);

/* lock the indexes */
#if defined(XB_INDEX_ANY)
#ifdef XB_LOCKING_ON
   i = NdxList;
   while( i && AutoLock )
   {
      if(( rc = i->index->LockIndex( F_SETLKW, F_WRLCK )) != XB_NO_ERROR ) 
      {
fprintf(stderr, "%s", DatabaseName.getData());
perror("failed index lock");
        return rc;
      }
      i = i->NextIx;
   }
#endif               /* XB_LOCKING_ON */
#endif

#if defined(XB_INDEX_ANY)
   /* for any unique indexes that were updated, verify no unique keys exist */
   i = NdxList;
   while( i )
   {
      if( i->index->UniqueIndex() )
      {
        if(( i->KeyUpdated = i->index->KeyWasChanged()) == 1 )
        {
          i->index->CreateKey(0, 0);
          if( i->index->FindKey() == XB_FOUND && i->index->GetCurDbfRec() != RecNo)
            xb_error(XB_KEY_NOT_UNIQUE);
        }
      }
      i = i->NextIx;
   }
#endif

#if defined(XB_INDEX_ANY)
   /* loop through deleting old index keys and adding new index keys */
   i = NdxList;
   while( i )
   {
      if( !i->index->UniqueIndex() )
         i->KeyUpdated = i->index->KeyWasChanged();
      if( i->KeyUpdated )
      {
         i->index->CreateKey( 1, 0 );      /* load key buf w/ old values */
         if((rc = i->index->DeleteKey( CurRec )) != XB_NO_ERROR)
         {
#ifdef XB_LOCKING_ON
           if( AutoLock )
           {
             LockDatabase( F_SETLK, F_UNLCK, RecNo );
             LockDatabase( F_SETLK, F_UNLCK, 0L );
           }
#if defined(XB_INDEX_ANY)
           i = NdxList;
           while( i && AutoLock )
           {
             i->index->LockIndex( F_SETLK, F_UNLCK );
             i = i->NextIx;
           }
#endif               /* XB_INDEX_ANY  */
#endif               /* XB_LOCKING_ON */
           rc = 10000;
           return rc;
         }

         i->index->CreateKey( 0, 0 );
         if(( rc = i->index->AddKey(CurRec)) != XB_NO_ERROR ) 
         {
#ifdef XB_LOCKING_ON
           if( AutoLock )
           {
             LockDatabase( F_SETLK, F_UNLCK, RecNo );
             LockDatabase( F_SETLK, F_UNLCK, 0L );
           }
#if defined(XB_INDEX_ANY)
           i = NdxList;
           while( i && AutoLock )
           {
             i->index->LockIndex( F_SETLK, F_UNLCK );
             i = i->NextIx;
           }
#endif               /* XB_INDEX_ANY  */
#endif               /* XB_LOCKING_ON */
           return rc;
         }
         i->index->TouchIndex();
      }
      i = i->NextIx;
   }
#endif                        /* XB_INDEX_ANY */

   if( fseek( fp, (long) HeaderLen+((RecNo-1L)*RecordLen),0 ))
     xb_error(XB_SEEK_ERROR);

   if( fwrite( RecBuf, RecordLen, 1, fp ) != 1 )
     xb_error(XB_WRITE_ERROR);

   /* calculate the latest header information */
   xbDate d;
   UpdateYY = d.YearOf() - 1900;
   if(XFV == 3)
      UpdateYY %= 100;  // dBASE III seems to do this, IV does not.  DTB
   UpdateMM = d.MonthOf();
   UpdateDD = d.DayOf( XB_FMT_MONTH );

   /* rewrite the header record */
   if(( rc = WriteHeader( 1 )) != XB_NO_ERROR )
      return rc;

#ifdef XB_LOCKING_ON
   if( AutoLock )
   {
      rc = LockDatabase( F_SETLK, F_UNLCK, RecNo );
      if(rc)
      {
fprintf(stderr, "%s", DatabaseName.getData());
perror("failed record unlock");
      }       
      rc = LockDatabase( F_SETLK, F_UNLCK, 0L );
      if(rc)
      {
fprintf(stderr, "%s", DatabaseName.getData());
perror("failed file unlock");
      }       
   }

#if defined(XB_INDEX_ANY)
   i = NdxList;
   while( i && AutoLock )
   {
      i->index->LockIndex( F_SETLK, F_UNLCK );
      i = i->NextIx;
   }
#endif               /* XB_INDEX_ANY  */
#endif               /* XB_LOCKING_ON */

//fprintf(stderr, "PutRecord End\n");
   CurRec = RecNo;
   DbfStatus = XB_OPEN;
   return XB_NO_ERROR;
}

/************************************************************************/
//! Delete the current record
/*!
  Marks the current record as deleted or if "real" deletes are turned
  on (xbDbf::RealDeleteOn()) will delete the record and add it to the
  free record list.  Normal dBase behavior is to simply mark the record
  as deleted; the record will actually be deleted when the the DBF file 
  "packed" (xbDbf::PackDatabase()).  If "real" deletes are not on, a 
  record may be undeleted using xbDbf::UndeleteRecord().
  
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::DeleteRecord( void )
{
   xbULong newCurRec = 0;
   xbShort rc = XB_NO_ERROR;

#if defined(XB_INDEX_ANY)
   xbIxList *i;
#endif

   if(!RecBuf)   
     xb_error(XB_INVALID_RECORD);
     
   if(CurRec < 1 || CurRec > NoOfRecs)
     xb_error(XB_INVALID_RECORD);
          
/* lock the database */
#ifdef XB_LOCKING_ON
   if( AutoLock )
   {
      if(( rc = LockDatabase( F_SETLKW, F_WRLCK, CurRec )) != XB_NO_ERROR ) 
      {
        return rc;
      }
      if(( rc = LockDatabase( F_SETLKW, F_WRLCK, 0L )) != XB_NO_ERROR )
      {
         LockDatabase( F_SETLK, F_UNLCK, CurRec );
         return rc;
      }
      
      if((rc = ReadHeader(1)) != XB_NO_ERROR)
      {
         if(AutoLock)
         {
            LockDatabase( F_SETLK, F_UNLCK, CurRec );
            LockDatabase( F_SETLK, F_UNLCK, 0L );
         }
         return rc;
      }
   }
#endif

/* lock the indexes */
#if defined(XB_INDEX_ANY) && defined(XB_LOCKING_ON) && defined(XB_REAL_DELETE)
   i = NdxList;
   while( i && AutoLock )
   {
      if(( rc = i->index->LockIndex( F_SETLKW, F_WRLCK )) != XB_NO_ERROR )
        return rc;
      i = i->NextIx;
   }
#endif

/* remove keys from indexes */
#if defined(XB_REAL_DELETE) && defined(XB_INDEX_ANY)
   if(RealDelete)
   {
     i = NdxList;
     while(i)
     {
        i->index->CreateKey(0, 0);      /* load key buf */
        if(i->index->GetCurDbfRec() == (xbULong)CurRec)
        {
          i->index->DeleteKey(CurRec);
          newCurRec = i->index->GetCurDbfRec();
        }
        else
          i->index->DeleteKey(CurRec);
        i->index->TouchIndex();
        i = i->NextIx;
     }
   }
#endif

   RecBuf[0] = 0x2a;

#ifdef XB_REAL_DELETE
//fprintf(stderr, "DeleteRecord() -> RealDelete = %d\n", RealDelete);
   if(RealDelete)
   {
#ifdef XB_MEMO_FIELDS
     //
     //  Delete memo data for memo fields.
     //
     for(int f = 0; f < NoOfFields; f++ )
       if(GetFieldType(f) == 'M' && MemoFieldExists(f))
         UpdateMemoData(f, 0, 0, F_SETLKW);
#endif
      xbase->PutULong(&RecBuf[1], FirstFreeRec);
      FirstFreeRec = CurRec;
      RealNumRecs--;
      WriteHeader(1);
   }
#endif

   if(!RealDelete)
   {
      if( DbfStatus != XB_UPDATED )
      {
         DbfStatus = XB_UPDATED;
         memcpy( RecBuf2, RecBuf, RecordLen );
      }

      rc = PutRecord( CurRec );
   }
   else
   {
      if(fseek( fp, (long) HeaderLen + ((CurRec - 1L) * RecordLen), 0))
         xb_error(XB_SEEK_ERROR);

      if(fwrite( RecBuf, RecordLen, 1, fp ) != 1 )
         xb_error(XB_WRITE_ERROR);

      //
      //  Attempt to read in the record for the current location
      //  in the active index.
      //
      CurRec = newCurRec;
      if(CurRec)
         rc = GetRecord(CurRec);
      else
         BlankRecord();
   }
      
#ifdef XB_LOCKING_ON
   if(AutoLock)
   {
      LockDatabase( F_SETLK, F_UNLCK, CurRec );
      LockDatabase( F_SETLK, F_UNLCK, 0L );
   }

#if defined(XB_INDEX_ANY) && defined(XB_REAL_DELETE)
   i = NdxList;
   while( i && AutoLock )
   {
      i->index->LockIndex( F_SETLK, F_UNLCK );
      i = i->NextIx;
   }
#endif               /* XB_INDEX_ANY  */
#endif               /* XB_LOCKING_ON */

   return rc;
}
/************************************************************************/
//! Undelete the current record
/*!
  Marks the currect record as not deleted (i.e. removes the flag indicating
  the record is deleted).  This method may not be used (and will return
  an error code) if "real" deletes are on.
  
  \returns One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
        <tr><td>XB_INVALID_RECORD</td><td>Invalid record number</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No error \\ \hline
   XB\_INVALID\_RECORD & Invalid record number \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::UndeleteRecord( void )
{
   xbShort rc;
   
#ifdef XB_REAL_DELETE
   if(RealDelete)
     xb_error(XB_INVALID_RECORD);
#endif
   if( RecBuf ) 
   {
      if( DbfStatus != XB_UPDATED )
      {
         DbfStatus = XB_UPDATED;
         memcpy( RecBuf2, RecBuf, RecordLen );
      }
  
      RecBuf[0] = 0x20;
      if(( rc = PutRecord( CurRec )) != 0 )
         return rc;
   }
   else
     xb_error(XB_INVALID_RECORD);

   return 0;
}
/************************************************************************/
//! Determine if current record is deleted
/*!
  \returns TRUE (1) if the current record is marked as deleted or FALSE
    (0) if not.
*/
xbShort xbDbf::RecordDeleted( void )
{
   if( RecBuf && RecBuf[0] == 0x2a )
      return 1;
   else
      return 0;
}
/************************************************************************/
//! Pack data file
/*!
*/
xbShort xbDbf::PackDatafiles(void (*statusFunc)(xbLong itemNum, xbLong numItems))
{
   xbShort rc, i;
   FILE *t;
   xbLong l;
   char *target, *source;
   xbString TempDbfName;
   char   * Buf = 0;
#ifdef XB_MEMO_FIELDS
   char tbuf[4];
#endif

#ifdef XB_MEMO_FIELDS
   xbLong   len, BufSize;
   xbString TempDbtName;
   xbShort  MemoFields;
#endif  /* XB_MEMO_FIELDS */

   xbDbf Temp( xbase );
   rc = xbase->DirectoryExistsInName( DatabaseName );
   if (rc) {
      TempDbfName.assign(DatabaseName, 0, rc);
      TempDbfName += "TMPXBASE.DBF";
   } else
     TempDbfName = "TMPXBASE.DBF";

   if (( t = fopen( TempDbfName, "w+b" )) == NULL )
     xb_open_error(TempDbfName);

   /* copy file header */
   if(( rc = fseek( fp, 0, SEEK_SET )) != 0 )
      xb_io_error(XB_SEEK_ERROR, TempDbfName);

   for( i = 0; i < HeaderLen; i++ )
      fputc( fgetc( fp ), t );
   fputc( 0x1a, t );

   if( fclose( t ) != 0 )
     xb_io_error(XB_CLOSE_ERROR, TempDbfName);

#ifdef XB_MEMO_FIELDS
   if(( MemoFields = MemoFieldsPresent()) > 0 )
   {

      TempDbtName = TempDbfName;
      TempDbtName.putAt(TempDbtName.len()-1, 'T');

      if ((t = fopen( TempDbtName, "w+b" )) == NULL)
         xb_open_error(TempDbtName);

      l = 1L;
      memset( tbuf, 0x00, 4 );
      xbase->PutLong( tbuf, l );

      if ((fwrite(&tbuf, 4, 1, t)) != 1)
         xb_io_error(XB_WRITE_ERROR, TempDbfName);

      if( MemoHeader.Version == 0x03 )
      {
         for( i = 0; i < 12; i++ ) fputc( 0x00, t );
         fputc( 0x03, t );
         for( i = 0; i < 495; i++ ) fputc( 0x00, t );
      } else {
         for( i = 0; i < 4; i++ ) fputc( 0x00, t );
         if ((fwrite(&MemoHeader.FileName, 8, 1, t)) != 1)
      xb_io_error(XB_WRITE_ERROR, TempDbfName);
         for( i = 0; i < 4; i++ ) fputc( 0x00, t );
         memset( tbuf, 0x00, 2 );
         xbase->PutShort( tbuf, MemoHeader.BlockSize );
         if ((fwrite(&tbuf, 2, 1, t)) != 1)
       xb_io_error(XB_WRITE_ERROR, TempDbfName);

         for( i = 22; i < MemoHeader.BlockSize; i++ ) fputc( 0x00, t );
      }

      if( fclose( t ) != 0 )
            xb_io_error(XB_CLOSE_ERROR, TempDbfName);
   }
#endif   /* XB_MEMO_FIELDS */

   /* reopen as database */
   if(( rc = Temp.OpenDatabase( TempDbfName )) != XB_NO_ERROR )
     return rc;

#ifdef XB_REAL_DELETE
   if(RealDelete)
     Temp.RealDeleteOn();
   Temp.FirstFreeRec = 0;
   Temp.RealNumRecs = 0;
#endif   
   Temp.ResetNoOfRecs();
   Temp.WriteHeader(2);          // flush NoOfRecs=0 to disk
   target = Temp.GetRecordBuf();
   source = GetRecordBuf();

   for( l = 1; l <= PhysicalNoOfRecords(); l++ )
   {
      if(statusFunc && (l == 1 || !(l % 100) || l == PhysicalNoOfRecords()))
         statusFunc(l, PhysicalNoOfRecords());

      if(( rc = GetRecord( l )) != XB_NO_ERROR )
        return rc;

      if( !RecordDeleted())
      {
         memcpy( target, source, GetRecordLen());

#ifdef XB_MEMO_FIELDS
         BufSize = 0L;
         Buf = NULL;

         for( i = 0; i < NoOfFields; i++ )
         {
            if( GetFieldType( i ) == 'M' && MemoFieldExists( i ))
            {
               Temp.PutLongField(i, 0L);
               len = GetMemoFieldLen( i );
               if( len > BufSize )
               {
                  if( BufSize )
                     free( Buf );
                  if ((Buf = (char *)malloc(len)) == NULL)
                              xb_memory_error;
                  BufSize = len;
               }
               GetMemoField( i, len, Buf, -1 );
               Temp.UpdateMemoData( i, len, Buf, -1 );
            }
         }
#endif
         if(( rc = Temp.AppendRecord()) != XB_NO_ERROR )
         {
           if(Buf) free(Buf);
           return rc;
         }
      }
   }
   if( Buf ) free( Buf );
   Temp.CloseDatabase();

   if (fclose(fp) != 0)
         xb_io_error(XB_CLOSE_ERROR, DatabaseName);

   if(remove(DatabaseName) != 0)
         xb_io_error(XB_WRITE_ERROR, DatabaseName);

   if(rename(TempDbfName, DatabaseName) != 0)
         xb_io_error(XB_WRITE_ERROR, TempDbfName);

#ifdef XB_MEMO_FIELDS
   if( MemoFields )
   {

 //     len = DatabaseName.len();
 //     len--;
 //     lb = DatabaseName[len];

      int len = DatabaseName.len() - 1;
      char lb = DatabaseName[len];

      if( lb == 'F' )
         DatabaseName.putAt(len, 'T');
      else
         DatabaseName.putAt(len, 't');

      if(fclose(mfp) != 0)       /* thanks Jourquin */
         xb_io_error(XB_CLOSE_ERROR, TempDbtName);

      if (remove(DatabaseName) != 0)
      {
         DatabaseName.putAt(len, lb);
         xb_io_error(XB_WRITE_ERROR, DatabaseName);
      }
      if( rename( TempDbtName, DatabaseName ) != 0 )
      {
         DatabaseName.putAt(len, lb);
         xb_io_error(XB_WRITE_ERROR, DatabaseName);
      }

      if(( mfp = fopen( DatabaseName, "r+b" )) == NULL )
        xb_open_error(DatabaseName);

      if(( rc = GetDbtHeader(1)) != 0 )
      {
        fclose( mfp );
        return rc;
      }
#ifdef XB_LOCKING_ON
        /* no buffering in multi user mode */
        setbuf( mfp, NULL );
#endif

      DatabaseName.putAt(len, lb);
   }

#endif /* XB_MEMO_FIELDS */

   if(( fp = fopen( DatabaseName, "r+b" )) == NULL )
     xb_open_error(DatabaseName);

#ifdef XB_LOCKING_ON
   /* no buffering in multi user mode */
   setbuf( fp, NULL );
#endif

   return XB_NO_ERROR;
}
/************************************************************************/
//! Pack the database
/*!
  This method removes all records marked for deletion from an Xbase (.DBF)
  file, reindexes any open index files, and also reorganizes any memo fields
  stored in a .DBT memo file.

  \param LockWaitOption One of the following:
    \htmlonly
      <p>
      <table border=2><tr><th>LockWaitOption</th><th>Description</th></tr>
        <tr><td>F_SETLK</td><td>Return immediately if the DBF file cannot be locked</td></tr>
        <tr><td>F_SETLKW</td><td>Wait for lock on DBF file to succeed</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{LockWaitOption} & \textbf{Description} \\ \hline \hline
        F\_SETLK & Return immediately if DBF file cannot be locked \\ \hline
        F\_SETLKW  & Wait for lock on DBF file to succeed \\ \hline
      \end{tabular}
    \endlatexonly
    
  \returns One of the following return codes:
    \htmlonly
      <p>
      <table border=2><tr><th>Return Code</th><th>Description</th></tr>
        <tr><td>XB_NO_ERROR</td><td>No error</td></tr>
   <tr><td>XB_CLOSE_ERROR</td><td>Unable to close intermediate work file</td></tr>
   <tr><td>XB_OPEN_ERROR</td><td>Could not open file</td></tr>
   <tr><td>XB_NO_MEMORY</td><td>Memory allocation error</td></tr>
        <tr><td>XB_WRITE_ERROR</td><td>Couldn't write to disk</td></tr>
   <tr><td>XB_SEEK_ERROR</td><td>Error seeking file</td></tr>
   <tr><td>XB_LOCK_FAILED</td><td>Unable to lock file or index</td></tr>
      </table>
    \endhtmlonly
    \latexonly
      \\
      \\
      \begin{tabular}{|l|l|} \hline
        \textbf{Return Code} & \textbf{Description} \\ \hline \hline
        XB\_NO\_ERROR & No Error \\ \hline
   XB\_CLOSE\_ERROR & Unable to close intermediate work file \\ \hline
        XB\_OPEN\_ERROR & Couldn't open the file \\ \hline
   XB\_NO\_MEMORY & Memory allocation error \\ \hline
        XB\_WRITE\_ERROR & Couldn't write to disk \\ \hline
   XB\_SEEK\_ERROR & Error seeking file \\ \hline
   XB\_LOCK\_FAILED & Unable to lock file or index \\ \hline
      \end{tabular}
    \endlatexonly
*/
xbShort xbDbf::PackDatabase(xbShort LockWaitOption,
                            void (*packStatusFunc)(xbLong itemNum, xbLong numItems),
                            void (*indexStatusFunc)(xbLong itemNum, xbLong numItems))
{
   xbShort rc;

   /* lock all open files and indexes */
   if(( rc = ExclusiveLock( LockWaitOption )) != XB_NO_ERROR ) return rc;

   if(( rc = PackDatafiles(packStatusFunc)) != XB_NO_ERROR )
   {
      ExclusiveUnlock();
      return rc;
   }

   /* refresh file header */
   if(( rc = ReadHeader(1)) != XB_NO_ERROR )
      return rc;

   if(( rc = RebuildAllIndices(indexStatusFunc)) != XB_NO_ERROR ) 
      return rc;

   ExclusiveUnlock();
   return XB_NO_ERROR;
}
/************************************************************************/
//! Copy DBF structure
/*!
*/
xbShort xbDbf::CopyDbfStructure(const char *NewFileName, xbShort Overlay) {
   xbShort rc, i;
   xbString ndfn;       /* new dbf file name */
   char  ch;
#ifdef XB_MEMO_FIELDS
   char buf[9];
   xbShort ct, NameLen;
   xbString MemoName;
#endif
   FILE  *t;

   /* build the new file name */
   rc = NameSuffixMissing( 1, NewFileName );
   ndfn = NewFileName;
   if( rc == 1 )
     ndfn += ".dbf";
   else if( rc == 2 )
     ndfn += ".DBF";

   /* check if the file exists and Overlay is on */
   if (((t = fopen( ndfn, "r" )) != NULL ) && !Overlay) {
      fclose(t);
      xb_io_error(XB_FILE_EXISTS, ndfn);
   }

   /* open new file */
   if ((t = fopen(ndfn, "w+b")) == NULL)
     xb_open_error(ndfn);

   /* copy the file header */
   if (( rc = fseek( fp, 0, SEEK_SET )) != 0 )
     xb_io_error(XB_SEEK_ERROR, ndfn);

   fputc( fgetc( fp ), t );

   /* do the date */
   xbDate d;
   ch = d.YearOf() - 1900;
   if(XFV == 3)
     ch %= 100;          // dBASE III+ does this, dBASE IV does not.
   fputc( ch, t );
   ch = d.MonthOf();
   fputc( ch, t );
   ch = d.DayOf( XB_FMT_MONTH );
   fputc( ch, t );

   /* record count */
   for( i = 0; i < 4; i++ ) fputc( 0x00, t );

   if((rc = fseek(fp, 7L, SEEK_CUR)) != 0) {
      fclose( t );
      xb_io_error(XB_SEEK_ERROR, DatabaseName);
   }
   for( i = 0; i < 4; i++ )
     fputc( fgetc( fp ), t );

   for( i = 0; i < 17; i++ )
     fputc( 0x00, t );

   if ((rc = fseek( fp, 17L, SEEK_CUR )) != 0) {
     fclose( t );
     xb_io_error(XB_SEEK_ERROR, DatabaseName);
   }

   for ( i = 29; i < HeaderLen; i++ )
     fputc( fgetc( fp ), t );

   fputc( 0x1a, t );
   fclose( t );

#ifdef XB_MEMO_FIELDS
   if( MemoFieldsPresent())
   {
      MemoName = ndfn;

      NameLen = MemoName.len();
      NameLen--;
      if( MemoName.getCharacter( NameLen ) == 'F' )
         MemoName.putAt(NameLen, 'T');
      else
         MemoName.putAt(NameLen, 't');

      if(( t = fopen( MemoName, "w+b" )) == NULL )
        xb_open_error(MemoName);

      memset( buf, 0x00, 4 );
      xbase->PutLong( buf, 1L );
      if(( fwrite( &buf, 4, 1, t )) != 1 ) 
      {
         fclose( t );
         xb_io_error(XB_WRITE_ERROR, ndfn);
      }
      if( MemoHeader.Version == 0x03 )
      {
         for( i = 0; i < 12; i++ ) fputc( 0x00, t );
         fputc( 0x03, t );
         for( i = 0; i < 495; i++ ) fputc( 0x00, t );
      }
      else
      {
         for( i = 0; i < 4; i++ ) fputc( 0x00, t );  // put 4 bytes 0x00
         memset( buf, 0x00, 9 );
         NameLen = ndfn.len();
         for( i = 0, ct = 0; i < NameLen; i++ )
           if( ndfn.getCharacter( i ) == PATH_SEPARATOR )
           {
             ct = i;
             ct++;
           }

         for( i = 0; i < 8 && ndfn[i+ct] != '.'; i++ )
            buf[i] = ndfn[i+ct];

         fwrite( &buf, 8, 1, t );
         for( i = 0; i < 4; i++ ) fputc( 0x00, t );
         memset( buf, 0x00, 2 );
         xbase->PutShort( buf, MemoHeader.BlockSize );
         if(( fwrite( &buf, 2, 1, t )) != 1 )
         {
            fclose(t);
            xb_io_error(XB_WRITE_ERROR, ndfn);
         }
         for( i = 22; i < MemoHeader.BlockSize; i++ ) fputc( 0x00, t );
      }
   }
   fclose( t );
#endif   // XB_MEMO_FIELDS
   return XB_NO_ERROR;
}
/************************************************************************/
//! Add index to list
/*!
  Adds the specified index to the list of indexes maintained by the 
  dbf.
  
  \param n index to add
  \param IndexName name of index
*/
#if defined(XB_INDEX_ANY)
xbShort xbDbf::AddIndexToIxList(xbIndex * n, const char *IndexName)
{
   xbIxList *i, *s, *t;

   if( !FreeIxList ){
      if((i = (xbIxList *) malloc(sizeof(xbIxList))) == NULL) {
        xb_memory_error;
      }
   }
   else
   {
      i = FreeIxList;
      FreeIxList = i->NextIx;
   } 
   memset(i, 0x00, sizeof(xbIxList));

   i->IxName  = IndexName;
   i->index   = n;

   s = NULL;
   t = NdxList;
   while( t && strcmp( t->IxName, IndexName ) < 0 )
   {
      s = t;
      t = t->NextIx;
   }
   i->NextIx = t;
   if( s == NULL )
      NdxList = i;
   else
      s->NextIx = i;
   return 0;
}
#endif
/************************************************************************/
//! Rebuild all index files
/*!
*/
xbShort xbDbf::RebuildAllIndices(void (*statusFunc)(xbLong itemNum, xbLong numItems))
{
#if defined(XB_INDEX_ANY)
  xbShort rc;
  xbIxList *n;

   n = NdxList;
   while( n )
   {
      if(( rc = n->index->ReIndex(statusFunc)) != XB_NO_ERROR )
      {
         ExclusiveUnlock();
         return rc;
      }
      n = n->NextIx;
   }
#endif
  return XB_NO_ERROR;
}
/************************************************************************/
//! Delete all records
/*!
*/
xbShort xbDbf::DeleteAll( xbShort Option )
{
   xbShort rc;

   if(( NoOfRecords()) == 0 ) 
      return XB_NO_ERROR;
   if(( rc = GetFirstRecord()) != XB_NO_ERROR )
      return rc;

   if( Option == 0 )   /* delete all option */
   {
     while( 1 ){
       if( !RecordDeleted())
          if(( rc = DeleteRecord()) != XB_NO_ERROR )
             return rc;
       if(( rc = GetNextRecord()) != XB_NO_ERROR )
          break;
     }
   }
   else   /* undelete all option */
   {
      while( 1 )
      {
         if( RecordDeleted())
            if(( rc = UndeleteRecord()) != XB_NO_ERROR )
               return rc;
#ifdef HAVE_EXCEPTIONS
         try {
#endif
         if(( rc = GetNextRecord()) != XB_NO_ERROR )
            break;
#ifdef HAVE_EXCEPTIONS
          } catch (xbEoFException &) {
            return XB_NO_ERROR;
          }
#endif
      }
   }
   if( rc == XB_EOF )
      return XB_NO_ERROR;
   else
      return rc;
}
/************************************************************************/
//! Delete all records and pack data file
/*!
*/
xbShort xbDbf::Zap( xbShort WaitOption )
{
#ifdef XB_MEMO_FIELDS
   xbShort MemosExist;
#endif

   xbShort NameLen, rc;
   xbString TempDbfName;

   rc = xbase->DirectoryExistsInName( DatabaseName );

   if (rc) {
     TempDbfName.assign(DatabaseName, 0, rc);
      TempDbfName += "TMPXBASE.DBF";
   } else
       TempDbfName = "TMPXBASE.DBF";

   if(( rc = CopyDbfStructure( TempDbfName, 1 )) != XB_NO_ERROR) {
      return rc;
   }

#ifdef XB_MEMO_FIELDS
   MemosExist = MemoFieldsPresent();
#endif

   if (( rc = ExclusiveLock( WaitOption )) != XB_NO_ERROR )
      return rc;

   if (( rc = remove( DatabaseName )) != 0 )
   {
      ExclusiveUnlock();
      xb_io_error(XB_WRITE_ERROR, DatabaseName);
   }

   if (( rc = rename( TempDbfName, DatabaseName )) != 0 ) {
      ExclusiveUnlock();
      xb_io_error(XB_WRITE_ERROR, DatabaseName);
   }

   if((fp = fopen( DatabaseName, "r+b" )) == NULL) {
     ExclusiveUnlock();
     xb_open_error(DatabaseName);
   }
#ifdef XB_LOCKING_ON
   setbuf( fp, NULL );
#endif
   ReadHeader( 1 );

#ifdef XB_MEMO_FIELDS
   if( MemosExist )
   {
      fclose( mfp );
      int dbnlen = DatabaseName.len() - 1;
      char lb = DatabaseName[dbnlen];
      if( lb == 'F' ) {
         DatabaseName.putAt(dbnlen, 'T');
    TempDbfName.putAt(dbnlen, 'T');
      } else {
         DatabaseName.putAt(dbnlen, 't');
    TempDbfName.putAt(dbnlen, 't');
      }

      if(( rc = remove( DatabaseName )) != 0 )
      {
         ExclusiveUnlock();
         xb_open_error(DatabaseName);
      }
      if(( rc = rename( TempDbfName, DatabaseName )) != 0 )
      {
   ExclusiveUnlock();
   xb_open_error(DatabaseName);
      }
      if(( mfp = fopen( DatabaseName, "r+b" )) == NULL ){
   ExclusiveUnlock();
   xb_open_error(DatabaseName);
      }
#ifdef XB_LOCKING_ON
      setbuf( mfp, NULL );
#endif

      GetDbtHeader(1);
      DatabaseName.putAt(dbnlen, lb);
   }      
#endif   // XB_MEMO_FIELDS

   if(( rc = RebuildAllIndices()) != XB_NO_ERROR )
   {
      ExclusiveUnlock();
      return rc;
   }
   ExclusiveUnlock();
   return XB_NO_ERROR;
}
/************************************************************************/
//! Remove an index from the list
/*!
*/
#if defined(XB_INDEX_ANY)
xbShort xbDbf::RemoveIndexFromIxList(xbIndex * n) {
   xbIxList *i, *s;

   i = NdxList;
   s = NULL;
   while( i )
   {
      if( i->index == n )
      {
         /* remove it from current chain */
         if( s )
           s->NextIx = i->NextIx;
         else
           NdxList = i->NextIx;

         /* add i to the current free chain */
         i->NextIx = FreeIxList;
         FreeIxList = i;
         FreeIxList->IxName = (const char *)NULL;
         FreeIxList->index = NULL;
         break;
      }
      else
      {
         s = i;
         i = i->NextIx;
      }
   }
   return XB_NO_ERROR;
} 
#endif

/************************************************************************/
//! Gets the number of records in the data file
/*!
*/
xbLong    
xbDbf::NoOfRecords(void)
{
  xbLong numRecs;

/* lock the database */
#ifdef XB_LOCKING_ON
  xbShort
    rc;

   if( AutoLock )
   {
      if(( rc = LockDatabase( F_SETLKW, F_RDLCK, 0L )) != XB_NO_ERROR )
      {
         return 0;
      }

      if((rc = ReadHeader(1)) != XB_NO_ERROR)
      {
         if(AutoLock)
         {
            LockDatabase( F_SETLK, F_UNLCK, 0L );
         }
         return 0;
      }
   }
#endif

#ifndef XB_REAL_DELETE
  numRecs = NoOfRecs;
#else
  numRecs = RealDelete ? RealNumRecs : NoOfRecs;
#endif

#ifdef XB_LOCKING_ON
   if(AutoLock)
   {
      LockDatabase( F_SETLK, F_UNLCK, 0L );
   }
#endif

  return numRecs;
}

/************************************************************************/
//! Get the physical number of records in the data file
/*!
*/
xbLong    
xbDbf::PhysicalNoOfRecords(void)
{
  xbLong numRecs;
    
/* lock the database */
#ifdef XB_LOCKING_ON
  xbShort
    rc;
    
   if( AutoLock )
   {
      if(( rc = LockDatabase( F_SETLKW, F_RDLCK, 0L )) != XB_NO_ERROR )
      {
         return 0;
      }
      
      if((rc = ReadHeader(1)) != XB_NO_ERROR)
      {
         if(AutoLock)
         {
            LockDatabase( F_SETLK, F_UNLCK, 0L );
         }
         return 0;
      }
   }
#endif

  numRecs = NoOfRecs;

#ifdef XB_LOCKING_ON
   if(AutoLock)
   {
      LockDatabase( F_SETLK, F_UNLCK, 0L );
   }
#endif

  return numRecs;
}

#if defined(XB_INDEX_ANY)
//! Get the number of currently open indexes for data file
/*!
*/
xbShort   
xbDbf::IndexCount(void)
{
  xbShort
    count;
    
  xbIxList
    *i;

  for(count = 0, i = NdxList; i; i = i->NextIx, count++)
    ;

  return count;
}

//! Get a specific index
/*!
*/
xbIndex *
xbDbf::GetIndex(xbShort indexNum)
{
  xbIxList
    *i;

  i = NdxList;
  while(indexNum && i)
  {
    indexNum--;
    i = i->NextIx;
  }

  if(i)
    return i->index;

  return 0;
}
#endif // XB_INDEX_ANY

void
xbDbf::Flush()
{
  if(fp)
    fflush(fp);

#ifdef XB_MEMO_FIELDS
  if(mfp)
    fflush(mfp);
#endif

#if defined(XB_INDEX_ANY)
  xbIxList
    *i;

  i = NdxList;
  while(i)
  {
    i->index->Flush();
    i = i->NextIx;
  }
#endif
}

