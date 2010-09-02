/*  $Id: dbf.h,v 1.15 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains the Class definition for a xbDBF object.

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


#ifndef __XB_DBF_H__
#define __XB_DBF_H__

#ifdef __GNUG__
#pragma interface
#endif

#ifdef __WIN32__
#include <xbase/xbconfigw32.h>
#else
#include <xbase/xbconfig.h>
#endif

#include <xbase/xtypes.h>
#include <xbase/xdate.h>

#include <iostream>
#include <stdio.h>

/*! \file dbf.h
*/

#if defined(XB_INDEX_ANY)
   class XBDLLEXPORT xbIndex;
   class XBDLLEXPORT xbNdx;
   class XBDLLEXPORT xbNtx;
#endif

/*****************************/
/* Field Types               */

#define XB_CHAR_FLD      'C'
#define XB_LOGICAL_FLD   'L'
#define XB_NUMERIC_FLD   'N'
#define XB_DATE_FLD      'D'
#define XB_MEMO_FLD      'M'
#define XB_FLOAT_FLD     'F'

/*****************************/
/* File Status Codes         */

#define XB_CLOSED  0
#define XB_OPEN    1
#define XB_UPDATED 2

/*****************************/
/* Other defines             */

#define XB_OVERLAY     1
#define XB_DONTOVERLAY 0

#define XB_CHAREOF  '\x1A'         /* end of DBF        */
#define XB_CHARHDR  '\x0D'         /* header terminator */

//! Used to define the fields in a database (DBF file).
/*!
  Generally one would define an xbSchema array to be passed
  to xbDbf::CreateDatabase() to define the fields in the database.

  For example, one might create a declaration as follows:
  
  \code  
  xbSchema MyRecord[] = 
  {
    { "FIRSTNAME", XB_CHAR_FLD,     15, 0 },
    { "LASTNAME",  XB_CHAR_FLD,     20, 0 },
    { "BIRTHDATE", XB_DATE_FLD,      8,  0 },
    { "AMOUNT",    XB_NUMERIC_FLD,   9,  2 },
    { "SWITCH",    XB_LOGICAL_FLD,   1,  0 },
    { "FLOAT1",    XB_FLOAT_FLD,     9,  2 },
    { "FLOAT2",    XB_FLOAT_FLD,     9,  1 },
    { "FLOAT3",    XB_FLOAT_FLD,     9,  2 },
    { "FLOAT4",    XB_FLOAT_FLD,     9,  3 },
    { "MEMO1",     XB_MEMO_FLD,     10, 0 },
    { "ZIPCODE",   XB_NUMERIC_FLD,   5,  0 },      
    { "",0,0,0 }
  };
  \endcode
  
  Note that the last xbSchema in an array must be a "null" entry like the 
  one above:
  
  \code
    { "",0,0,0 }
  \endcode
  
  To indicate the end of the array.
*/
struct XBDLLEXPORT xbSchema {
   char      FieldName[11];
   char      Type;
// xbUShort  FieldLen;       /* does not work */
// xbUShort  NoOfDecs;       /* does not work */
   unsigned  char FieldLen;  /* fields are stored as one byte on record*/
   unsigned  char NoOfDecs;
};

//! Defines a field in an XBase file header (DBF file header)
/*!
  This structure is only used internally by the xbDbf class.
*/
struct XBDLLEXPORT xbSchemaRec {
   char     FieldName[11];
   char     Type;            /* field type */
   char     *Address;        /* pointer to field in record buffer 1 */
// xbUShort FieldLen;        /* does not work */
// xbUShort NoOfDecs;        /* does not work */
   unsigned char FieldLen;   /* fields are stored as one byte on record */
   unsigned char NoOfDecs;
   char     *Address2;       /* pointer to field in record buffer 2 */
   char     *fp;             /* pointer to null terminated buffer for field */
                             /* see method GetString */
   xbShort  LongFieldLen;    /* to handle long field lengths */
};

//! xbIxList struct
/*!
  Internal use only.
*/
struct XBDLLEXPORT xbIxList {
   xbIxList * NextIx;
   xbString IxName;
#if defined(XB_INDEX_ANY)
   xbIndex  * index;
   xbShort  Unique;
   xbShort  KeyUpdated;
#endif
};

//! xbMH struct
/*!
  Internal use only.
*/

#ifdef XB_MEMO_FIELDS
struct XBDLLEXPORT xbMH{                      /* memo header                    */
   xbLong  NextBlock;             /* pointer to next block to write */
   char    FileName[8];           /* name of dbt file               */
   char    Version;               /* not sure                       */
   xbShort BlockSize;             /* memo file block size           */
};
#endif

//! xbDbf class
/*!
  The xbDbf class encapsulates an xbase DBF database file.  It includes
  all dbf access, field access, and locking methods.
*/
class XBDLLEXPORT xbDbf {

public:
   xbDbf( xbXBase * );
   virtual ~xbDbf();

   xbXBase  *xbase;               /* linkage to main base class */
//   char EofChar[10];

/* datafile methods */
#if defined(XB_INDEX_ANY)
   xbShort   AddIndexToIxList(xbIndex *, const char *IndexName);
   xbShort   RemoveIndexFromIxList( xbIndex * );
#endif
   xbShort   AppendRecord( void );
   xbShort   BlankRecord( void );
   xbLong    CalcCheckSum( void );
   xbShort   CloseDatabase(bool deleteIndexes = 0);
   xbShort   CopyDbfStructure( const char *, xbShort );
   xbShort   CreateDatabase( const char * Name, xbSchema *, const xbShort Overlay );
   //! Return the current position in the dbf file
   /*!
   */
   xbLong    DbfTell( void ) { return ftell( fp ); }
   //! Delete all records
   /*!
   */
   xbShort   DeleteAllRecords( void ) { return DeleteAll(0); }
   xbShort   DeleteRecord( void );
#ifdef XBASE_DEBUG
   xbShort   DumpHeader( xbShort );
#endif
   xbShort   DumpRecord( xbULong );
   //! Return number of fields
   /*!
   */
   xbLong    FieldCount( void ) { return NoOfFields; }
   //! Return Dbf name
   /*!
   */
   xbString& GetDbfName( void ) { return DatabaseName; }
   //! Return status
   /*!
   */
   xbShort   GetDbfStatus( void ) { return DbfStatus; }
   xbShort   GetFirstRecord( void );
   xbShort   GetLastRecord( void );
   xbShort   GetNextRecord( void );
   xbShort   GetPrevRecord( void );
   //! Return current record number
   /*!
   */
   xbLong    GetCurRecNo( void ) { return CurRec; }
   xbShort   GetRecord( xbULong );
   //! Return a pointer to the record buffer
   /*!
   */
   char *    GetRecordBuf( void ) { return RecBuf; }
   //! Return record length
   /*!
   */
   xbShort   GetRecordLen( void ) { return RecordLen; }
   xbShort   NameSuffixMissing( xbShort, const char * );
   xbLong    NoOfRecords( void );
   xbLong    PhysicalNoOfRecords(void);
   xbShort   OpenDatabase( const char * );
   xbShort   PackDatabase(xbShort LockWaitOption,
                          void (*packStatusFunc)(xbLong itemNum, xbLong numItems) = 0,
                          void (*indexStatusFunc)(xbLong itemNum, xbLong numItems) = 0);
   xbShort   PutRecord(void); // Put record to current position
   xbShort   PutRecord(xbULong);
   xbShort   RebuildAllIndices(void (*statusFunc)(xbLong itemNum, xbLong numItems) = 0);
   xbShort   RecordDeleted( void );
   //! Set number of records to zero????
   /*!
   */
   void      ResetNoOfRecs( void ) { NoOfRecs = 0L; }
   xbShort   SetVersion( xbShort );
   //! Undelete all records
   /*!
   */
   xbShort   UndeleteAllRecords( void ) { return DeleteAll(1); }
   xbShort   UndeleteRecord( void );
   xbShort   Zap( xbShort );

/* field methods */
   const char *GetField(xbShort FieldNo) const; // Using internal static buffer
   const char *GetField(const char *Name) const;
   xbShort   GetField( xbShort FieldNo, char *Buf) const;
   xbShort   GetRawField(const xbShort FieldNo, char *Buf) const;
   xbShort   GetField( xbShort FieldNo, char *Buf, xbShort RecBufSw) const;
   xbShort   GetField( const char *Name, char *Buf) const;
   xbShort   GetRawField(const char *Name, char *Buf) const;
   xbShort   GetField( const char *Name, char *Buf, xbShort RecBufSw) const;
   xbShort   GetField(xbShort FieldNo, xbString&, xbShort RecBufSw ) const;
   xbShort   GetFieldDecimal( const xbShort );
   xbShort   GetFieldLen( const xbShort );
   char *    GetFieldName( const xbShort );
   xbShort   GetFieldNo( const char * FieldName ) const;
   char      GetFieldType( const xbShort FieldNo ) const;
   xbShort   GetLogicalField( const xbShort FieldNo );
   xbShort   GetLogicalField( const char * FieldName );

   char *    GetStringField( const xbShort FieldNo );
   char *    GetStringField( const char * FieldName );

   xbShort   PutField( const xbShort, const char * );
   xbShort   PutRawField( const xbShort FieldNo, const char *buf );
   xbShort   PutField( const char *Name, const char *buf);
   xbShort   PutRawField( const char *Name, const char *buf );
   xbShort   ValidLogicalData( const char * );
   xbShort   ValidNumericData( const char * );

   xbLong    GetLongField( const char *FieldName) const;
   xbLong    GetLongField( const xbShort FieldNo) const;
   xbShort   PutLongField( const xbShort, const xbLong );
   xbShort   PutLongField( const char *, const xbLong);

   xbFloat   GetFloatField( const char * FieldName );
   xbFloat   GetFloatField( const xbShort FieldNo );
   xbShort   PutFloatField( const char *, const xbFloat);
   xbShort   PutFloatField( const xbShort, const xbFloat);

   xbDouble  GetDoubleField(const char *);
   xbDouble  GetDoubleField(const xbShort, xbShort RecBufSw = 0);
   xbShort   PutDoubleField(const char *, const xbDouble);
   xbShort   PutDoubleField(const xbShort, const xbDouble);

#ifdef XB_LOCKING_ON
   xbShort   LockDatabase( const xbShort, const xbShort, const xbULong );
   xbShort   ExclusiveLock( const xbShort );
   xbShort   ExclusiveUnlock( void );

#ifndef HAVE_FCNTL
   xbShort   UnixToDosLockCommand( const xbShort WaitOption,
             const xbShort LockType ) const;
#endif

#else
   xbShort   LockDatabase( const xbShort, const xbShort, const xbLong )
     { return XB_NO_ERROR; }
   xbShort   ExclusiveLock( const xbShort ) { return XB_NO_ERROR; };
   xbShort   ExclusiveUnlock( void )      { return XB_NO_ERROR; };
#endif

   //! Turn autolock on
   /*!
   */
   void    AutoLockOn( void )  { AutoLock = 1; }
   //! Turn autolock off
   /*!
   */
   void    AutoLockOff( void ) { AutoLock = 0; }
   //! Return whether or not autolocking is on or off
   /*!
   */
   xbShort GetAutoLock(void) { return AutoLock; }

#ifdef XB_MEMO_FIELDS
   xbShort   GetMemoField( const xbShort FieldNo,const xbLong len,
             char * Buf, const xbShort LockOption );
   xbLong    GetMemoFieldLen( const xbShort FieldNo );
   xbShort   GetFPTField( const xbShort FieldNo,const xbLong len,
             char * Buf, const xbShort LockOption );
   xbLong    GetFPTFieldLen( const xbShort FieldNo );
   xbShort   UpdateMemoData( const xbShort FieldNo, const xbLong len,
              const char * Buf, const xbShort LockOption );
   xbShort   MemoFieldExists( const xbShort FieldNo ) const;
   xbShort   LockMemoFile( const xbShort WaitOption, const xbShort LockType );
   xbShort   MemoFieldsPresent( void ) const;
   xbLong    CalcLastDataBlock();
   xbShort   FindBlockSetInChain( const xbLong BlocksNeeded, const xbLong
               LastDataBlock, xbLong & Location, xbLong &PreviousNode );
   xbShort   GetBlockSetFromChain( const xbLong BlocksNeeded, const xbLong
               Location, const xbLong PreviousNode );

#ifdef XBASE_DEBUG
   xbShort   DumpMemoFreeChain( void );
   void      DumpMemoHeader( void ) const;
   void      DumpMemoBlock( void ) const;
#endif
#endif

   //! Turn on "real" deletes
   /*!
     This should be done before creating a database (with 
    xbDbf::CreateDatatabase()) and thereafter before opening
    a database with xbDbfCreateDatabase().
    
    You cannot "turn on" real deletes once a database has been created
    and records added.
   */
   void      RealDeleteOn(void) { RealDelete = 1; if(fp) ReadHeader(1); }
   /*! Turn off "real" deletes
   */
   void      RealDeleteOff(void) { RealDelete = 0; if(fp) ReadHeader(1); }
   //! Return whether "real" deletes are on or off
   /*!
     Use this to determine if "real deletes" are being used with
    the database.
   */
   xbShort   GetRealDelete(void) { return RealDelete; }

#if defined(XB_INDEX_ANY)
   xbShort   IndexCount(void);
   xbIndex   *GetIndex(xbShort indexNum);
#endif

   void      Flush();

protected:
   xbString DatabaseName;
   xbShort  XFV;                  /* xBASE file version            */
   xbShort  NoOfFields;
   char   DbfStatus;              /* 0 = closed
                                     1 = open
                                     2 = updates pending           */
   FILE   *fp;                    /* file pointer                  */
   xbSchemaRec *SchemaPtr;        /* Pointer to field data         */
   char   *RecBuf;                /* Pointer to record buffer      */
   char   *RecBuf2;               /* Pointer to original rec buf   */

#ifdef XB_MEMO_FIELDS
   FILE    *mfp;                  /* memo file pointer             */
   void    *mbb;                  /* memo block buffer             */
   xbMH     MemoHeader;           /* memo header structure         */

   xbShort  mfield1;              /* memo block field one FF       */
   xbShort  MStartPos;            /* memo start pos of data        */
   xbLong   MFieldLen;            /* memo length of data           */
   xbLong   NextFreeBlock;        /* next free block in free chain */
   xbLong   FreeBlockCnt;         /* count of free blocks this set */

   xbLong   MNextBlockNo;         /* free block chain              */
   xbLong   MNoOfFreeBlocks;      /* free block chain              */

   xbLong   CurMemoBlockNo;       /* Current block no loaded       */
#endif

/* Next seven variables are read directly off the database header */
/* Don't change the order of the following seven items            */
   char   Version;
   char   UpdateYY;
   char   UpdateMM;
   char   UpdateDD;
//   xbLong   NoOfRecs;
//   xbShort  HeaderLen;
//   xbShort  RecordLen;

   xbULong  NoOfRecs;
   xbUShort HeaderLen;
   xbUShort RecordLen;

//#ifdef XB_REAL_DELETE
   xbULong  FirstFreeRec;
   xbULong  RealNumRecs;
//#endif

   xbIxList * MdxList;
   xbIxList * NdxList;
   xbIxList * FreeIxList;
   xbULong  CurRec;               /* Current record or zero   */
   xbShort  AutoLock;             /* Auto update option 0 = off  */

//#ifdef XB_REAL_DELETE
   xbShort  RealDelete;           /* real delete option 0 = off */
//#endif

#ifdef XB_LOCKING_ON
   xbShort CurLockType;           /* current type of file lock */
   xbShort CurLockCount;          /* number of current file locks */
   xbULong CurLockedRecNo;        /* currently locked record no */
   xbShort CurRecLockType;        /* current type of rec lock held (F_RDLOCK or F_WRLCK) */
   xbShort CurRecLockCount;       /* number of current record locks */
   xbShort CurMemoLockType;       /* current type of memo lock */
   xbShort CurMemoLockCount;      /* number of current memo locks */
#endif

   xbShort   DeleteAll( xbShort );
   void    InitVars( void );
   xbShort   PackDatafiles(void (*statusFunc)(xbLong itemNum, xbLong numItems) = 0);
   xbShort   ReadHeader( xbShort );
   xbShort   WriteHeader( const xbShort );

#ifdef XB_MEMO_FIELDS
   xbShort   AddMemoData( const xbShort FieldNo, const xbLong Len, const char * Buf );
   xbShort   CreateMemoFile( void );
   xbShort   DeleteMemoField( const xbShort FieldNo );
   xbShort   GetDbtHeader( const xbShort Option );
   xbShort   GetMemoBlockSize( void ) { return MemoHeader.BlockSize; }
   xbShort   OpenMemoFile( void );
   xbShort   OpenFPTFile(void);
   xbShort   PutMemoData( const xbLong StartBlock, const xbLong BlocksNeeded,
             const xbLong Len, const char * Buf );
   xbShort   ReadMemoBlock( const xbLong BlockNo, const xbShort Option);
   xbShort   SetMemoBlockSize( const xbShort );
   xbShort   UpdateHeadNextNode( void ) const;
   xbShort   WriteMemoBlock( const xbLong BlockNo, const xbShort Option );
   xbShort   IsType3Dbt( void ) const { return( Version==(char)0x83 ? 1:0 ); }
   xbShort   IsType4Dbt( void ) const
            {return (( Version==(char)0x8B || Version==(char)0x8E ) ? 1:0 );}
#endif
};
#endif      // __XB_DBF_H__


