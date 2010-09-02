/*  $Id: lock.cpp,v 1.8 2003/08/16 19:59:39 gkunkel Exp $

    Xbase project source code

    This file contains the basic Xbase routines for locking Xbase files

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

#ifndef XB_LOCKING_ON
XBDLLEXPORT_DATA(const int)
  xbF_SETLK = 0,
  xbF_SETLKW = 1;
#endif

#ifdef XB_LOCKING_ON

#include <fcntl.h>
#include <stdio.h>

#ifdef HAVE_SYS_TYPES_H
#include <sys/types.h>           /* BSDI BSD/OS 3.1 */
#endif

#ifdef HAVE_SYS_LOCKING_H
#include <sys/locking.h>
#endif

#ifdef HAVE_IO_H
#include <io.h>
#endif

#include <xbase/xbexcept.h>

#include <errno.h>

XBDLLEXPORT_DATA(const int)
  xbF_SETLK = F_SETLK,
  xbF_SETLKW = F_SETLKW;

/*! \file lock.cpp
*/

/************************************************************************/
//! UnixToDosLockCommand
/*!
  \param WaitOption
  \param LockType
*/
#ifndef HAVE_FCNTL
xbShort xbDbf::UnixToDosLockCommand( const xbShort WaitOption,
    const xbShort LockType ) const
// this method converts the unix locking commands into as close of 
// a dos lock command as is possible
{
  if( LockType == LK_LOCK )
    return LK_UNLCK;
  else if( WaitOption == F_SETLK )
    return LK_NBLCK;
  else 
    return LK_LOCK;
}
#endif

#ifdef HAVE_FCNTL
//
//  This method handles the case where fcntl fails because it is interrupted
//  for some reason.  In this case it will return -1 and set errno to
//  EINTR.  If it does this, we must try again.
static int 
DoFcntl(int fd, int cmd, struct flock *lock)
{
  int
    rc;

  do
  {
    rc = fcntl(fd, cmd, lock);
  } while(rc == -1 && errno == EINTR);

  return rc;
}
#endif

/************************************************************************/
//! Short description
/*!
  \param WaitOption
  \param LockType
  \param LRecNo
*/
xbShort xbDbf::LockDatabase( const xbShort WaitOption, const xbShort LockType, 
    const xbULong LRecNo )
{
   /*   
      if WaitOption = F_SETLK  - return immediately 
                      F_SETLKW - wait until lock function executes

      if LockType   = F_RDLCK - do a read / shared lock
                      F_WRLCK - do a write / exclusive lock
                      F_UNLCK - do an unlock

      if LRecNo     = 0L  - lock the dbf header
                    = 1-n - lock appropriate record

      if locking routines fail, look in errno or do perror() for explanation
      
      Nested record locking support added Oct 26, 1998 by Derry Bryson
   */
#ifdef HAVE_FCNTL
  struct flock fl;
  fl.l_type = LockType;

#else
  long length;
  int  whence, Cmd;
  long offset;
#endif

//fprintf(stderr, "LockDatabase: %s %s CurLockCount=%d\n", LRecNo ? "Record" : "File" ,
// LockType == F_RDLCK ? "F_RDLCK" : LockType == F_WRLCK ? "F_WRLCK" : "F_UNLCK",
// CurLockCount);
   if( LRecNo > NoOfRecs )
       xb_error(XB_INVALID_RECORD);

   if( LRecNo == 0L )
   {
      /*
      **  This is a file lock.  The following code (and some below)
      **  allows nesting of file lock calls by only locking on the
      **  first lock call (unless upgrading a read lock to a write lock)
      **  and counting the number of locks to allow unnesting of locks.
      */
      if(CurLockType != -1)
      {
         if(LockType != F_UNLCK)
         {
           /*
           **  Allow changing a read lock into a write lock, but
           **  if a write lock is already held just say success.
           */
           if(CurLockType == F_WRLCK || CurLockType == LockType)
           {
//fprintf(stderr, "LockDatabase: nested lock succeeds\n");           
              CurLockCount++;
              return XB_NO_ERROR;
           }
        }
        else
        {
          CurLockCount--;
          /*
          **  If there are still outstanding locks, just indicate success.
          */
          if(CurLockCount)
          {
//fprintf(stderr, "LockDatabase:  nested unlock returns with remaining locks\n");          
            return XB_NO_ERROR;
          }            
        }
      }
        
#ifdef HAVE_FCNTL
      fl.l_whence = SEEK_SET;
      fl.l_start = 0L;
      fl.l_len = 7L;
#else
      whence = SEEK_SET;
      offset = 0L;
      length = 7L;
#endif
   }
#if 0   
   /*
   **  I don't understand why this code is here (hence I have removed it)
   **  as it causes locks which are never removed - Derry Bryson
   */
   else if ( CurRec )
   {
#ifdef HAVE_FCNTL
      fl.l_whence = SEEK_CUR;
      fl.l_start = (HeaderLen+(RecordLen*(LRecNo-1))) - DbfTell();
      fl.l_len = 1L;
#else
      whence = SEEK_CUR;
      offset = (HeaderLen+(RecordLen*(LRecNo-1))) - DbfTell();
      length = 1L;
#endif
   }
#endif   
   else   /* CurRec = 0 */
   {
      /*
      **  This is a record lock.  The following code (and some below)
      **  allows nesting of record lock calls by only locking on the
      **  first lock call (unless upgrading a read lock to a write lock)
      **  and counting the number of locks to allow unnesting of locks.
      */
      if(CurLockedRecNo)
      {
        if(LockType != F_UNLCK)
        {
           /*
           **  Allow changing a read lock into a write lock, but
           **  if a write lock is already held just say success.
           */
           if(CurRecLockType == F_WRLCK || CurRecLockType == LockType)
           {
//fprintf(stderr, "LockDatabase: nested lock succeeds\n");           
              CurRecLockCount++;
              return XB_NO_ERROR;
           }
        }
        else
        {
          CurRecLockCount--;
          /*
          **  If there are still outstanding locks, just indicate success.
          */
          if(CurRecLockCount)
          {
//fprintf(stderr, "LockDatabase:  nested unlock returns with remaining locks\n");          
            return XB_NO_ERROR;
          }            
        }
      }
        
#ifdef HAVE_FCNTL
      fl.l_whence = SEEK_SET;
      fl.l_start = HeaderLen + (RecordLen*(LRecNo-1) );
      fl.l_len = 1L;
#else
      whence = SEEK_SET;
      offset = HeaderLen + (RecordLen*(LRecNo-1));
      length = 1L;
#endif
   }

#ifdef HAVE_FCNTL
   if(DoFcntl(fileno(fp), WaitOption, &fl) == -1)
{
//fprintf(stderr, "LockDatabase:  failed!\n");
      xb_error(XB_LOCK_FAILED)
}      
   else
   {
      if(LRecNo) /* record lock */
      {
         if(LockType != F_UNLCK)
         {
            CurLockedRecNo = LRecNo;
            CurRecLockType = LockType;
            CurRecLockCount++;
         }
         else if(!CurRecLockCount)
         {
            CurLockedRecNo = 0;
            CurRecLockType = -1;
         }
      }
      else
      {
         if(LockType != F_UNLCK)
         {
            CurLockType = LockType;
            CurLockCount++;
         }
         else if (!CurLockCount)
            CurLockType = -1;
      }
//fprintf(stderr, "LockDatabase: success!\n");      
      return XB_NO_ERROR;
   }
#else
   if( fseek( fp, offset, whence ) != 0 )
       xb_error(XB_SEEK_ERROR);
   Cmd = UnixToDosLockCommand( WaitOption, LockType );
   if( locking( fileno( fp ), Cmd, length ) != 0 )
       xb_error(XB_LOCK_FAILED)
   else
   {
      if(LRecNo) /* record lock */
      {
         if(LockType != F_UNLCK)
         {
            CurLockedRecNo = LRecNo;
            CurRecLockType = LockType;
            CurRecLockCount++;
         }
         else if(!CurRecLockCount)
         {
            CurLockedRecNo = 0;
            CurRecLockType = -1;
         }
      }
      else
      {
         if(LockType != F_UNLCK)
         {
            CurLockType = LockType;
            CurLockCount++;
         }
         else if (!CurLockCount)
            CurLockType = -1;
      }
      return XB_NO_ERROR;
   }
#endif
} 
/************************************************************************/
//! Short Description
/*!
  \param WaitOption
  \param LockType
*/
#ifdef XB_INDEX_ANY

xbShort xbIndex::LockIndex( const xbShort WaitOption,
             const xbShort LockType )
{
   /*  This method locks the first 512 bytes of the index file,
       effectively locking the file from other processes that are
       using the locking protocols

      if WaitOption = F_SETLK  - return immediately 
                      F_SETLKW - wait until lock function executes

      if LockType   = F_RDLCK - do a read / shared lock
                      F_WRLCK - do a write / exclusive lock
                      F_UNLCK - do an unlock

      if locking routines fail, look in errno or do perror() for explanation
   */
   
//fprintf(stderr, "LockIndex\n");
   /*
   **  Support nested index locking.
   */
   if(CurLockCount)
   {
     if(LockType != F_UNLCK)
     {
        /*
        **  Allow changing a read lock into a write lock, but
        **  if a write lock is already held or the current lock
        **  type is the same as the new lock type just say success.
        */
        if(CurLockType == F_WRLCK || CurLockType == LockType)
        {
           CurLockCount++;
           return XB_NO_ERROR;
        }
     }
     else
     {
        /*
        **  This is an unlock
        */
        CurLockCount--;
        
        /*
        **  If there are still outstanding locks, just indicate success.
        */
        if(CurLockCount)
           return XB_NO_ERROR;
     }
   }
        
#ifdef HAVE_FCNTL
   struct flock fl;

   fl.l_type = LockType;
   fl.l_whence = SEEK_SET;
   fl.l_start = 0L;
//   fl.l_len = XB_NDX_NODE_SIZE;
   fl.l_len = 1;

   if(DoFcntl( fileno( indexfp ), WaitOption, &fl ) == -1 )
       xb_error(XB_LOCK_FAILED)
   else
   {
      if(LockType != F_UNLCK)
      {
         CurLockType = LockType;
         CurLockCount++;
      }
      else if(!CurLockCount)
         CurLockType = -1;
      return XB_NO_ERROR;
   }
#else
   if( fseek( indexfp, 0L, SEEK_SET ) != 0 )
     return XB_SEEK_ERROR;
   if( locking( fileno( indexfp ),
    dbf->UnixToDosLockCommand( WaitOption, LockType ),XB_NDX_NODE_SIZE ) != 0 )
       xb_error(XB_LOCK_FAILED)
   else
   {
      if(LockType != F_UNLCK)
      {
         CurLockType = LockType;
         CurLockCount++;
      }
      else if(!CurLockCount)
         CurLockType = -1;
      return XB_NO_ERROR;
   }
#endif
}
#endif   /* XB_INDEX_ANY  */
/************************************************************************/
//! Short description
/*!
  \param WaitOption
  \param LockType
*/
#ifdef XB_MEMO_FIELDS 
xbShort xbDbf::LockMemoFile( const xbShort WaitOption, const xbShort LockType )
{
   /*  This method locks the first 4 bytes of the memo file,
       effectively locking the file from other processes that are
       using the locking protocols

       The first four bytes point to the free block chain

      if WaitOption = F_SETLK  - return immediately 
                      F_SETLKW - wait until lock function executes

      if LockType   = F_RDLCK - do a read / shared lock
                      F_WRLCK - do a write / exclusive lock
                      F_UNLCK - do an unlock

      if locking routines fail, look in errno or do perror() for explanation
   */

   /*
   **  Support nested index locking.
   */
   if(CurMemoLockCount)
   {
     if(LockType != F_UNLCK)
     {
        /*
        **  Allow changing a read lock into a write lock, but
        **  if a write lock is already held or the current lock
        **  type is the same as the new lock type just say success.
        */
        if(CurMemoLockType == F_WRLCK || CurMemoLockType == LockType)
        {
           CurMemoLockCount++;
           return XB_NO_ERROR;
        }
     }
     else
     {
        /*
        **  This is an unlock
        */
        CurMemoLockCount--;
        
        /*
        **  If there are still outstanding locks, just indicate success.
        */
        if(CurMemoLockCount)
           return XB_NO_ERROR;
     }
   }
        
#ifdef HAVE_FCNTL
   struct flock fl;

   fl.l_type   = LockType;
   fl.l_whence = SEEK_SET;
   fl.l_start  = 0L;
   fl.l_len    = 4L;

   if(DoFcntl( fileno( mfp ), WaitOption, &fl ) == -1 )
       xb_error(XB_LOCK_FAILED)
   else
   {
      if(LockType != F_UNLCK)
      {
         CurMemoLockType = LockType;
         CurMemoLockCount++;
      }
      else if(!CurMemoLockCount)
         CurMemoLockType = -1;
      return XB_NO_ERROR;
   }
#else
   if( fseek( mfp , 0L, SEEK_SET ) != 0 )
       xb_error(XB_SEEK_ERROR);

   if( locking( fileno( mfp ),
    UnixToDosLockCommand( WaitOption, LockType ), 4L ) != 0 )
       xb_error(XB_LOCK_FAILED)
   else
   {
      if(LockType != F_UNLCK)
      {
         CurMemoLockType = LockType;
         CurMemoLockCount++;
      }
      else if(!CurMemoLockCount)
         CurMemoLockType = -1;
      return XB_NO_ERROR;
   }
#endif
}
#endif   /* XB_MEMO_FIELDS  */
/***********************************************************************/
//! Short description
/*!
  \param LockWaitOption
*/
xbShort xbDbf::ExclusiveLock( const xbShort LockWaitOption )
{
   /* this routine locks all files and indexes for a database file */
   /* if it fails, no locks are left on (theoretically)            */
   xbIxList *i;
   xbShort rc;

   AutoLockOff();
   if(( rc = LockDatabase( LockWaitOption, F_WRLCK, 0 )) != XB_NO_ERROR )
      return rc;

#ifdef XB_MEMO_FIELDS
   if( MemoFieldsPresent())
      if(( rc = LockMemoFile( LockWaitOption, F_WRLCK )) != XB_NO_ERROR )
        return rc;
#endif

#ifdef XB_INDEX_ANY
   i = NdxList;
   while( i ) 
   {
#ifdef HAVE_EXCEPTIONS
       try {
#endif
      if(( rc = i->index->LockIndex( LockWaitOption, F_WRLCK )) != XB_NO_ERROR )
      {
         ExclusiveUnlock();
#ifndef HAVE_EXCEPTIONS
             return rc;
#endif
      }

#ifdef HAVE_EXCEPTIONS
       } catch (xbException &x) {
          ExclusiveUnlock();
          xb_error(XB_LOCK_FAILED);
       }
#endif
      i = i->NextIx;
   }
#endif

   return XB_NO_ERROR;
}  
/***********************************************************************/
//! Short description
/*!
*/
xbShort xbDbf::ExclusiveUnlock( void )
{
   /* this routine unlocks all files and indexes for a database file */

   xbIxList *i;

   LockDatabase( F_SETLK, F_UNLCK, 0 );

#ifdef XB_MEMO_FIELDS
   if( MemoFieldsPresent())
      LockMemoFile( F_SETLK, F_UNLCK );
#endif

#ifdef XB_INDEX_ANY
   i = NdxList;
   while( i ) 
   {
      i->index->LockIndex( F_SETLK, F_UNLCK );
      i = i->NextIx;
   }
#endif

   AutoLockOn();
   return XB_NO_ERROR;
}  
/***********************************************************************/
#endif  /* XB_LOCKING_ON */
