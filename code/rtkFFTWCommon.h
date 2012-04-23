/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itkFFTWCommon.h,v $
  Language:  C++
  Date:      $Date: 2008-12-21 19:13:11 $
  Version:   $Revision: 1.8 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef __rtkFFTWCommon_h
#define __rtkFFTWCommon_h

#include <itkConfigure.h>

#if defined(USE_FFTWF) || defined(USE_FFTWD)
#include "fftw3.h"
#endif
#include <itkFastMutexLock.h>

namespace rtk
{
namespace fftw
{
/**
 * \class Interface
 * \brief Wrapper for FFTW API
 */
template <typename TPixel>
class  Proxy
{
  // empty -- only double and float specializations work
protected:
  Proxy() {}; 
  ~Proxy() {};
};

#if defined(USE_FFTWF)

template <>
class Proxy<float>
{
public:
  typedef float         PixelType;
  typedef fftwf_complex ComplexType;
  typedef fftwf_plan    PlanType;
  typedef Proxy<float>  Self;
  // a global lock to ensure thread safety
  static itk::FastMutexLock::Pointer lock;
  static void Lock()
    {
    if( Self::lock.IsNull() )
      {
      // init stuff
      Self::lock = itk::FastMutexLock::New();
      Self::lock->Lock();
      fftwf_init_threads();
      }
    else
      {
      Self::lock->Lock();
      }
       
    }
  static void Unlock()
    {
    Self::lock->Unlock();
    }
   

  static PlanType Plan_dft_c2r_1d(int n,
                                  ComplexType *in,
                                  PixelType *out,
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_c2r_1d(n,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_c2r_2d(int nx, 
                                  int ny,
                                  ComplexType *in,
                                  PixelType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_c2r_2d(nx,ny,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_c2r_3d(int nx, 
                                  int ny,
                                  int nz,
                                  ComplexType *in, 
                                  PixelType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_c2r_3d(nx,ny,nz,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_c2r(int rank, 
                               const int *n,
                               ComplexType *in, 
                               PixelType *out, 
                               unsigned flags,
                               int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_c2r(rank,n,in,out,flags);
    Unlock();
    return plan;
    }

  static PlanType Plan_dft_r2c_1d(int n,
                                  PixelType *in,
                                  ComplexType *out,
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_r2c_1d(n,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_r2c_2d(int nx, 
                                  int ny,
                                  PixelType *in,
                                  ComplexType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_r2c_2d(nx,ny,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_r2c_3d(int nx, 
                                  int ny,
                                  int nz,
                                  PixelType *in, 
                                  ComplexType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_r2c_3d(nx,ny,nz,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_r2c(int rank, 
                               const int *n,
                               PixelType *in, 
                               ComplexType *out, 
                               unsigned flags,
                               int threads=1)
    {
    Lock();
    fftwf_plan_with_nthreads(threads);
    PlanType plan = fftwf_plan_dft_r2c(rank,n,in,out,flags);
    Unlock();
    return plan;
    }

  static void Execute(PlanType p)
    {
    fftwf_execute(p);
    }
  static void DestroyPlan(PlanType p)
    {
    Lock();
    fftwf_destroy_plan(p);
    Unlock();
    }
};

#endif // USE_FFTWF


#if defined(USE_FFTWD)
template <> 
class Proxy<double>
{
public:
  typedef double        PixelType;
  typedef fftw_complex  ComplexType;
  typedef fftw_plan     PlanType;
  typedef Proxy<double> Self;

  // a global lock to ensure thread safety
  static itk::FastMutexLock::Pointer lock;
  static void Lock()
    {
    if( Self::lock.IsNull() )
      {
      // init stuff
      Self::lock = itk::FastMutexLock::New();
      Self::lock->Lock();
      fftw_init_threads();
      }
    else
      {
      lock->Lock();
      }
       
    }
  static void Unlock()
    {
    lock->Unlock();
    }
   
  static PlanType Plan_dft_c2r_1d(int n,
                                  ComplexType *in,
                                  PixelType *out,
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_c2r_1d(n,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_c2r_2d(int nx, 
                                  int ny,
                                  ComplexType *in,
                                  PixelType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_c2r_2d(nx,ny,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_c2r_3d(int nx, 
                                  int ny,
                                  int nz,
                                  ComplexType *in, 
                                  PixelType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_c2r_3d(nx,ny,nz,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_c2r(int rank, 
                               const int *n,
                               ComplexType *in, 
                               PixelType *out, 
                               unsigned flags,
                               int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_c2r(rank,n,in,out,flags);
    Unlock();
    return plan;
    }

  static PlanType Plan_dft_r2c_1d(int n,
                                  PixelType *in,
                                  ComplexType *out,
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_r2c_1d(n,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_r2c_2d(int nx, 
                                  int ny,
                                  PixelType *in,
                                  ComplexType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_r2c_2d(nx,ny,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_r2c_3d(int nx, 
                                  int ny,
                                  int nz,
                                  PixelType *in, 
                                  ComplexType *out, 
                                  unsigned flags,
                                  int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_r2c_3d(nx,ny,nz,in,out,flags);
    Unlock();
    return plan;
    }
  static PlanType Plan_dft_r2c(int rank, 
                               const int *n,
                               PixelType *in, 
                               ComplexType *out, 
                               unsigned flags,
                               int threads=1)
    {
    Lock();
    fftw_plan_with_nthreads(threads);
    PlanType plan = fftw_plan_dft_r2c(rank,n,in,out,flags);
    Unlock();
    return plan;
    }
  static void Execute(PlanType p)
    {
    fftw_execute(p);
    }
  static void DestroyPlan(PlanType p)
    {
    Lock();
    fftw_destroy_plan(p);
    Unlock();
    }
};
#endif
}
}
#endif
