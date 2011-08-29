#ifndef __itkOpenCLFDKBackProjectionImageFilter_h
#define __itkOpenCLFDKBackProjectionImageFilter_h

#include "itkFDKBackProjectionImageFilter.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace itk
{

class ITK_EXPORT OpenCLFDKBackProjectionImageFilter :
  public FDKBackProjectionImageFilter< itk::Image<float,3>, itk::Image<float,3> >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<float,3>                                ImageType;
  typedef OpenCLFDKBackProjectionImageFilter                 Self;
  typedef FDKBackProjectionImageFilter<ImageType, ImageType> Superclass;
  typedef SmartPointer<Self>                                 Pointer;
  typedef SmartPointer<const Self>                           ConstPointer;

  typedef ImageType::RegionType        OutputImageRegionType;
  typedef itk::Image<float, 2>         ProjectionImageType;
  typedef ProjectionImageType::Pointer ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(OpenCLFDKBackProjectionImageFilter, ImageToImageFilter);

  /** Function to allocate memory on device */
  void InitDevice();

  /** Function to sync memory from device to host and free device memory */
  void CleanUpDevice();

protected:
  OpenCLFDKBackProjectionImageFilter();
  virtual ~OpenCLFDKBackProjectionImageFilter() {}

  void GenerateData();

private:
  OpenCLFDKBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                     //purposely not implemented

  cl_context       m_Context;
  cl_command_queue m_CommandQueue;
  cl_mem           m_DeviceMatrix;
  cl_mem           m_DeviceVolume;
  cl_mem           m_DeviceProjection;
  cl_program       m_Program;
  cl_kernel        m_Kernel;
};

} // end namespace itk

#endif
