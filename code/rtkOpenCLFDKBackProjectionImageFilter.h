/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __rtkOpenCLFDKBackProjectionImageFilter_h
#define __rtkOpenCLFDKBackProjectionImageFilter_h

#include "rtkFDKBackProjectionImageFilter.h"

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

namespace rtk
{

class ITK_EXPORT OpenCLFDKBackProjectionImageFilter :
  public FDKBackProjectionImageFilter< itk::Image<float,3>, itk::Image<float,3> >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<float,3>                                ImageType;
  typedef OpenCLFDKBackProjectionImageFilter                 Self;
  typedef FDKBackProjectionImageFilter<ImageType, ImageType> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

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

} // end namespace rtk

#endif
