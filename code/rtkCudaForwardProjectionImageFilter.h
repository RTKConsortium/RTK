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

#ifndef __rtkCudaForwardProjectionImageFilter_h
#define __rtkCudaForwardProjectionImageFilter_h

#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkWin32Header.h"

/** \class CudaForwardProjectionImageFilter
 * \brief TODO
 *
 * TODO
 *
 * \author TODO 
 *
 * \ingroup Projector CudaImageToImageFilter
 */

namespace rtk
{

class rtkcuda_EXPORT CudaForwardProjectionImageFilter :
  public ForwardProjectionImageFilter< itk::Image<float,3>, itk::Image<float,3> >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<float,3>                                ImageType;
  typedef CudaForwardProjectionImageFilter                   Self;
  typedef ForwardProjectionImageFilter<ImageType, ImageType> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  typedef ImageType::RegionType        OutputImageRegionType;
  typedef itk::Image<float, 2>         ProjectionImageType;
  typedef ProjectionImageType::Pointer ProjectionImagePointer;
  typedef itk::Vector<float,3>         VectorType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaForwardProjectionImageFilter, ImageToImageFilter);

  /** Function to allocate memory on device */
  void InitDevice();

  /** Function to synchronize memory from device to host and free device memory */
  void CleanUpDevice();

  /** Boolean to keep the hand on the memory management of the GPU. Default is
   * off. If on, the user must call manually InitDevice and CleanUpDevice. */
  itkGetMacro(ExplicitGPUMemoryManagementFlag, bool);
  itkSetMacro(ExplicitGPUMemoryManagementFlag, bool);

protected:
  CudaForwardProjectionImageFilter();
  ~CudaForwardProjectionImageFilter() {};

  void GenerateData();

private:
  //purposely not implemented
  CudaForwardProjectionImageFilter(const Self&);
  void operator=(const Self&);

  int                m_VolumeDimension[3];
  int                m_ProjectionDimension[2];
  float *            m_DeviceVolume;
  float *            m_DeviceProjection;
  float *            m_DeviceMatrix;
  bool               m_ExplicitGPUMemoryManagementFlag;
}; // end of class

} // end namespace rtk

#endif
