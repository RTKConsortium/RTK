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

#ifndef __rtkCudaBackProjectionImageFilter_h
#define __rtkCudaBackProjectionImageFilter_h

#include "rtkBackProjectionImageFilter.h"

namespace rtk
{

/** \class CudaBackProjectionImageFilter
 * \brief Cuda version of the  backprojection.
 *
 * GPU-based implementation of the backprojection step of the
 * [Feldkamp, Davis, Kress, 1984] algorithm for filtered backprojection
 * reconstruction of cone-beam CT images with a circular source trajectory.
 *
 * \author Simon Rit
 *
 * \ingroup Projector CudaImageToImageFilter
 */
class ITK_EXPORT CudaBackProjectionImageFilter :
  public BackProjectionImageFilter< itk::Image<float,3>, itk::Image<float,3> >
{
public:
  /** Standard class typedefs. */
  typedef itk::Image<float,3>                                ImageType;
  typedef CudaBackProjectionImageFilter                      Self;
  typedef BackProjectionImageFilter<ImageType, ImageType>    Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  typedef ImageType::RegionType            OutputImageRegionType;
  typedef itk::Image<float, 2>             ProjectionImageType;
  typedef ProjectionImageType::Pointer     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaBackProjectionImageFilter, ImageToImageFilter);

  /** Function to allocate memory on device */
  void InitDevice();

  /** Function to synchronize memory from device to host and free device memory */
  void CleanUpDevice();

protected:
  CudaBackProjectionImageFilter();
  virtual ~CudaBackProjectionImageFilter() {};

  void GenerateData();

  /** Boolean to keep the hand on the memory management of the GPU. Default is
   * off. If on, the user must call manually InitDevice and CleanUpDevice. */
  itkGetMacro(ExplicitGPUMemoryManagementFlag, bool);
  itkSetMacro(ExplicitGPUMemoryManagementFlag, bool);

private:
  CudaBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented

  bool       m_ExplicitGPUMemoryManagementFlag;
  int        m_VolumeDimension[3];
  int        m_ProjectionDimension[2];
  float *    m_DeviceVolume;
  float *    m_DeviceProjection;
  float *    m_DeviceMatrix;
};

} // end namespace rtk

#endif
