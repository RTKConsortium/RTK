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

#ifndef rtkCudaWarpBackProjectionImageFilter_h
#define rtkCudaWarpBackProjectionImageFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaWarpBackProjectionImageFilter
 * \brief Voxel-based backprojection into warped volume implemented in CUDA
 *
 * GPU-based implementation of the voxel-based backprojection, assuming
 * a deformation of the volume.
 *
 * \test rtksarttest.cxx
 *
 * \author Cyril Mory
 *
 * \ingroup Projector CudaImageToImageFilter
 */
class rtkcuda_EXPORT CudaWarpBackProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  BackProjectionImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float,3>                             ImageType;
  typedef itk::CudaImage<itk::CovariantVector<float, 3>, 3>   DVFType;
  typedef BackProjectionImageFilter< ImageType, ImageType>    BackProjectionImageFilterType;
  typedef CudaWarpBackProjectionImageFilter                   Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
                     BackProjectionImageFilterType>           Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  typedef ImageType::RegionType            OutputImageRegionType;
  typedef itk::CudaImage<float, 2>         ProjectionImageType;
  typedef ProjectionImageType::Pointer     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaWarpBackProjectionImageFilter, Superclass);

  /** Input projection stack */
  void SetInputProjectionStack(const InputImageType* ProjectionStack);
  InputImageType::Pointer GetInputProjectionStack();

  /** Input displacement vector field */
  void SetInputVolume(const InputImageType* Volume);
  InputImageType::Pointer GetInputVolume();

  /** Input displacement vector field */
  void SetDisplacementField(const DVFType* MVF);
  DVFType::Pointer GetDisplacementField();

protected:
  CudaWarpBackProjectionImageFilter();
  virtual ~CudaWarpBackProjectionImageFilter() {};

  virtual void GenerateInputRequestedRegion();

  virtual void GPUGenerateData();

private:
  CudaWarpBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented
};

} // end namespace rtk

#endif
