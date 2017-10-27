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

#ifndef rtkCudaWarpImageFilter_h
#define rtkCudaWarpImageFilter_h

#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkWarpImageFilter.h>
#include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaWarpImageFilter
 * \brief Cuda version of the WarpImageFilter
 *
 * Deform an image using a Displacement Vector Field. GPU-based implementation
 *
 * \test rtkwarptest
 *
 * \author Cyril Mory
 *
 * \ingroup CudaImageToImageFilter
 */
class CudaWarpImageFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
    itk::WarpImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>, itk::CudaImage<itk::CovariantVector<float, 3>, 3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float,3>                              ImageType;
  typedef itk::CovariantVector<float, 3>                       DisplacementVectorType;
  typedef itk::CudaImage<DisplacementVectorType, 3>            DVFType;
  typedef itk::WarpImageFilter< ImageType, ImageType, DVFType> WarpImageFilterType;
  typedef CudaWarpImageFilter                                  Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
                     WarpImageFilterType>                      Superclass;
  typedef itk::SmartPointer<Self>                              Pointer;
  typedef itk::SmartPointer<const Self>                        ConstPointer;

  typedef ImageType::RegionType            OutputImageRegionType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaWarpImageFilter, Superclass);

protected:
  rtkcuda_EXPORT CudaWarpImageFilter();
  virtual ~CudaWarpImageFilter() {};

  virtual void GPUGenerateData();

private:
  CudaWarpImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented
};

} // end namespace rtk

#endif
