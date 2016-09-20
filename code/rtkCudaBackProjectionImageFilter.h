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

#ifndef rtkCudaBackProjectionImageFilter_h
#define rtkCudaBackProjectionImageFilter_h

#include "rtkBackProjectionImageFilter.h"
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaInPlaceImageFilter.h>

namespace rtk
{

/** \class CudaBackProjectionImageFilter
 * \brief Cuda version of the  backprojection.
 *
 * GPU-based implementation of the backprojection step of the
 * [Feldkamp, Davis, Kress, 1984] algorithm for filtered backprojection
 * reconstruction of cone-beam CT images with a circular source trajectory.
 *
 * \test rtksarttest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup Projector CudaImageToImageFilter
 */
class CudaBackProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  BackProjectionImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float,3>                          ImageType;
  typedef BackProjectionImageFilter< ImageType, ImageType> BackProjectionImageFilterType;
  typedef CudaBackProjectionImageFilter                    Self;
  typedef itk::CudaInPlaceImageFilter<ImageType, ImageType,
                     BackProjectionImageFilterType>        Superclass;
  typedef itk::SmartPointer<Self>                          Pointer;
  typedef itk::SmartPointer<const Self>                    ConstPointer;

  typedef ImageType::RegionType            OutputImageRegionType;
  typedef itk::CudaImage<float, 2>         ProjectionImageType;
  typedef ProjectionImageType::Pointer     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaBackProjectionImageFilter, Superclass);

protected:
  rtkcuda_EXPORT CudaBackProjectionImageFilter();
  virtual ~CudaBackProjectionImageFilter() {};

  virtual void GPUGenerateData();

private:
  CudaBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented
};

} // end namespace rtk

#endif
