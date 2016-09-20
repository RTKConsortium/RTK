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
#ifndef rtkCudaCropImageFilter_h
#define rtkCudaCropImageFilter_h

#include <itkCropImageFilter.h>
#include "rtkWin32Header.h"

#include <itkCudaImage.h>
#include <itkCudaImageToImageFilter.h>

namespace rtk
{
/** \class CudaCropImageFilter
 * \brief Decrease the image size by cropping the image by an itk::Size at
 * both the upper and lower bounds of the largest possible region.
 *
 * CropImageFilter changes the image boundary of an image by removing
 * pixels outside the target region.  The target region is not specified in
 * advance, but calculated in BeforeThreadedGenerateData().
 *
 * This filter uses CropImageFilter to perform the cropping.
 *
 * \author Marc Vila
 *
 * \ingroup CudaImageToImageFilter
 *
 */
class CudaCropImageFilter :
  public itk::CudaImageToImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  itk::CropImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float,3>                                                                    ImageType;
  typedef CudaCropImageFilter                                                                        Self;
  typedef itk::CropImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> >                   Superclass;
  typedef itk::CudaImageToImageFilter<itk::CudaImage<float,3>, itk::CudaImage<float,3>, Superclass > GPUSuperclass;
  typedef itk::SmartPointer< Self >                                                                  Pointer;
  typedef itk::SmartPointer< const Self >                                                            ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaCropImageFilter, ImageToImageFilter);

protected:
  rtkcuda_EXPORT CudaCropImageFilter();
  virtual ~CudaCropImageFilter() {};

  virtual void GPUGenerateData();

private:
  CudaCropImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);  //purposely not implemented

}; // end of class
} // end namespace rtk

#endif
