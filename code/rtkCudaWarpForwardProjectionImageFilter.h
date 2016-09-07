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

#ifndef rtkCudaWarpForwardProjectionImageFilter_h
#define rtkCudaWarpForwardProjectionImageFilter_h

#include "rtkForwardProjectionImageFilter.h"
#include "itkCudaInPlaceImageFilter.h"
#include "itkCudaUtil.h"
#include "rtkWin32Header.h"

/** \class CudaWarpForwardProjectionImageFilter
 * \brief Trilinear interpolation forward projection in warped volume implemented in CUDA
 *
 * CudaWarpForwardProjectionImageFilter is similar to
 * CudaForwardProjectionImageFilter, but assumes the object has
 * undergone a known deformation, and compensates for it during the
 * forward projection. It amounts to bending the trajectories of the rays.
 *
 * \author Cyril Mory
 *
 * \ingroup Projector CudaImageToImageFilter
 */

namespace rtk
{

class rtkcuda_EXPORT CudaWarpForwardProjectionImageFilter :
  public itk::CudaInPlaceImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3>,
  ForwardProjectionImageFilter< itk::CudaImage<float,3>, itk::CudaImage<float,3> > >
{
public:
  /** Standard class typedefs. */
  typedef itk::CudaImage<float, 3>                                            InputImageType;
  typedef itk::CovariantVector<float, 3>                                      DisplacementVectorType;
  typedef CudaWarpForwardProjectionImageFilter                              Self;
  typedef ForwardProjectionImageFilter<InputImageType, InputImageType>                  Superclass;
  typedef itk::CudaInPlaceImageFilter<InputImageType, InputImageType, Superclass >      GPUSuperclass;
  typedef itk::SmartPointer<Self>                                             Pointer;
  typedef itk::SmartPointer<const Self>                                       ConstPointer;
  typedef itk::CudaImage<DisplacementVectorType, 3>                           DVFType;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(CudaWarpForwardProjectionImageFilter, ForwardProjectionImageFilter);

  /** Input projection stack */
  void SetInputProjectionStack(const InputImageType* ProjectionStack);
  InputImageType::Pointer GetInputProjectionStack();
  
  /** Input displacement vector field */
  void SetInputVolume(const InputImageType* Volume);
  InputImageType::Pointer GetInputVolume();
  
  /** Input displacement vector field */
  void SetDisplacementField(const DVFType* DVF);
  DVFType::Pointer GetDisplacementField();

  /** Set step size along ray (in mm). Default is 1 mm. */
  itkGetConstMacro(StepSize, double);
  itkSetMacro(StepSize, double);

protected:
  CudaWarpForwardProjectionImageFilter();
  ~CudaWarpForwardProjectionImageFilter() {};

  virtual void GenerateInputRequestedRegion();
  
  virtual void GPUGenerateData();

private:
  //purposely not implemented
  CudaWarpForwardProjectionImageFilter(const Self&);
  void operator=(const Self&);

  double             m_StepSize;
}; // end of class

} // end namespace rtk

#endif
