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

#ifndef rtkIterativeConeBeamReconstructionFilter_h
#define rtkIterativeConeBeamReconstructionFilter_h

// Forward projection filters
#include "rtkConfiguration.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#include "rtkJosephForwardAttenuatedProjectionImageFilter.h"
// Back projection filters
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephBackAttenuatedProjectionImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaForwardProjectionImageFilter.h"
  #include "rtkCudaBackProjectionImageFilter.h"
  #include "rtkCudaRayCastBackProjectionImageFilter.h"
#endif

namespace rtk
{

/** \class IterativeConeBeamReconstructionFilter
 * \brief Mother class for cone beam reconstruction filters which
 * need runtime selection of their forward and back projection filters
 *
 * IterativeConeBeamReconstructionFilter defines methods to set the forward
 * and/or back projection filter(s) of a IterativeConeBeamReconstructionFilter
 * at runtime
 *
 * \author Cyril Mory
 *
 * \ingroup ReconstructionAlgorithm
 */
template<class TOutputImage,
         class ProjectionStackType=TOutputImage>
class ITK_EXPORT IterativeConeBeamReconstructionFilter :
  public itk::ImageToImageFilter<TOutputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef IterativeConeBeamReconstructionFilter               Self;
  typedef itk::ImageToImageFilter<TOutputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                             Pointer;
  typedef itk::SmartPointer<const Self>                       ConstPointer;

  /** Convenient typedefs */
  typedef ProjectionStackType                                                     VolumeType;
  typedef enum {FP_UNKNOWN=-1, FP_JOSEPH=0, FP_CUDARAYCAST=2, FP_JOSEPHATTENUATED=3}                                    ForwardProjectionType;
  typedef enum {BP_UNKNOWN=-1, BP_VOXELBASED=0,BP_JOSEPH=1,BP_CUDAVOXELBASED=2,BP_CUDARAYCAST=4,BP_JOSEPHATTENUATED=5} BackProjectionType;

  /** Typedefs of each subfilter of this composite filter */
  typedef rtk::ForwardProjectionImageFilter< VolumeType, ProjectionStackType >    ForwardProjectionFilterType;
  typedef rtk::BackProjectionImageFilter< ProjectionStackType, VolumeType >       BackProjectionFilterType;
  typedef typename ForwardProjectionFilterType::Pointer                           ForwardProjectionPointerType;
  typedef typename BackProjectionFilterType::Pointer                              BackProjectionPointerType;

  /** Standard New method. */
  itkNewMacro(Self)

  /** Runtime information support. */
  itkTypeMacro(IterativeConeBeamReconstructionFilter, itk::ImageToImageFilter)

  /** Accessors to forward and backprojection types. */
  virtual void SetForwardProjectionFilter (ForwardProjectionType fwtype);
  ForwardProjectionType GetForwardProjectionFilter () { return m_CurrentForwardProjectionConfiguration; }
  virtual void SetBackProjectionFilter (BackProjectionType bptype);
  BackProjectionType GetBackProjectionFilter () { return m_CurrentBackProjectionConfiguration; }

protected:
  IterativeConeBeamReconstructionFilter();
  virtual ~IterativeConeBeamReconstructionFilter() ITK_OVERRIDE {}

  /** Creates and returns an instance of the back projection filter.
   * To be used in SetBackProjectionFilter. */
  virtual BackProjectionPointerType InstantiateBackProjectionFilter (int bptype);

  /** Creates and returns an instance of the forward projection filter.
   * To be used in SetForwardProjectionFilter. */
  virtual ForwardProjectionPointerType InstantiateForwardProjectionFilter (int fwtype);

  /** Internal variables storing the current forward
    and back projection methods */
  ForwardProjectionType m_CurrentForwardProjectionConfiguration;
  BackProjectionType    m_CurrentBackProjectionConfiguration;

private:
  //purposely not implemented
  IterativeConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkIterativeConeBeamReconstructionFilter.hxx"
#endif

#endif
