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

#ifndef rtkFDKWarpBackProjectionImageFilter_h
#define rtkFDKWarpBackProjectionImageFilter_h

#include "rtkFDKBackProjectionImageFilter.h"

#include <itkBarrier.h>

namespace rtk
{

/** \class FDKWarpBackProjectionImageFilter
 * \brief CPU version of the warp backprojection of motion-compensated FDK.
 *
 * The deformation is described by the TDeformation template parameter. This
 * type must implement the function SetFrame and returns a 3D deformation
 * vector field. FDKWarpBackProjectionImageFilter loops over the projections,
 * sets the frame number, updates the deformation and compose the resulting
 * deformation with the projection matrix. One thus obtain a warped
 * backprojection that is used in motion-compensated cone-beam CT
 * reconstruction. This has been described in [Rit et al, TMI, 2009] and
 * [Rit et al, Med Phys, 2009].
 *
 * \test rtkmotioncompensatedfdktest.cxx
 *
 * \author Simon Rit
 *
 * \ingroup Projector
 */
template <class TInputImage, class TOutputImage, class TDeformation>
class ITK_EXPORT FDKWarpBackProjectionImageFilter :
  public FDKBackProjectionImageFilter<TInputImage,TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKWarpBackProjectionImageFilter                       Self;
  typedef FDKBackProjectionImageFilter<TInputImage,TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                                Pointer;
  typedef itk::SmartPointer<const Self>                          ConstPointer;
  typedef typename TInputImage::PixelType                        InputPixelType;
  typedef typename TOutputImage::RegionType                      OutputImageRegionType;

  typedef TDeformation                      DeformationType;
  typedef typename DeformationType::Pointer DeformationPointer;

  typedef rtk::ProjectionGeometry<TOutputImage::ImageDimension>     GeometryType;
  typedef typename GeometryType::Pointer                            GeometryPointer;
  typedef typename GeometryType::MatrixType                         ProjectionMatrixType;
  typedef itk::Image<InputPixelType, TInputImage::ImageDimension-1> ProjectionImageType;
  typedef typename ProjectionImageType::Pointer                     ProjectionImagePointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FDKWarpBackProjectionImageFilter, FDKBackProjectionImageFilter);

  /** Set the deformation. */
  itkGetMacro(Deformation, DeformationPointer);
  itkSetMacro(Deformation, DeformationPointer);

protected:
  FDKWarpBackProjectionImageFilter():m_DeformationUpdateError(false) {};
  ~FDKWarpBackProjectionImageFilter() {}

  void BeforeThreadedGenerateData() ITK_OVERRIDE;

  void ThreadedGenerateData( const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId ) ITK_OVERRIDE;

private:
  FDKWarpBackProjectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);                   //purposely not implemented

  DeformationPointer    m_Deformation;
  itk::Barrier::Pointer m_Barrier;
  bool                  m_DeformationUpdateError;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkFDKWarpBackProjectionImageFilter.hxx"
#endif

#endif
