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
 * \ingroup RTK Projector
 */
template <class TInputImage, class TOutputImage, class TDeformation>
class ITK_EXPORT FDKWarpBackProjectionImageFilter : public FDKBackProjectionImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(FDKWarpBackProjectionImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(FDKWarpBackProjectionImageFilter);
#endif

  /** Standard class type alias. */
  using Self = FDKWarpBackProjectionImageFilter;
  using Superclass = FDKBackProjectionImageFilter<TInputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  using InputPixelType = typename TInputImage::PixelType;
  using OutputImageRegionType = typename TOutputImage::RegionType;

  using DeformationType = TDeformation;
  using DeformationPointer = typename DeformationType::Pointer;

  using GeometryType = rtk::ProjectionGeometry<TOutputImage::ImageDimension>;
  using GeometryPointer = typename GeometryType::Pointer;
  using ProjectionMatrixType = typename GeometryType::MatrixType;
  using ProjectionImageType = itk::Image<InputPixelType, TInputImage::ImageDimension - 1>;
  using ProjectionImagePointer = typename ProjectionImageType::Pointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(FDKWarpBackProjectionImageFilter, FDKBackProjectionImageFilter);

  /** Set the deformation. */
  itkGetMacro(Deformation, DeformationPointer);
  itkSetObjectMacro(Deformation, DeformationType);

protected:
  FDKWarpBackProjectionImageFilter();
  ~FDKWarpBackProjectionImageFilter() override = default;

  void
  GenerateData() override;

private:
  DeformationPointer m_Deformation;
};

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkFDKWarpBackProjectionImageFilter.hxx"
#endif

#endif
