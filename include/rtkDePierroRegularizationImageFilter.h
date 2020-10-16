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

#ifndef rtkDePierroRegularizationImageFilter_h
#define rtkDePierroRegularizationImageFilter_h

#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkDivideImageFilter.h>
#include <itkImageKernelOperator.h>
#include <itkNeighborhoodOperatorImageFilter.h>
#include <itkConstantBoundaryCondition.h>
#include "rtkConstantImageSource.h"

namespace rtk
{

/** \class DePierroRegularizationImageFilter
 * \brief Implements a regularization for MLEM/OSEM reconstruction.
 *
 * Perform the quadratic penalization described in [De Pierro, IEEE TMI, 1995] for
 * MLEM/OSEM reconstruction.
 *
 * This filter takes the k and k+1 updates of the classic MLEM/OSEM algorithm as
 * inputs and return the regularization factor.
 *
 * \dot
 * digraph DePierroRegularizationImageFilter {
 *
 *  Input0 [ label="Input 0 (Update k of MLEM)"];
 *  Input0 [shape=Mdiamond];
 *  Input1 [label="Input 1 (Update k+1 of MLEM)"];
 *  Input1 [shape=Mdiamond];
 *  Input2 [label="Input 2 (Backprojection of one)"];
 *  Input2 [shape=Mdiamond];
 *  Output [label="Output (Regularization factor)"];
 *  Output [shape=Mdiamond];
 *  KernelImage [ label="KernelImage"];
 *  KernelImage [shape=Mdiamond];
 *
 *  node [shape=box];
 *  ImageKernelOperator [ label="itk::ImageKernelOperator" URL="\ref itk::ImageKernelOperator"];
 *  NOIF[ label="itk::NeighborhoodOperatorImageFilter" URL="\ref itk::NeighborhoodOperatorImageFilter"];
 *  Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
 *  Multiply1 [ label="itk::MultiplyImageFilter (by constant)" URL="\ref itk::MultiplyImageFilter"];
 *  Multiply2 [ label="itk::MultiplyImageFilter (by constant)" URL="\ref itk::MultiplyImageFilter"];
 *  CustomBinary [ label="itk::BinaryGeneratorImageFilter ((A + sqrt(A*A + B))/2)" URL="\ref
 * itk::BinaryGeneratorImageFilter"]; Input2 -> Subtract; KernelImage -> ImageKernelOperator; ImageKernelOperator ->
 * NOIF; Input0 -> NOIF; NOIF -> Multiply1; Multiply1 -> Subtract; Subtract -> CustomBinary; Input1 -> Multiply2;
 *  Multiply2 -> CustomBinary;
 *  CustomBinary -> Output;
 *  }
 * \enddot
 *
 * \author Antoine Robert
 *
 * \ingroup RTK ReconstructionAlgorithm
 */
template <class TInputImage, class TOutputImage = TInputImage>
class ITK_EXPORT DePierroRegularizationImageFilter : public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
#if ITK_VERSION_MAJOR == 5 && ITK_VERSION_MINOR == 1
  ITK_DISALLOW_COPY_AND_ASSIGN(DePierroRegularizationImageFilter);
#else
  ITK_DISALLOW_COPY_AND_MOVE(DePierroRegularizationImageFilter);
#endif

  /** Standard class type alias. */
  using Self = DePierroRegularizationImageFilter;
  using Superclass = itk::ImageToImageFilter<TOutputImage, TOutputImage>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Some convenient type alias. */
  using InputImageType = TInputImage;
  using InputImagePointerType = typename TInputImage::Pointer;
  using OutputImageType = TOutputImage;
  using InputPixelType = typename TInputImage::PixelType;

  /** ImageDimension constants */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;

  /** Typedefs of each subfilter of this composite filter */
  using MultiplyImageFilterType = itk::MultiplyImageFilter<InputImageType, InputImageType>;
  using MultpiplyImageFilterPointerType = typename MultiplyImageFilterType::Pointer;
  using ConstantVolumeSourceType = rtk::ConstantImageSource<InputImageType>;
  using ConstantVolumeSourcePointerType = typename ConstantVolumeSourceType::Pointer;
  using SubtractImageFilterType = itk::SubtractImageFilter<InputImageType, InputImageType>;
  using SubtractImageFilterPointerType = typename SubtractImageFilterType::Pointer;
  using ImageKernelOperatorType = itk::ImageKernelOperator<InputPixelType, InputImageDimension>;
  using NOIFType = itk::NeighborhoodOperatorImageFilter<InputImageType, InputImageType>;
  using NOIFPointerType = typename NOIFType::Pointer;
  using CustomBinaryFilterType = itk::BinaryGeneratorImageFilter<InputImageType, InputImageType, OutputImageType>;
  using CustomBinaryFilterPointerType = typename CustomBinaryFilterType::Pointer;

  /** Typedef for the boundary condition */
  using BoundaryCondition = itk::ConstantBoundaryCondition<InputImageType>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods). */
  itkTypeMacro(DePierroRegularizationImageFilter, itk::ImageToImageFilter);

  /** Get / Set the hyper parameter for the regularization */
  itkGetMacro(Beta, double);
  itkSetMacro(Beta, double);

protected:
  DePierroRegularizationImageFilter();
  ~DePierroRegularizationImageFilter() override = default;

  void
  GenerateInputRequestedRegion() override;

  void
  GenerateOutputInformation() override;

  void
  GenerateData() override;

  MultpiplyImageFilterPointerType m_MultiplyConstant1ImageFilter;
  MultpiplyImageFilterPointerType m_MultiplyConstant2ImageFilter;
  ConstantVolumeSourcePointerType m_KernelImage;
  ConstantVolumeSourcePointerType m_DefaultNormalizationVolume;
  SubtractImageFilterPointerType  m_SubtractImageFilter;
  BoundaryCondition               m_BoundsCondition;
  ImageKernelOperatorType         m_KernelOperator;
  NOIFPointerType                 m_ConvolutionFilter;
  CustomBinaryFilterPointerType   m_CustomBinaryFilter;

private:
  double m_Beta{ 0.01 };

}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "rtkDePierroRegularizationImageFilter.hxx"
#endif

#endif
