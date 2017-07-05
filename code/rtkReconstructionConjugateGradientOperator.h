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

#ifndef rtkReconstructionConjugateGradientOperator_h
#define rtkReconstructionConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>
#include <itkAddImageFilter.h>

#include "rtkConstantImageSource.h"

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkLaplacianImageFilter.h"

#ifdef RTK_USE_CUDA
  #include "rtkCudaConstantVolumeSource.h"
  #include "rtkCudaLaplacianImageFilter.h"
#endif

namespace rtk
{

  /** \class ReconstructionConjugateGradientOperator
   * \brief Implements the operator A used in conjugate gradient reconstruction
   *
   * This filter implements the operator A used in the conjugate gradient reconstruction method,
   * which attempts to find the f that minimizes
   * || sqrt(D) (Rf -p) ||_2^2 + gamma || grad f ||_2^2,
   * with R the forward projection operator,
   * p the measured projections, and D the displaced detector weighting operator.
   *
   * With gamma=0, this it is similar to the ART and SART methods. The difference lies
   * in the algorithm employed to minimize this cost function. ART uses the
   * Kaczmarz method (projects and back projects one ray at a time),
   * SART the block-Kaczmarz method (projects and back projects one projection
   * at a time), and ConjugateGradient a conjugate gradient method
   * (projects and back projects all projections together).
   *
   * This filter takes in input f and outputs R_t D R f + gamma Laplacian f
   * If m_Regularized is false (default), regularization is ignored, and gamma is considered null 
   *
   * \dot
   * digraph ReconstructionConjugateGradientOperator {
   *
   * Input0 [ label="Input 0 (Volume)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Input2 [label="Input 2 (Weights)"];
   * Input2 [shape=Mdiamond];
   * Input3 [label="Input Support Mask"];
   * Input3 [shape=Mdiamond];
   * Output [label="Output (Volume)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * ConstantVolumeSource [label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
   * ConstantProjectionsSource [label="rtk::ConstantImageSource" URL="\ref rtk::ConstantImageSource"];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   * Multiply [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
   * MultiplyInput [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
   * MultiplyOutput [ label="itk::MultiplyImageFilter" URL="\ref itk::MultiplyImageFilter"];
   * Laplacian [ label="rtk::LaplacianImageFilter" URL="\ref rtk::LaplacianImageFilter"];
   * MultiplyLaplacian [ label="itk::MultiplyImageFilter (by gamma)" URL="\ref itk::MultiplyImageFilter"];
   * Add [ label="itk::AddImageFilter" URL="\ref itk::AddImageFilter"];
   *
   * Input0 -> MultiplyInput;
   * Input3 -> MultiplyInput;
   * MultiplyInput -> ForwardProjection;
   * ConstantProjectionsSource -> ForwardProjection;
   * ConstantVolumeSource -> BackProjection;
   * ForwardProjection -> Multiply;
   * Input2 -> Multiply;
   * Multiply -> BackProjection;
   * BackProjection -> Add;
   * Input3 -> MultiplyOutput;
   * MultiplyInput -> Laplacian;
   * Laplacian -> MultiplyLaplacian;
   * MultiplyLaplacian -> Add;
   * Add -> MultiplyOutput;
   * MultiplyOutput -> Output;
   * }
   * \enddot
   *
   * \test rtkconjugategradienttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage >
class ReconstructionConjugateGradientOperator : public ConjugateGradientOperator< TOutputImage >
{
public:
  /** Standard class typedefs. */
  typedef ReconstructionConjugateGradientOperator    Self;
  typedef ConjugateGradientOperator< TOutputImage >  Superclass;
  typedef itk::SmartPointer< Self >                  Pointer;
#ifdef RTK_USE_CUDA
  typedef itk::CudaImage<itk::CovariantVector<typename TOutputImage::ValueType, TOutputImage::ImageDimension>, TOutputImage::ImageDimension > GradientImageType;
#else
  typedef itk::Image<itk::CovariantVector<typename TOutputImage::ValueType, TOutputImage::ImageDimension >, TOutputImage::ImageDimension > GradientImageType;
#endif

  /** Method for creation through the object factory. */
  itkNewMacro(Self)

  /** Run-time type information (and related methods). */
  itkTypeMacro(rtkReconstructionConjugateGradientOperator, ConjugateGradientOperator)

  typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >    BackProjectionFilterType;
  typedef typename BackProjectionFilterType::Pointer                      BackProjectionFilterPointer;

  typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage > ForwardProjectionFilterType;
  typedef typename ForwardProjectionFilterType::Pointer                   ForwardProjectionFilterPointer;

  typedef rtk::ConstantImageSource<TOutputImage>                          ConstantSourceType;
  typedef itk::MultiplyImageFilter<TOutputImage>                          MultiplyFilterType;
  typedef itk::AddImageFilter<TOutputImage>                               AddFilterType;

  typedef rtk::LaplacianImageFilter<TOutputImage, GradientImageType>      LaplacianFilterType;

  typedef typename TOutputImage::Pointer                                  OutputImagePointer;

  /** Set the backprojection filter*/
  void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

  /** Set the forward projection filter*/
  void SetForwardProjectionFilter (const ForwardProjectionFilterPointer _arg);

  /** Set the support mask, if any, for support constraint in reconstruction */
  void SetSupportMask(const TOutputImage *SupportMask);
  typename TOutputImage::ConstPointer GetSupportMask();

  /** Set the geometry of both m_BackProjectionFilter and m_ForwardProjectionFilter */
  itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)
  
  /** If Regularized, perform laplacian-based regularization during 
  *  reconstruction (gamma is the strength of the regularization) */
  itkSetMacro(Regularized, bool)
  itkGetMacro(Regularized, bool)
  itkSetMacro(Gamma, float)
  itkGetMacro(Gamma, float)

protected:
  ReconstructionConjugateGradientOperator();
  ~ReconstructionConjugateGradientOperator() {}

  /** Does the real work. */
  void GenerateData() ITK_OVERRIDE;

  /** Member pointers to the filters used internally (for convenience)*/
  BackProjectionFilterPointer            m_BackProjectionFilter;
  ForwardProjectionFilterPointer         m_ForwardProjectionFilter;

  typename ConstantSourceType::Pointer              m_ConstantProjectionsSource;
  typename ConstantSourceType::Pointer              m_ConstantVolumeSource;
  typename MultiplyFilterType::Pointer              m_MultiplyProjectionsFilter;
  typename MultiplyFilterType::Pointer              m_MultiplyOutputVolumeFilter;
  typename MultiplyFilterType::Pointer              m_MultiplyInputVolumeFilter;
  typename MultiplyFilterType::Pointer              m_MultiplyLaplacianFilter;
  typename AddFilterType::Pointer                   m_AddFilter;
  typename LaplacianFilterType::Pointer             m_LaplacianFilter;
  typename MultiplyFilterType::Pointer              m_MultiplySupportMaskFilter;

  /** Member attributes */
  rtk::ThreeDCircularProjectionGeometry::Pointer    m_Geometry;
  bool                                              m_Regularized;
  float                                             m_Gamma; //Strength of the regularization

  /** Pointers to intermediate images, used to simplify complex branching */
  typename TOutputImage::Pointer                    m_FloatingInputPointer, m_FloatingOutputPointer;

  /** When the inputs have the same type, ITK checks whether they occupy the
   * same physical space or not. Obviously they dont, so we have to remove this check */
  void VerifyInputInformation() ITK_OVERRIDE {}

  /** The volume and the projections must have different requested regions */
  void GenerateInputRequestedRegion() ITK_OVERRIDE;
  void GenerateOutputInformation() ITK_OVERRIDE;

private:
  ReconstructionConjugateGradientOperator(const Self &); //purposely not implemented
  void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkReconstructionConjugateGradientOperator.hxx"
#endif

#endif
