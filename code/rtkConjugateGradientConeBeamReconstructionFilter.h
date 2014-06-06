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

#ifndef __rtkConjugateGradientConeBeamReconstructionFilter_h
#define __rtkConjugateGradientConeBeamReconstructionFilter_h

#include <itkMultiplyImageFilter.h>
#include <itkTimeProbe.h>

#include "rtkConjugateGradientImageFilter.h"
#include "rtkReconstructionConjugateGradientOperator.h"
#include "rtkIterativeConeBeamReconstructionFilter.h"
#include "rtkThreeDCircularProjectionGeometry.h"
#include "rtkDisplacedDetectorImageFilter.h"

namespace rtk
{
  /** \class ConjugateGradientConeBeamReconstructionFilter
   * \brief Implements ConjugateGradient
   *
   * This filter implements the ConjugateGradient method.
   * ConjugateGradient attempts to find the f that minimizes || Rf -p ||_2^2, with R the
   * forward projection operator and p the measured projections.
   * In this it is similar to the ART and SART methods. The difference lies
   * in the algorithm employed to minimize this cost function. ART uses the
   * Kaczmarz method (projects and back projects one ray at a time),
   * SART the block-Kaczmarz method (projects and back projects one projection
   * at a time), and ConjugateGradient a conjugate gradient method
   * (projects and back projects all projections together).
   *
   * \dot
   * digraph ConjugateGradientConeBeamReconstructionFilter {
   *
   * Input0 [ label="Input 0 (Volume)"];
   * Input0 [shape=Mdiamond];
   * Input1 [label="Input 1 (Projections)"];
   * Input1 [shape=Mdiamond];
   * Output [label="Output (Volume)"];
   * Output [shape=Mdiamond];
   *
   * node [shape=box];
   * ZeroMultiplyVolume [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * BeforeZeroMultiplyVolume [label="", fixedsize="false", width=0, height=0, shape=none];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * ConjugateGradient[ label="rtk::ConjugateGradientImageFilter" URL="\ref rtk::ConjugateGradientImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=None];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> ConjugateGradient;
   * Input1 -> BackProjection;
   * ZeroMultiplyVolume -> BackProjection;
   * BackProjection -> ConjugateGradient;
   * ConjugateGradient -> Output;
   * }
   * \enddot
   *
   * \test rtksirttest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage >
class ConjugateGradientConeBeamReconstructionFilter : public rtk::IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef ConjugateGradientConeBeamReconstructionFilter                      Self;
    typedef IterativeConeBeamReconstructionFilter<TOutputImage, TOutputImage>  Superclass;
    typedef itk::SmartPointer< Self >                                          Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ConjugateGradientConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated */
    void SetInputVolume(const TOutputImage* Volume);

    /** The gated measured projections */
    void SetInputProjectionStack(const TOutputImage* Projection);

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >  ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                    ForwardProjectionFilterPointer;
    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >     BackProjectionFilterType;
    typedef rtk::ConjugateGradientImageFilter<TOutputImage>                  ConjugateGradientFilterType;
    typedef itk::MultiplyImageFilter<TOutputImage>                           MultiplyVolumeFilterType;
    typedef rtk::ReconstructionConjugateGradientOperator<TOutputImage>       CGOperatorFilterType;
    typedef rtk::DisplacedDetectorImageFilter<TOutputImage>                  DisplacedDetectorFilterType;

    /** Pass the ForwardProjection filter to the conjugate gradient operator */
    void SetForwardProjectionFilter (int _arg);

    /** Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the B of AX=B */
    void SetBackProjectionFilter (int _arg);

    /** Pass the geometry to all filters needing it */
    itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

    itkSetMacro(NumberOfIterations, int)
    itkGetMacro(NumberOfIterations, int)

    itkSetMacro(MeasureExecutionTimes, bool)
    itkGetMacro(MeasureExecutionTimes, bool)

protected:
    ConjugateGradientConeBeamReconstructionFilter();
    ~ConjugateGradientConeBeamReconstructionFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename MultiplyVolumeFilterType::Pointer                                  m_ZeroMultiplyVolumeFilter;
    typename ConjugateGradientFilterType::Pointer                               m_ConjugateGradientFilter;
    typename CGOperatorFilterType::Pointer                                      m_CGOperator;
    typename ForwardProjectionImageFilter<TOutputImage, TOutputImage>::Pointer  m_ForwardProjectionFilter;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilter;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilterForB;
    typename DisplacedDetectorFilterType::Pointer                               m_DisplacedDetectorFilter;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    ConjugateGradientConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    ThreeDCircularProjectionGeometry::Pointer m_Geometry;

    int   m_NumberOfIterations;
    bool  m_MeasureExecutionTimes;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkConjugateGradientConeBeamReconstructionFilter.txx"
#endif

#endif
