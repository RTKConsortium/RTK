#ifndef __rtkSIRTConjugateGradientOperator_h
#define __rtkSIRTConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"

#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

  /** \class SIRTConjugateGradientOperator
   * \brief Implements the operator A used in SIRT
   *
   * This filter implements the operator A used in the SIRT method.
   * SIRT attempts to find the f that minimizes || Rf -p ||_2^2, with R the
   * forward projection operator and p the measured projections.
   * In this it is similar to the ART and SART methods. The difference lies
   * in the algorithm employed to minimize this cost function. ART uses the
   * Kaczmarz method (projects and back projects one ray at a time),
   * SART the block-Kaczmarz method (projects and back projects one projection
   * at a time), and SIRT a steepest descent or conjugate gradient method
   * (projects and back projects all projections together).
   *
   * This filter takes in input f and outputs R_t R f
   *
   * \dot
   * digraph SIRTConjugateGradientOperator {
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
   * ZeroMultiplyProjections [label="itk::MultiplyImageFilter (by zero)" URL="\ref itk::MultiplyImageFilter"];
   * BeforeZeroMultiplyVolume [label="", fixedsize="false", width=0, height=0, shape=none];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=None];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> ForwardProjection;
   * Input1 -> ZeroMultiplyProjections;
   * ZeroMultiplyProjections -> ForwardProjection;
   * ZeroMultiplyVolume -> BackProjection;
   * ForwardProjection -> BackProjection;
   * BackProjection -> Output;
   *
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
class SIRTConjugateGradientOperator : public ConjugateGradientOperator< TOutputImage >
{
public:
    /** Standard class typedefs. */
    typedef SIRTConjugateGradientOperator             Self;
    typedef ConjugateGradientOperator< TOutputImage > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(rtkSIRTConjugateGradientOperator, ConjugateGradientOperator)

    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >    BackProjectionFilterType;
    typedef typename BackProjectionFilterType::Pointer                      BackProjectionFilterPointer;

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage > ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                   ForwardProjectionFilterPointer;

    typedef itk::MultiplyImageFilter<TOutputImage>                          MultiplyFilterType;

    /** Set the backprojection filter*/
    void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

    /** Set the forward projection filter*/
    void SetForwardProjectionFilter (const ForwardProjectionFilterPointer _arg);

    /** Set the geometry of both m_BackProjectionFilter and m_ForwardProjectionFilter */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

protected:
    SIRTConjugateGradientOperator();
    ~SIRTConjugateGradientOperator(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    BackProjectionFilterPointer            m_BackProjectionFilter;
    ForwardProjectionFilterPointer         m_ForwardProjectionFilter;

    typename MultiplyFilterType::Pointer              m_MultiplyFilter;
    typename MultiplyFilterType::Pointer              m_ZeroMultiplyProjectionFilter;
    typename MultiplyFilterType::Pointer              m_ZeroMultiplyVolumeFilter;

    /** When the inputs have the same type, ITK checks whether they occupy the
    * same physical space or not. Obviously they dont, so we have to remove this check
    */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    SIRTConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSIRTConjugateGradientOperator.txx"
#endif

#endif
