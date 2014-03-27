#ifndef __rtkADMMTotalVariationConjugateGradientOperator_h
#define __rtkADMMTotalVariationConjugateGradientOperator_h

#include <itkMultiplyImageFilter.h>
#include <itkSubtractImageFilter.h>

#include "rtkConjugateGradientOperator.h"
#include "rtkBackProjectionImageFilter.h"
#include "rtkForwardProjectionImageFilter.h"
#include "rtkForwardDifferenceGradientImageFilter.h"
#include "rtkBackwardDifferenceDivergenceImageFilter.h"

#include "rtkThreeDCircularProjectionGeometry.h"

namespace rtk
{

  /** \class ADMMTotalVariationConjugateGradientOperator
   * \brief Implements the operator A used in the conjugate gradient step
   * of ADMM reconstruction with total variation regularization
   *
   * This filter implements the operator A used in the conjugate gradient step
   * of a reconstruction method based on compressed sensing. The method attempts
   * to find the f that minimizes || Rf -p ||_2^2 + alpha * TV(f), with R the
   * forward projection operator, p the measured projections, and TV the total variation.
   * Details on the method and the calculations can be found in
   *
   * Mory, C., B. Zhang, V. Auvray, M. Grass, D. Schafer, F. Peyrin, S. Rit, P. Douek,
   * and L. Boussel. “ECG-Gated C-Arm Computed Tomography Using L1 Regularization.”
   * In Proceedings of the 20th European Signal Processing Conference (EUSIPCO), 2728–32, 2012.
   *
   * This filter takes in input f and outputs R_t R f - beta div(grad(f)).
   *
   * \dot
   * digraph ADMMTotalVariationConjugateGradientOperator {
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
   * Subtract [ label="itk::SubtractImageFilter" URL="\ref itk::SubtractImageFilter"];
   * Multiply [ label="itk::MultiplyImageFilter (by lambda)" URL="\ref itk::MultiplyImageFilter"];
   * Gradient [ label="rtk::ForwardDifferenceGradientImageFilter" URL="\ref rtk::ForwardDifferenceGradientImageFilter"];
   * Divergence [ label="rtk::BackwardDifferenceDivergenceImageFilter" URL="\ref rtk::BackwardDifferenceDivergenceImageFilter"];
   * BackProjection [ label="rtk::BackProjectionImageFilter" URL="\ref rtk::BackProjectionImageFilter"];
   * ForwardProjection [ label="rtk::ForwardProjectionImageFilter" URL="\ref rtk::ForwardProjectionImageFilter"];
   *
   * Input0 -> BeforeZeroMultiplyVolume [arrowhead=None];
   * BeforeZeroMultiplyVolume -> ZeroMultiplyVolume;
   * BeforeZeroMultiplyVolume -> ForwardProjection;
   * BeforeZeroMultiplyVolume -> Gradient;
   * Input1 -> ZeroMultiplyProjections;
   * ZeroMultiplyProjections -> ForwardProjection;
   * ZeroMultiplyVolume -> BackProjection;
   * ForwardProjection -> BackProjection;
   * BackProjection -> Subtract;
   * Gradient -> Divergence;
   * Divergence -> Multiply;
   * Multiply -> Subtract;
   * Subtract -> Output;
   *
   * }
   * \enddot
   *
   * \test rtkadmmtvtest.cxx
   *
   * \author Cyril Mory
   *
   * \ingroup ReconstructionAlgorithm
   */

template< typename TOutputImage >
class ADMMTotalVariationConjugateGradientOperator : public ConjugateGradientOperator< TOutputImage >
{
public:
    /** Standard class typedefs. */
    typedef ADMMTotalVariationConjugateGradientOperator             Self;
    typedef ConjugateGradientOperator< TOutputImage > Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(rtkADMMTotalVariationConjugateGradientOperator, ConjugateGradientOperator)

    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >    BackProjectionFilterType;
    typedef typename BackProjectionFilterType::Pointer                      BackProjectionFilterPointer;

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage > ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                   ForwardProjectionFilterPointer;

    typedef itk::MultiplyImageFilter<TOutputImage>                          MultiplyFilterType;
    typedef itk::SubtractImageFilter<TOutputImage>                          SubtractFilterType;
    typedef rtk::ForwardDifferenceGradientImageFilter<TOutputImage>         GradientFilterType;
    typedef rtk::BackwardDifferenceDivergenceImageFilter
                          <typename GradientFilterType::OutputImageType>    DivergenceFilterType;

    /** Set the backprojection filter*/
    void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

    /** Set the forward projection filter*/
    void SetForwardProjectionFilter (const ForwardProjectionFilterPointer _arg);

    /** Set the geometry of both m_BackProjectionFilter and m_ForwardProjectionFilter */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Set the regularization parameter */
    itkSetMacro(Beta, float)

protected:
    ADMMTotalVariationConjugateGradientOperator();
    ~ADMMTotalVariationConjugateGradientOperator(){}

//    typename TOutputImage::ConstPointer GetInputVolumeSeries();
//    typename ProjectionStackType::ConstPointer GetInputProjectionStack();

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    BackProjectionFilterPointer            m_BackProjectionFilter;
    ForwardProjectionFilterPointer         m_ForwardProjectionFilter;

    typename SubtractFilterType::Pointer              m_SubtractFilter;
    typename MultiplyFilterType::Pointer              m_MultiplyFilter;
    typename MultiplyFilterType::Pointer              m_ZeroMultiplyProjectionFilter;
    typename MultiplyFilterType::Pointer              m_ZeroMultiplyVolumeFilter;
    typename DivergenceFilterType::Pointer            m_DivergenceFilter;
    typename GradientFilterType::Pointer              m_GradientFilter;
    float m_Beta;

    /** When the inputs have the same type, ITK checks whether they occupy the
    * same physical space or not. Obviously they dont, so we have to remove this check
    */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    ADMMTotalVariationConjugateGradientOperator(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

};
} //namespace RTK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkADMMTotalVariationConjugateGradientOperator.txx"
#endif

#endif
