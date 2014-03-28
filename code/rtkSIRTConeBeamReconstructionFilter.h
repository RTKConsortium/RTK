#ifndef __rtkSIRTConeBeamReconstructionFilter_h
#define __rtkSIRTConeBeamReconstructionFilter_h

#include <itkMultiplyImageFilter.h>

#include "rtkConjugateGradientImageFilter.h"
#include "rtkSIRTConjugateGradientOperator.h"

#include "rtkRayCastInterpolatorForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#include "rtkJosephForwardProjectionImageFilter.h"
#ifdef USE_CUDA
  #include "rtkCudaForwardProjectionImageFilter.h"
  #include "rtkCudaBackProjectionImageFilter.h"
#endif

#include "rtkThreeDCircularProjectionGeometry.h"
#include "itkTimeProbe.h"

namespace rtk
{
template< typename TOutputImage >
class SIRTConeBeamReconstructionFilter : public itk::ImageToImageFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef SIRTConeBeamReconstructionFilter             Self;
    typedef itk::ImageToImageFilter<TOutputImage, TOutputImage> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(SIRTConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated */
    void SetInputVolume(const TOutputImage* Volume);

    /** The gated measured projections */
    void SetInputProjectionStack(const TOutputImage* Projection);

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >  ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                    ForwardProjectionFilterPointer;
    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >     BackProjectionFilterType;
    typedef rtk::ConjugateGradientImageFilter<TOutputImage>                  ConjugateGradientFilterType;
    typedef itk::MultiplyImageFilter<TOutputImage>                           MultiplyVolumeFilterType;
    typedef rtk::SIRTConjugateGradientOperator<TOutputImage>                 CGOperatorFilterType;

    /** Pass the ForwardProjection filter to the conjugate gradient operator */
    void ConfigureForwardProjection (int _arg);

    /** Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the B of AX=B */
    void ConfigureBackProjection (int _arg);

    /** Pass the geometry to all filters needing it */
    itkSetMacro(Geometry, ThreeDCircularProjectionGeometry::Pointer)

    itkSetMacro(NumberOfIterations, int)
    itkGetMacro(NumberOfIterations, int)

    itkSetMacro(MeasureExecutionTimes, bool)
    itkGetMacro(MeasureExecutionTimes, bool)

protected:
    SIRTConeBeamReconstructionFilter();
    ~SIRTConeBeamReconstructionFilter(){}

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename MultiplyVolumeFilterType::Pointer                                  m_ZeroMultiplyVolumeFilter;
    typename ConjugateGradientFilterType::Pointer                               m_ConjugateGradientFilter;
    typename CGOperatorFilterType::Pointer                                      m_CGOperator;
    typename ForwardProjectionImageFilter<TOutputImage, TOutputImage>::Pointer  m_ForwardProjectionFilter;
    typename BackProjectionImageFilter<TOutputImage, TOutputImage>::Pointer     m_BackProjectionFilterForConjugateGradient, m_BackProjectionFilter;

    /** The inputs of this filter have the same type (float, 3) but not the same meaning
    * It is normal that they do not occupy the same physical space. Therefore this check
    * must be removed */
    void VerifyInputInformation(){}

    /** The volume and the projections must have different requested regions
    */
    void GenerateInputRequestedRegion();
    void GenerateOutputInformation();

private:
    SIRTConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    int m_NumberOfIterations;
    bool m_MeasureExecutionTimes;
    ThreeDCircularProjectionGeometry::Pointer m_Geometry;
    int m_CurrentForwardProjectionConfiguration;
    int m_CurrentBackProjectionConfiguration;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkSIRTConeBeamReconstructionFilter.txx"
#endif

#endif
