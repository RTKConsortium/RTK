#ifndef __rtkADMMTotalVariationConeBeamReconstructionFilter_h
#define __rtkADMMTotalVariationConeBeamReconstructionFilter_h

#include <itkImageToImageFilter.h>
#include <itkAddImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkMultiplyImageFilter.h>

//#include "rtkForwardDifferenceGradientImageFilter.h"
//#include "rtkBackwardDifferenceDivergenceImageFilter.h"
#include "rtkConjugateGradientImageFilter.h"
#include "rtkSoftThresholdTVImageFilter.h"

#include "rtkADMMTotalVariationConjugateGradientOperator.h"

//#include "rtkBackProjectionImageFilter.h"
//#include "rtkForwardProjectionImageFilter.h"
#include "rtkRayCastInterpolatorForwardProjectionImageFilter.h"
#include "rtkJosephBackProjectionImageFilter.h"
#ifdef USE_CUDA
  #include "rtkCudaForwardProjectionImageFilter.h"
  #include "rtkCudaBackProjectionImageFilter.h"
#endif

#include "rtkThreeDCircularProjectionGeometry.h"
#include "itkTimeProbe.h"

namespace rtk
{
template< typename TOutputImage >
class ADMMTotalVariationConeBeamReconstructionFilter : public itk::ImageToImageFilter<TOutputImage, TOutputImage>
{
public:
    /** Standard class typedefs. */
    typedef ADMMTotalVariationConeBeamReconstructionFilter             Self;
    typedef itk::ImageToImageFilter<TOutputImage, TOutputImage> Superclass;
    typedef itk::SmartPointer< Self >        Pointer;

    /** Method for creation through the object factory. */
    itkNewMacro(Self)

    /** Run-time type information (and related methods). */
    itkTypeMacro(ADMMTotalVariationConeBeamReconstructionFilter, itk::ImageToImageFilter)

    /** The 3D image to be updated */
    void SetInputVolume(const TOutputImage* Volume);

    /** The gated measured projections */
    void SetInputProjectionStack(const TOutputImage* Projection);

    typedef rtk::ForwardProjectionImageFilter< TOutputImage, TOutputImage >               ForwardProjectionFilterType;
    typedef typename ForwardProjectionFilterType::Pointer                                 ForwardProjectionFilterPointer;
    typedef rtk::BackProjectionImageFilter< TOutputImage, TOutputImage >                  BackProjectionFilterType;

    typedef rtk::ConjugateGradientImageFilter<TOutputImage>                               ConjugateGradientFilterType;
    typedef rtk::ForwardDifferenceGradientImageFilter<TOutputImage>                       ImageGradientFilterType;
    typedef rtk::BackwardDifferenceDivergenceImageFilter
        <typename ImageGradientFilterType::OutputImageType>                               ImageDivergenceFilterType;
    typedef rtk::SoftThresholdTVImageFilter
        <typename ImageGradientFilterType::OutputImageType>                               SoftThresholdTVFilterType;
    typedef itk::AddImageFilter<TOutputImage>                                             AddVolumeFilterType;
    typedef itk::AddImageFilter<typename ImageGradientFilterType::OutputImageType>        AddGradientsFilterType;
    typedef itk::MultiplyImageFilter<TOutputImage>                                        MultiplyVolumeFilterType;
    typedef itk::MultiplyImageFilter<typename ImageGradientFilterType::OutputImageType>   MultiplyGradientFilterType;
    typedef itk::SubtractImageFilter<typename ImageGradientFilterType::OutputImageType>   SubtractGradientsFilterType;
    typedef rtk::ADMMTotalVariationConjugateGradientOperator<TOutputImage>                CGOperatorFilterType;

    /** Pass the ForwardProjection filter to the conjugate gradient operator */
    void ConfigureForwardProjection (int _arg);

    /** Pass the backprojection filter to the conjugate gradient operator and to the back projection filter generating the B of AX=B */
    void ConfigureBackProjection (int _arg);

    /** Pass the geometry to all filters needing it */
    void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

    /** Increase the value of Beta at each iteration */
    void SetBetaForCurrentIteration(int iter);

    itkSetMacro(alpha, float)
    itkGetMacro(alpha, float)

    itkSetMacro(beta, float)
    itkGetMacro(beta, float)

    itkSetMacro(AL_iterations, float)
    itkGetMacro(AL_iterations, float)

    itkSetMacro(CG_iterations, float)
    itkGetMacro(CG_iterations, float)

    itkSetMacro(MeasureExecutionTimes, bool)
    itkGetMacro(MeasureExecutionTimes, bool)

protected:
    ADMMTotalVariationConeBeamReconstructionFilter();
    ~ADMMTotalVariationConeBeamReconstructionFilter(){}

//    typename TOutputImage::ConstPointer GetInputVolume();
//    typename TOutputImage::Pointer GetInputProjectionStack();

    /** Does the real work. */
    virtual void GenerateData();

    /** Member pointers to the filters used internally (for convenience)*/
    typename SubtractGradientsFilterType::Pointer                               m_SubtractFilter1, m_SubtractFilter2;
    typename MultiplyVolumeFilterType::Pointer                                  m_MultiplyFilter;
    typename MultiplyVolumeFilterType::Pointer                                  m_ZeroMultiplyVolumeFilter;
    typename MultiplyGradientFilterType::Pointer                                m_ZeroMultiplyGradientFilter;
    typename ImageGradientFilterType::Pointer                                   m_GradientFilter1, m_GradientFilter2;
    typename AddVolumeFilterType::Pointer                                       m_AddVolumeFilter;
    typename AddGradientsFilterType::Pointer                                    m_AddGradientsFilter;
    typename ImageDivergenceFilterType::Pointer                                 m_DivergenceFilter;
    typename ConjugateGradientFilterType::Pointer                               m_ConjugateGradientFilter;
    typename SoftThresholdTVFilterType::Pointer                                 m_SoftThresholdFilter;
//    typename ConjugateGradientOperator< TOutputImage>::Pointer                  m_CGOperator; //Mother class
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
    ADMMTotalVariationConeBeamReconstructionFilter(const Self &); //purposely not implemented
    void operator=(const Self &);  //purposely not implemented

    float m_alpha, m_beta;
    int m_AL_iterations, m_CG_iterations;
    bool m_MeasureExecutionTimes;

};
} //namespace ITK


#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkADMMTotalVariationConeBeamReconstructionFilter.txx"
#endif

#endif
