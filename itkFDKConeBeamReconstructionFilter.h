#ifndef __itkFDKConeBeamReconstructionFilter_h
#define __itkFDKConeBeamReconstructionFilter_h

#include "itkFDKWeightProjectionFilter.h"
#include "itkFFTRampImageFilter.h"
#include "itkFDKBackProjectionImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkTimeProbe.h>

/** \class FDKConeBeamReconstructionFilter
 * \brief Implements Feldkamp, David and Kress cone-beam reconstruction
 *
 * FDKConeBeamReconstructionFilter is a mini-pipeline filter which combines
 * the different steps of the FDK cone-beam reconstruction filter:
 * - itk::FDKWeightProjectionFilter for 2D weighting of the projections,
 * - itk::FFTRampImageFilter for ramp filtering,
 * - itk::FDKBackProjectionImageFilter for backprojection.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage, class TFFTPrecision=double>
class ITK_EXPORT FDKConeBeamReconstructionFilter :
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FDKConeBeamReconstructionFilter               Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef itk::ExtractImageFilter< InputImageType, OutputImageType >                 ExtractFilterType;
  typedef itk::FDKWeightProjectionFilter< InputImageType, OutputImageType >          WeightFilterType;
  typedef itk::FFTRampImageFilter< OutputImageType, OutputImageType, TFFTPrecision > RampFilterType;
  typedef itk::FDKBackProjectionImageFilter< OutputImageType, OutputImageType >      BackProjectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FDKConeBeamReconstructionFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  virtual ThreeDCircularProjectionGeometry::Pointer GetGeometry();
  virtual void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

  /** Get pointer to the ramp filter used by the feldkamp reconstruction */
  typename RampFilterType::Pointer GetRampFilter() { return m_RampFilter; }

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of cone-beam projection images processed
      simultaneously. Default is 4. */
  itkGetMacro(ProjectionSubsetSize, unsigned int);
  itkSetMacro(ProjectionSubsetSize, unsigned int);

protected:
  FDKConeBeamReconstructionFilter();
  ~FDKConeBeamReconstructionFilter(){}

  virtual void GenerateInputRequestedRegion();

  void GenerateOutputInformation();

  void GenerateData();

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer m_ExtractFilter;
  typename WeightFilterType::Pointer m_WeightFilter;
  typename RampFilterType::Pointer m_RampFilter;
  typename BackProjectionFilterType::Pointer m_BackProjectionFilter;

private:
  //purposely not implemented
  FDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Number of projections processed at a time. */
  unsigned int m_ProjectionSubsetSize;

  /** Probes to time reconstruction */
  itk::TimeProbe m_PreFilterProbe;
  itk::TimeProbe m_FilterProbe;
  itk::TimeProbe m_BackProjectionProbe;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFDKConeBeamReconstructionFilter.txx"
#endif

#endif
