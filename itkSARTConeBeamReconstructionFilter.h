#ifndef __itkSARTConeBeamReconstructionFilter_h
#define __itkSARTConeBeamReconstructionFilter_h

#include "itkBackProjectionImageFilter.h"
#include "itkForwardProjectionImageFilter.h"

#include <itkExtractImageFilter.h>
#include <itkMultiplyByConstantImageFilter.h>
#include <itkSubtractImageFilter.h>
#include <itkTimeProbe.h>

/** \class SARTConeBeamReconstructionFilter
 * \brief Implements the Simultaneous Algebraic Reconstruction Technique [Andersen, 1984]
 *
 * SARTConeBeamReconstructionFilter is a mini-pipeline filter which combines
 * the different steps of the SART cone-beam reconstruction, mainly:
 * - ExtractFilterType to work on one projection at a time
 * - ForwardProjectionImageFilter,
 * - SubtractImageFilter,
 * - BackProjectionImageFilter.
 * The input stack of projections is processed piece by piece (the size is
 * controlled with ProjectionSubsetSize) via the use of itk::ExtractImageFilter
 * to extract sub-stacks.
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT SARTConeBeamReconstructionFilter :
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef SARTConeBeamReconstructionFilter              Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Typedefs of each subfilter of this composite filter */
  typedef itk::ExtractImageFilter< InputImageType, OutputImageType >                     ExtractFilterType;
  typedef itk::MultiplyByConstantImageFilter< OutputImageType, double, OutputImageType > MultiplyFilterType;
  typedef itk::ForwardProjectionImageFilter< OutputImageType, OutputImageType >          ForwardProjectionFilterType;
  typedef itk::SubtractImageFilter< OutputImageType, OutputImageType >                   SubtractFilterType;
  typedef itk::BackProjectionImageFilter< OutputImageType, OutputImageType >             BackProjectionFilterType;
  typedef typename BackProjectionFilterType::Pointer                                     BackProjectionFilterPointer;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(SARTConeBeamReconstructionFilter, ImageToImageFilter);

  /** Get / Set the object pointer to projection geometry */
  virtual ThreeDCircularProjectionGeometry::Pointer GetGeometry();
  virtual void SetGeometry(const ThreeDCircularProjectionGeometry::Pointer _arg);

  void PrintTiming(std::ostream& os) const;

  /** Get / Set the number of iterations. Default is 3. */
  itkGetMacro(NumberOfIterations, unsigned int);
  itkSetMacro(NumberOfIterations, unsigned int);

  /** Get / Set the convergence factor. Default is 0.3. */
  itkGetMacro(Lambda, double);
  itkSetMacro(Lambda, double);

  /** Set and init the backprojection filter. Default is voxel based backprojection. */
  virtual void SetBackProjectionFilter (const BackProjectionFilterPointer _arg);

protected:
  SARTConeBeamReconstructionFilter();
  ~SARTConeBeamReconstructionFilter(){}

  virtual void GenerateInputRequestedRegion();

  void GenerateOutputInformation();

  void GenerateData();

  /** The two inputs should not be in the same space so there is nothing
   * to verify. */
  virtual void VerifyInputInformation() {}

  /** Pointers to each subfilter of this composite filter */
  typename ExtractFilterType::Pointer m_ExtractFilter;
  typename MultiplyFilterType::Pointer m_ZeroMultiplyFilter;
  typename ForwardProjectionFilterType::Pointer m_ForwardProjectionFilter;
  typename SubtractFilterType::Pointer m_SubtractFilter;
  typename MultiplyFilterType::Pointer m_MultiplyFilter;
  typename BackProjectionFilterType::Pointer m_BackProjectionFilter;

private:
  //purposely not implemented
  SARTConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);

  /** Number of projections processed at a time. */
  unsigned int m_NumberOfIterations;

  /** Convergence factor according to Andersen's publications which relates
   * to the step size of the gradient descent. Default 0.3, Must be in (0,2). */
  double m_Lambda;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkSARTConeBeamReconstructionFilter.txx"
#endif

#endif
