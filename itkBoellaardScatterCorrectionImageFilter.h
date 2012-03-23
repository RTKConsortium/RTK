#ifndef __itkBoellaardScatterCorrectionImageFilter_h
#define __itkBoellaardScatterCorrectionImageFilter_h

#include "itkInPlaceImageFilter.h"
#include "rtkConfiguration.h"

/** \class BoellaardScatterCorrectionImageFilter
 *
 * TODO
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT BoellaardScatterCorrectionImageFilter :
  public InPlaceImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef BoellaardScatterCorrectionImageFilter         Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                     InputImageType;
  typedef TOutputImage                                    OutputImageType;
  typedef typename OutputImageType::RegionType            OutputImageRegionType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(BoellaardScatterCorrectionImageFilter, ImageToImageFilter);

  /** Get / Set the air threshold on projection images */
  itkGetMacro(AirThreshold, double);
  itkSetMacro(AirThreshold, double);

  /** Get / Set the scatter-to-primary ratio on projection images */
  itkGetMacro(ScatterToPrimaryRatio, double);
  itkSetMacro(ScatterToPrimaryRatio, double);

  /** Get / Set the non-negativity constraint threshold */
  itkGetMacro(NonNegativityConstraintThreshold, double);
  itkSetMacro(NonNegativityConstraintThreshold, double);

protected:
  BoellaardScatterCorrectionImageFilter();
  ~BoellaardScatterCorrectionImageFilter(){}

  /** Requires full projection images to estimate scatter */
  virtual void EnlargeOutputRequestedRegion(DataObject *itkNotUsed(output));
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, ThreadIdType threadId );

  /** Split the output's RequestedRegion into "num" pieces, returning
   * region "i" as "splitRegion". Reimplemented from ImageSource to ensure
   * that each thread covers entire projections. */
  virtual int SplitRequestedRegion(int i, int num, OutputImageRegionType& splitRegion);

private:
  BoellaardScatterCorrectionImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);               //purposely not implemented

  /** Air threshold on projection images. */
  double m_AirThreshold;

  /** Scatter to primary ratio */
  double m_ScatterToPrimaryRatio;

  /** Non-negativity constraint threshold */
  double m_NonNegativityConstraintThreshold;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkBoellaardScatterCorrectionImageFilter.txx"
#endif

#endif
