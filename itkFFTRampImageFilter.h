#ifndef __itkFFTRampImageFilter_h
#define __itkFFTRampImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkConceptChecking.h>
#include "rtkConfiguration.h"

/** \class FFTRampImageFilter
 * \brief Implements the ramp image filter of the filtered backprojection algorithm.
 *
 * The filter code is based on FFTConvolutionImageFilter by Gaetan Lehmann
 * (see http://hdl.handle.net/10380/3154)
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage, class TFFTPrecision=double>
class ITK_EXPORT FFTRampImageFilter :
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef FFTRampImageFilter Self;

  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;

  typedef SmartPointer<Self>       Pointer;
  typedef SmartPointer<const Self> ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                         InputImageType;
  typedef TOutputImage                        OutputImageType;
  typedef TFFTPrecision                       FFTPrecisionType;
  typedef typename InputImageType::Pointer    InputImagePointer;
  typedef typename InputImageType::PixelType  InputImagePixelType;
  typedef typename OutputImageType::Pointer   OutputImagePointer;
  typedef typename OutputImageType::PixelType OutputImagePixelType;
  typedef typename itk::Image<TFFTPrecision,
                              TInputImage::ImageDimension > FFTInputImageType;
  typedef typename FFTInputImageType::Pointer FFTInputImagePointer;
  typedef typename itk::Image<std::complex<TFFTPrecision>,
                              TInputImage::ImageDimension > FFTOutputImageType;
  typedef typename FFTOutputImageType::Pointer FFTOutputImagePointer;
  typedef typename InputImageType::RegionType  RegionType;
  typedef typename InputImageType::IndexType   IndexType;
  typedef typename InputImageType::SizeType    SizeType;

  /** ImageDimension constants */
  itkStaticConstMacro(InputImageDimension, unsigned int,
                      TInputImage::ImageDimension);
  itkStaticConstMacro(OutputImageDimension, unsigned int,
                      TOutputImage::ImageDimension);
  itkStaticConstMacro(ImageDimension, unsigned int,
                      TOutputImage::ImageDimension);

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(FFTRampImageFilter, ImageToImageFilter);

  /**
   * Set/Get the greatest prime factor allowed on the size of the padded image.
   * The filter increase the size of the image to reach a size with the greatest
   * prime factor smaller or equal to the specified value. The default value is
   * 13, which is the greatest prime number for which the FFT are precomputed
   * in FFTW, and thus gives very good performance.
   * A greatest prime factor of 2 produce a size which is a power of 2, and thus
   * is suitable for vnl base fft filters.
   * A greatest prime factor of 1 or less - typically 0 - disable the extra padding.
   *
   * Warning: this parameter is not used (and useful) only when ITK is built with
   * FFTW support.
   */
  itkGetConstMacro(GreatestPrimeFactor, int);
  itkSetMacro(GreatestPrimeFactor, int);

#ifdef ITK_USE_CONCEPT_CHECKING
  /** Begin concept checking */
  itkConceptMacro(InputHasPixelTraitsCheck,
                  (Concept::HasPixelTraits<InputImagePixelType>) );
  itkConceptMacro(InputHasNumericTraitsCheck,
                  (Concept::HasNumericTraits<InputImagePixelType>) );
  /** End concept checking */
#endif

  /** Set/Get the percentage of the image widthfeathered with data to correct
    * for truncation.
    */
  itkGetConstMacro(TruncationCorrection, double);
  itkSetMacro(TruncationCorrection, double);

  /** Set/Get the Hann window frequency. 0 (default) disables it */
  itkGetConstMacro(HannCutFrequency, double);
  itkSetMacro(HannCutFrequency, double);

  /** Set/Get the Cosine Cut window frequency. 0 (default) disables it */
  itkGetConstMacro(CosineCutFrequency, double);
  itkSetMacro(CosineCutFrequency, double);

  /** Set/Get the Hamming window frequency. 0 (default) disables it */
  itkGetConstMacro(HammingFrequancy, double);
  itkSetMacro(HammingFrequancy, double);
protected:
  FFTRampImageFilter();
  ~FFTRampImageFilter(){}

  void GenerateInputRequestedRegion();

  virtual void BeforeThreadedGenerateData();

  virtual void ThreadedGenerateData( const RegionType& outputRegionForThread, ThreadIdType threadId );

  /** Pad the inputRegion region of the input image and returns a pointer to the new padded image.
    * Padding includes a correction for truncation [Ohnesorge, Med Phys, 2000].
    * centralRegion is the region of the returned image which corresponds to inputRegion.
    */
  FFTInputImagePointer PadInputImageRegion(const RegionType &inputRegion);

  void PrintSelf(std::ostream& os, Indent indent) const;

  bool IsPrime( int n ) const;

  int GreatestPrimeFactor( int n ) const;

  /** Creates and return a pointer to one line of the ramp kernel in Fourier space.
   *  Used in generate data functions. */
  FFTOutputImagePointer GetFFTRampKernel(const int width);

  /** Pre compute weights for truncation correction in a lookup table. The index
    * is the distance to the original image border.
    * Careful: the function is not thread safe but it does nothing if the weights have
    * already been computed.
    */
  void UpdateTruncationMirrorWeights();

private:
  FFTRampImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&);     //purposely not implemented

  /** Percentage of the image width which is feathered with data to correct for truncation.
    * 0 means no correction.
    */
  double m_TruncationCorrection;
  typename std::vector<TFFTPrecision> m_TruncationMirrorWeights;
  int GetTruncationCorrectionExtent();

  /**
   * Greatest prime factor of the FFT input.
   */
  int m_GreatestPrimeFactor;

  /**
   * Cut frequency of Hann, Cosine and Hamming windows. The first one which is
   * non-zero is used.
   */
  double m_HannCutFrequency;
  double m_CosineCutFrequency;
  double m_HammingFrequancy;

}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkFFTRampImageFilter.txx"
#endif

#endif
