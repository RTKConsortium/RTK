#ifndef __itkElektaSynergyRawToAttenuationImageFilter_h
#define __itkElektaSynergyRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCropImageFilter.h>
#include "itkElektaSynergyLutImageFilter.h"

/** \class RawToAttenuationImageFilter
 * \brief Convert raw Elekta Synergy data to attenuation images
 *
 * This composite filter composes the operations required to convert
 * a raw image from the Elekta Synergy cone-beam CT scanner to 
 * attenuation images usable in standard reconstruction algorithms,*
 * e.g. Feldkamp algorithm.
 *
 * \author Simon Rit
 */
namespace itk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT ElektaSynergyRawToAttenuationImageFilter:
  public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawToAttenuationImageFilter      Self;
  typedef ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef SmartPointer<Self>                            Pointer;
  typedef SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage                                   InputImageType;
  typedef TOutputImage                                  OutputImageType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawToAttenuationImageFilter, ImageToImageFilter);

protected:
  ElektaSynergyRawToAttenuationImageFilter();
  ~ElektaSynergyRawToAttenuationImageFilter(){}

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  void GenerateOutputInformation();

  /** Single-threaded version of GenerateData.  This filter delegates
   * to other filters. */
  void GenerateData();

private:
  ElektaSynergyRawToAttenuationImageFilter(const Self&); //purposely not implemented
  void operator=(const Self&); //purposely not implemented

  typedef itk::ElektaSynergyLutImageFilter<InputImageType, OutputImageType> LutFilterType;
  typedef itk::CropImageFilter<OutputImageType, OutputImageType> CropFilterType;

  typename LutFilterType::Pointer m_LutFilter;
  typename CropFilterType::Pointer m_CropFilter;
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkElektaSynergyRawToAttenuationImageFilter.txx"
#endif

#endif
