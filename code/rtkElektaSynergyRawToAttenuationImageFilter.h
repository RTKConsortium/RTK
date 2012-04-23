#ifndef __rtkElektaSynergyRawToAttenuationImageFilter_h
#define __rtkElektaSynergyRawToAttenuationImageFilter_h

#include <itkImageToImageFilter.h>
#include <itkCropImageFilter.h>

#include "rtkElektaSynergyLutImageFilter.h"
#include "rtkBoellaardScatterCorrectionImageFilter.h"

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
namespace rtk
{

template<class TInputImage, class TOutputImage=TInputImage>
class ITK_EXPORT ElektaSynergyRawToAttenuationImageFilter :
  public itk::ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  /** Standard class typedefs. */
  typedef ElektaSynergyRawToAttenuationImageFilter           Self;
  typedef itk::ImageToImageFilter<TInputImage, TOutputImage> Superclass;
  typedef itk::SmartPointer<Self>                            Pointer;
  typedef itk::SmartPointer<const Self>                      ConstPointer;

  /** Some convenient typedefs. */
  typedef TInputImage  InputImageType;
  typedef TOutputImage OutputImageType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(ElektaSynergyRawToAttenuationImageFilter, itk::ImageToImageFilter);

protected:
  ElektaSynergyRawToAttenuationImageFilter();
  ~ElektaSynergyRawToAttenuationImageFilter(){
  }

  /** Apply changes to the input image requested region. */
  virtual void GenerateInputRequestedRegion();

  void GenerateOutputInformation();

  /** Single-threaded version of GenerateData.  This filter delegates
   * to other filters. */
  void GenerateData();

private:
  //purposely not implemented
  ElektaSynergyRawToAttenuationImageFilter(const Self&);
  void operator=(const Self&);

  typedef itk::CropImageFilter<InputImageType, InputImageType>                       CropFilterType;
  typedef rtk::BoellaardScatterCorrectionImageFilter<InputImageType, InputImageType> ScatterFilterType;
  typedef rtk::ElektaSynergyLutImageFilter<InputImageType, OutputImageType>          LutFilterType;

  typename LutFilterType::Pointer m_LutFilter;
  typename CropFilterType::Pointer m_CropFilter;
  typename ScatterFilterType::Pointer m_ScatterFilter;
}; // end of class

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkElektaSynergyRawToAttenuationImageFilter.txx"
#endif

#endif
