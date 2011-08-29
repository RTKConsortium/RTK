#ifndef __itkOpenCLFDKConeBeamReconstructionFilter_h
#define __itkOpenCLFDKConeBeamReconstructionFilter_h

#include "itkFDKConeBeamReconstructionFilter.h"
#include "itkOpenCLFDKBackProjectionImageFilter.h"

/** \class OpenCLFDKConeBeamReconstructionFilter
 * \brief Implements Feldkamp, David and Kress cone-beam reconstruction using OpenCL
 *
 * Replaces ramp and backprojection in FDKConeBeamReconstructionFilter with
 * - OpenCLFDKBackProjectionImageFilter.
 * Also take care to create the reconstructed volume on the GPU at the beginning and
 * transfers it at the end.
 *
 * \author Simon Rit
 */
namespace itk
{

class ITK_EXPORT OpenCLFDKConeBeamReconstructionFilter :
  public FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float >
{
public:
  /** Standard class typedefs. */
  typedef OpenCLFDKConeBeamReconstructionFilter                                              Self;
  typedef FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float > Superclass;
  typedef SmartPointer<Self>                                                                 Pointer;
  typedef SmartPointer<const Self>                                                           ConstPointer;

  /** Typedefs of subfilters which have been implemented with OpenCL */
  typedef itk::OpenCLFDKBackProjectionImageFilter BackProjectionFilterType;

  /** Standard New method. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(OpenCLFDKConeBeamReconstructionFilter, ImageToImageFilter);

protected:
  OpenCLFDKConeBeamReconstructionFilter();
  ~OpenCLFDKConeBeamReconstructionFilter(){}

  void GenerateData();

private:
  //purposely not implemented
  OpenCLFDKConeBeamReconstructionFilter(const Self&);
  void operator=(const Self&);
}; // end of class

} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkOpenCLFDKConeBeamReconstructionFilter.txx"
#endif

#endif
