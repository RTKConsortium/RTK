/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef __rtkOpenCLFDKConeBeamReconstructionFilter_h
#define __rtkOpenCLFDKConeBeamReconstructionFilter_h

#include "rtkFDKConeBeamReconstructionFilter.h"
#include "rtkOpenCLFDKBackProjectionImageFilter.h"

/** \class OpenCLFDKConeBeamReconstructionFilter
 * \brief Implements Feldkamp, David and Kress cone-beam reconstruction using OpenCL
 *
 * Replaces ramp and backprojection in FDKConeBeamReconstructionFilter with
 * - OpenCLFDKBackProjectionImageFilter.
 * Also take care to create the reconstructed volume on the GPU at the beginning and
 * transfers it at the end.
 *
 * \author Simon Rit
 *
 * \ingroup FDKConeBeamReconstructionFilter
 */
namespace rtk
{

class ITK_EXPORT OpenCLFDKConeBeamReconstructionFilter :
  public FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float >
{
public:
  /** Standard class typedefs. */
  typedef OpenCLFDKConeBeamReconstructionFilter                                              Self;
  typedef FDKConeBeamReconstructionFilter< itk::Image<float,3>, itk::Image<float,3>, float > Superclass;
  typedef itk::SmartPointer<Self>                                                                 Pointer;
  typedef itk::SmartPointer<const Self>                                                           ConstPointer;

  /** Typedefs of subfilters which have been implemented with OpenCL */
  typedef rtk::OpenCLFDKBackProjectionImageFilter BackProjectionFilterType;

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

} // end namespace rtk

#ifndef ITK_MANUAL_INSTANTIATION
#include "rtkOpenCLFDKConeBeamReconstructionFilter.txx"
#endif

#endif
