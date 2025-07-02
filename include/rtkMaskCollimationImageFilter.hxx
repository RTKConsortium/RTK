/*=========================================================================
 *
 *  Copyright RTK Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

#ifndef rtkMaskCollimationImageFilter_hxx
#define rtkMaskCollimationImageFilter_hxx


#include <itkImageRegionIteratorWithIndex.h>

#include "rtkHomogeneousMatrix.h"
#include "rtkProjectionsRegionConstIteratorRayBased.h"
#include "rtkMacro.h"

namespace rtk
{

template <class TInputImage, class TOutputImage>
MaskCollimationImageFilter<TInputImage, TOutputImage>::MaskCollimationImageFilter()
  : m_Geometry(nullptr)
{}

template <class TInputImage, class TOutputImage>
void
MaskCollimationImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
MaskCollimationImageFilter<TInputImage, TOutputImage>::BeforeThreadedGenerateData()
{
  if (this->GetGeometry()->GetGantryAngles().size() != this->GetOutput()->GetLargestPossibleRegion().GetSize()[2])
    itkExceptionMacro(<< "Number of projections in the input stack and the geometry object differ.");
}

template <class TInputImage, class TOutputImage>
void
MaskCollimationImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  // Iterators on input and output
  using InputRegionIterator = ProjectionsRegionConstIteratorRayBased<TInputImage>;
  InputRegionIterator * itIn = nullptr;
  itIn = InputRegionIterator::New(this->GetInput(), outputRegionForThread, m_Geometry);
  itk::ImageRegionIteratorWithIndex<TOutputImage> itOut(this->GetOutput(), outputRegionForThread);

  // Go over each projection
  unsigned int npixperslice = 0;
  npixperslice = outputRegionForThread.GetSize(0) * outputRegionForThread.GetSize(1);
  for (typename OutputImageRegionType::SizeValueType iProj = outputRegionForThread.GetIndex(2);
       iProj < outputRegionForThread.GetIndex(2) + outputRegionForThread.GetSize(2);
       iProj++)
  {
    for (unsigned int pix = 0; pix < npixperslice; pix++, itIn->Next(), ++itOut)
    {
      using PointType = typename InputRegionIterator::PointType;
      PointType source = itIn->GetSourcePosition();
      double    sourceNorm = source.GetNorm();
      PointType pixel = itIn->GetPixelPosition();

      PointType pixelToSource(source - pixel);
      PointType sourceDir = source / sourceNorm;

      // Get the 3D position of the ray and the main plane
      // (at 0,0,0 and orthogonal to source to center)
      const double mag = sourceNorm / (pixelToSource * sourceDir);
      PointType    intersection = source - mag * pixelToSource;

      // The first coordinate on the plane is measured along the detector
      // orthogonal to the plane defined by the detector source direction
      // and the secon detector vector
      PointType v;
      v[0] = this->GetGeometry()->GetRotationMatrix(iProj)[m_RotationAxisIndex][0];
      v[1] = this->GetGeometry()->GetRotationMatrix(iProj)[m_RotationAxisIndex][1];
      v[2] = this->GetGeometry()->GetRotationMatrix(iProj)[m_RotationAxisIndex][2];
      PointType u = CrossProduct(v, sourceDir);
      u /= u.GetNorm();
      double ucoord = u * intersection;

      // The second coordinate is measured along the orthogonal
      v = CrossProduct(sourceDir, u);
      double vcoord = v * intersection;

      if (ucoord > -1. * this->GetGeometry()->GetCollimationUInf()[iProj] &&
          ucoord < this->GetGeometry()->GetCollimationUSup()[iProj] &&
          vcoord > -1. * this->GetGeometry()->GetCollimationVInf()[iProj] &&
          vcoord < this->GetGeometry()->GetCollimationVSup()[iProj])
      {
        itOut.Set(itIn->Get());
      }
      else
      {
        itOut.Set(0.);
      }
    }
  }

  delete itIn;
}

} // end namespace rtk

#endif
