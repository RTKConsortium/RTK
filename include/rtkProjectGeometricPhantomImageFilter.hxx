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

#ifndef rtkProjectGeometricPhantomImageFilter_hxx
#define rtkProjectGeometricPhantomImageFilter_hxx

#include "rtkForbildPhantomFileReader.h"
#include "rtkRayConvexIntersectionImageFilter.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

namespace rtk
{
template <class TInputImage, class TOutputImage>
ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>::ProjectGeometricPhantomImageFilter()
{
  m_RotationMatrix.SetIdentity();
}

template <class TInputImage, class TOutputImage>
void
ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() const
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull())
    itkExceptionMacro(<< "Geometry has not been set.");
}

template <class TInputImage, class TOutputImage>
void
ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  // Reading figure config file
  if (!m_ConfigFile.empty())
  {
    auto reader = rtk::ForbildPhantomFileReader::New();
    reader->SetFilename(m_ConfigFile);
    reader->GenerateOutputInformation();
    this->m_GeometricPhantom = reader->GetGeometricPhantom();
  }

  // Check that it's not empty
  const GeometricPhantom::ConvexShapeVector & cov = m_GeometricPhantom->GetConvexShapes();
  if (cov.empty())
    itkExceptionMacro(<< "Empty phantom");

  // Create one add filter per convex object
  std::vector<typename itk::ImageSource<TOutputImage>::Pointer> projectors;
  for (const auto & convexShape : cov)
  {
    ConvexShape::Pointer co = convexShape->Clone();
    co->Rotate(m_RotationMatrix);
    co->Translate(m_OriginOffset);
    co->Rescale(m_PhantomScale);
    for (size_t i = 0; i < m_PlaneDirections.size(); i++)
      co->AddClipPlane(m_PlaneDirections[i], m_PlanePositions[i]);

    if (!projectors.empty())
    {
      using RCOIType = RayConvexIntersectionImageFilter<TOutputImage, TOutputImage>;
      auto rcoi = RCOIType::New();
      rcoi->SetInput(projectors.back()->GetOutput());
      rcoi->SetGeometry(this->GetGeometry());
      rcoi->SetConvexShape(co);
      projectors.push_back(rcoi.GetPointer());
    }
    else
    {
      using RCOIType = RayConvexIntersectionImageFilter<TInputImage, TOutputImage>;
      auto rcoi = RCOIType::New();
      rcoi->SetInput(this->GetInput());
      rcoi->SetGeometry(this->GetGeometry());
      rcoi->SetConvexShape(co);
      projectors.push_back(rcoi.GetPointer());
    }
  }

  projectors.back()->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());
  projectors.back()->Update();
  this->GraftOutput(projectors.back()->GetOutput());
}

template <class TInputImage, class TOutputImage>
void
ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>::AddClipPlane(const VectorType & dir,
                                                                            const ScalarType & pos)
{
  for (size_t i = 0; i < m_PlaneDirections.size(); i++)
  {
    if (dir == m_PlaneDirections[i] && pos == m_PlanePositions[i])
      return;
  }
  m_PlaneDirections.push_back(dir);
  m_PlanePositions.push_back(pos);
}

template <class TInputImage, class TOutputImage>
void
ProjectGeometricPhantomImageFilter<TInputImage, TOutputImage>::SetClipPlanes(const std::vector<VectorType> & dir,
                                                                             const std::vector<ScalarType> & pos)
{
  m_PlaneDirections = dir;
  m_PlanePositions = pos;
}

} // end namespace rtk

#endif
