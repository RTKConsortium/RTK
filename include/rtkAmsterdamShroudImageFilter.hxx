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

#ifndef rtkAmsterdamShroudImageFilter_hxx
#define rtkAmsterdamShroudImageFilter_hxx

#include <itkImageFileWriter.h>
#include <itkImageRegionIterator.h>
#include "rtkHomogeneousMatrix.h"

namespace rtk
{

template <class TInputImage>
AmsterdamShroudImageFilter<TInputImage>
::AmsterdamShroudImageFilter():
  m_UnsharpMaskSize(17),
  m_Geometry(ITK_NULLPTR),
  m_Corner1(0.),
  m_Corner2(0.)
{
  m_DerivativeFilter = DerivativeType::New();
  m_NegativeFilter = NegativeType::New();
  m_ThresholdFilter = ThresholdType::New();
  m_SumFilter = SumType::New();
  m_ConvolutionFilter = ConvolutionType::New();
  m_SubtractFilter = SubtractType::New();
  m_PermuteFilter = PermuteType::New();

  m_NegativeFilter->SetInput( m_DerivativeFilter->GetOutput() );
  m_ThresholdFilter->SetInput( m_NegativeFilter->GetOutput() );
  m_SumFilter->SetInput( m_ThresholdFilter->GetOutput() );
  m_ConvolutionFilter->SetInput( m_SumFilter->GetOutput() );
  m_SubtractFilter->SetInput1( m_SumFilter->GetOutput() );
  m_SubtractFilter->SetInput2( m_ConvolutionFilter->GetOutput() );
  m_PermuteFilter->SetInput( m_SubtractFilter->GetOutput() );

  m_DerivativeFilter->SetOrder(DerivativeType::FirstOrder);
  m_DerivativeFilter->SetDirection(1);
  m_DerivativeFilter->SetSigma(4);

  m_NegativeFilter->SetConstant(-1.0);
  m_NegativeFilter->InPlaceOn();

  m_ThresholdFilter->SetUpper(0.0);
  m_ThresholdFilter->SetOutsideValue(0.0);
  m_ThresholdFilter->InPlaceOn();

  m_SumFilter->SetProjectionDimension(0);

  // The permute filter is used to allow streaming because ITK splits the output image in the last direction
  typename PermuteType::PermuteOrderArrayType order;
  order[0] = 1;
  order[1] = 0;
  m_PermuteFilter->SetOrder(order);

  // Create default kernel
  UpdateUnsharpMaskKernel();
}

template <class TInputImage>
void
AmsterdamShroudImageFilter<TInputImage>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();

  if ( !outputPtr || !inputPtr)
    {
    return;
    }

  m_DerivativeFilter->SetInput( this->GetInput() );
  m_PermuteFilter->UpdateOutputInformation();
  outputPtr->CopyInformation( m_PermuteFilter->GetOutput() );
}

template <class TInputImage>
void
AmsterdamShroudImageFilter<TInputImage>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    {
    return;
    }
  m_DerivativeFilter->SetInput( this->GetInput() );
  m_PermuteFilter->GetOutput()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion() );
  m_PermuteFilter->GetOutput()->PropagateRequestedRegion();
}

template<class TInputImage>
void
AmsterdamShroudImageFilter<TInputImage>
::GenerateData()
{
  // If there is a geometry set, use it to crop the projections
  if( m_Geometry.GetPointer() )
    CropOutsideProjectedBox();

  unsigned int kernelWidth;
  kernelWidth = m_ConvolutionFilter->GetKernelImage()->GetLargestPossibleRegion().GetSize()[1];
  if(kernelWidth != m_UnsharpMaskSize)
    UpdateUnsharpMaskKernel();

  // If the convolution filter is not updated independently of the rest of the
  // pipeline, there is a bug in the streaming and it does not behave as
  // expected: the largest possible region is modified of the output of the
  // convolution filter is modified! I don't have an explanation for this but
  // this seems to fix the problem (SR).
  m_ConvolutionFilter->Update();

  m_PermuteFilter->Update();
  this->GraftOutput( m_PermuteFilter->GetOutput() );
}

template<class TInputImage>
void
AmsterdamShroudImageFilter<TInputImage>
::UpdateUnsharpMaskKernel()
{
  // Unsharp mask: difference between image and moving average
  // m_ConvolutionFilter computes the moving average
  typename TOutputImage::Pointer kernel = TOutputImage::New();
  typename TOutputImage::RegionType region;
  region.SetIndex(0, 0);
  region.SetIndex(1, (int)m_UnsharpMaskSize/-2);
  region.SetSize(0, 1);
  region.SetSize(1, m_UnsharpMaskSize);
  kernel->SetRegions(region);
  kernel->Allocate();
  kernel->FillBuffer(1./m_UnsharpMaskSize);
  m_ConvolutionFilter->SetKernelImage( kernel );
}

template<class TInputImage>
void
AmsterdamShroudImageFilter<TInputImage>
::CropOutsideProjectedBox()
{
  // The crop is performed after derivation for the sake of simplicity
  m_DerivativeFilter->Update();

  typename TInputImage::RegionType reg;
  reg = m_DerivativeFilter->GetOutput()->GetRequestedRegion();

  typedef typename itk::ImageRegionIterator<TInputImage> OutputIterator;
  OutputIterator it(m_DerivativeFilter->GetOutput(), reg);

  // Prepare the 8 corners of the box
  std::vector<GeometryType::HomogeneousVectorType> corners;
  for(unsigned int i=0; i<8; i++)
    {
    GeometryType::HomogeneousVectorType corner;
    if(i/4 == 0)
      corner[2] = m_Corner1[2];
    else
      corner[2] = m_Corner2[2];
    if((i/2)%2 == 0)
      corner[1] = m_Corner1[1];
    else
      corner[1] = m_Corner2[1];
    if(i%2 == 0)
      corner[0] = m_Corner1[0];
    else
      corner[0] = m_Corner2[0];
    corner[3] = 1.;
    corners.push_back(corner);
    }

  for(int iProj=reg.GetIndex(2);
          iProj<reg.GetIndex(2)+(int)reg.GetSize(2);
          iProj++)
    {
    // Project and keep the inferior and superior 2d corner
    itk::ContinuousIndex<double, 3> pCornerInf, pCornerSup;
    GeometryType::MatrixType matrix;
    matrix = m_Geometry->GetMatrices()[iProj].GetVnlMatrix();
    for(unsigned int ci=0; ci<8; ci++)
      {
      typename TInputImage::PointType pCorner(0.);
      vnl_vector< double > pCornerVnl = m_Geometry->GetMatrices()[iProj].GetVnlMatrix()* corners[ci].GetVnlVector();
      for(unsigned int i=0; i<2; i++)
        pCorner[i] = pCornerVnl[i] / pCornerVnl[2];
      itk::ContinuousIndex<double, 3> pCornerI;
      this->GetInput()->TransformPhysicalPointToContinuousIndex(pCorner, pCornerI);
      if(ci==0)
        {
        pCornerInf = pCornerI;
        pCornerSup = pCornerI;
        }
      else
        {
        for(int i=0; i<2; i++)
          {
          pCornerInf[i] = std::min(pCornerInf[i], pCornerI[i]);
          pCornerSup[i] = std::max(pCornerSup[i], pCornerI[i]);
          }
        }
      }

    // Set to 0 all pixels outside 2D projected box
    for(unsigned int j=0;
                     j<reg.GetSize(1);
                     j++)
      {
      for(unsigned int i=0;
                       i<reg.GetSize(0);
                       i++, ++it)
        {
        if(i<pCornerInf[0] || i>pCornerSup[0] ||
           j<pCornerInf[1] || j>pCornerSup[1])
          it.Set(0);
        }
      }
    }
}

} // end namespace rtk

#endif
