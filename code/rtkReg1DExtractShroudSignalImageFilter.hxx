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

#ifndef rtkReg1DExtractShroudSignalImageFilter_hxx
#define rtkReg1DExtractShroudSignalImageFilter_hxx

#include <itkExtractImageFilter.h>
#include <itkTranslationTransform.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkMeanSquaresImageToImageMetric.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkImageRegistrationMethod.h>
#include <itkImageDuplicator.h>

namespace rtk
{

template<class TInputPixel, class TOutputPixel>
Reg1DExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::Reg1DExtractShroudSignalImageFilter()
{
}

template<class TInputPixel, class TOutputPixel>
void
Reg1DExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::GenerateInputRequestedRegion()
{
  typename Superclass::InputImagePointer  inputPtr =
    const_cast< TInputImage * >( this->GetInput() );
  if ( !inputPtr )
    {
    return;
    }
  inputPtr->SetRequestedRegion(inputPtr->GetLargestPossibleRegion());
}

template<class TInputPixel, class TOutputPixel>
void
Reg1DExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::GenerateOutputInformation()
{
  // get pointers to the input and output
  typename Superclass::InputImageConstPointer inputPtr  = this->GetInput();
  typename Superclass::OutputImagePointer     outputPtr = this->GetOutput();

  if ( !outputPtr || !inputPtr)
  {
    return;
  }
  typename TOutputImage::RegionType outRegion;
  typename TOutputImage::RegionType::SizeType outSize;
  typename TOutputImage::RegionType::IndexType outIdx;
  outSize[0] = this->GetInput()->GetLargestPossibleRegion().GetSize()[1];
  outIdx[0] = this->GetInput()->GetLargestPossibleRegion().GetIndex()[1];
  outRegion.SetSize(outSize);
  outRegion.SetIndex(outIdx);

  const typename TInputImage::SpacingType &
    inputSpacing = inputPtr->GetSpacing();
  typename TOutputImage::SpacingType outputSpacing;
  outputSpacing[0] = inputSpacing[1];
  outputPtr->SetSpacing(outputSpacing);

  typename TOutputImage::DirectionType outputDirection;
  outputDirection[0][0] = 1;
  outputPtr->SetDirection(outputDirection);

  const typename TInputImage::PointType &
    inputOrigin = inputPtr->GetOrigin();
  typename TOutputImage::PointType outputOrigin;
  outputOrigin[0] = inputOrigin[1];
  outputPtr->SetOrigin(outputOrigin);

  outputPtr->SetLargestPossibleRegion(outRegion);
}

template<class TInputPixel, class TOutputPixel>
TOutputPixel
Reg1DExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::register1D(RegisterImageType* f, RegisterImageType* m)
{
  typedef itk::TranslationTransform<TOutputPixel, 1>                                TransformType;
  typedef itk::RegularStepGradientDescentOptimizer                                  OptimizerType;
  typedef itk::MeanSquaresImageToImageMetric<RegisterImageType, RegisterImageType>  MetricType;
  typedef itk::LinearInterpolateImageFunction<RegisterImageType, TOutputPixel>      InterpolatorType;
  typedef itk::ImageRegistrationMethod<RegisterImageType, RegisterImageType>        RegistrationType;

  typename MetricType::Pointer metric = MetricType::New();
  typename TransformType::Pointer transform = TransformType::New();
  typename OptimizerType::Pointer optimizer = OptimizerType::New();
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();
  typename RegistrationType::Pointer registration = RegistrationType::New();

  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetInterpolator(interpolator);

  registration->SetFixedImage(f);
  registration->SetMovingImage(m);
  registration->SetFixedImageRegion(f->GetLargestPossibleRegion());

  typedef typename RegistrationType::ParametersType ParametersType;
  ParametersType initialParameters(transform->GetNumberOfParameters());
  // Initial offset along X
  initialParameters[0] = itk::NumericTraits<TOutputPixel>::Zero;

  registration->SetInitialTransformParameters(initialParameters);
  optimizer->SetMaximumStepLength(1.00);
  optimizer->SetMinimumStepLength(0.1);

  // Set a stopping criterion
  optimizer->SetNumberOfIterations(1000);

  registration->Update();

  return registration->GetLastTransformParameters()[0];
}

template<class TInputPixel, class TOutputPixel>
void
Reg1DExtractShroudSignalImageFilter<TInputPixel, TOutputPixel>
::GenerateData()
{
  this->AllocateOutputs();

  typedef itk::ExtractImageFilter<TInputImage, RegisterImageType>  ExtractFilterType;
  typedef itk::ImageDuplicator<RegisterImageType>                  DuplicatorType;

  typename TInputImage::ConstPointer input = this->GetInput();
  typename TInputImage::RegionType inputRegion = input->GetLargestPossibleRegion();
  typename TInputImage::SizeType inputSize = inputRegion.GetSize();

  typename ExtractFilterType::Pointer extractor = ExtractFilterType::New();
  extractor->SetInput(input);

  typename TInputImage::RegionType extractRegion;
  typename TInputImage::SizeType extractSize = inputRegion.GetSize();
  typename TInputImage::IndexType extractIdx = inputRegion.GetIndex();

  extractSize[1] = 0;
  extractIdx[1] = 0;
  extractRegion.SetSize(extractSize);
  extractRegion.SetIndex(extractIdx);
  extractor->SetExtractionRegion(extractRegion);
  extractor->SetDirectionCollapseToIdentity();
  extractor->Update();
  typename DuplicatorType::Pointer duplicator = DuplicatorType::New();
  duplicator->SetInputImage(extractor->GetOutput());
  duplicator->Update();
  typename RegisterImageType::Pointer prev = duplicator->GetOutput();
  TOutputPixel pos = itk::NumericTraits<TOutputPixel>::Zero;

  typename Superclass::OutputImagePointer output = this->GetOutput();
  output->Allocate();
  typename TOutputImage::RegionType::IndexType outputIdx;
  outputIdx[0] = 0;
  (*output)[outputIdx] = pos;
  for (unsigned int i = 1; i < inputSize[1]; ++i)
  {
    extractIdx[1] = i;
    extractRegion.SetSize(extractSize);
    extractRegion.SetIndex(extractIdx);
    extractor->SetExtractionRegion(extractRegion);
    extractor->SetDirectionCollapseToIdentity();
    extractor->Update();
    pos -= register1D(prev, extractor->GetOutput());
    outputIdx[0] = i;
    (*output)[outputIdx] = pos;
    duplicator->SetInputImage(extractor->GetOutput());
    duplicator->Update();
    prev = duplicator->GetOutput();
  }

}

} // end of namespace rtk
#endif
