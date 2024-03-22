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

#ifndef rtkPILineImageFilter_hxx
#define rtkPILineImageFilter_hxx

#include "math.h"

#include <rtkPILineImageFilter.h>

#include <rtkHomogeneousMatrix.h>

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkPixelTraits.h>

#include <itkPowellOptimizer.h>

namespace rtk
{

int POWELL_CALLS_TO_GET_VALUE = 0;

class PowellPILineCostFunction : public itk::SingleValuedCostFunction
{
public:
  using Self = PowellPILineCostFunction;
  using Superclass = itk::SingleValuedCostFunction;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;
  itkNewMacro(Self);
  itkTypeMacro(PowellPILineCostFunction, SingleValuedCostFunction);

  /** Get and Set macro*/
  itkGetMacro(DistanceToOrigin, double);
  itkSetMacro(DistanceToOrigin, double);

  itkGetMacro(Gamma, double);
  itkSetMacro(Gamma, double);

  itkGetMacro(HelixPitch, double);
  itkSetMacro(HelixPitch, double);

  itkGetMacro(HelixRadius, double);
  itkSetMacro(HelixRadius, double);

  itkGetMacro(AxialPosition, double);
  itkSetMacro(AxialPosition, double);

  enum
  {
    SpaceDimension = 1
  };

  using ParametersType = Superclass::ParametersType;
  using DerivativeType = Superclass::DerivativeType;
  using MeasureType = Superclass::MeasureType;

protected:
  PowellPILineCostFunction();


  void
  GetDerivative(const ParametersType &, DerivativeType &) const override
  {}

  MeasureType
  GetValue(const ParametersType & parameters) const override
  {
    ++POWELL_CALLS_TO_GET_VALUE;

    double s = parameters[0];

    double tmp1 = m_HelixRadius - m_DistanceToOrigin * cos(m_Gamma - s);
    double alpha = atan2(m_DistanceToOrigin * sin(m_Gamma - s), tmp1);
    double tmp3 = (1 + (pow(m_DistanceToOrigin, 2) - pow(m_HelixRadius, 2)) / (2 * m_HelixRadius * tmp1));

    MeasureType measure = pow(m_AxialPosition - m_HelixPitch * ((M_PI - 2 * alpha) * tmp3 + s), 2);

    return measure;
  }

  unsigned int
  GetNumberOfParameters() const override
  {
    return SpaceDimension;
  }

private:
  double m_DistanceToOrigin;
  double m_Gamma;
  double m_HelixPitch;
  double m_HelixRadius;
  double m_AxialPosition;
};

PowellPILineCostFunction::PowellPILineCostFunction() {}

template <class TInputImage, class TOutputImage>
void
PILineImageFilter<TInputImage, TOutputImage>::VerifyPreconditions() ITKv5_CONST
{
  this->Superclass::VerifyPreconditions();

  if (this->m_Geometry.IsNull() || !this->m_Geometry->GetTheGeometryIsVerified())
    itkExceptionMacro(<< "Geometry has not been set or not been verified");
}


template <class TInputImage, class TOutputImage>
void
PILineImageFilter<TInputImage, TOutputImage>::DynamicThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread)
{
  const unsigned int Dimension = TInputImage::ImageDimension;

  // Iterators
  OutputIteratorType itOut(this->GetOutput(), outputRegionForThread);
  InputIteratorType  itIn(this->GetInput(), outputRegionForThread);

  typename InputImageType::SpacingType spacing = this->GetInput()->GetSpacing();
  typename InputImageType::PointType   origin = this->GetInput()->GetOrigin();

  for (itOut.GoToBegin(), itIn.GoToBegin(); !itOut.IsAtEnd(); ++itOut, ++itIn)
  {
    OutputImageIndexType index = itOut.GetIndex();
    double               x = origin[0] + spacing[0] * index[0];
    double               y = origin[1] + spacing[1] * index[1];
    double               z = origin[2] + spacing[2] * index[2];

    double r = sqrt(pow(x, 2) + pow(z, 2));
    double gamma = atan2(x, z);
    double h = this->GetGeometry()->GetHelixPitch() / (2 * M_PI);
    double R = this->GetGeometry()->GetHelixRadius();

    using OptimizerType = itk::PowellOptimizer;

    // Declaration of an itkOptimizer
    OptimizerType::Pointer itkOptimizer = OptimizerType::New();


    // Declaration of the CostFunction
    PowellPILineCostFunction::Pointer cost_fun = PowellPILineCostFunction::New();
    cost_fun->SetDistanceToOrigin(r);
    cost_fun->SetGamma(gamma);
    cost_fun->SetHelixPitch(h);
    cost_fun->SetHelixRadius(R);
    cost_fun->SetAxialPosition(y);


    itkOptimizer->SetCostFunction(cost_fun);


    using ParametersType = PowellPILineCostFunction::ParametersType;

    // We start at 0
    ParametersType initialPosition(1); // 1D minimization

    initialPosition[0] = 0.;

    itkOptimizer->SetMaximize(false);
    itkOptimizer->SetStepLength(1);
    itkOptimizer->SetStepTolerance(0.001);
    itkOptimizer->SetValueTolerance(0.01);
    itkOptimizer->SetMaximumIteration(1000);

    itkOptimizer->SetInitialPosition(initialPosition);

    try
    {
      itkOptimizer->StartOptimization();
    }
    catch (const itk::ExceptionObject & e)
    {
      std::cout << "Exception thrown ! " << std::endl;
      std::cout << "An error occurred during Optimization" << std::endl;
      std::cout << "Location    = " << e.GetLocation() << std::endl;
      std::cout << "Description = " << e.GetDescription() << std::endl;
      // return EXIT_FAILURE;
    }

    double finalPosition = itkOptimizer->GetCurrentPosition()[0];

    // We set the two bounds of the PI Line
    using VectorType = itk::Vector<float, 2>;
    VectorType vector;
    // Lower bound
    vector[0] = finalPosition;
    // Upper bound
    double tmp = atan2(r * sin(gamma - finalPosition), R - r * cos(gamma - finalPosition));
    vector[1] = finalPosition + M_PI - 2 * tmp;

    itOut.Set(vector);
  }
}

} // end namespace rtk

#endif
