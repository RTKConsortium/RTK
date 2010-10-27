#ifndef __itkFDKBackProjectionImageFilter_txx
#define __itkFDKBackProjectionImageFilter_txx

#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>

namespace itk
{

template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::BeforeThreadedGenerateData()
{
  UpdateAngularWeights();
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, 
                       int threadId )
{
  const unsigned int Dimension = TInputImage::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);

  // Create interpolator, could be any interpolation
  typedef itk::LinearInterpolateImageFunction< ProjectionImageType, double > InterpolatorType;
  typename InterpolatorType::Pointer interpolator = InterpolatorType::New();

  // Ramp factor is the correction for ramp filter which did not account for the divergence of the beam
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer());
  double rampFactor = geometry->GetSourceToDetectorDistance() / geometry->GetSourceToIsocenterDistance();
  rampFactor *= 0.5; // Factor 1/2 in eq 176, page 106, Kak & Slaney

  // Iterators on volume input and output
  typedef ImageRegionConstIterator<TInputImage> InputRegionIterator;
  InputRegionIterator itIn(this->GetInput(), outputRegionForThread);
  typedef ImageRegionIteratorWithIndex<TOutputImage> OutputRegionIterator;
  OutputRegionIterator itOut(this->GetOutput(), outputRegionForThread);

  // Rotation center (assumed to be at 0 yet)
  typename TInputImage::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Continuous index at which we interpolate
  ContinuousIndex<double, Dimension-1> pointProj;

  // Go over each projection
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = this->GetProjection(iProj, m_AngularWeights[iProj] * rampFactor);
    interpolator->SetInputImage(projection);

    // Index to index matrix normalized to have a correct backprojection weight (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj, projection);
    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    // Go over each voxel
    itIn.GoToBegin();
    itOut.GoToBegin();
    while(!itIn.IsAtEnd())
      {
      // Compute projection index
      for(unsigned int i=0; i<Dimension-1; i++)
        {
        pointProj[i] = matrix[i][Dimension];
        for(unsigned int j=0; j<Dimension; j++)
          pointProj[i] += matrix[i][j] * itOut.GetIndex()[j];
        }

      // Apply perspective
      double perspFactor = matrix[Dimension-1][Dimension];
      for(unsigned int j=0; j<Dimension; j++)
        perspFactor += matrix[Dimension-1][j] * itOut.GetIndex()[j];
      perspFactor = 1/perspFactor;
      for(unsigned int i=0; i<Dimension-1; i++)
        pointProj[i] = pointProj[i]*perspFactor;

      // Interpolate if in projection
      if( interpolator->IsInsideBuffer(pointProj) )
        {
        if (iProj)
          itOut.Set( itOut.Get() + perspFactor*perspFactor*interpolator->EvaluateAtContinuousIndex(pointProj) );
        else
          itOut.Set( itIn.Get() + perspFactor*perspFactor*interpolator->EvaluateAtContinuousIndex(pointProj) );
        }

      ++itIn;
      ++itOut;
      }
    }
}

template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::UpdateAngularWeights()
{
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer());
  unsigned int nProj = geometry->GetRotationAngles().size();
  m_AngularWeights.resize(nProj);

  // Special management of single or empty dataset
  const double degreesToRadians = vcl_atan(1.0) / 45.0;
  if(nProj==1)
    m_AngularWeights[0] = 0.5 * degreesToRadians * 360;
  if(nProj<2)
    return;
    
  // Otherwise we sort the angles in a multimap
  std::multimap<double,unsigned int> angles;
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    double angle = geometry->GetRotationAngles()[iProj];
    angle = angle-360*floor(angle/360); // between -360 and 360
    if(angle<0) angle += 360;           // between 0    and 360
    angles.insert(std::pair<double, unsigned int>(angle, iProj));
    }

  // We then go over the sorted angles and deduce the angular weight
  std::multimap<double,unsigned int>::const_iterator prev = angles.begin(),
                                                     curr = angles.begin(),
                                                     next = angles.begin();
  next++;

  //First projection wraps the angle of the last one
  m_AngularWeights[curr->second] = 0.5 * degreesToRadians * ( next->first - angles.rbegin()->first + 360 );
  curr++; next++;

  //Rest of the angles
  while(next!=angles.end())
  {
    m_AngularWeights[curr->second] = 0.5 * degreesToRadians * ( next->first - prev->first );
    prev++; curr++; next++;
  }

  //Last projection wraps the angle of the first one
  m_AngularWeights[curr->second] = 0.5 * degreesToRadians * ( angles.begin()->first + 360 - prev->first );
}

} // end namespace itk


#endif
