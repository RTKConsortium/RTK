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
  this->SetTranspose(false);
}

/**
 * GenerateData performs the accumulation
 */
template <class TInputImage, class TOutputImage>
void
FDKBackProjectionImageFilter<TInputImage,TOutputImage>
::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, int threadId )
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

/*
    // Optimized version
    if (abs(matrix[1][0])<1e-10 && abs(matrix[2][0])<1e-10)
      {
      OptimizedBackprojection( outputRegionForThread, matrix, projection);
      continue;
      }
*/
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
::OptimizedBackprojection(const OutputImageRegionType& region, const ProjectionMatrixType& matrix, const ProjectionImagePointer projection)
{
  typename ProjectionImageType::SizeType pSize = projection->GetLargestPossibleRegion().GetSize();

  // Continuous index at which we interpolate
  double u, v, w;
  int ui, vi;
  double du;

  for(int k=region.GetIndex(2); k<region.GetIndex(2)+(int)region.GetSize(2); k++)
    {
    for(int j=region.GetIndex(1); j<region.GetIndex(1)+(int)region.GetSize(1); j++)
      {
      int i = region.GetIndex(0);
      u = matrix[0][0] * i + matrix[0][1] * j + matrix[0][2] * k + matrix[0][3];
      v =                    matrix[1][1] * j + matrix[1][2] * k + matrix[1][3];
      w =                    matrix[2][1] * j + matrix[2][2] * k + matrix[2][3];

      //Apply perspective
      w = 1/w;
      u *= w;
      v *= w;        
      du = w * matrix[0][0];
      w *= w;

#define BILINEAR_BACKPROJECTION
#ifdef BILINEAR_BACKPROJECTION
      double u1, u2, v1, v2;
      vi = Math::Floor<double>(v);
      if(vi>=0 && vi<(int)projection->GetLargestPossibleRegion().GetSize(1)-1)
        {
        v1 = v-vi;
        v2 = 1.0-v1;
#else
      vi = Math::Round<double>(v);
      if(vi>=0 && vi<(int)projection->GetLargestPossibleRegion().GetSize(1))
        {
#endif

        typename TInputImage::PixelType * in = projection->GetBufferPointer() + vi * pSize[0];
        typename TOutputImage::PixelType * out = this->GetOutput()->GetBufferPointer() + i + region.GetSize(0) * (j + k * region.GetSize(1) );

        // Innermost loop
        for(i=0; i<(int)region.GetSize(0); i++, u+=du, out++)
          {
#ifdef BILINEAR_BACKPROJECTION
          ui = Math::Floor<double>(u);          
          if(ui>=0 && ui<(int)projection->GetLargestPossibleRegion().GetSize(0)-1)
            {
            u1 = u-ui;
            u2 = 1.0-u1;
            *out += w * (v2 * (u2 * *(in+ui)          + u1 * *(in+ui+1)) +
                         v1 * (u2 * *(in+ui+pSize[0]) + u1 * *(in+ui+pSize[0]+1)));
            }
#else
          ui = Math::Round<double>(u);          
          if(ui>=0 && ui<(int)projection->GetLargestPossibleRegion().GetSize(0))
            {
            *out += w * *(in+ui);
            }
#endif
          } //i
        }
      } //j
    } //k
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

  //Detect short scans, warn the user until implementation of adequate weighting
  for(unsigned int iProj=0; iProj<nProj; iProj++)
    {
    if( m_AngularWeights[iProj] > 10*degreesToRadians )
      {
      itkWarningMacro(<<"Short scan detected, current implementation does not account for it"<<std::endl);
      m_AngularWeights[iProj] = vnl_math_min(m_AngularWeights[(nProj+iProj-1)%nProj], m_AngularWeights[(iProj+1)%nProj]); 
      }
    }
}

} // end namespace itk


#endif
