namespace rtk
{

template < class TCoordRep, unsigned int VBoxDimension >
bool
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::Evaluate( const VectorType& rayDirection )
{
  // http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
  // BI <-> m_BoxMin
  // Bh <-> m_BoxMax
  // Ro <-> m_RayOrigin
  // Rd <-> rayDirection
  // Tnear <-> m_NearestDistance
  // Tfar <-> m_FarthestDistance
  m_RayDirection = rayDirection;
  m_NearestDistance = itk::NumericTraits< TCoordRep >::NonpositiveMin();
  m_FarthestDistance = itk::NumericTraits< TCoordRep >::max();
  TCoordRep T1, T2, invRayDir;
  for(unsigned int i=0; i<VBoxDimension; i++)
    {
    if(rayDirection[i] == itk::NumericTraits< TCoordRep >::ZeroValue())
      if(m_RayOrigin[i]<m_BoxMin[i] || m_RayOrigin[i]>m_BoxMax[i])
        return false;

    invRayDir = 1/rayDirection[i];
    T1 = (m_BoxMin[i] - m_RayOrigin[i]) * invRayDir;
    T2 = (m_BoxMax[i] - m_RayOrigin[i]) * invRayDir;
    if(T1>T2) std::swap( T1, T2 );
    if(T1>m_NearestDistance) m_NearestDistance = T1;
    if(T2<m_FarthestDistance) m_FarthestDistance = T2;
    if(m_NearestDistance>m_FarthestDistance) return false;
    if(m_FarthestDistance<0) return false;
    }
  return true;
}

template < class TCoordRep, unsigned int VBoxDimension >
void
RayBoxIntersectionFunction<TCoordRep, VBoxDimension>
::SetBoxFromImage( ImageBaseConstPointer img )
{
  if(VBoxDimension != img->GetImageDimension())
    itkGenericExceptionMacro(<< "Box and image dimensions must agree");

  // Box corner 1
  m_BoxMin = img->GetOrigin().GetVectorFromOrigin();
  m_BoxMin -= img->GetSpacing() * 0.5;

  // Box corner 2
  m_BoxMax = m_BoxMin;
  for(unsigned int i=0; i<VBoxDimension; i++)
    m_BoxMax[i] += img->GetSpacing()[i] * img->GetLargestPossibleRegion().GetSize()[i];

  // Sort
  for(unsigned int i=0; i<VBoxDimension; i++)
    if(m_BoxMin[i]>m_BoxMax[i])
      std::swap( m_BoxMin[i], m_BoxMax[i] );
}

} // namespace rtk
