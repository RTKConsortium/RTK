namespace rtk {

//--------------------------------------------------------------------
template< unsigned int TDimension >
void
ProjectionGeometry< TDimension >
::PrintSelf( std::ostream& os, itk::Indent indent ) const
{
  os << "List of projection matrices:" << std::endl;
  for(unsigned int i=0; i<m_Matrices.size(); i++)
    {
    os << indent << "Matrix #" << i << ": "
       << m_Matrices[i] << std::endl;
    }
}

}
