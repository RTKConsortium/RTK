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

#include <fstream>
#include <stdlib.h>
#include <itksys/RegularExpression.hxx>
#include "rtkForbildPhantomFileReader.h"
#include "rtkQuadricShape.h"
#include "rtkBoxShape.h"
#include "rtkIntersectionOfConvexShapes.h"

namespace rtk
{
void
ForbildPhantomFileReader
::GenerateOutputInformation()
{
  m_GeometricPhantom = GeometricPhantom::New();

  std::ifstream     myFile;
  myFile.open( m_Filename.c_str() );
  if ( !myFile.is_open() )
    {
    itkGenericExceptionMacro("Error opening file" << m_Filename);
    }
  while ( !myFile.eof() )
    {
    // A figure is between curly brackets
    std::string       line;
    getline(myFile, line, '{');
    getline(myFile, line, '}');

    // Find fig
    std::string regex = "\\[ *([a-zA-Z_]*):";
    itksys::RegularExpression re;
    if(!re.compile(regex.c_str()))
      itkExceptionMacro(<< "Could not compile " << regex);
    if(!re.find(line.c_str()))
      continue;
    const std::string fig = re.match(1);

    // Find density
    ScalarType rho;
    if(!FindParameterInString("rho", line, rho))
      itkGenericExceptionMacro("Could not find rho in " << line);

    // Find center
    m_Center.Fill(0.);
    FindParameterInString("x", line, m_Center[0]);
    FindParameterInString("y", line, m_Center[1]);
    FindParameterInString("z", line, m_Center[2]);

    // Find radius
    if(fig == "Sphere")
      CreateForbildSphere(line);
    else if(fig == "Box")
      CreateForbildBox(line);
    else if(fig.substr(0,8) == "Cylinder")
      CreateForbildCylinder(line, fig);
    else if(fig.substr(0,10) == "Ellipt_Cyl")
      CreateForbildElliptCyl(line, fig);
    else if(fig.substr(0,9) == "Ellipsoid")
      CreateForbildEllipsoid(line, fig);
    else if(fig.substr(0,9) == "Cone")
      CreateForbildCone(line, fig);

    // Density
    ScalarType density = rho;
    for(size_t i=0; i<m_GeometricPhantom->GetConvexShapes().size(); i++)
      {
      if(m_GeometricPhantom->GetConvexShapes()[i]->IsInside(m_Center))
        density -= m_GeometricPhantom->GetConvexShapes()[i]->GetDensity();
      }
    for(size_t i=0; i<m_Unions.size(); i++)
      {
      if(m_Unions[i]->IsInside(m_Center))
        density -= m_Unions[i]->GetDensity();
      }
    if(m_ConvexShape.IsNotNull())
      {
      m_ConvexShape->SetDensity(density);
      m_GeometricPhantom->AddConvexShape(m_ConvexShape);
      FindClipPlanes(line);
      FindUnions(line);
      m_ConvexShape = ConvexShape::Pointer(ITK_NULLPTR);
      }
    }
  for(size_t i=0; i<m_Unions.size(); i++)
    m_GeometricPhantom->AddConvexShape(m_Unions[i]);
  myFile.close();
}

void
ForbildPhantomFileReader
::CreateForbildSphere(const std::string &s)
{
  ScalarType r = 0.;
  if(!FindParameterInString("r", s, r))
    itkExceptionMacro(<< "Could not find r (radius) in "<< s);
  VectorType axes;
  axes.Fill(r);
  QuadricShape::Pointer q = QuadricShape::New();
  q->SetEllipsoid(m_Center, axes);
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader
::CreateForbildBox(const std::string &s)
{
  VectorType length;
  if(!FindParameterInString("dx", s, length[0]))
    itkExceptionMacro(<< "Could not find dx in "<< s);
  if(!FindParameterInString("dy", s, length[1]))
    itkExceptionMacro(<< "Could not find dy in "<< s);
  if(!FindParameterInString("dz", s, length[2]))
    itkExceptionMacro(<< "Could not find dz in "<< s);
  BoxShape::Pointer b = BoxShape::New();
  b->SetBoxMin(m_Center - 0.5 * length);
  b->SetBoxMax(m_Center + 0.5 * length);
  m_ConvexShape = b.GetPointer();
}

void
ForbildPhantomFileReader
::CreateForbildCylinder(const std::string &s, const std::string &fig)
{
  ScalarType l = 0.;
  if(!FindParameterInString("l", s, l))
    itkExceptionMacro(<< "Could not find l (length) in "<< s);
  ScalarType r = 0.;
  if(!FindParameterInString("r", s, r))
    itkExceptionMacro(<< "Could not find r (radius) in "<< s);
  VectorType axes;
  axes.Fill(r);
  VectorType planeDir;
  planeDir.Fill(0.);
  ConvexShape::RotationMatrixType rot;
  rot.SetIdentity();
  if(fig == "Cylinder_x")
    {
    axes[0] = 0.;
    planeDir[0] = 1.;
    }
  else if(fig == "Cylinder_y")
    {
    axes[1] = 0.;
    planeDir[1] = 1.;
    }
  else if(fig == "Cylinder_z")
    {
    axes[2] = 0.;
    planeDir[2] = 1.;
    }
  else //Cylinder
    {
    axes[0] = 0.;
    planeDir[0] = 1.;
    VectorType dir;
    if(!FindVectorInString("axis", s, dir))
      itkExceptionMacro(<< "Could not find axis in "<< s);
    rot = ComputeRotationMatrixBetweenVectors(planeDir, dir);
    }
  QuadricShape::Pointer q = QuadricShape::New();
  q->SetEllipsoid(VectorType(0.), axes);
  q->AddClipPlane(planeDir, 0.5*l);
  q->AddClipPlane(-1.*planeDir, 0.5*l);
  q->Rotate(rot);
  q->Translate(m_Center);
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader
::CreateForbildElliptCyl(const std::string &s, const std::string &fig)
{
  ScalarType l = 0.;
  if(!FindParameterInString("l", s, l))
    itkExceptionMacro(<< "Could not find l (length) in "<< s);
  VectorType axes;
  axes.Fill(0.);
  size_t found = 0;
  if(FindParameterInString("dx", s, axes[0]))
    found++;
  if(FindParameterInString("dy", s, axes[1]))
    found++;
  if(FindParameterInString("dz", s, axes[2]))
    found++;
  if(found!=2)
    itkExceptionMacro(<< "Exactly two among dx dy dz are required for "
                      << fig << ", " << found << " found in " << s);
  VectorType planeDir;
  for(unsigned int i=0; i<Dimension; i++)
    planeDir[i] = (axes[i]==0.)?1.:0.;

  QuadricShape::Pointer q = QuadricShape::New();
  q->SetEllipsoid(VectorType(0.), axes);
  q->AddClipPlane(planeDir, 0.5*l);
  q->AddClipPlane(-1.*planeDir, 0.5*l);
  if(fig == "Ellipt_Cyl")
    {
    VectorType a_x;
    if(!FindVectorInString("a_x", s, a_x))
      itkExceptionMacro(<< "Could not find a_x in "<< s);
    a_x /= a_x.GetNorm();
    VectorType a_y;
    if(!FindVectorInString("a_x", s, a_y))
      itkExceptionMacro(<< "Could not find a_y in "<< s);
    a_y /= a_y.GetNorm();
    VectorType a_z = CrossProduct(a_x, a_y);
    RotationMatrixType rot;
    for(unsigned int i=0; i<Dimension; i++)
      {
      rot[0][i] = a_x[i];
      rot[1][i] = a_y[i];
      rot[2][i] = a_z[i];
      }
    q->Rotate(rot);
    }
  q->Translate(m_Center);
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader
::CreateForbildEllipsoid(const std::string &s, const std::string &fig)
{
  VectorType axes;
  if(!FindParameterInString("dx", s, axes[0]))
    itkExceptionMacro(<< "Could not find dx in "<< s);
  if(!FindParameterInString("dy", s, axes[1]))
    itkExceptionMacro(<< "Could not find dy in "<< s);
  if(!FindParameterInString("dz", s, axes[2]))
    itkExceptionMacro(<< "Could not find dz in "<< s);
  QuadricShape::Pointer q = QuadricShape::New();
  q->SetEllipsoid(VectorType(0.), axes);

  if(fig == "Ellipsoid_free")
    {
    RotationMatrixType rot;
    VectorType dirx, diry, dirz;
    bool bx = FindVectorInString("a_x", s, dirx);
    bool by = FindVectorInString("a_y", s, diry);
    bool bz = FindVectorInString("a_z", s, dirz);
    if(bx) dirx /= dirx.GetNorm();
    if(by) diry /= diry.GetNorm();
    if(bz) dirz /= dirz.GetNorm();
    if(!bx) dirx = CrossProduct(diry,dirz);
    if(!by) diry = CrossProduct(dirz,dirx);
    if(!bz) dirz = CrossProduct(dirx,diry);
    for(unsigned int i=0; i<Dimension; i++)
      {
      rot[i][0] = dirx[i];
      rot[i][1] = diry[i];
      rot[i][2] = dirz[i];
      }
    q->Rotate(rot);
    }
  q->Translate(m_Center);
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader
::CreateForbildCone(const std::string & /*s*/, const std::string & /*fig*/)
{
  itkExceptionMacro(<< "Cones have not been implemented (yet).");
}

void
ForbildPhantomFileReader
::CreateForbildTetrahedron(const std::string & /*s*/)
{
}

bool
ForbildPhantomFileReader
::FindParameterInString(const std::string &name, const std::string &s, ScalarType & param)
{
  std::string regex = std::string("  *") + name + std::string(" *= *([-+0-9.]*)");
  itksys::RegularExpression re;
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  bool bFound = re.find(s.c_str());
  if(bFound)
    param = atof(re.match(1).c_str());
  return bFound;
}

bool
ForbildPhantomFileReader
::FindVectorInString(const std::string &name,const std::string &s, VectorType & vec)
{
  std::string regex = std::string(" *") + name +
      std::string(" *\\( *([-+0-9.]*) *, *([-+0-9.]*) *, *([-+0-9.]*) *\\)");
  itksys::RegularExpression re;
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  bool bFound = re.find(s.c_str());
  if(bFound)
    {
    for(size_t i=0; i<3; i++)
      {
      vec[i] = atof(re.match(i+1).c_str());
      }
    }
  return bFound;
}

ForbildPhantomFileReader::RotationMatrixType
ForbildPhantomFileReader
::ComputeRotationMatrixBetweenVectors(const VectorType& source, const VectorType & dest) const
{
  VectorType s = source / source.GetNorm();
  VectorType d = dest / dest.GetNorm();
  RotationMatrixType r;
  r.SetIdentity();
  ScalarType c = s*d;
  if( fabs(c)-1. == itk::NumericTraits<ScalarType>::ZeroValue() )
    {
    return r;
    }
  VectorType v = CrossProduct(s, d);
  ConvexShape::RotationMatrixType vx;
  vx.Fill(0.);
  vx[0][1] = -v[2];
  vx[1][0] =  v[2];
  vx[0][2] =  v[1];
  vx[2][0] = -v[1];
  vx[1][2] =  v[0];
  vx[2][1] = -v[0];
  r += vx;
  r += vx * vx * 1. / (1.+c);
  return r;
}

void
ForbildPhantomFileReader
::FindClipPlanes(const std::string &s)
{
  // of the form r(x,y,z) > expr
  std::string regex(" +r *\\( *([-+0-9.]*) *, *([-+0-9.]*) *, *([-+0-9.]*) *\\) *([<>]) *([-+0-9.]*)");
  itksys::RegularExpression re;
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  const char * currs = s.c_str();
  while(re.find(currs))
    {
    VectorType vec;
    for(size_t i=0; i<3; i++)
      {
      vec[i] = atof(re.match(i+1).c_str());
      }
    vec /= vec.GetNorm();
    ScalarType sign = (re.match(4) == std::string("<"))?1.:-1.;
    ScalarType expr = atof(re.match(5).c_str());
    m_ConvexShape->AddClipPlane(sign*vec, sign*expr);
    currs += re.end();
    }

  // of the form x>expr or x<expr
  regex = " +x *([<>]) *([-+0-9.]*)";
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  currs = s.c_str();
  while(re.find(currs))
    {
    VectorType vec;
    vec.Fill(0.);
    vec[0] = 1.;
    ScalarType sign = (re.match(1) == std::string("<"))?1.:-1.;
    ScalarType expr = atof(re.match(2).c_str());
    m_ConvexShape->AddClipPlane(sign*vec, sign*expr);
    currs += re.end();
    }

  // of the form y>expr or y<expr
  regex = " +y *([<>]) *([-+0-9.]*)";
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  currs = s.c_str();
  while(re.find(currs))
    {
    VectorType vec;
    vec.Fill(0.);
    vec[1] = 1.;
    ScalarType sign = (re.match(1) == std::string("<"))?1.:-1.;
    ScalarType expr = atof(re.match(2).c_str());
    m_ConvexShape->AddClipPlane(sign*vec, sign*expr);
    currs += re.end();
    }

  // of the form z>expr or z<expr
  regex = " +z *([<>]) *([-+0-9.]*)";
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  currs = s.c_str();
  while(re.find(currs))
    {
    VectorType vec;
    vec.Fill(0.);
    vec[2] = 1.;
    ScalarType sign = (re.match(1) == std::string("<"))?1.:-1.;
    ScalarType expr = atof(re.match(2).c_str());
    m_ConvexShape->AddClipPlane(sign*vec, sign*expr);
    currs += re.end();
    }
}

void
ForbildPhantomFileReader
::FindUnions(const std::string &s)
{
  std::string regex(" +union *= *([-0-9]*)");
  itksys::RegularExpression re;
  if(!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  const char *currs = s.c_str();
  while(re.find(currs))
    {
    currs += re.end();
    IntersectionOfConvexShapes::Pointer ico = IntersectionOfConvexShapes::New();
    ico->AddConvexShape(m_ConvexShape);
    size_t len = m_GeometricPhantom->GetConvexShapes().size();
    int u = atoi(re.match(1).c_str());
    size_t pos = len+u-1;
    ico->AddConvexShape(m_GeometricPhantom->GetConvexShapes()[pos]);
    if(m_ConvexShape->GetDensity() !=
       m_GeometricPhantom->GetConvexShapes()[pos]->GetDensity())
      itkExceptionMacro(<< "Cannot unionize objects of different density in " << s);
    ico->SetDensity(-1.*m_ConvexShape->GetDensity());
    m_Unions.push_back(ico.GetPointer());
    }
}

} // namespace rtk
