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

#include "math.h"
#include <cstdlib>
#include <clocale>
#include <fstream>
#include <itksys/RegularExpression.hxx>

#include "rtkForbildPhantomFileReader.h"
#include "rtkQuadricShape.h"
#include "rtkBoxShape.h"
#include "rtkIntersectionOfConvexShapes.h"

namespace rtk
{
void
ForbildPhantomFileReader::GenerateOutputInformation()
{
  // Save locale setting
  const std::string oldLocale = setlocale(LC_NUMERIC, nullptr);
  setlocale(LC_NUMERIC, "C");
  m_GeometricPhantom = GeometricPhantom::New();

  std::ifstream myFile;
  myFile.open(m_Filename.c_str());
  if (!myFile.is_open())
  {
    itkGenericExceptionMacro("Error opening file" << m_Filename);
  }
  while (!myFile.eof())
  {
    // A figure is between curly brackets
    std::string line;
    getline(myFile, line, '{');
    getline(myFile, line, '}');

    // Find fig
    std::string               regex = "\\[ *([a-zA-Z_]*):";
    itksys::RegularExpression re;
    if (!re.compile(regex.c_str()))
      itkExceptionMacro(<< "Could not compile " << regex);
    if (!re.find(line.c_str()))
      continue;
    const std::string fig = re.match(1);

    // Find density
    ScalarType rho = NAN;
    if (!FindParameterInString("rho", line, rho))
      itkGenericExceptionMacro("Could not find rho in " << line);

    // Find center
    m_Center.Fill(0.);
    FindParameterInString("x", line, m_Center[0]);
    FindParameterInString("y", line, m_Center[1]);
    FindParameterInString("z", line, m_Center[2]);

    // Find radius
    if (fig == "Sphere")
      CreateForbildSphere(line);
    else if (fig == "Box")
      CreateForbildBox(line);
    else if (fig.substr(0, 8) == "Cylinder")
      CreateForbildCylinder(line, fig);
    else if (fig.substr(0, 10) == "Ellipt_Cyl")
      CreateForbildElliptCyl(line, fig);
    else if (fig.substr(0, 9) == "Ellipsoid")
      CreateForbildEllipsoid(line, fig);
    else if (fig.substr(0, 4) == "Cone")
      CreateForbildCone(line, fig);

    // Density
    ScalarType density = rho;
    for (const auto & convexShape : m_GeometricPhantom->GetConvexShapes())
    {
      if (convexShape->IsInside(m_Center))
        density -= convexShape->GetDensity();
    }
    for (const auto & m_Union : m_Unions)
    {
      if (m_Union->IsInside(m_Center))
        density -= m_Union->GetDensity();
    }
    if (m_ConvexShape.IsNotNull())
    {
      m_ConvexShape->SetDensity(density);
      FindClipPlanes(line);
      m_GeometricPhantom->AddConvexShape(m_ConvexShape);
      FindUnions(line);
      m_ConvexShape = ConvexShape::Pointer(nullptr);
    }
  }
  for (const auto & m_Union : m_Unions)
    m_GeometricPhantom->AddConvexShape(m_Union);
  myFile.close();
  setlocale(LC_NUMERIC, oldLocale.c_str());
}

void
ForbildPhantomFileReader::CreateForbildSphere(const std::string & s)
{
  ScalarType r = 0.;
  if (!FindParameterInString("r", s, r))
    itkExceptionMacro(<< "Could not find r (radius) in " << s);
  VectorType axes;
  axes.Fill(r);
  auto q = QuadricShape::New();
  q->SetEllipsoid(m_Center, axes);
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader::CreateForbildBox(const std::string & s)
{
  VectorType length;
  if (!FindParameterInString("dx", s, length[0]))
    itkExceptionMacro(<< "Could not find dx in " << s);
  if (!FindParameterInString("dy", s, length[1]))
    itkExceptionMacro(<< "Could not find dy in " << s);
  if (!FindParameterInString("dz", s, length[2]))
    itkExceptionMacro(<< "Could not find dz in " << s);
  auto b = BoxShape::New();
  b->SetBoxMin(m_Center - 0.5 * length);
  b->SetBoxMax(m_Center + 0.5 * length);
  m_ConvexShape = b.GetPointer();
}

void
ForbildPhantomFileReader::CreateForbildCylinder(const std::string & s, const std::string & fig)
{
  ScalarType l = 0.;
  if (!FindParameterInString("l", s, l))
    itkExceptionMacro(<< "Could not find l (length) in " << s);
  ScalarType r = 0.;
  if (!FindParameterInString("r", s, r))
    itkExceptionMacro(<< "Could not find r (radius) in " << s);
  VectorType axes;
  axes.Fill(r);
  VectorType planeDir;
  planeDir.Fill(0.);
  ConvexShape::RotationMatrixType rot;
  rot.SetIdentity();
  if (fig == "Cylinder_x")
  {
    axes[0] = 0.;
    planeDir[0] = 1.;
  }
  else if (fig == "Cylinder_y")
  {
    axes[1] = 0.;
    planeDir[1] = 1.;
  }
  else if (fig == "Cylinder_z")
  {
    axes[2] = 0.;
    planeDir[2] = 1.;
  }
  else // Cylinder
  {
    axes[0] = 0.;
    planeDir[0] = 1.;
    VectorType dir;
    if (!FindVectorInString("axis", s, dir))
      itkExceptionMacro(<< "Could not find axis in " << s);
    rot = ComputeRotationMatrixBetweenVectors(planeDir, dir);
  }
  auto q = QuadricShape::New();
  q->SetEllipsoid(itk::MakePoint(0., 0., 0.), axes);
  q->AddClipPlane(planeDir, 0.5 * l);
  q->AddClipPlane(-1. * planeDir, 0.5 * l);
  q->Rotate(rot);
  q->Translate(m_Center.GetVectorFromOrigin());
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader::CreateForbildElliptCyl(const std::string & s, const std::string & fig)
{
  ScalarType l = 0.;
  if (!FindParameterInString("l", s, l))
    itkExceptionMacro(<< "Could not find l (length) in " << s);
  VectorType axes;
  axes.Fill(0.);
  size_t found = 0;
  if (FindParameterInString("dx", s, axes[0]))
    found++;
  if (FindParameterInString("dy", s, axes[1]))
    found++;
  if (FindParameterInString("dz", s, axes[2]))
    found++;
  if (found != 2)
    itkExceptionMacro(<< "Exactly two among dx dy dz are required for " << fig << ", " << found << " found in " << s);
  VectorType planeDir;
  for (unsigned int i = 0; i < Dimension; i++)
    planeDir[i] = (axes[i] == 0.) ? 1. : 0.;

  auto q = QuadricShape::New();
  q->SetEllipsoid(itk::MakePoint(0., 0., 0.), axes);
  q->AddClipPlane(planeDir, 0.5 * l);
  q->AddClipPlane(-1. * planeDir, 0.5 * l);
  if (fig == "Ellipt_Cyl")
  {
    VectorType a_x;
    bool       a_x_Found = FindVectorInString("a_x", s, a_x);
    VectorType a_y;
    bool       a_y_Found = FindVectorInString("a_y", s, a_y);
    VectorType a_z;
    bool       a_z_Found = FindVectorInString("axis", s, a_z);
    if (a_x_Found && a_y_Found)
    {
      a_x /= a_x.GetNorm();
      a_y /= a_y.GetNorm();
      a_z = CrossProduct(a_x, a_y);
    }
    else if (a_x_Found && a_z_Found)
    {
      a_x /= a_x.GetNorm();
      a_z /= a_z.GetNorm();
      a_y = CrossProduct(a_z, a_x);
    }
    else if (a_y_Found && a_z_Found)
    {
      a_y /= a_y.GetNorm();
      a_z /= a_z.GetNorm();
      a_x = CrossProduct(a_y, a_z);
    }
    else
    {
      itkExceptionMacro(<< "Could not find 2 vectors among a_x, a_y and axis in " << s);
    }
    RotationMatrixType rot;
    for (unsigned int i = 0; i < Dimension; i++)
    {
      rot[i][0] = a_x[i];
      rot[i][1] = a_y[i];
      rot[i][2] = a_z[i];
    }
    q->Rotate(rot);
  }
  q->Translate(m_Center.GetVectorFromOrigin());
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader::CreateForbildEllipsoid(const std::string & s, const std::string & fig)
{
  VectorType axes;
  if (!FindParameterInString("dx", s, axes[0]))
    itkExceptionMacro(<< "Could not find dx in " << s);
  if (!FindParameterInString("dy", s, axes[1]))
    itkExceptionMacro(<< "Could not find dy in " << s);
  if (!FindParameterInString("dz", s, axes[2]))
    itkExceptionMacro(<< "Could not find dz in " << s);
  auto q = QuadricShape::New();
  q->SetEllipsoid(itk::MakePoint(0., 0., 0.), axes);

  if (fig == "Ellipsoid_free")
  {
    RotationMatrixType rot;
    VectorType         dirx, diry, dirz;
    bool               bx = FindVectorInString("a_x", s, dirx);
    bool               by = FindVectorInString("a_y", s, diry);
    bool               bz = FindVectorInString("a_z", s, dirz);
    if (bx)
      dirx /= dirx.GetNorm();
    if (by)
      diry /= diry.GetNorm();
    if (bz)
      dirz /= dirz.GetNorm();
    if (!bx)
      dirx = CrossProduct(diry, dirz);
    if (!by)
      diry = CrossProduct(dirz, dirx);
    if (!bz)
      dirz = CrossProduct(dirx, diry);
    for (unsigned int i = 0; i < Dimension; i++)
    {
      rot[i][0] = dirx[i];
      rot[i][1] = diry[i];
      rot[i][2] = dirz[i];
    }
    q->Rotate(rot);
  }
  q->Translate(m_Center.GetVectorFromOrigin());
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader::CreateForbildCone(const std::string & s, const std::string & fig)
{
  ScalarType l = 0.;
  if (!FindParameterInString("l", s, l))
    itkExceptionMacro(<< "Could not find l (length) in " << s);
  size_t     found = 0;
  ScalarType r1 = 0.;
  if (!FindParameterInString("r1", s, r1))
  {
    itkExceptionMacro(<< "Missing radius r1 in " << fig << ", " << found << " found in " << s);
  }
  ScalarType r2 = 0.;
  if (!FindParameterInString("r2", s, r2))
  {
    itkExceptionMacro(<< "Missing radius r2 in " << fig << ", " << found << " found in " << s);
  }

  VectorType axes;
  axes.Fill(r1);
  VectorType planeDir;
  planeDir[0] = 0.;
  planeDir[1] = 0.;
  planeDir[2] = 1.;

  auto      q = QuadricShape::New();
  PointType spatialOffset;
  spatialOffset[0] = 0.;
  spatialOffset[1] = 0.;
  if (r2 > r1)
  {
    axes[2] = -l * r1 / (r2 - r1);
    q->AddClipPlane(-1. * planeDir, axes[2]);
    q->AddClipPlane(planeDir, l - axes[2]);
    spatialOffset[2] = axes[2] - 0.5 * l;
  }
  else
  {
    axes[2] = -l * r1 / (r1 - r2);
    q->AddClipPlane(-1. * planeDir, -axes[2]);
    q->AddClipPlane(planeDir, l + axes[2]);
    spatialOffset[2] = -axes[2] - 0.5 * l;
  }
  q->SetEllipsoid(itk::MakePoint(0., 0., 0.), axes);
  q->SetJ(0.);

  RotationMatrixType rot;
  rot.Fill(0.);
  if (fig == "Cone_x")
  {
    rot[0][2] = 1.;
    rot[1][0] = 1.;
    rot[2][1] = 1.;
  }
  else if (fig == "Cone_y")
  {
    rot[0][1] = 1.;
    rot[1][2] = 1.;
    rot[2][0] = 1.;
  }
  else if (fig == "Cone_z")
  {
    rot.SetIdentity();
  }
  else if (fig == "Cone")
  {
    VectorType dir;
    if (!FindVectorInString("axis", s, dir))
      itkExceptionMacro(<< "Could not find axis in " << s);
    rot = ComputeRotationMatrixBetweenVectors(planeDir, dir);
  }
  else
  {
    itkExceptionMacro(<< "Unknown figure: " << fig);
  }
  q->Rotate(rot);
  VectorType translation = m_Center.GetVectorFromOrigin() + rot * spatialOffset.GetVectorFromOrigin();
  q->Translate(translation);
  m_ConvexShape = q.GetPointer();
}

void
ForbildPhantomFileReader::CreateForbildTetrahedron(const std::string & /*s*/)
{}

bool
ForbildPhantomFileReader::FindParameterInString(const std::string & name, const std::string & s, ScalarType & param)
{
  std::string               regex = std::string("  *") + name + std::string(" *= *([-+0-9.]*)");
  itksys::RegularExpression re;
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  bool bFound = re.find(s.c_str());
  if (bFound)
    param = std::stod(re.match(1).c_str());
  return bFound;
}

bool
ForbildPhantomFileReader::FindVectorInString(const std::string & name, const std::string & s, VectorType & vec)
{
  std::string regex = std::string(" *") + name + std::string(" *\\( *([-+0-9.]*) *, *([-+0-9.]*) *, *([-+0-9.]*) *\\)");
  itksys::RegularExpression re;
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  bool bFound = re.find(s.c_str());
  if (bFound)
  {
    for (size_t i = 0; i < 3; i++)
    {
      vec[i] = std::stod(re.match(i + 1).c_str());
    }
  }
  return bFound;
}

ForbildPhantomFileReader::RotationMatrixType
ForbildPhantomFileReader::ComputeRotationMatrixBetweenVectors(const VectorType & source, const VectorType & dest) const
{
  // https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
  VectorType         s = source / source.GetNorm();
  VectorType         d = dest / dest.GetNorm();
  RotationMatrixType r;
  r.SetIdentity();
  ScalarType c = s * d;
  if (itk::Math::abs(c) - 1. == itk::NumericTraits<ScalarType>::ZeroValue())
  {
    return r;
  }
  VectorType                      v = CrossProduct(s, d);
  ConvexShape::RotationMatrixType vx;
  vx.Fill(0.);
  vx[0][1] = -v[2];
  vx[0][2] = v[1];
  vx[1][0] = v[2];
  vx[1][2] = -v[0];
  vx[2][0] = -v[1];
  vx[2][1] = v[0];
  r += vx;
  r += vx * vx * 1. / (1. + c);
  return r;
}

void
ForbildPhantomFileReader::FindClipPlanes(const std::string & s)
{
  // of the form r(x,y,z) > expr
  std::string               regex(" +r *\\( *([-+0-9.]*) *, *([-+0-9.]*) *, *([-+0-9.]*) *\\) *([<>]) *([-+0-9.]*)");
  itksys::RegularExpression re;
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  const char * currs = s.c_str();
  while (re.find(currs))
  {
    VectorType vec;
    for (size_t i = 0; i < 3; i++)
    {
      vec[i] = std::stod(re.match(i + 1).c_str());
    }
    vec /= vec.GetNorm();
    ScalarType sign = (re.match(4) == std::string("<")) ? 1. : -1.;
    ScalarType expr = std::stod(re.match(5).c_str());
    m_ConvexShape->AddClipPlane(sign * vec, sign * expr);
    currs += re.end();
  }

  // of the form x>expr or x<expr
  regex = " +x *([<>]) *([-+0-9.]*)";
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  currs = s.c_str();
  while (re.find(currs))
  {
    VectorType vec;
    vec.Fill(0.);
    vec[0] = 1.;
    ScalarType sign = (re.match(1) == std::string("<")) ? 1. : -1.;
    ScalarType expr = std::stod(re.match(2).c_str());
    m_ConvexShape->AddClipPlane(sign * vec, sign * expr);
    currs += re.end();
  }

  // of the form y>expr or y<expr
  regex = " +y *([<>]) *([-+0-9.]*)";
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  currs = s.c_str();
  while (re.find(currs))
  {
    VectorType vec;
    vec.Fill(0.);
    vec[1] = 1.;
    ScalarType sign = (re.match(1) == std::string("<")) ? 1. : -1.;
    ScalarType expr = std::stod(re.match(2).c_str());
    m_ConvexShape->AddClipPlane(sign * vec, sign * expr);
    currs += re.end();
  }

  // of the form z>expr or z<expr
  regex = " +z *([<>]) *([-+0-9.]*)";
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  currs = s.c_str();
  while (re.find(currs))
  {
    VectorType vec;
    vec.Fill(0.);
    vec[2] = 1.;
    ScalarType sign = (re.match(1) == std::string("<")) ? 1. : -1.;
    ScalarType expr = std::stod(re.match(2).c_str());
    m_ConvexShape->AddClipPlane(sign * vec, sign * expr);
    currs += re.end();
  }
}

void
ForbildPhantomFileReader::FindUnions(const std::string & s)
{
  std::string               regex(" +union *= *([-0-9]*)");
  itksys::RegularExpression re;
  if (!re.compile(regex.c_str()))
    itkExceptionMacro(<< "Could not compile " << regex);
  const char * currs = s.c_str();
  m_UnionWith.push_back(-1);
  while (re.find(currs))
  {
    currs += re.end();
    auto ico = IntersectionOfConvexShapes::New();
    ico->AddConvexShape(m_ConvexShape);
    size_t len = m_GeometricPhantom->GetConvexShapes().size();
    int    u = std::stoi(re.match(1).c_str());
    size_t pos = len + u - 1;
    ico->AddConvexShape(m_GeometricPhantom->GetConvexShapes()[pos]);
    if (m_ConvexShape->GetDensity() != m_GeometricPhantom->GetConvexShapes()[pos]->GetDensity())
      itkExceptionMacro(<< "Cannot unionize objects of different density in " << s);
    ico->SetDensity(-1. * m_ConvexShape->GetDensity());

    m_UnionWith.back() = pos;
    m_Unions.push_back(ico.GetPointer());

    // Handles the union of three objects. Union of more objects would require
    // the implementation of the inclusion-exclusion formula
    // https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle
    if (m_UnionWith[pos] != -1)
    {
      ico = IntersectionOfConvexShapes::New();
      ico->AddConvexShape(m_ConvexShape);
      ico->AddConvexShape(m_GeometricPhantom->GetConvexShapes()[m_UnionWith[pos]]);
      ico->SetDensity(-1. * m_ConvexShape->GetDensity());
      m_Unions.push_back(ico.GetPointer());

      ico = IntersectionOfConvexShapes::New();
      ico->AddConvexShape(m_ConvexShape);
      ico->AddConvexShape(m_GeometricPhantom->GetConvexShapes()[pos]);
      ico->AddConvexShape(m_GeometricPhantom->GetConvexShapes()[m_UnionWith[pos]]);
      ico->SetDensity(m_ConvexShape->GetDensity());
      m_Unions.push_back(ico.GetPointer());
    }
  }
}

} // namespace rtk
