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

#include "rtkJosephBackProjectionImageFilter.h"

namespace rtk
{

template<>
void
rtk::JosephBackProjectionImageFilter<itk::VectorImage<float, 3>,
                                     itk::VectorImage<float, 3>,
                                     Functor::SplatWeightMultiplication< float, double, float >,
                                     itk::VectorImage<float, 2> >
::BilinearSplat(const itk::VariableLengthVector<float> rayValue,
                const double stepLengthInVoxel,
                const double voxelSize,
                float *pxiyi,
                float *pxsyi,
                float *pxiys,
                float *pxsys,
                const double x,
                const double y,
                const int ox,
                const int oy)
{
    int ix = vnl_math_floor(x);
    int iy = vnl_math_floor(y);
    int idx = ix*ox + iy*oy;
    CoordRepType lx = x - ix;
    CoordRepType ly = y - iy;
    CoordRepType lxc = 1.-lx;
    CoordRepType lyc = 1.-ly;

  for (int component=0; component<rayValue.GetSize(); component++)
    {
    pxiyi[idx + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lxc * lyc);
    pxsyi[idx + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lx * lyc);
    pxiys[idx + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lxc * ly);
    pxsys[idx + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lx * ly);
    }
}


template<>
void
rtk::JosephBackProjectionImageFilter<itk::VectorImage<float, 3>,
                                     itk::VectorImage<float, 3>,
                                     Functor::SplatWeightMultiplication< float, double, float >,
                                     itk::VectorImage<float, 2> >
::BilinearSplatOnBorders(const itk::VariableLengthVector<float> rayValue,
                         const double stepLengthInVoxel,
                         const double voxelSize,
                         float *pxiyi,
                         float *pxsyi,
                         float *pxiys,
                         float *pxsys,
                         const double x,
                         const double y,
                         const int ox,
                         const int oy,
                         const double minx,
                         const double miny,
                         const double maxx,
                         const double maxy)
{
    int ix = vnl_math_floor(x);
    int iy = vnl_math_floor(y);
    int idx = ix*ox + iy*oy;
    CoordRepType lx = x - ix;
    CoordRepType ly = y - iy;
    CoordRepType lxc = 1.-lx;
    CoordRepType lyc = 1.-ly;

    int offset_xi = 0;
    int offset_yi = 0;
    int offset_xs = 0;
    int offset_ys = 0;

    if(ix < minx) offset_xi = ox;
    if(iy < miny) offset_yi = oy;
    if(ix >= maxx) offset_xs = -ox;
    if(iy >= maxy) offset_ys = -oy;

  for (int component=0; component<rayValue.GetSize(); component++)
    {
    pxiyi[idx + offset_xi + offset_yi + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lxc * lyc);
    pxsyi[idx + offset_xi + offset_yi + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lx * lyc);
    pxiys[idx + offset_xi + offset_yi + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lxc * ly);
    pxsys[idx + offset_xi + offset_yi + component] += m_SplatWeightMultiplication(rayValue[component], stepLengthInVoxel, voxelSize, lx * ly);
    }
}

} // end namespace rtk
