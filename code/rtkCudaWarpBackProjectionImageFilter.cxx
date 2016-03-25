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

#include "rtkCudaWarpBackProjectionImageFilter.h"
#include "rtkCudaUtilities.hcu"
#include "rtkCudaWarpBackProjectionImageFilter.hcu"

#include <itkImageRegionIterator.h>
#include <itkImageRegionConstIterator.h>
#include <itkMacro.h>

namespace rtk
{

CudaWarpBackProjectionImageFilter
::CudaWarpBackProjectionImageFilter()
{
  this->SetNumberOfRequiredInputs(1);
}

void
CudaWarpBackProjectionImageFilter
::SetInputVolume(const ImageType* Volume)
{
  this->SetInput(0, const_cast<ImageType*>(Volume));
}

void
CudaWarpBackProjectionImageFilter
::SetInputProjectionStack(const ImageType* ProjectionStack)
{
  this->SetInput(1, const_cast<ImageType*>(ProjectionStack));
}

void
CudaWarpBackProjectionImageFilter
::SetDisplacementField(const DVFType* MVF)
{
  this->SetInput("DisplacementField", const_cast<DVFType*>(MVF));
}

CudaWarpBackProjectionImageFilter::ImageType::Pointer
CudaWarpBackProjectionImageFilter
::GetInputVolume()
{
  return static_cast< ImageType * >
          ( this->itk::ProcessObject::GetInput(0) );
}

CudaWarpBackProjectionImageFilter::ImageType::Pointer
CudaWarpBackProjectionImageFilter
::GetInputProjectionStack()
{
  return static_cast< ImageType * >
          ( this->itk::ProcessObject::GetInput(1) );
}

CudaWarpBackProjectionImageFilter::DVFType::Pointer
CudaWarpBackProjectionImageFilter
::GetDisplacementField()
{
  return static_cast< DVFType * >
          ( this->itk::ProcessObject::GetInput("DisplacementField") );
}

void
CudaWarpBackProjectionImageFilter
::GenerateInputRequestedRegion()
{
  Superclass::GenerateInputRequestedRegion();

  // Since we do not know where the DVF points, the
  // whole input projection is required.
  // However, the volume's requested region
  // computed by Superclass::GenerateInputRequestedRegion()
  // is the requested region for the DVF
  this->GetInputVolume()->SetRequestedRegion(this->GetOutput()->GetRequestedRegion());

  // Compute the intersection betwee the DVF's largest possible region and
  // the requested region
  DVFType::RegionType largest = this->GetDisplacementField()->GetLargestPossibleRegion();
  largest.Crop(this->GetInputVolume()->GetRequestedRegion());
  this->GetDisplacementField()->SetRequestedRegion(largest);

  this->GetInputProjectionStack()->SetRequestedRegionToLargestPossibleRegion();
}


void
CudaWarpBackProjectionImageFilter
::GPUGenerateData()
{
  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetSize(Dimension-1);
  const unsigned int iFirstProj = this->GetInputProjectionStack()->GetLargestPossibleRegion().GetIndex(Dimension-1);

  // Ramp factor is the correction for ramp filter which did not account for the
  // divergence of the beam
//  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer() );

  // Rotation center (assumed to be at 0 yet)
  ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  itk::ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInputVolume()->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxVol;
  matrixIdxVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    matrixIdxVol[i][3] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    rotCenterIndex[i] -= this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    }

  // Cuda convenient format for dimensions
  int projectionSize[2];
  projectionSize[0] = this->GetInputProjectionStack()->GetBufferedRegion().GetSize()[0];
  projectionSize[1] = this->GetInputProjectionStack()->GetBufferedRegion().GetSize()[1];

  int volumeSize[3];
  volumeSize[0] = this->GetOutput()->GetBufferedRegion().GetSize()[0];
  volumeSize[1] = this->GetOutput()->GetBufferedRegion().GetSize()[1];
  volumeSize[2] = this->GetOutput()->GetBufferedRegion().GetSize()[2];

  int inputDVFSize[3];
  inputDVFSize[0] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[0];
  inputDVFSize[1] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[1];
  inputDVFSize[2] = this->GetDisplacementField()->GetBufferedRegion().GetSize()[2];

  // Split the DVF into three images (one per component)
  ImageType::Pointer xCompDVF = ImageType::New();
  ImageType::Pointer yCompDVF = ImageType::New();
  ImageType::Pointer zCompDVF = ImageType::New();
  ImageType::RegionType largest = this->GetDisplacementField()->GetLargestPossibleRegion();
  xCompDVF->SetRegions(largest);
  yCompDVF->SetRegions(largest);
  zCompDVF->SetRegions(largest);
  xCompDVF->Allocate();
  yCompDVF->Allocate();
  zCompDVF->Allocate();
  itk::ImageRegionIterator<ImageType>     itxComp(xCompDVF, largest);
  itk::ImageRegionIterator<ImageType>     ityComp(yCompDVF, largest);
  itk::ImageRegionIterator<ImageType>     itzComp(zCompDVF, largest);
  itk::ImageRegionConstIterator<DVFType>       itDVF(this->GetDisplacementField(), largest);
  while(!itDVF.IsAtEnd())
    {
      itxComp.Set(itDVF.Get()[0]);
      ityComp.Set(itDVF.Get()[1]);
      itzComp.Set(itDVF.Get()[2]);
      ++itxComp;
      ++ityComp;
      ++itzComp;
      ++itDVF;
    }


  // Transform matrices that we will need during the warping process
  itk::Matrix<double, 4, 4> indexInputToPPInputMatrix;
  itk::Matrix<double, 4, 4> indexInputToIndexDVFMatrix;
  itk::Matrix<double, 4, 4> PPInputToIndexInputMatrix;

  indexInputToPPInputMatrix = rtk::GetIndexToPhysicalPointMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix()
                              * matrixIdxVol.GetVnlMatrix();

  indexInputToIndexDVFMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetDisplacementField().GetPointer() ).GetVnlMatrix()
                              * rtk::GetIndexToPhysicalPointMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix()
                              * matrixIdxVol.GetVnlMatrix();

  PPInputToIndexInputMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix();

  // Convert the matrices to arrays of floats (skipping the last line, as we don't care)
  float fIndexInputToIndexDVFMatrix[12];
  float fPPInputToIndexInputMatrix[12];
  float fIndexInputToPPInputMatrix[12];
  for (int j = 0; j < 12; j++)
    {
    fIndexInputToIndexDVFMatrix[j] = (float) indexInputToIndexDVFMatrix[j/4][j%4];
    fPPInputToIndexInputMatrix[j] = (float) PPInputToIndexInputMatrix[j/4][j%4];
    fIndexInputToPPInputMatrix[j] = (float) indexInputToPPInputMatrix[j/4][j%4];
    }

  // Load the required images onto the GPU (handled by the CudaDataManager)
  float *pin  = *(float**)( this->GetInput()->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pout = *(float**)( this->GetOutput()->GetCudaDataManager()->GetGPUBufferPointer() );

  float *stackGPUPointer = *(float**)( this->GetInputProjectionStack()->GetCudaDataManager()->GetGPUBufferPointer() );
  ptrdiff_t projSize = this->GetInputProjectionStack()->GetBufferedRegion().GetSize()[0] *
                       this->GetInputProjectionStack()->GetBufferedRegion().GetSize()[1];
  stackGPUPointer += projSize * (iFirstProj-this->GetInputProjectionStack()->GetBufferedRegion().GetIndex()[2]);

  float *pinxDVF = *(float**)( xCompDVF->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pinyDVF = *(float**)( yCompDVF->GetCudaDataManager()->GetGPUBufferPointer() );
  float *pinzDVF = *(float**)( zCompDVF->GetCudaDataManager()->GetGPUBufferPointer() );

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++, stackGPUPointer += projSize)
    {
    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);

    // We correct the matrix for non zero indexes
    itk::Matrix<double, 3, 3> matrixIdxProj;
    matrixIdxProj.SetIdentity();
    for(unsigned int i=0; i<2; i++)
      //SR: 0.5 for 2D texture
      matrixIdxProj[i][2] = -1*(this->GetInputProjectionStack()->GetBufferedRegion().GetIndex()[i])+0.5;

    matrix = matrixIdxProj.GetVnlMatrix() * matrix.GetVnlMatrix() * matrixIdxVol.GetVnlMatrix();

    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    float fMatrix[12];
    for (int j = 0; j < 12; j++)
      fMatrix[j] = matrix[j/4][j%4];

    CUDA_warp_back_project(projectionSize,
                      volumeSize,
                      inputDVFSize,
                      fMatrix,
                      pin,
                      pout,
                      stackGPUPointer,
                      pinxDVF,
                      pinyDVF,
                      pinzDVF,
                      fIndexInputToIndexDVFMatrix,
                      fPPInputToIndexInputMatrix,
                      fIndexInputToPPInputMatrix
                      );
    pin = pout;
    }
}

} // end namespace rtk
