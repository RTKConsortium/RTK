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
#include <itkImageAlgorithm.h>

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
::SetDisplacementField(const DVFType* DVF)
{
  this->SetInput("DisplacementField", const_cast<DVFType*>(DVF));
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
  this->GetInputProjectionStack()->SetRequestedRegionToLargestPossibleRegion();

  // To compute the DVF's requested region, we use the same method
  // as in itk::WarpImageFilter, except if the ITK version is too old to
  // contain the required methods
#if ITK_VERSION_MAJOR < 4 || (ITK_VERSION_MAJOR == 4 && ITK_VERSION_MINOR < 8)
  this->GetDisplacementField()->SetRequestedRegionToLargestPossibleRegion();
#else 
  // If the output and the deformation field have the same
  // information, just propagate up the output requested region for the
  // deformation field. Otherwise, it is non-trivial to determine
  // the smallest region of the deformation field that fully
  // contains the physical space covered by the output's requested
  // region, se we do the easy thing and request the largest possible region
  DVFType::Pointer    fieldPtr = this->GetDisplacementField();
  ImageType::Pointer  outputPtr = this->GetOutput();
  if ( fieldPtr.IsNotNull() )
    {
    // tolerance for origin and spacing depends on the size of pixel
    // tolerance for direction is a fraction of the unit cube.
    const itk::SpacePrecisionType coordinateTol = this->GetCoordinateTolerance() * outputPtr->GetSpacing()[0]; // use first dimension spacing

    bool DefFieldSameInformation =
       (outputPtr->GetOrigin().GetVnlVector().is_equal(fieldPtr->GetOrigin().GetVnlVector(), coordinateTol))
    && (outputPtr->GetSpacing().GetVnlVector().is_equal(fieldPtr->GetSpacing().GetVnlVector(), coordinateTol))
    && (outputPtr->GetDirection().GetVnlMatrix().as_ref().is_equal(fieldPtr->GetDirection().GetVnlMatrix(), this->GetDirectionTolerance()));

    if (DefFieldSameInformation)
      {
      fieldPtr->SetRequestedRegion( outputPtr->GetRequestedRegion() );
      }
    else
      {
      typedef DVFType::RegionType DisplacementRegionType;

      DisplacementRegionType fieldRequestedRegion = itk::ImageAlgorithm::EnlargeRegionOverBox(outputPtr->GetRequestedRegion(),
                                                                                         outputPtr.GetPointer(),
                                                                                         fieldPtr.GetPointer());
      fieldPtr->SetRequestedRegion( fieldRequestedRegion );
      }
    if ( !fieldPtr->VerifyRequestedRegion() )
      {
      fieldPtr->SetRequestedRegion( fieldPtr->GetLargestPossibleRegion() );
      }
    }
#endif
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
  int projectionSize[3];
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

  // Transform matrices that we will need during the warping process
  itk::Matrix<double, 4, 4> indexInputToIndexDVFMatrix;
  itk::Matrix<double, 4, 4> PPInputToIndexInputMatrix;
  itk::Matrix<double, 4, 4> indexInputToPPInputMatrix;

  indexInputToIndexDVFMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetDisplacementField().GetPointer() ).GetVnlMatrix()
                              * rtk::GetIndexToPhysicalPointMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix()
                              * matrixIdxVol.GetVnlMatrix();

  PPInputToIndexInputMatrix = rtk::GetPhysicalPointToIndexMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix();

  indexInputToPPInputMatrix = rtk::GetIndexToPhysicalPointMatrix( this->GetInputVolume().GetPointer() ).GetVnlMatrix()
                              * matrixIdxVol.GetVnlMatrix();

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
  float *pDVF = *(float**)( this->GetDisplacementField()->GetCudaDataManager()->GetGPUBufferPointer() );

  float *stackGPUPointer = *(float**)( this->GetInputProjectionStack()->GetCudaDataManager()->GetGPUBufferPointer() );
  ptrdiff_t projSize = this->GetInputProjectionStack()->GetBufferedRegion().GetSize()[0] *
                       this->GetInputProjectionStack()->GetBufferedRegion().GetSize()[1];
  stackGPUPointer += projSize * (iFirstProj-this->GetInputProjectionStack()->GetBufferedRegion().GetIndex()[2]);

  // Allocate a large matrix to hold the matrix of all projections
  // fMatrix is for flat detector, the other two are for cylindrical
  float *fMatrix = new float[12 * nProj];
  float *fvolIndexToProjPP = new float[12 * nProj];
  float *fprojPPToProjIndex = new float[9];

  // Projection physical point to projection index matrix
  itk::Matrix<double, 3, 3> projPPToProjIndex = GetProjectionPhysicalPointToProjectionIndexMatrix();

  // Correction for non-zero indices in the projections
  itk::Matrix<double, 3, 3> matrixIdxProj;
  matrixIdxProj.SetIdentity();
  for(unsigned int i=0; i<2; i++)
    //SR: 0.5 for 2D texture
    matrixIdxProj[i][2] = -1*(this->GetInput(1)->GetBufferedRegion().GetIndex()[i])+0.5;

  projPPToProjIndex = matrixIdxProj.GetVnlMatrix() * projPPToProjIndex.GetVnlMatrix();

  for (int j = 0; j < 9; j++)
    fprojPPToProjIndex[j] = projPPToProjIndex[j/3][j%3];

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Volume index to projection physical point matrix
    // normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType volIndexToProjPP = GetVolumeIndexToProjectionPhysicalPointMatrix(iProj);

    // Correction for non-zero indices in the volume
    volIndexToProjPP = volIndexToProjPP.GetVnlMatrix() * matrixIdxVol.GetVnlMatrix();

    double perspFactor = volIndexToProjPP[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += volIndexToProjPP[Dimension-1][j] * rotCenterIndex[j];
    volIndexToProjPP /= perspFactor;

    ProjectionMatrixType matrix = ProjectionMatrixType(projPPToProjIndex.GetVnlMatrix() * volIndexToProjPP.GetVnlMatrix());

    // Fill float arrays with matrices coefficients, to be passed to GPU
    for (int j = 0; j < 12; j++)
      {
      fvolIndexToProjPP[j + (iProj-iFirstProj) * 12] = volIndexToProjPP[j/4][j%4];
      fMatrix[j + (iProj-iFirstProj) * 12] = matrix[j/4][j%4];
      }
    }

  for (unsigned int i=0; i<nProj; i+=SLAB_SIZE)
    {
    // If nProj is not a multiple of SLAB_SIZE, the last slab will contain less than SLAB_SIZE projections
    projectionSize[2] = std::min(nProj-i, (unsigned int)SLAB_SIZE);

    // Run the back projection with a slab of SLAB_SIZE or less projections
    CUDA_warp_back_project( projectionSize,
                            volumeSize,
                            inputDVFSize,
                            fMatrix + 12 * i,
                            fvolIndexToProjPP + 12 * i,
                            fprojPPToProjIndex,
                            pin,
                            pout,
                            stackGPUPointer + projSize * i,
                            pDVF,
                            fIndexInputToIndexDVFMatrix,
                            fPPInputToIndexInputMatrix,
                            fIndexInputToPPInputMatrix,
                            this->m_Geometry->GetRadiusCylindricalDetector()
                            );

    // Re-use the output as input
    pin = pout;
    }

  delete[] fMatrix;
  delete[] fvolIndexToProjPP;
  delete[] fprojPPToProjIndex;
}

} // end namespace rtk
