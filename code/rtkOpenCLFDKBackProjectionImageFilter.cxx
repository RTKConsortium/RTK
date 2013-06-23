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

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include "rtkOpenCLFDKBackProjectionImageFilter.h"

#include <itkImageRegionConstIterator.h>
#include <itkImageRegionIteratorWithIndex.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMacro.h>
#include "rtkOpenCLUtilities.h"

namespace rtk
{

OpenCLFDKBackProjectionImageFilter
::OpenCLFDKBackProjectionImageFilter()
{
}

void
OpenCLFDKBackProjectionImageFilter
::InitDevice()
{
  // OpenCL init (platform, device, context and command queue)
  std::vector<cl_platform_id> platforms = GetListOfOpenCLPlatforms();
  std::vector<cl_device_id>   devices = GetListOfOpenCLDevices(platforms[0]);
  cl_context_properties       properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platforms[0], 0};

  cl_int error;
  m_Context = clCreateContext(properties, devices.size(), &(devices[0]), NULL, NULL, &error);
  if(error != CL_SUCCESS)
    itkExceptionMacro(<< "Could not create OpenCL context, error code: " << error);

  m_CommandQueue = clCreateCommandQueue(m_Context, devices[0], CL_QUEUE_PROFILING_ENABLE, &error);
  if(error != CL_SUCCESS)
    itkExceptionMacro(<< "Could not create OpenCL command queue, error code: " << error);

  // OpenCL memory allocation
  m_DeviceMatrix = clCreateBuffer(m_Context,
                                  CL_MEM_READ_ONLY,
                                  sizeof(cl_float) * 12,
                                  NULL,
                                  &error);
  if(error != CL_SUCCESS)
    itkExceptionMacro(<< "Could not allocate OpenCL matrix buffer, error code: " << error);

  size_t volBytes = this->GetOutput()->GetRequestedRegion().GetNumberOfPixels() * sizeof(float);
  m_DeviceVolume = clCreateBuffer(m_Context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  volBytes,
                                  (void*)this->GetInput()->GetBufferPointer(),
                                  &error);
  if(error != CL_SUCCESS)
    itkExceptionMacro(<< "Could not allocate OpenCL volume buffer, error code: " << error);

  cl_image_format projFormat;
  projFormat.image_channel_order = CL_INTENSITY;
  projFormat.image_channel_data_type = CL_FLOAT;
  m_DeviceProjection =  clCreateImage2D(m_Context,
                                        CL_MEM_READ_ONLY,
                                        &projFormat,
                                        this->GetInput(1)->GetLargestPossibleRegion().GetSize()[0],
                                        this->GetInput(1)->GetLargestPossibleRegion().GetSize()[1],
                                        0,
                                        NULL,
                                        &error);
  if(error != CL_SUCCESS)
    itkExceptionMacro(<< "Could not allocate OpenCL projection image, error code: " << error);

  CreateAndBuildOpenCLProgramFromSourceFile("rtkOpenCLFDKBackProjectionImageFilter.cl",
                                            m_Context, m_Program);

  m_Kernel = clCreateKernel(m_Program, "OpenCLFDKBackProjectionImageFilterKernel", &error);
  if(error != CL_SUCCESS)
    itkExceptionMacro(<< "Could not create OpenCL kernel, error code: " << error);

  // Set kernel parameters
  cl_uint4 volumeDim;
  volumeDim.s[0] = this->GetOutput()->GetRequestedRegion().GetSize()[0];
  volumeDim.s[1] = this->GetOutput()->GetRequestedRegion().GetSize()[1];
  volumeDim.s[2] = this->GetOutput()->GetRequestedRegion().GetSize()[2];
  volumeDim.s[3] = 1;
  OPENCL_CHECK_ERROR( clSetKernelArg(m_Kernel, 0, sizeof(cl_mem), &m_DeviceVolume) );
  OPENCL_CHECK_ERROR( clSetKernelArg(m_Kernel, 1, sizeof(cl_mem), &m_DeviceMatrix) );
  OPENCL_CHECK_ERROR( clSetKernelArg(m_Kernel, 2, sizeof(cl_mem), &m_DeviceProjection) );
  OPENCL_CHECK_ERROR( clSetKernelArg(m_Kernel, 3, sizeof(cl_uint4), &volumeDim) );
}

void
OpenCLFDKBackProjectionImageFilter
::CleanUpDevice()
{
  size_t volBytes = this->GetOutput()->GetRequestedRegion().GetNumberOfPixels() * sizeof(float);

  OPENCL_CHECK_ERROR( clReleaseProgram(m_Program) );
  OPENCL_CHECK_ERROR( clFinish(m_CommandQueue) );
  OPENCL_CHECK_ERROR( clEnqueueReadBuffer (m_CommandQueue,
                                           m_DeviceVolume,
                                           CL_TRUE,
                                           0,
                                           volBytes,
                                           this->GetOutput()->GetBufferPointer(),
                                           0,
                                           NULL,
                                           NULL) );
  OPENCL_CHECK_ERROR( clReleaseMemObject(m_DeviceProjection) );
  OPENCL_CHECK_ERROR( clReleaseMemObject(m_DeviceVolume) );
  OPENCL_CHECK_ERROR( clReleaseMemObject(m_DeviceMatrix) );
  OPENCL_CHECK_ERROR( clReleaseCommandQueue(m_CommandQueue) );
  OPENCL_CHECK_ERROR( clReleaseContext(m_Context) );
}

void
OpenCLFDKBackProjectionImageFilter
::GenerateData()
{
  this->AllocateOutputs();

  const unsigned int Dimension = ImageType::ImageDimension;
  const unsigned int nProj = this->GetInput(1)->GetLargestPossibleRegion().GetSize(Dimension-1);
  const unsigned int iFirstProj = this->GetInput(1)->GetLargestPossibleRegion().GetIndex(Dimension-1);

  // Ramp factor is the correction for ramp filter which did not account for the
  // divergence of the beam
  const GeometryPointer geometry = dynamic_cast<GeometryType *>(this->GetGeometry().GetPointer() );

  // Rotation center (assumed to be at 0 yet)
  ImageType::PointType rotCenterPoint;
  rotCenterPoint.Fill(0.0);
  itk::ContinuousIndex<double, Dimension> rotCenterIndex;
  this->GetInput(0)->TransformPhysicalPointToContinuousIndex(rotCenterPoint, rotCenterIndex);

  // Include non-zero index in matrix
  itk::Matrix<double, 4, 4> matrixIdxVol;
  matrixIdxVol.SetIdentity();
  for(unsigned int i=0; i<3; i++)
    {
    matrixIdxVol[i][3] = this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    rotCenterIndex[i] -= this->GetOutput()->GetRequestedRegion().GetIndex()[i];
    }

  // Go over each projection
  for(unsigned int iProj=iFirstProj; iProj<iFirstProj+nProj; iProj++)
    {
    // Extract the current slice
    ProjectionImagePointer projection = this->GetProjection(iProj);

    // Index to index matrix normalized to have a correct backprojection weight
    // (1 at the isocenter)
    ProjectionMatrixType matrix = GetIndexToIndexProjectionMatrix(iProj);

    // We correct the matrix for non zero indexes
    itk::Matrix<double, 3, 3> matrixIdxProj;
    matrixIdxProj.SetIdentity();
    for(unsigned int i=0; i<2; i++)
      //SR: 0.5 for 2D texture
      matrixIdxProj[i][2] = -1*(projection->GetBufferedRegion().GetIndex()[i])+0.5;

    matrix = matrixIdxProj.GetVnlMatrix() * matrix.GetVnlMatrix() * matrixIdxVol.GetVnlMatrix();

    double perspFactor = matrix[Dimension-1][Dimension];
    for(unsigned int j=0; j<Dimension; j++)
      perspFactor += matrix[Dimension-1][j] * rotCenterIndex[j];
    matrix /= perspFactor;

    float fMatrix[12];
    for (int j = 0; j < 12; j++)
      fMatrix[j] = matrix[j/4][j%4];
    OPENCL_CHECK_ERROR( clEnqueueWriteBuffer (m_CommandQueue,
                                              m_DeviceMatrix,
                                              CL_TRUE,
                                              0,
                                              12*sizeof(float),
                                              fMatrix,
                                              0,
                                              NULL,
                                              NULL) );

    const size_t origin[3] = {0,0,0};
    size_t       region[3];
    region[0] = this->GetInput(1)->GetRequestedRegion().GetSize()[0];
    region[1] = this->GetInput(1)->GetRequestedRegion().GetSize()[1];
    region[2] = 1;
    OPENCL_CHECK_ERROR( clEnqueueWriteImage(m_CommandQueue,
                                            m_DeviceProjection,
                                            CL_TRUE,
                                            origin,
                                            region,
                                            0,
                                            0,
                                            projection->GetBufferPointer(),
                                            0,
                                            NULL,
                                            NULL) );

    // Execute kernel
    cl_event events[2];
    size_t   local_work_size = 256;
    size_t   global_work_size = this->GetOutput()->GetRequestedRegion().GetNumberOfPixels();
    OPENCL_CHECK_ERROR( clEnqueueNDRangeKernel(m_CommandQueue,
                                               m_Kernel,
                                               1,
                                               NULL,
                                               &global_work_size,
                                               &local_work_size,
                                               0,
                                               NULL,
                                               &events[0]) );
    OPENCL_CHECK_ERROR( clWaitForEvents(1, &events[0]) );
    OPENCL_CHECK_ERROR( clReleaseEvent(events[0]) );
    }
}

} // end namespace rtk
