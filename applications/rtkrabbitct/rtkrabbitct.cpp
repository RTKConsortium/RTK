/** RabbitCT - Version 1.0

  RabbitCT enables easy benchmarking of backprojection algorithms.
*/

// include the required header files
#include <iostream>
#include <fstream>
#include <stdio.h>

#include "rabbitct.h"

#include "rtkCudaFDKBackProjectionImageFilter.hcu"
#include "rtkMacro.h"

//#define WRITE_OUTPUT
#ifdef WRITE_OUTPUT
#  include <itkImportImageFilter.h>
#  include <itkImageFileWriter.h>
#endif //WRITE_OUTPUT

int    img_dim[2];
int    vol_dim[3];
float *dev_vol;
float *dev_img;
float *dev_matrix;

/** \brief Initialization routine.

  This method is required for initializing the data required for
  backprojection. It is called right before the first iteration.
  Here any time intensive preliminary computations and
  initializations should be performed.
*/
FNCSIGN bool RCTLoadAlgorithm(RabbitCtGlobalData* rcgd)
{
  img_dim[0] = rcgd->S_x;
  img_dim[1] = rcgd->S_y;

  vol_dim[0] = rcgd->L;
  vol_dim[1] = rcgd->L;
  vol_dim[2] = rcgd->L;

  CUDA_reconstruct_conebeam_init(img_dim, vol_dim, dev_vol, dev_img, dev_matrix);

  // calculate the number of voxels
  int N = rcgd->L * rcgd->L * rcgd->L;

  // allocate the required volume
  rcgd->f_L = new float[N];

  return true;
}

/** \brief Finish routine.

  This method is called after the last projection image. Here
  it should be made sure the the rcgd->out_volume pointer
  is set correctly.
*/
FNCSIGN bool RCTFinishAlgorithm(RabbitCtGlobalData* rcgd)
{
  CUDA_reconstruct_conebeam_cleanup(vol_dim, rcgd->f_L, dev_vol, dev_img, dev_matrix);
  return true;
}

/** \brief Cleanup routine.

  This method can be used to clean up the allocated
  data required for backprojection. It is called just before
  the benchmark finishes.
*/
FNCSIGN bool RCTUnloadAlgorithm(RabbitCtGlobalData* rcgd)
{
#ifdef WRITE_OUTPUT
  //Import
  itk::ImportImageFilter<float,3>::RegionType volRegion;

  itk::ImportImageFilter<float,3>::RegionType::IndexType volIndex;
  volIndex.Fill(0.0);
  volRegion.SetIndex(volIndex);

  itk::ImportImageFilter<float,3>::RegionType::SizeType volSize;
  volSize.Fill(rcgd->L);
  volRegion.SetSize(volSize);

  itk::ImportImageFilter<float,3>::Pointer vol = itk::ImportImageFilter<float, 3>::New();
  vol->SetRegion(volRegion);
  vol->SetSpacing(itk::Vector<double, 3>(rcgd->R_L) );
  vol->SetImportPointer(rcgd->f_L, rcgd->L * rcgd->L * rcgd->L, false);

  // Write
  typedef itk::ImageFileWriter<  itk::Image<float,3> > WriterType;
  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName( "rtkrabbitct.mhd" );
  writer->SetInput( vol->GetOutput() );
  writer->Update();
#endif //WRITE_OUTPUT

  // delete the previously allocated volume
  delete [] (rcgd->f_L);
  rcgd->f_L = ITK_NULLPTR;
  return true;
}

/** \brief Backprojection iteration.

  This function is the C++ implementation of the pseudo-code
  in the technical note.
*/
FNCSIGN bool RCTAlgorithmBackprojection(RabbitCtGlobalData* rcgd)
{
  double matrix[12];

  //Transpose
  for (unsigned int j=0; j<3; j++)
    for (unsigned int i=0; i<4; i++)
      matrix[j*4+i] = rcgd->A_n[i*3+j];

  //Transform from volume coordinate to index
  for (unsigned int j=0; j<3; j++)
    {
    for (unsigned int i=0; i<3; i++)
      matrix[j*4+3] += matrix[j*4+i]*rcgd->O_L;
    matrix[j*4+0] *= rcgd->R_L;
    matrix[j*4+1] *= rcgd->R_L;
    matrix[j*4+2] *= rcgd->R_L;
    }

  // Texture coordinates (point 0.0,0.0 is at 0.5,0.5)
  for (unsigned int j=0; j<4; j++)
    for (unsigned int i=0; i<2; i++)
      matrix[i*4+j] += matrix[8+j]*0.5;

  // To float
  float fMatrix[12];
  for(unsigned int i=0; i<12; i++)
    fMatrix[i] = matrix[i];

  CUDA_reconstruct_conebeam (img_dim, vol_dim, rcgd->I_n, fMatrix, dev_vol, dev_img, dev_matrix);

  return true;
}

