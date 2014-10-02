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
#ifndef __rtkForwardWarpImageFilter_txx
#define __rtkForwardWarpImageFilter_txx
#include "rtkForwardWarpImageFilter.h"

// Put the helper classes in an anonymous namespace so that it is not
// exposed to the user

namespace
{
//nameless namespace

//=========================================================================================================================
//helper class 1 to allow a threaded execution: add contributions of input to output and update weights
//=========================================================================================================================
template<class InputImageType, class OutputImageType, class DeformationFieldType> class HelperClass1 : public itk::ImageToImageFilter<InputImageType, OutputImageType>
{

public:
  /** Standard class typedefs. */
  typedef HelperClass1  Self;
  typedef itk::ImageToImageFilter<InputImageType,OutputImageType> Superclass;
  typedef itk::SmartPointer<Self>         Pointer;
  typedef itk::SmartPointer<const Self>   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro( HelperClass1, ImageToImageFilter );

  /** Constants for the image dimensions */
  itkStaticConstMacro(ImageDimension, unsigned int,InputImageType::ImageDimension);


  //Typedefs
  typedef typename OutputImageType::PixelType        OutputPixelType;
  typedef itk::Image<double, ImageDimension > WeightsImageType;
  typedef itk::Image<itk::SimpleFastMutexLock, ImageDimension > MutexImageType;
  //===================================================================================
  //Set methods
  void SetWeights(const typename WeightsImageType::Pointer input) {
    m_Weights = input;
    this->Modified();
  }
  void SetDeformationField(const typename DeformationFieldType::Pointer input) {
    m_DeformationField=input;
    this->Modified();
  }
  void SetMutexImage(const typename MutexImageType::Pointer input) {
    m_MutexImage=input;
    this->Modified();
    m_ThreadSafe=true;
  }

  //Get methods
  typename WeightsImageType::Pointer GetWeights() {
    return m_Weights;
  }

  /** Typedef to describe the output image region type. */
  typedef typename OutputImageType::RegionType OutputImageRegionType;

protected:
  HelperClass1();
  ~HelperClass1() {};

  //the actual processing
  void BeforeThreadedGenerateData();
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));

  //member data
  typename  itk::Image< double, ImageDimension>::Pointer m_Weights;
  typename DeformationFieldType::Pointer m_DeformationField;
  typename MutexImageType::Pointer m_MutexImage;
  bool m_ThreadSafe;

};



//=========================================================================================================================
//Member functions of the helper class 1
//=========================================================================================================================


//=========================================================================================================================
//Empty constructor
template<class InputImageType, class OutputImageType, class DeformationFieldType >
HelperClass1<InputImageType, OutputImageType, DeformationFieldType>::HelperClass1()
{
  m_ThreadSafe=false;
}


//=========================================================================================================================
//Before threaded data
template<class InputImageType, class OutputImageType, class DeformationFieldType >
void HelperClass1<InputImageType, OutputImageType, DeformationFieldType>::BeforeThreadedGenerateData()
{
  //Since we will add, put to zero!
  this->GetOutput()->FillBuffer(itk::NumericTraits<double>::Zero);
  this->GetWeights()->FillBuffer(itk::NumericTraits<double>::Zero);
}


//=========================================================================================================================
//update the output for the outputRegionForThread
template<class InputImageType, class OutputImageType, class DeformationFieldType >
void HelperClass1<InputImageType, OutputImageType, DeformationFieldType>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId) )
{

  //Get pointer to the input
  typename InputImageType::ConstPointer inputPtr = this->GetInput();

  //Get pointer to the output
  typename OutputImageType::Pointer outputPtr = this->GetOutput();
  //typename OutputImageType::SizeType size=outputPtr->GetLargestPossibleRegion().GetSize();

  //Iterators over input and deformation field
  typedef itk::ImageRegionConstIteratorWithIndex<InputImageType> InputImageIteratorType;
  typedef itk::ImageRegionIterator<DeformationFieldType> DeformationFieldIteratorType;

  //define them over the outputRegionForThread
  InputImageIteratorType inputIt(inputPtr, outputRegionForThread);
  DeformationFieldIteratorType fieldIt(m_DeformationField,outputRegionForThread);

  //Initialize
  typename InputImageType::IndexType index;
  itk::ContinuousIndex<double,ImageDimension> contIndex;
  typename InputImageType::PointType point;
  typedef typename DeformationFieldType::PixelType DisplacementType;
  DisplacementType displacement;
  fieldIt.GoToBegin();
  inputIt.GoToBegin();

  //define some temp variables
  signed long baseIndex[ImageDimension];
  double distance[ImageDimension];
  for(unsigned int i=0; i<ImageDimension; i++) distance[i] = 0.0; // to avoid warning
  unsigned int dim, counter, upper;
  double overlap, totalOverlap;
  typename OutputImageType::IndexType neighIndex;

  //Find the number of neighbors
  unsigned int neighbors =  1 << ImageDimension;


  //==================================================================================================
  //Loop over the region and add the intensities to the output and the weight to the weights
  //==================================================================================================
  while( !inputIt.IsAtEnd() ) {

    // get the input image index
    index = inputIt.GetIndex();
    inputPtr->TransformIndexToPhysicalPoint( index, point );

    // get the required displacement
    displacement = fieldIt.Get();

    // compute the required output image point
    for(unsigned int j = 0; j < ImageDimension; j++ ) point[j] += displacement[j];


    // Update the output and the weights
    if(outputPtr->TransformPhysicalPointToContinuousIndex(point, contIndex ) ) {
      for(dim = 0; dim < ImageDimension; dim++) {
        // The following  block is equivalent to the following line without
        // having to call floor. For positive inputs!!!
        // baseIndex[dim] = (long) vcl_floor(contIndex[dim] );
        baseIndex[dim] = (long) contIndex[dim];
        distance[dim] = contIndex[dim] - double( baseIndex[dim] );
      }

      //Add contribution for each neighbor
      totalOverlap = itk::NumericTraits<double>::Zero;
      for( counter = 0; counter < neighbors ; counter++ ) {
        overlap = 1.0;          // fraction overlap
        upper = counter;  // each bit indicates upper/lower neighbour

        // get neighbor index and overlap fraction
        for( dim = 0; dim < ImageDimension; dim++ ) {
          if ( upper & 1 ) {
            neighIndex[dim] = baseIndex[dim] + 1;
            overlap *= distance[dim];
          } else {
            neighIndex[dim] = baseIndex[dim];
            overlap *= 1.0 - distance[dim];
          }
          upper >>= 1;
        }

        //Set neighbor value only if overlap is not zero
        if( (overlap>0.0)) // &&
          // 			(static_cast<unsigned int>(neighIndex[0])<size[0]) &&
          // 			(static_cast<unsigned int>(neighIndex[1])<size[1]) &&
          // 			(static_cast<unsigned int>(neighIndex[2])<size[2]) &&
          // 			(neighIndex[0]>=0) &&
          // 			(neighIndex[1]>=0) &&
          // 			(neighIndex[2]>=0) )
        {

          if (! m_ThreadSafe) {
            //Set the pixel and weight at neighIndex
            outputPtr->SetPixel(neighIndex, outputPtr->GetPixel(neighIndex) + overlap * static_cast<OutputPixelType>(inputIt.Get()));
            m_Weights->SetPixel(neighIndex, m_Weights->GetPixel(neighIndex) + overlap);

          } else {
            //Entering critilal section: shared memory
            m_MutexImage->GetPixel(neighIndex).Lock();

            //Set the pixel and weight at neighIndex
            outputPtr->SetPixel(neighIndex, outputPtr->GetPixel(neighIndex) + overlap * static_cast<OutputPixelType>(inputIt.Get()));
            m_Weights->SetPixel(neighIndex, m_Weights->GetPixel(neighIndex) + overlap);

            //Unlock
            m_MutexImage->GetPixel(neighIndex).Unlock();

          }
          //Add to total overlap
          totalOverlap += overlap;
        }

        //check for totaloverlap: not very likely
        if( totalOverlap == 1.0 ) {
          // finished
          break;
        }
      }
    }

    ++fieldIt;
    ++inputIt;
  }


}



//=========================================================================================================================
//helper class 2 to allow a threaded execution of normalisation by the weights
//=========================================================================================================================
template<class InputImageType, class OutputImageType>
class HelperClass2 : public itk::ImageToImageFilter<InputImageType, OutputImageType>
{

public:
  /** Standard class typedefs. */
  typedef HelperClass2  Self;
  typedef itk::ImageToImageFilter<InputImageType,OutputImageType> Superclass;
  typedef itk::SmartPointer<Self>         Pointer;
  typedef itk::SmartPointer<const Self>   ConstPointer;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Run-time type information (and related methods) */
  itkTypeMacro( HelperClass2, ImageToImageFilter );

  /** Constants for the image dimensions */
  itkStaticConstMacro(ImageDimension, unsigned int,InputImageType::ImageDimension);

  //Typedefs
  typedef typename  InputImageType::PixelType        InputPixelType;
  typedef typename  OutputImageType::PixelType        OutputPixelType;
  typedef itk::Image<double, ImageDimension > WeightsImageType;


  //Set methods
  void SetWeights(const typename WeightsImageType::Pointer input) {
    m_Weights = input;
    this->Modified();
  }
  void SetEdgePaddingValue(OutputPixelType value) {
    m_EdgePaddingValue = value;
    this->Modified();
  }

  /** Typedef to describe the output image region type. */
  typedef typename OutputImageType::RegionType OutputImageRegionType;

protected:
  HelperClass2();
  ~HelperClass2() {};


  //the actual processing
  virtual void ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId));


  //member data
  typename     WeightsImageType::Pointer m_Weights;
  OutputPixelType m_EdgePaddingValue;
} ;



//=========================================================================================================================
//Member functions of the helper class 2
//=========================================================================================================================


//=========================================================================================================================
//Empty constructor
template<class InputImageType, class OutputImageType >
HelperClass2<InputImageType, OutputImageType>::HelperClass2()
{
  m_EdgePaddingValue=static_cast<OutputPixelType>(0.0);
}


//=========================================================================================================================
//update the output for the outputRegionForThread
template<class InputImageType, class OutputImageType > void
HelperClass2<InputImageType, OutputImageType>::ThreadedGenerateData(const OutputImageRegionType& outputRegionForThread, itk::ThreadIdType itkNotUsed(threadId) )
{

  //Get pointer to the input
  typename InputImageType::ConstPointer inputPtr = this->GetInput();

  //Get pointer to the output
  typename OutputImageType::Pointer outputPtr = this->GetOutput();

  //Iterators over input, weigths  and output
  typedef itk::ImageRegionConstIterator<InputImageType> InputImageIteratorType;
  typedef itk::ImageRegionIterator<OutputImageType> OutputImageIteratorType;
  typedef itk::ImageRegionIterator<WeightsImageType> WeightsImageIteratorType;

  //define them over the outputRegionForThread
  OutputImageIteratorType outputIt(outputPtr, outputRegionForThread);
  InputImageIteratorType inputIt(inputPtr, outputRegionForThread);
  WeightsImageIteratorType weightsIt(m_Weights, outputRegionForThread);


  //==================================================================================================
  //loop over the output and normalize the input, remove holes
  OutputPixelType neighValue;
  double zero = itk::NumericTraits<double>::Zero;
  while (!outputIt.IsAtEnd()) {
    //the weight is not zero
    if (weightsIt.Get() != zero) {
      //divide by the weight
      outputIt.Set(static_cast<OutputPixelType>(inputIt.Get()/weightsIt.Get()));
    }

    //copy the value of the  neighbour that was just processed
    else {
      if(!outputIt.IsAtBegin()) {
        //go back
        --outputIt;
        neighValue=outputIt.Get();
        ++outputIt;
        outputIt.Set(neighValue);
      } else {
        //DD("is at begin, setting edgepadding value");
        outputIt.Set(m_EdgePaddingValue);
      }
    }
    ++weightsIt;
    ++outputIt;
    ++inputIt;

  }//end while
}//end member


}//end nameless namespace



namespace rtk
{

//=========================================================================================================================
// The rest is the ForwardWarpImageFilter
//=========================================================================================================================

//=========================================================================================================================
//constructor
template <class InputImageType, class OutputImageType, class DeformationFieldType>
ForwardWarpImageFilter<InputImageType, OutputImageType, DeformationFieldType>::ForwardWarpImageFilter()
{
  // mIsUpdated=false;
  m_NumberOfThreadsIsGiven=false;
  m_EdgePaddingValue=static_cast<PixelType>(0.0);
  m_ThreadSafe=false;
  m_Verbose=false;
}


//=========================================================================================================================
//Update
template <class InputImageType, class OutputImageType, class DeformationFieldType>
void ForwardWarpImageFilter<InputImageType, OutputImageType, DeformationFieldType>::GenerateData()
{

  //Get the properties of the input
  typename InputImageType::ConstPointer inputPtr=this->GetInput();
  typename WeightsImageType::RegionType region;
  typename WeightsImageType::RegionType::SizeType size=inputPtr->GetLargestPossibleRegion().GetSize();
  region.SetSize(size);
  typename OutputImageType::IndexType start;
  for (unsigned int i =0; i< ImageDimension ; i ++)start[i]=0;
  region.SetIndex(start);

  //Allocate the weights
  typename WeightsImageType::Pointer weights=ForwardWarpImageFilter::WeightsImageType::New();
  weights->SetRegions(region);
  weights->Allocate();
  weights->SetSpacing(inputPtr->GetSpacing());


  //===========================================================================
  //warp is divided in in two loops, for each we call a threaded helper class
  //1. Add contribution of input to output and update weights
  //2. Normalize the output by the weight and remove holes
  //===========================================================================

  //===========================================================================
  //1. Add contribution of input to output and update weights

  //Define an internal image type in double  precision
  typedef itk::Image<double, ImageDimension> InternalImageType;

  //Call threaded helper class 1
  typedef HelperClass1<InputImageType, InternalImageType, DeformationFieldType> HelperClass1Type;
  typename HelperClass1Type::Pointer helper1=HelperClass1Type::New();

  //Set input
  if(m_NumberOfThreadsIsGiven)helper1->SetNumberOfThreads(m_NumberOfThreads);
  helper1->SetInput(inputPtr);
  helper1->SetDeformationField(m_DeformationField);
  helper1->SetWeights(weights);

  //Threadsafe?
  if(m_ThreadSafe) {
    //Allocate the mutex image
    typename MutexImageType::Pointer mutex=ForwardWarpImageFilter::MutexImageType::New();
    mutex->SetRegions(region);
    mutex->Allocate();
    mutex->SetSpacing(inputPtr->GetSpacing());
    helper1->SetMutexImage(mutex);
    if (m_Verbose) std::cout <<"Forwarp warping using a thread-safe algorithm" <<std::endl;
  } else  if(m_Verbose)std::cout <<"Forwarp warping using a thread-unsafe algorithm" <<std::endl;

  //Execute helper class
  helper1->Update();

  //Get the output
  typename InternalImageType::Pointer temp= helper1->GetOutput();

  //For clarity
  weights=helper1->GetWeights();


  //===========================================================================
  //2. Normalize the output by the weights and remove holes
  //Call threaded helper class
  typedef HelperClass2<InternalImageType, OutputImageType> HelperClass2Type;
  typename HelperClass2Type::Pointer helper2=HelperClass2Type::New();

  //Set temporary output as input
  if(m_NumberOfThreadsIsGiven)helper2->SetNumberOfThreads(m_NumberOfThreads);
  helper2->SetInput(temp);
  helper2->SetWeights(weights);
  helper2->SetEdgePaddingValue(m_EdgePaddingValue);

  //Execute helper class
  helper2->Update();

  //Set the output
  this->SetNthOutput(0, helper2->GetOutput());
}

}

#endif
