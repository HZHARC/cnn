#include"convolution.h"


__global__
void convolution(  double ** d_dataBuffer, 
                   double ** d_outputBuffer, 
                   double ** d_kernels, 
                   size_t dataBufferSize,
                   size_t outputBufferSize,
                   size_t kernelSize){


        size_t mydataIdx = blockIdx.x / kernelSize;
        size_t mykernelIdx = blockIdx.x % kernelSize;
        size_t myoutputIdx = blockIdx.x;
        

        __shared__ double s_kernel[KERNEL_SIZE];
        if ( threadIdx.x < KERNEL_SIZE)
                s_kernel[threadIdx.x] = d_kernels[mykernelIdx][threadIdx.x];
        __syncthreads();


        // we may encounter with negative index, so all int here
        int myIdx = 0; // the pixel index within the single image in the 1d array
        // the pixel's location in the image
        int myX = 0;
        int myY = 0;

        // the relative coordinates for each pixel
        //int relativeX = 0;
        //int relativeY = 0;


        // and note the top left pixel is considered as the origin.
        int offsetX = 0;
        int offsetY = 0;
        // pixel index -- the absolute coordinates for the pixel
        int pixelX = 0; 
        int pixelY = 0;

        /* value containers used in the loop */
        double sum = 0; 


        for ( size_t iLoops = 0; iLoops < 75 ; ++ iLoops){
                myIdx = iLoops * blockDim.x + threadIdx.x;
                myX = myIdx % IMG_WIDTH;
                myY = myIdx / IMG_WIDTH;
                offsetX = myX - (KERNEL_WIDTH - 1) / 2;
                offsetY = myY - (KERNEL_HEIGHT - 1) / 2;
                sum = 0;
                
                for ( size_t iPixels = 0; iPixels < KERNEL_SIZE; ++iPixels){


                        pixelX = iPixels % KERNEL_WIDTH + offsetX;
                        pixelY = iPixels / KERNEL_WIDTH + offsetY;

                        if ( pixelX >= 0 && pixelY >= 0)          // only update the sum when the pixel coordinate is valid -- considering the out-bound pixels as zero
                                sum += d_dataBuffer[mydataIdx][pixelY * IMG_WIDTH + pixelX] * s_kernel[iPixels];

                }
                // write the result to the output array
                d_outputBuffer[myoutputIdx][myIdx] = sum / KERNEL_SIZE;
        }

}
void batch(vector< vector < Matrix<Double, Dynamic, Dynamic, RowMajor> > > dataset, vector<  Matrix<Double, Dynamic, Dynamic, RowMajor> > kernels, vector < vector < Matrix<Double, Dynamic, Dynamic, RowMajor> > > output ){

        

        /* STEP 0 : Memory Allocation */
        // CPU
        size_t inputDatasetSize = dataset.size() * dataset[0].size() ;  // the total number of pictures in the input dataset
        size_t outputDatasetSize = dataset.size() * dataset[0].size() * kernels.size() ;   // the total number of pictures in the output dataset

        size_t kernelSize = kernels.size();
        size_t dataBufferSize = BLKNUM / kernelSize;  // the number of images we deal with in each kernel call (each channel is considered as a image here and this arrangement may make some blocks idle, but the number of idle blocks is less than the number of kernels;
        size_t outputBufferSize = dataBufferSize * kernelSize;

                // note BLKNUM is the total number of blocks available on the card , and it's defined in the .h file
        double** h_dataBuffer = new double*[dataBufferSize]; // this array stores the address of the leading element for each matrix that we are going to the GPU in each kernel call
        double** h_outputStorage = new double*[outputBufferSize];  // similar as above, storing the address of the leading entry of the output container

        // GPU
        // we apply similar memory management as what we did to the host
        double ** d_kernels = new double*[kernelSize];
        double ** d_dataBuffer = new double*[dataBufferSize];
        double ** d_outputBuffer = new double*[outputBufferSize];

        // allocating and preloading the kernels to the gpu memory
        for ( size_t i = 0; i < kernelSize; ++i){
                checkCudaErrors(cudaMalloc(&d_kernels[i], sizeof(double) * KERNEL_WIDTH *  KERNEL_HEIGHT));
                checkCudaErrors(cudaMemcpy(d_kernels[i], kernels[i].data(), sizeof(double) * KERNEL_WIDTH * KERNEL_HEIGHT, cudaMemcpyHostToDevice));
        }
        
        for (size_t i = 0 ; i < dataBufferSize; ++i){
                checkCudaErrors(cudaMalloc(&d_dataBuffer[i], sizeof(double) * IMG_WIDTH * IMG_HEIGHT));
        }

        for ( size_t i = 0; i < outputBufferSize; ++i){
                checkCudaErrors(cudaMalloc(&d_outputBuffer[i], sizeof(double) * IMG_HEIGHT * IMG_WIDTH));
        }


        /* STEP 2: Convolving */
        

        size_t cursor = 0; // the index of the NEXT image to be sent to the GPU. Assuming that the inner vector sizes are all the same -- the number of channels for each data image are the same
        size_t NumChannelsPerImg = dataset[0].size();

        /* the transformation formulas are : 
         * cursor = outervectorIdx * NumChannelsPerImg + innervectorIdx
         * outervectorIdx = cursor / NumChannelsPerImg
         * innervectorIdx = cursor % NumChannelsPerImg */

        size_t inputStartingIdx = 0;
        size_t totalNumIteration = inputDatasetSize / dataBufferSize; // then in this case totalNumIteration * dataBufferSize should be smaller than the inputDatasetSize, we will deal with the left overs at the end

        
        for (size_t i = 0; i < totalNumIteration; ++i){
                // first copy the data to GPU
                inputStartingIdx = cursor; // the index of the leading input image for this loop
                // then the index of the last image is inputStartingIdx + dataBufferSize - 1
                for (size_t j = 0; j < dataBufferSize; ++j){
                        checkCudaErrors(cudaMemcpy(d_dataBuffer[j], dataset[cursor/NumChannelsPerImg][cursor % NumChannelsPerImg].data(), sizeof(double) * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice));
                        ++cursor;
                }
                // then launching kernel to do the computation
                convolution<<<BLKNUM, 1024>>>(d_dataBuffer, d_outputBuffer, d_kernels, dataBufferSize, dataBufferSize * kernelSize, kernelSize);


                // finally copy the result out
               // in the gpu, we store the convolved picture derived from the same image close to each other. eg, if the input image sequence is ABC and there are three kernel in total, then the output sequence is A1A2A3B1B2B3C1C2C3, and we need to convert the order to A1B1C1A2B2C2....
                for ( size_t j = 0; j < outputDatasetSize; ++j){
                        // need to check : the function call to write into eigen
                       // the second class image index is j/kernelSize + inputStartingIdx and the kernel index is j % kernelSize 
                        // then the outervectorIdx =( j/kernelSize + inputStartingIdx) /NumChannelsPerImg 
                        // and the innervectorIdx = j % kernelSize * NumChannelsPerImg + ( j / kernelSize + inputStartingIdx) % NumChannelsPerImg
                       checkCudaErrors(cudaMemcpy( output[ (j/kernelSize + inputStartingIdx) /NumChannelsPerImg][j % kernelSize * NumChannelsPerImg + ( j / kernelSize + inputStartingIdx) % NumChannelsPerImg].data(), d_outputBuffer[j], sizeof(double) * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDeviceToHost)) ;
                }

        }

        // deal with the leftovers
        // the number of leftovers is less than kernelSize
        // copy in 
        size_t lastInputSize = inputDatasetSize - cursor;
        for (size_t i = 0; i < lastInputSize; ++i){
                checkCudaErrors(cudaMemcpy(d_dataBuffer[i], dataset[(cursor + i)/NumChannelsPerImg][(cursor + i ) % NumChannelsPerImg].data(), sizeof(double) * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyHostToDevice));
        }
        // launch kernel
        convolution<<<lastInputSize, 1024>>>(d_dataBuffer, d_outputBuffer, d_kernels, lastInputSize, lastInputSize * kernelSize, kernelSize);

        // copy out
        for (size_t i = 0; i < lastInputSize * kernelSize; ++i){
                checkCudaErrors(cudaMemcpy( output[ (i/kernelSize + cursor) /NumChannelsPerImg][i % kernelSize * NumChannelsPerImg + ( i / kernelSize + cursor) % NumChannelsPerImg].data(), d_outputBuffer[i], sizeof(double) * IMG_HEIGHT * IMG_WIDTH, cudaMemcpyDeviceToHost)) ;
        }

        return;

}
