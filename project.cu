/*--------------------------------------------------------------------------------------------------*/
/*                                                                                                  */
/*       Alberto Quesada Aranda                                                                     */
/*       Åbo Akademi University                                                                     */
/*       Advanced Computer Graphics and Graphics Hardware                                           */
/*                                                                                                  */
/*       Two-Point angula correlation code                                                          */
/*       Input: two list of galaxies                                                                */
/*       Output: .txt file with the data for generate the histogram                                 */
/*                                                                                                  */
/*       Base code taken from: https://github.com/djbard/ccogs/tree/master/angular_correlation      */
/*                                                                                                  */
/*--------------------------------------------------------------------------------------------------*/


#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<math.h>
#include<unistd.h>
#include<cuda_runtime.h>
#include<time.h>

using namespace std;

#define SUBMATRIX_SIZE 16384 // parallel threads: 32 blocks * 512 threads/block = 16384 threads
#define DEFAULT_NBINS 256  // num of bins for the histogram

#define	arcm_to_rad 1/3437.7468  // (1/60)*(pi/180), convert from arcm to rad
#define conv_angle 57.2957795; // 180/pi, convert from rad to degrees

// variables for calculate the execution time
static clock_t start_time;
static double elapsed;

/*------------------------------------------------------------------
	Kernel to calculate angular distances
-------------------------------------------------------------------*/

__global__ void distance(volatile float *a0, volatile float *d0, volatile float *a1, volatile float *d1, int xind, int yind, int max_xind, int max_yind, volatile int *dev_hist, float hist_min, float hist_max, int nbins, float bin_width, bool two_different_files=1) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;  // idx is the thread id, it must range to 32blocks * 512threads/block = 16384 threads
    idx += xind;    // allow the thread to know which submatrix has to calculate

    // printf("%d  %d  %d\n", blockIdx.x, threadIdx.x, blockDim.x);
    // blockIdx.x [0-31]
    // blockDim.x 512
    // threadIdx.x [0-511]

    __shared__ int shared_hist[DEFAULT_NBINS+2];    // shared vector for save the results within a block of threads

    // initialize it only once in each block
    if(threadIdx.x==0)
    {
        for (int i=0;i<nbins+2;i++)
            shared_hist[i] = 0;
    }

    // before starting the calculations we need to be sure that the shared_hist is initialized; if not, we take the risk
    // of loss calculations because we don't know the order in which the threads are executed and is possible that a thread
    // performs the calculations and write the results in shared_hist before the thread 0 has initialized it.
    __syncthreads();

    // if NUM_GALAXIES0 % SUBMATRIX_SIZE != 0 in the last submatrix we will have more threads than needed calculations, therefore
    // we won't perform calculations with those threads.
    if (idx<max_xind) {

        float dist, alpha1, delta1, a_diff;
        float alpha0 = a0[idx];
        float delta0 = d0[idx];

        bool do_calc = 1;
        int bin_index = 0; 

        // each kernel will calculate the angle between one galaxy of the first input data (idx) and all the galaxies within [yind-ymax]
        int ymax = yind + SUBMATRIX_SIZE; // ymax will be the end of the submatrix that we are calculating of the second galaxies input

        // we have to take care and if ymax > NUM_GALAXIES1, we stop the calculations bucle at that point
        if (ymax>max_yind)
            ymax = max_yind;

        // we will perform the same calculation between input0[idx] and every input1[] (range [yind-ymax])
        for(int i=yind; i<ymax; i++) 
        {
            // if the two input files are different (DR case) we have to perform all N*N calculations
            if (two_different_files) {
                do_calc = 1;
            }
            // if the two input files are the same (DD, RR cases) we have to perform N*(N-1)/2 calculations
            else {
                if (xind != yind) {
                    do_calc = 1;
                }
                else {
                    if(idx > i)
                        do_calc=1;
                    else
                        do_calc=0;
                }
            }

            if (do_calc)
            {              
                alpha1 = a1[i];
                delta1 = d1[i];
                a_diff = alpha0 - alpha1;
                dist = acos(sin(delta0)*sin(delta1) + cos(alpha0)*cos(alpha1)*cos(a_diff));
                dist *= conv_angle; // convert from rad to degrees

                // check in which bin we have to include the angle calculated
                if(dist < hist_min) // underflow
                    bin_index = 0; 
                else if(dist >= hist_max) // overflow
                    bin_index = nbins + 1;
                else {
                    bin_index = int((dist-hist_min)/bin_width) + 1;
                }

                // more than one thread could try to write its result in the shared_hist at the same time and in the same memory
                // location. We need an atomic operation for prevent the loss of data
                atomicAdd(&shared_hist[bin_index],1); // increment by one the corresponding histogram bin 

            }
        }
    }

    // before copy the results of each block to the global histogram we have to be sure that all the threads within the block have
    // ended its calculations
    __syncthreads();

    // only one thread (0) will write the results of the block to which it belongs to the global histogram
    // for avoid the need of another atomic operation, our global histogram save the result of each block successively :
    // [block[0], block[1] .... block [31]]
    if(threadIdx.x==0)
    {
        for(int i=0;i<nbins+2;i++)
            dev_hist[i+(blockIdx.x*(nbins+2))]=shared_hist[i]; 
    }

}


/*------------------------------------------------------------------
	Calculations and call to kernel
-------------------------------------------------------------------*/

int calc(FILE *infile0, FILE *infile1, FILE *outfile, int nbins, float hist_lower_range, float hist_upper_range, float hist_bin_width, bool two_different_files){
    
	//d_ means device -> GPU, h_ means host -> CPU
    float *d_alpha0, *d_delta0, *d_alpha1, *d_delta1;
    float *h_alpha0, *h_delta0, *h_alpha1, *h_delta1;
    int NUM_GALAXIES0, NUM_GALAXIES1;

    // reading the data of the input files
    // first we read the number of galaxies of each file
    fscanf(infile0, "%d", &NUM_GALAXIES0);
    fscanf(infile1, "%d", &NUM_GALAXIES1);

    // calculate the size of the array needed for save in memory all the galaxies
    int size_of_galaxy_array0 = NUM_GALAXIES0 * sizeof(float);    
	int size_of_galaxy_array1 = NUM_GALAXIES1 * sizeof(float); 

	printf("SIZE 0 # GALAXIES: %d\n",NUM_GALAXIES0);
  	printf("SIZE 1 # GALAXIES: %d\n",NUM_GALAXIES1);

  	// allocate space for the galaxies data in global memory
    h_alpha0 = (float*)malloc(size_of_galaxy_array0);
    h_delta0 = (float*)malloc(size_of_galaxy_array0);
    h_alpha1 = (float*)malloc(size_of_galaxy_array1);
    h_delta1 = (float*)malloc(size_of_galaxy_array1);

    float temp0, temp1;

    // reading and saving the galaxies data in radians
    for(int i=0; i<NUM_GALAXIES0; i++)
    {
        fscanf(infile0, "%f %f", &temp0, &temp1);
        h_alpha0[i] = temp0 * arcm_to_rad;
        h_delta0[i] = temp1 * arcm_to_rad;
    }

    for(int i=0; i<NUM_GALAXIES1; i++)
    {
        fscanf(infile1, "%f %f", &temp0, &temp1);
        h_alpha1[i] = temp0 * arcm_to_rad;
        h_delta1[i] = temp1 * arcm_to_rad;
    }

    // defining dimensions for the grid and block
    dim3 grid, block;
    grid.x = 8192/(DEFAULT_NBINS); // number of blocks = 32
    block.x = SUBMATRIX_SIZE/grid.x; // number of threads/block = 512

    // allocating the histograms
    /* I will need 3 arrays for the histograms:
        - hist : for each submatrix, save the results of each thread block seccuentially (global memory)
        - dev_hist : the same as hist, but in GPU memory
        - hist_array : save the global result of all the submatrix in global memory
    */

    int *hist, *dev_hist;

    int size_hist = grid.x * (nbins+2); // I use +2, one for underflow and other for overflow
    int size_hist_bytes = size_hist*sizeof(int);

    // allocating and initializing to 0 hist in global mem
    hist = (int*)malloc(size_hist_bytes);
    memset(hist, 0, size_hist_bytes);

    // allocating and initializing to 0 dev_hist in GPU mem
    cudaMalloc((void **) &dev_hist, (size_hist_bytes));
    cudaMemset(dev_hist, 0, size_hist_bytes);

    unsigned long  *hist_array;

    // allocating and initializing to 0 the array for the final histogram (the sum of each submatrix partial result)
    int hist_array_size = (nbins+2) * sizeof(unsigned long);
    hist_array =  (unsigned long*)malloc(hist_array_size);
    memset(hist_array,0,hist_array_size); 

    // allocating memory in GPU for save the galaxies data
    cudaMalloc((void **) &d_alpha0, size_of_galaxy_array0 );
    cudaMalloc((void **) &d_delta0, size_of_galaxy_array0 );
    cudaMalloc((void **) &d_alpha1, size_of_galaxy_array1 );
    cudaMalloc((void **) &d_delta1, size_of_galaxy_array1 );

    // check to see if we allocated enough memory.
    if (0==d_alpha0 || 0==d_delta0 || 0==d_alpha1 || 0==d_delta1 || 0==dev_hist)
    {
        printf("couldn't allocate enough memory in GPU\n");
        return 1;
    }

    // initialize array to all 0's
    /*cudaMemset(d_alpha0,0,size_of_galaxy_array0);
    cudaMemset(d_delta0,0,size_of_galaxy_array0);
    cudaMemset(d_alpha1,0,size_of_galaxy_array1);
    cudaMemset(d_delta1,0,size_of_galaxy_array1);*/

    // copy galaxies data to GPU
    cudaMemcpy(d_alpha0, h_alpha0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta0, h_delta0, size_of_galaxy_array0, cudaMemcpyHostToDevice );
    cudaMemcpy(d_alpha1, h_alpha1, size_of_galaxy_array1, cudaMemcpyHostToDevice );
    cudaMemcpy(d_delta1, h_delta1, size_of_galaxy_array1, cudaMemcpyHostToDevice );

    int x, y;
    int num_submatrices_x = NUM_GALAXIES0 / SUBMATRIX_SIZE;
    int num_submatrices_y = NUM_GALAXIES1 / SUBMATRIX_SIZE;

    // if NUM_GALAXIES % SUBMATRIX_SIZE != 0, we will need one submatrix more (not the whole submatrix) for perform all the calculations
    if (NUM_GALAXIES0%SUBMATRIX_SIZE != 0) {
        num_submatrices_x += 1;
    }
    if (NUM_GALAXIES1%SUBMATRIX_SIZE != 0) {
        num_submatrices_y += 1;
    }

    int bin_index = 0;
    // explanation of the iterations in the documentation
    for(int k = 0; k < num_submatrices_y; k++)
    {
        y = k*SUBMATRIX_SIZE;

        int jmax = 0;
        
        // if the two files are the same, then only loop over the upper half of the matrix of operations
        if (two_different_files == 0)
            jmax = k;

        for(int j = jmax; j < num_submatrices_x; j++)
        {
            x = j*SUBMATRIX_SIZE; 

            // set the histogram to all zeros each time.
            cudaMemset(dev_hist,0,size_hist_bytes);

            // call to the kernel
            distance<<<grid,block>>>(d_alpha0, d_delta0,d_alpha1, d_delta1, x, y, NUM_GALAXIES0, NUM_GALAXIES1, dev_hist, hist_lower_range, hist_upper_range, nbins, hist_bin_width, two_different_files);

            // copy the results from GPU memory to global mem
            cudaMemcpy(hist, dev_hist, size_hist_bytes, cudaMemcpyDeviceToHost);

            // add together the results of each block in a single histogram
            for(int m=0; m<size_hist; m++)
            {
                bin_index = m%(nbins+2); // range it to [0-258]
                hist_array[bin_index] += hist[m];
            }    
        }  
    }

    unsigned long total = 0;

    // write in the output file the range of the bin of the histogram and the number of galaxies included in that range
    // start in k = 1 and finish before nbins+1 for avoid the over/underflow
    float lo = hist_lower_range;
    float hi = 0;
    for(int k=1; k<nbins+1; k++)
    {
        hi = lo + hist_bin_width;

        fprintf(outfile, "%.3e %.3e %lu \n",lo,hi,hist_array[k]);
        total += hist_array[k];

        lo = hi;
    }
    printf("total: %lu \n", total);

    // close opened files
    fclose(infile0);
    fclose(infile1);
    fclose(outfile);

    // free global memory
    free(h_alpha0);
    free(h_delta0);
    free(h_alpha1);
    free(h_delta1);
    free(hist);

    // free GPU memory
    cudaFree(d_alpha0);
    cudaFree(d_delta0);  
    cudaFree(d_alpha1);
    cudaFree(d_delta1);  
    cudaFree(dev_hist);

    return 0;
}  

/*------------------------------------------------------------------
	MAIN
-------------------------------------------------------------------*/

int main(int argc, char **argv) {

    start_time = clock(); // start the timer

    int nbins = DEFAULT_NBINS; // 256 bins --> 0-64º --> 64/0.25 = 256
    float bin_width = 0.25;
    float lower_range = 0.0000001;
    float upper_range = nbins * bin_width; // 256 * 0.25 = 64
    bool different_files = 1;

    if (argc != 3) {
        printf("\nMust pass two input files.\n");
        exit(1);
    }
    
    // opening the input files and creating the output file
    FILE *infile0, *infile1, *outfile;
    infile0 = fopen(argv[optind],"r");
    infile1 = fopen(argv[optind+1],"r");
    outfile = fopen("output.txt", "w");

    // if the input files are the same (DD, RR) the calculations needed are different from the DR case
    if (strcmp(argv[optind],argv[optind+1]) == 0) {
        different_files = 0;
        printf("Input files are the same!\n");
    }

    calc(infile0, infile1, outfile, nbins, lower_range, upper_range, bin_width, different_files);

    // total time 
    elapsed = clock() - start_time;
    elapsed = elapsed / CLOCKS_PER_SEC;
    printf("Execution time: %f \n", elapsed);

    return 0;
}

