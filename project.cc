#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include <unistd.h>
#include<cmath>
#include<time.h>

using namespace std;

#define DEFAULT_NBINS 256

#define CONV_FACTOR 57.2957795 // 180/pi
#define arcm_to_rad 1/3437.7468  // (1/60)*(pi/180), convert from arcm to rad

// variables for calculate the execution time
static clock_t start_time;
static double elapsed;

int main(int argc, char **argv)
{
    start_time = clock(); // start the timer

    float hist_lower_range = 0.0000001;
    int nbins = DEFAULT_NBINS;
    float hist_bin_width = 0.25;
    float hist_upper_range = nbins*hist_bin_width;

    if (argc < 2) {
        printf("\nMust pass in at least two input files on command line!\n");
        exit(1);
    }

    float *h_alpha0, *h_delta0;
    float *h_alpha1, *h_delta1;

    FILE *infile0, *infile1, *outfile ;
    infile0 = fopen(argv[optind],"r");
    infile1 = fopen(argv[optind+1],"r");
    outfile = fopen("outputc.txt", "w");

    bool two_different_files = 1;
    if (strcmp(argv[1],argv[2])==0)
    {
        two_different_files = 0;
        printf("Using the same file!\n");
    }

    int NUM_GALAXIES0;
    int NUM_GALAXIES1;
    fscanf(infile0, "%d", &NUM_GALAXIES0);

    int size_of_galaxy_array = NUM_GALAXIES0 * sizeof(float);    
    printf("SIZE 0 # GALAXIES: %d\n",NUM_GALAXIES0);

    h_alpha0 = (float*)malloc(size_of_galaxy_array);
    h_delta0 = (float*)malloc(size_of_galaxy_array);
    float temp0, temp1;

    for(int i=0; i<NUM_GALAXIES0; i++)
    {
        fscanf(infile0, "%f %f", &temp0, &temp1);
        h_alpha0[i] = temp0*arcm_to_rad;
        h_delta0[i] = temp1*arcm_to_rad;
        
    }

    fscanf(infile1, "%d", &NUM_GALAXIES1);
    printf("SIZE 1 # GALAXIES: %d\n",NUM_GALAXIES1);

    size_of_galaxy_array = NUM_GALAXIES1 * sizeof(float);    
    h_alpha1 = (float*)malloc(size_of_galaxy_array);
    h_delta1 = (float*)malloc(size_of_galaxy_array);

    for(int i=0; i<NUM_GALAXIES1; i++)
    {
        fscanf(infile1, "%f %f", &temp0, &temp1);
        h_alpha1[i] = temp0*arcm_to_rad;
        h_delta1[i] = temp1*arcm_to_rad;
    }

    unsigned long *hist;
    float hist_min = hist_lower_range;
    float hist_max = hist_upper_range;
    float bin_width = hist_bin_width;

    int size_hist = (nbins+2);
    int size_hist_bytes = size_hist*sizeof(unsigned long);

    hist = (unsigned long*)malloc(size_hist_bytes);
    memset(hist, 0, size_hist_bytes);

    int x, y;
    float dist = 0;

    int bin_index = 0;
    for(int i = 0; i < NUM_GALAXIES0; i++)
    {
        for(int j = 0; j < NUM_GALAXIES1; j++)
        {
           
            bool do_calc = 1;
            if (two_different_files)
            {
                do_calc = 1;
            }
            else // Doing the same file
            {
                if(i > j)
                    do_calc=1;
                else
                    do_calc=0;
            }
            
            if (do_calc)
            {
                float dist, a_diff;

                a_diff = h_alpha0[i] - h_alpha1[j];
                dist = (sin(h_delta0[i])*sin(h_delta1[j]) + cos(h_alpha0[i])*cos(h_alpha1[j])*cos(a_diff));
                
                //I don't knwo why with this code sometimes the dist is bigger than 1, so the arcos(n>1) = NAN
                if (dist > 1)
                    dist = 1;

                dist = acos(dist);
                dist *= CONV_FACTOR;

                if(dist < hist_min)
                    bin_index = 0;
                else if(dist >= hist_max)
                    bin_index = nbins + 1;
                else
                    bin_index = int(dist/bin_width) + 1;

                hist[bin_index]++;
            }
        }
    }  

    unsigned long total = 0;
    float bins_mid = 0;

    float lo = hist_lower_range;
    float hi = 0;
    for(int k=1; k<nbins+1; k++) {
        hi = lo + hist_bin_width;   
            
        fprintf(outfile, "%.3e %.3e %lu \n",lo,hi,hist[k]);
        total += hist[k];

        lo = hi;
    }

    printf("total: %lu \n", total);

    fclose(infile0);
    fclose(infile1);
    fclose(outfile);

    free(h_alpha0);
    free(h_delta0);
    free(h_alpha1);
    free(h_delta1);
    free(hist);

    // total time 
    elapsed = clock() - start_time;
    elapsed = elapsed / CLOCKS_PER_SEC;
    printf("Execution time: %f \n", elapsed);

    return 0;
}  
