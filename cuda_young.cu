
#include <stdio.h>

const int TILE_DIM = 32;
const int BLOCK_ROWS = 8;

__global__ void transposeCoalesced(double *odata, const double *idata, int rows,int cols)
{
  	__shared__ double tile[TILE_DIM][TILE_DIM+1];

  	int x = blockIdx.x * TILE_DIM + threadIdx.x;
  	int y = blockIdx.y * TILE_DIM + threadIdx.y;

	//  if (x >= cols||y >= rows){
	//      return;
	//  }

  	int maxJ = TILE_DIM;
  	int maxJ2 = TILE_DIM;
  	int otherMaxJ = rows - y;
  	if (maxJ > otherMaxJ)
    	maxJ = otherMaxJ;


  	if ( x < cols ){
    		for (int j = 0; j < maxJ; j += BLOCK_ROWS)
     		tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*cols + x];
  	}
  	__syncthreads();

  	x = blockIdx.y * TILE_DIM + threadIdx.x;  // transpose block offset
  	y = blockIdx.x * TILE_DIM + threadIdx.y;

  	int otherMaxJ2 = cols - y;
  	if (maxJ2 > otherMaxJ2){
      	maxJ2 = otherMaxJ2;
  	}
  	if ( x < rows){
   		for (int j = 0; j < maxJ2; j += BLOCK_ROWS)
      	 	odata[(y+j)*rows + x] = tile[threadIdx.x][threadIdx.y + j];
  	}

}



__global__ void cuconvolve_youngCausal(double * in, double * out, int rows, int columns, double B, double *bf) 
{    


	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    	if(idx<columns)
	{
    		/* Compute first 3 output elements */  
    		out[idx] = B*in[idx];

    		out[idx+columns] = B*in[idx+columns] + bf[2]*out[idx];

    		out[idx+2*columns] = B*in[idx+2*columns] + (bf[1]*out[idx]+bf[2]*out[idx+columns]);
    
    		/* Recursive computation of output in forward direction using filter parameters bf and B */
    		for(int i=3; i<rows; i++) 
   		{
        		out[idx+i*columns] = B*in[idx+i*columns];

        		for(int j=0; j<3; j++) 
	  		{
            		out[idx+i*columns] += bf[j]*out[idx + (i-(3-j))*columns];
        		}
    		}

	}  
}

__global__ void cuconvolve_youngAnticausal(double * in, double * out, int rows, int columns, double B, double *bb) 
{
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    
    	int total = columns*(rows-1);

    	if(idx<columns)
	{
    		/* Compute last 3 output elements */
    		out[total + idx] = B*in[total + idx];

    		out[total + idx - columns] = B*in[total + idx - columns] + bb[0]*out[total + idx];

    		out[total + idx - 2*columns] = B*in[total + idx - 2*columns] + (bb[0]*out[total + idx - columns]+bb[1]*out[total + idx]);
    
    		/* Recursive computation of output in backward direction using filter parameters bb and B */
    		for (int i=3; i<rows-1; i++) 
    		{
        		out[total + idx - i*columns] = B*in[total + idx - i*columns];
        		for (int j=0; j<3; j++) 
	  		{
            		out[total + idx - i*columns] += bb[j]*out[total + idx - (i-(j+1))*columns];
        		}
    		}
   	}       
}

extern "C"
void cudaYoung(double * in, double * out, int rows, int columns, double *bf, double *bb, double B) 
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

    /** \brief Array to store output of Causal filter convolution */

	double *d_input, *d_output, *d_bf, *d_bb;
	cudaMalloc((void**) &d_input, rows*columns*sizeof(double)); 
	cudaMalloc((void**) &d_output, rows*columns*sizeof(double)); 
	
	cudaMalloc((void**) &d_bf, rows*columns*sizeof(double));
	cudaMalloc((void**) &d_bb, rows*columns*sizeof(double));

	cudaMemcpy(d_input, in, rows*columns*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bf, bf, 3*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bb, bb, 3*sizeof(double), cudaMemcpyHostToDevice);


	dim3 dimGrid1((columns+TILE_DIM-1)/TILE_DIM,(rows+TILE_DIM-1)/TILE_DIM, 1);
	dim3 dimGrid2((rows+TILE_DIM-1)/TILE_DIM,(columns+TILE_DIM-1)/TILE_DIM, 1);
	dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);

	// -------- Convolve Rows----------

	transposeCoalesced<<< dimGrid1, dimBlock>>>(d_output, d_input, rows, columns);

    	cuconvolve_youngCausal<<<rows/256 + 1 , 256>>>(d_output, d_input, columns, rows, B, d_bf);

    	cuconvolve_youngAnticausal<<<rows/256 + 1, 256>>>(d_input, d_output, columns, rows, B, d_bb);

	// -------- Convolve Columns ----------

	transposeCoalesced<<< dimGrid2, dimBlock>>>(d_input, d_output, columns, rows);

    	cuconvolve_youngCausal<<<columns/256 + 1, 256>>>(d_input, d_output, rows, columns, B, d_bf);

    	cuconvolve_youngAnticausal<<<columns/256 + 1, 256>>>(d_output, d_input, rows, columns, B, d_bb);

	cudaMemcpy(in, d_input, rows*columns*sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop);

  	cudaEventSynchronize(stop);
  	float milliseconds = 0;
  	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Execution time elapsed: %f ms\n", milliseconds);

	cudaFree(d_input);
	cudaFree(d_output);
	cudaFree(d_bf);
	cudaFree(d_bb);
}



