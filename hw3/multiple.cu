#include <stdlib.h>
#include <stdio.h>

#include "timer.h"
#include "cuda_utils.h"

typedef float dtype;

#define N_ (8 * 1024 * 1024)
#define MAX_THREADS 256
#define MAX_BLOCKS 64

#define MIN(x,y) ((x < y) ? x : y)


/* return the next power of 2 number that is larger than x */
unsigned int nextPow2( unsigned int x ) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

/* find out # of threads and # thread blocks for a particular kernel */
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (whichKernel < 3)
    {
				/* 1 thread per element */
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
				/* 1 thread per 2 elements */
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
		/* limit the total number of threads */
    if (whichKernel == 5)
        blocks = MIN(maxBlocks, blocks);
}

/* special type of reduction to account for floating point error */
dtype reduce_cpu(dtype *data, int n) {
    dtype sum = data[0];
    dtype c = (dtype)0.0;
    for (int i = 1; i < n; i++)
    {
        dtype y = data[i] - c;
        dtype t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}


__global__ void
kernel5(dtype *g_idata, dtype *g_odata, unsigned int n)
{
	__shared__ dtype scratch[MAX_THREADS];

	unsigned int numOfThreads = blockDim.x * gridDim.x;
	unsigned int bid = blockIdx.x;
	unsigned int i = bid * blockDim.x + threadIdx.x;

	// initialize the scratch to its own element:
	scratch[threadIdx.x] = g_idata[i];
	// loop until you add the required number of elements
	int offset = numOfThreads;
	while( (i + offset) < n) {
		scratch[threadIdx.x] += g_idata[i + offset];
		offset += numOfThreads; // 
	}

	__syncthreads ();

	for(unsigned int s = (blockDim.x >> 1); s > 32; s = s >> 1) {
		if(threadIdx.x < s) {
			scratch[threadIdx.x] += scratch[threadIdx.x + s];
		}
		__syncthreads ();
	}

	if (threadIdx.x < 32) {

		volatile dtype *scratchUnroll = scratch;

		if (n > 64) {
			scratchUnroll[threadIdx.x] += scratchUnroll[threadIdx.x + 32];
		}
		if (n > 32) {
			scratchUnroll[threadIdx.x] += scratchUnroll[threadIdx.x + 16];
		}
		scratchUnroll[threadIdx.x] += scratchUnroll[threadIdx.x + 8];
		scratchUnroll[threadIdx.x] += scratchUnroll[threadIdx.x + 4];
		scratchUnroll[threadIdx.x] += scratchUnroll[threadIdx.x + 2];
		scratchUnroll[threadIdx.x] += scratchUnroll[threadIdx.x + 1];		
	}

	if(threadIdx.x == 0) {
		g_odata[bid] = scratch[0]; // the blocks overwrite the first "numOfBlocks" elements in the output array. each block writes at the block idx location. 
	}
}




int 
main(int argc, char** argv)
{
	int i;

	/* data structure */
	dtype *h_idata, h_odata, h_cpu;
	dtype *d_idata, *d_odata;	

	/* timer */
	struct stopwatch_t* timer = NULL;
	long double t_kernel_5, t_cpu;

	/* which kernel are we running */
	int whichKernel;

	/* number of threads and thread blocks */
	int threads, blocks;

  int N;
  if(argc > 1) {
    N = atoi (argv[1]);
    printf("N: %d\n", N);
  } else {
    N = N_;
    printf("N: %d\n", N);
  }

	/* naive kernel */
	whichKernel = 5;
	getNumBlocksAndThreads (whichKernel, N, MAX_BLOCKS, MAX_THREADS, 
													blocks, threads);

	/* initialize timer */
	stopwatch_init ();
	timer = stopwatch_create ();

	/* allocate memory */
	h_idata = (dtype*) malloc (N * sizeof (dtype));
	CUDA_CHECK_ERROR (cudaMalloc (&d_idata, N * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMalloc (&d_odata, blocks * sizeof (dtype)));

	/* Initialize array */
	srand48(time(NULL));
	for(i = 0; i < N; i++) {
		h_idata[i] = drand48() / 100000;
	}
	CUDA_CHECK_ERROR (cudaMemcpy (d_idata, h_idata, N * sizeof (dtype), 
																cudaMemcpyHostToDevice));
	
	/* ================================================== */
	/* GPU kernel */
	dim3 gb(blocks, 1, 1);
	dim3 tb(threads, 1, 1);

	/* warm up */	
	kernel5 <<<gb, tb>>> (d_idata, d_odata, N);
	cudaThreadSynchronize ();

	stopwatch_start (timer);

	/* execute kernel */
	kernel5 <<<gb, tb>>> (d_idata, d_odata, N);
	//printf("S: %d\r\n", N);
	//printf("Blocks: %d; Threads: %d\r\n", blocks, threads);
	int s = blocks; // here s is 64
	while(s > 1) {
		threads = 0;
		blocks = 0;
		getNumBlocksAndThreads (whichKernel, s, MAX_BLOCKS, MAX_THREADS, 
														blocks, threads);

		dim3 gb(blocks, 1, 1);
		dim3 tb(threads, 1, 1);

		kernel5 <<<gb, tb>>> (d_odata, d_odata, s);

		s = (s + threads * 2 - 1) / (threads * 2);
	}
	cudaThreadSynchronize ();

	t_kernel_5 = stopwatch_stop (timer);
	fprintf (stdout, "Time to execute multiple add GPU reduction kernel: %Lg secs\n", t_kernel_5);

  double bw = (N * sizeof(dtype)) / (t_kernel_5 * 1e9);
  fprintf (stdout, "Effective bandwidth: %.2lf GB/s\n", bw);

	/* copy result back from GPU */
	CUDA_CHECK_ERROR (cudaMemcpy (&h_odata, d_odata, sizeof (dtype), 
																cudaMemcpyDeviceToHost));
	/* ================================================== */

	/* ================================================== */
	/* CPU kernel */
	stopwatch_start (timer);
	h_cpu = reduce_cpu (h_idata, N);
	t_cpu = stopwatch_stop (timer);
	fprintf (stdout, "Time to execute naive CPU reduction: %Lg secs\n",
         	 t_cpu);
	/* ================================================== */

	if(abs (h_odata - h_cpu) > 1e-5) {
    fprintf(stderr, "FAILURE: GPU: %f  CPU: %f\n", h_odata, h_cpu);
	} else {
    printf("SUCCESS: GPU: %f  CPU: %f\n", h_odata, h_cpu);
	}

	return 0;
}