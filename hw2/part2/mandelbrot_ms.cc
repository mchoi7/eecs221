/**
 *  \file mandelbrot_ms.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>

#include "render.hh"
#include <mpi.h> 

#define DATA_TAG 1
#define FINISH_TAG 0 

int
mandelbrot(double x, double y) {
  int maxit = 511;
  double cx = x;
  double cy = y;
  double newx, newy;

  int it = 0;
  for (it = 0; it < maxit && (x*x + y*y) < 4; ++it) {
    newx = x*x - y*y + cx;
    newy = 2*x*y + cy;
    x = newx;
    y = newy;
  }
  return it;
}

void slave(int rank, int width, double minX, double jt, double minY, double it) {

	// create the send-buffer:
	int sendbuf[width + 1];

	MPI_Status status;
	int rowToCompute;
	double x, y;

	while (true) {
		
		MPI_Recv(&rowToCompute, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

		if (status.MPI_TAG == FINISH_TAG) {

			return;

		} else {

			y = minY + (rowToCompute * it);
			x = minX;
			for (int p = 0; p < width; ++p) {
				sendbuf[p] = mandelbrot(x, y);
				x += jt;
			}
			sendbuf[width] = rowToCompute; // appending row number to the very end
			MPI_Send(sendbuf, width + 1, MPI_INT, 0, DATA_TAG, MPI_COMM_WORLD);

		}

	}
}

void master(int height, int width, double minX, double jt, double minY, double it, int numOfProc) {

	double t_start = MPI_Wtime();

	// create the final buffer:
	int * finalBuf;
	finalBuf = new int [height*width];

	MPI_Status status;
	int recvbuf[width + 1];
	int nextRow = 0;

	// seed the slaves:
	for (int i = 1; i < numOfProc; i++) {
		MPI_Send(&nextRow, 1, MPI_INT, i, DATA_TAG, MPI_COMM_WORLD);
		nextRow++;
	}

	// loop to get results from slaves and re-assign them new work:
	while (nextRow < height) {
		MPI_Recv(recvbuf, width + 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Send(&nextRow, 1, MPI_INT, status.MPI_SOURCE, DATA_TAG, MPI_COMM_WORLD);
		// store the received result:
		memcpy(finalBuf + (recvbuf[width] * width), recvbuf, width * sizeof(int));
		nextRow++;
	}
	
	/* numOfProc-1 SLAVES ARE GOING TO BE ACTIVE AT ALL TIMES */

	// receive results from trailing work and terminate processes:
	for (int i = 1; i < numOfProc; i++) {
		MPI_Recv(recvbuf, width + 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
		MPI_Send(0, 0, MPI_INT, status.MPI_SOURCE, FINISH_TAG, MPI_COMM_WORLD);
		// store the received result:
		memcpy(finalBuf + (recvbuf[width] * width), recvbuf, width * sizeof(int));
	}

	/* RENDER THE RESULTS */
	gil::rgb8_image_t img(height, width);
  	auto img_view = gil::view(img);
  	for (int k = 0; k < height; ++k) {
      for (int p = 0; p < width; ++p) {
      	img_view(p, k) = render(finalBuf[ (k * width) + p] / 512.0); 
      }
  	}

  	double t_elapsed = MPI_Wtime() - t_start;
  	printf("Total time: %f\r\n", t_elapsed);

  	gil::png_write_view("mandelbrot-ms.png", const_view(img));

}

int
main (int argc, char* argv[])
{
  double minX = -2.1;
  double maxX = 0.7;
  double minY = -1.25;
  double maxY = 1.25;
  
  int height, width;
  if (argc == 3) {
    height = atoi (argv[1]);
    width = atoi (argv[2]);
    assert (height > 0 && width > 0);
  } else {
    fprintf (stderr, "usage: %s <height> <width>\n", argv[0]);
    fprintf (stderr, "where <height> and <width> are the dimensions of the image.\n");
    return -1;
  }

  double it = (maxY - minY)/height;
  double jt = (maxX - minX)/width;

  /* INITIALIZE MPI ENVIRONMENT */
  MPI_Init(&argc, &argv);

  int numOfProc;
  int rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);

  if (rank == 0) {
  	
  	master(height, width, minX, jt, minY, it, numOfProc); // invoke the master

  } else {
  	
  	slave(rank, width, minX, jt, minY, it); // invoke a slave

  }

  MPI_Finalize();

  return 0;

}

/* eof */