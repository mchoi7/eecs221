/**
 *  \file mandelbrot_joe.cc
 *
 *  \brief Implement your parallel mandelbrot set in this file.
 */

#include <iostream>
#include <cstdlib>

#include "render.hh"
#include <mpi.h>

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
  double x, y;

  /* INITIALIZE MPI ENVIRONMENT */
  MPI_Init(&argc, &argv);

  double t_start, t_elapsed; // timing variables

  int numOfProc;  // total number of processes
  int rank;       // id of the process
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numOfProc);

  if (rank == 0) {
  	t_start = MPI_Wtime(); // start the timer on the root process
  	printf("Number of MPI processes: %d\r\n", numOfProc);
  }

  int rowsPerProc = height/numOfProc;
  int dataCnt = rowsPerProc * width; // send-buffer size
  int sendbuf[dataCnt];

  /* PER PROCESS COMPUTATION */
  // process rank should compute on rows: rank * (height / numOfProc), rank * (height / numOfProc) + 1, ..., rank * (height / numOfProc) + (height / numOfProc - 1)

  y = minY + (rank * rowsPerProc * it); // starting y
  for (int i = 0; i < rowsPerProc; ++i) {
  	x = minX;
  	for (int j = 0; j < width; ++j) {
  		sendbuf[ (i * width) + j] = mandelbrot(x, y);
  		x += jt;
  	}
  	y += it;
  }

  int * trailingRows = NULL;
  int numOfTrailingRows;
  // only root will operate on traling rows:
  if (rank == 0) {
  	numOfTrailingRows = height - (rowsPerProc * numOfProc); // compute the number of trailing rows
  	trailingRows = new int[numOfTrailingRows * width]; // buffer for the trailing rows

  }

  /* BARRIER */
  MPI_Barrier(MPI_COMM_WORLD);

  /* GATHERING DATA FROM ALL PROCESSES */
  int * recvbuf = NULL;
  if (rank == 0) {
  	recvbuf = new int[numOfProc * dataCnt];
  }
  MPI_Gather(sendbuf, dataCnt, MPI_INT, recvbuf, dataCnt, MPI_INT, 0, MPI_COMM_WORLD);

  /* ROOT PROCESSES ALL RESULTS */
  if (rank == 0) {

  	gil::rgb8_image_t img(height, width);
    auto img_view = gil::view(img);

    // aggregate data from different processes:
    for (int k = 0; k < (rowsPerProc * numOfProc); ++k) {
    	for (int p = 0; p < width; ++p) {
    		img_view(p, k) = render(recvbuf[ (k * width) + p] / 512.0);
    	}
    }
    // deal with trailing rows, if any:
    for (int k = 0; k < numOfTrailingRows; ++k) {
    	for (int p = 0; p < width; ++p) {
    		img_view(p, k + (rowsPerProc * numOfProc) ) = render(trailingRows[ (k * width) + p] / 512.0);
    	}
    }

    t_elapsed = MPI_Wtime () - t_start; // compute the overall time taken
    printf("Total time: %f\r\n", t_elapsed);

    gil::png_write_view("mandelbrot-joe.png", const_view(img));

  }

  MPI_Finalize();

  return 0;
}

/* eof */

