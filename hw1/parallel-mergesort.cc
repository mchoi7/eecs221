/**
 *  \file parallel-mergesort.cc
 *
 *  \brief Implement your parallel mergesort in this file.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "sort.hh"

#include <string.h>
#include <omp.h> 

int compare (const void* a, const void* b)
{
  keytype ka = *(const keytype *)a;
  keytype kb = *(const keytype *)b;
  if (ka < kb)
    return -1;
  else if (ka == kb)
    return 0;
  else
    return 1;
} 

void serialMergeForParallel(keytype* T, int p1, int r1, int p2, int r2, keytype* A, int p3) {

	// T[p1] is the first element of the first sub-array
	// T[r1] is the last element of the first sub-array
	// T[p2] is the first element of the second sub-array
	// T[r2] is the last element of the second sub-array

	// A[p3] is the first element where the right value should be written to

	// control variables:
	int i = p1;
	int j = p2;
	int k = p3;

	while ( (i <= r1) && (j <= r2) ) {

		if ( T[i] <= T[j] ) {

			A[k] = T[i];
			i++;

		} else {

			A[k] = T[j];
			j++;

		}

		k++;

	}

	// either array might have elements left
	// if the first array has elements left:
	while (i <= r1) {

		A[k] = T[i];
		i++;
		k++;

	}
	// if the second array has elements left:
	while (j <= r2) {
		
		A[k] = T[j];
		j++;
		k++;

	}

}

void merge(keytype* A, int start, int middle, int end, keytype* temp)
{
	// A[start] is the first element in the first sub-array
	// A[middle] is the last element in the first sub-array
	// A[end] is the last element in the second sub-array
	// temp is the array that will hold all the final elements

	//keytype temp[n1+n2]; // THIS CAN CAUSE STACK-SMASHING ATTACKS FOR REALLY LARGE VALUES OF N1+N2

	// index control variables:
	int i = start; // for first sub-array
	int j = middle + 1; // for second sub-array
	int k = start; // for temp array

	// merge the elements looking at both arrays:
	while ( (i <= middle) && (j <= end) ) {

		if( A[i] <= A[j] ) {

			temp[k] = A[i];
			i++;

		} else {

			temp[k] = A[j];
			j++;

		}

		k++;
	}

	// either array might have elements left
	// if the first array has elements left:
	while (i <= middle) {

		temp[k] = A[i];
		i++;
		k++;

	}
	// if the second array has elements left:
	while (j <= end) {
		
		temp[k] = A[j];
		j++;
		k++;

	}

	// now we need to copy the merged result into the original array:
	memcpy(A + start, temp + start, (end - start + 1) * sizeof(keytype) ); // THIS COULD BE CONTENTION 

}

int max(int a, int b) {
	return (a > b)?a:b;
}

int binarySearch(keytype key, keytype* sub, int p, int r) {
	// sub[p] is the first element of the sub-array
	// sub[r] is the last element of the sub-array

	int low = p;
	int high = max(p, r + 1);
	int mid;

	while (low < high) {
		
		mid = (low + high) / 2;

		if (key <= sub[mid]) {

			high = mid;

		} else {

			low = mid + 1;

		}
	}
	
	return high; // returns the index such that all the elements below this index are lower than the key

}

void parallelMerge(keytype* T, int p1, int r1, int p2, int r2, keytype* A, int p3, int base) {
	// T[p1] is the first element of the first sub-array
	// T[r1] is the last element of the first sub-array
	// T[p2] is the first element of the second sub-array
	// T[r2] is the last element of the second sub-array  

	// T[p1..r1] AND T[p2..r2] are sorted sub-arrays

	// A[p3] is the first element of array that holds the final value

	int n1 = r1 - p1 + 1; // number of elements in the first sub-array
	int n2 = r2 - p2 + 1; // number of elements in the second sub-array

	int N = n1 + n2;

	if (N <= base) {
		// serialize the merge:
		serialMergeForParallel(T, p1, r1, p2, r2, A, p3);

	} else {
		// divide and conquer the merge operation:

		if (n1 < n2) { // complying to our assumption

		  int temp;

		  // exchange p1 and p2
		  temp = p1;
		  p1 = p2;
		  p2 = temp;

		  // exchange r1 and r2 
		  temp = r1;
		  r1 = r2;
		  r2 = temp;

		  // exchange n1 and n2
		  temp = n1;
		  n1 = n2;
		  n2 = temp;

		}

		if (n1 == 0) { // if both the arrays are empty

			return;

		} else {

			int q1 = (p1 + r1) / 2;
			int q2 = binarySearch(T[q1], T, p2, r2);
			int q3 = p3 + (q1 - p1) + (q2 - p2);
			A[q3] = T[q1];

			#pragma omp task
      {
			  parallelMerge(T, p1, q1 - 1, p2, q2 - 1, A, p3, base);
      }
			parallelMerge(T, q1 + 1, r1, q2, r2, A, q3 + 1, base); 
      #pragma omp taskwait
		}
	}

}

void mergeSort(keytype* A, int start, int end, keytype* temp, int base) {

	int numel = end - start + 1;

	if ( numel <= base) {

		// sort the sub-array in place and return:
		qsort(A + start, numel, sizeof(keytype), compare);
		return;

	} else {

		// divide up the task:

		// A[middle] is the last element of the first sub-array 
		int middle = start + (end - start) / 2; // want to avoid (start + end) /2

    #pragma omp task
    {
		  mergeSort(A, start, middle, temp, base); // recursively divide up the first sub-array on a separate thread
		}
    mergeSort(A, middle + 1, end, temp, base); // recursively divide up the second sub-array on the same thread
    
    #pragma omp taskwait // wait for both tasks to sync up here before proceding to the merge
		//merge(A, start, middle, end, temp); // merge the sorted sub-arrays sequentially
		parallelMerge(A, start, middle, middle + 1, end, temp, start, base); // merge the sorted sub-arrays parallely
		memcpy(A + start, temp + start, (end - start + 1) * sizeof(keytype));
	}

}

void parallelSort (int N, keytype* A) // the result goes into A
{
	keytype* temp;
	temp = new keytype[N]; // on heap of master, shared by all threads

  #pragma omp parallel
  {
  	int numOfThreads = omp_get_num_threads();
    #pragma omp master
    {
      printf("Number of threads spawned: %d\n", numOfThreads);
    }

	  #pragma omp single
    {
      mergeSort(A, 0, N-1, temp, N/numOfThreads); // call the recursive mergeSort with a single thread
    }

  }

}
/* eof */
