/**
 *  \file sort.hh
 *
 *  \brief Interface to sorting arrays of keys ('keytype' values).
 */

#if !defined (INC_SORT_HH)
#define INC_SORT_HH /*!< sort.hh already included */

/** 'keytype' is the primitive type for sorting keys */
typedef unsigned long keytype; // typedef to alias data-types

/**
 *  Sorts an input array containing N keys, A[0:N-1]. The sorted
 *  output overwrites the input array.
 */
void sequentialSort (int N, keytype* A); 										// IMPLEMNTED USING qsort

/**
 *  Sorts an input array containing N keys, A[0:N-1]. The sorted
 *  output overwrites the input array. This is the routine YOU will
 *  implement; see 'parallel-qsort.cc'.
 */
void parallelSort (int N, keytype* A);											// TODO!

/** Returns a new uninitialized array of length N */
keytype* newKeys (int N);														// IMPLEMENTED

/** Returns a new copy of A[0:N-1] */
keytype* newCopy (int N, const keytype* A);										// IMPLEMENTED

/**
 *  Checks whether A[0:N-1] is in fact sorted, and if not, aborts the
 *  program.
 */
void assertIsSorted (int N, const keytype* A); 									// IMPLEMENTED

/**
 *  Checks whether A[0:N-1] == B[0:N-1]. If not, aborts the program.
 */
void assertIsEqual (int N, const keytype* A, const keytype* B); 				// IMPLEMENTED

#endif

/* eof */
