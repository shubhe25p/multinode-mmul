//
// (C) 2021, E. Wes Bethel
// mpi_2dmesh.cpp

// usage:
//      mpi_2dmesh [args as follows]
// 
// command line arguments: 
//    -g 1|2|3  : domain decomp method, where 1=row-slab decomp, 2=column-slab decomp, 
//          3=tile decomp (OPTIONAL, default is -g 1, row-slab decomp)
//    -i filename : name of datafile containing raw unsigned bytes 
//          (OPTIONAL, default input filename in mpi_2dmesh.hpp)
//    -x XXX : specify the number of columns in the mesh, the width (REQUIRED)
//    -y YYY : specify the number of rows in the mesh, the height (REQUIRED)
//    -o filename : where output results will be written (OPTIONAL, default output
//          filename in mpi_2dmesh.hpp)
//    -a 1|2 : the action to perform: 1 means perform per-tile processing then write
//       output showing results, 2 means generate an output file with cells labeled as to
//       which rank they belong to (depends on -g 1|2|3 setting)
//       (OPTIONAL, default: -a 1, perform the actual processing)
//    -v : a flag that will trigger printing out the 2D vector array of Tile2D (debug)
//       (OPTIONAL, default value is no verbose debug output)
//
// Assumptions:
//
// Grid decompositions:
//       When creating tile-based decomps, will compute the number of tiles in each
//       dimension as sqrt(nranks); please use a number of ranks that is the square
//       of an integer, e.g., 4, 9, 16, 25, etc.

#include <iostream>
#include <vector>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <math.h>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <random>
#include <string.h>
#include <cblas.h>

#include "mpi_2dmesh.hpp"  // for AppState and Tile2D class

#define DEBUG_TRACE 0 

void fill(double* p, int n) {
    static std::random_device rd;
    static std::default_random_engine gen(rd());
    static std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (int i = 0; i < n; ++i)
        p[i] = i%14;
}

void printArray(double *A, int n, int m)
{
   for (int i=0;i<m;i++)
   {
      for (int j=0;j<n;j++)
      {
         printf("%6.4f ", A[i*n+j]);
      }
      printf("\n");
   }
   printf("\n");
}
int
parseArgs(int ac, char *av[], AppState *as)
{
   int rstat = 0;
   int c;

   while ( (c = getopt(ac, av, "va:g:x:y:i:")) != -1) {
      switch(c) {
         case 'a': {
                      int action = std::atoi(optarg == NULL ? "-1" : optarg);
                      if (action != MESH_PROCESSING && action != MESH_LABELING_ONLY)
                      {
                         printf("Error parsing command line: -a %s is an undefined action \n",optarg);
                         rstat = 1;
                      }
                      else
                         as->action=action;
                      break;
                   }

         case 'g': {
                      int decomp = std::atoi(optarg == NULL ? "-1" : optarg);
                      if (decomp != ROW_DECOMP && decomp != COLUMN_DECOMP && decomp != TILE_DECOMP)
                      {
                         printf("Error parsing command line: -g %s is an undefined decomposition\n",optarg);
                         rstat = 1;
                      }
                      else
                         as->decomp=decomp;
                      break;
                   }
         case 'x' : {
                       int xval = std::atoi(optarg == NULL ? "-1" : optarg);
                       if  (xval < 0 )
                       {
                          printf(" Error parsing command line: %d is not a valid mesh width \n",xval);
                          rstat = 1;
                       }
                       else
                          as->global_mesh_size[0] = xval;
                       break;
                    }
         case 'y' : {
                       int yval = std::atoi(optarg == NULL ? "-1" : optarg);
                       if  (yval < 0 )
                       {
                          printf(" Error parsing command line: %d is not a valid mesh height \n",yval);
                          rstat = 1;
                       }
                       else
                          as->global_mesh_size[1] = yval;
                       break;
                    }
         case 'i' : {
                       strcpy(as->input_filename, optarg);
                       break;
                    }
         case 'o' : {
                       strcpy(as->output_filename, optarg);
                       break;
                    }
         case 'v' : {
                        as->debug = 1;
                       break;
                    }
      } // end switch

   } // end while

   return rstat;
}

//
// computeMeshDecomposition:
// input: AppState *as - nranks, mesh size, etc.
// computes: tileArray, a 2D vector of Tile2D objects
//
// assumptions:
// - For tile decompositions, will use sqrt(nranks) tiles per axis
//
void
computeMeshDecomposition(AppState *as, vector < vector < Tile2D > > *tileArray) {
   int xtiles, ytiles;
   int ntiles;

   if (as->decomp == ROW_DECOMP) { // this is where B will go
      // in a row decomposition, each tile width is the same as the mesh width
      // the mesh is decomposed along height
      xtiles = 1;

      //  set up the y coords of the tile boundaries
      ytiles = as->nranks/2;
      int ylocs[ytiles+1];
      int ysize = 2 * as->global_mesh_size[1] / as->nranks; // size of each tile in y

      int yval=0;
      for (int i=0; i<ytiles; i++, yval+=ysize) {
         ylocs[i] = yval;
      }
      ylocs[ytiles] = as->global_mesh_size[1];

      // then, create tiles along the y axis
      for(int j = 0; j < ytiles; j++){
         int rank = j;
         for (int i=0; i<ytiles; i++)
         {
            vector < Tile2D > tiles;
            int width =  as->global_mesh_size[0];
            int height = ylocs[i+1]-ylocs[i];
            Tile2D t = Tile2D(0, ylocs[i], width, height, rank);
            rank += ytiles;
            tiles.push_back(t);
            tileArray->push_back(tiles);
         }
      }
   }
   else if (as->decomp == COLUMN_DECOMP) { // this is where A will go
      // in a columne decomposition, each tile height is the same as the mesh height
      // the mesh is decomposed along width
      ytiles = 1;

      // set up the x coords of the tile boundaries
      xtiles = as->nranks/2; 
      int xlocs[xtiles+1];
      int xsize = 2 * as->global_mesh_size[0] / as->nranks; // size of each tile in x

      int xval=0;
      for (int i=0; i<xtiles; i++, xval+=xsize) {
         xlocs[i] = xval;
      }
      xlocs[xtiles] = as->global_mesh_size[0];

      // then, create tiles along the x axis
      vector < Tile2D > tile_row;
      int rank = 0;
      for(int j=0;j<xtiles;j++){
      for (int i=0; i<xtiles; i++)
      {
         int width =  xlocs[i+1]-xlocs[i];
         int height = as->global_mesh_size[1];
         Tile2D t = Tile2D(xlocs[i], 0, width, height, rank);
         rank++;
         tile_row.push_back(t);
      }
      }
      tileArray->push_back(tile_row);
   }
   else // assume as->decom == TILE_DECOMP (this is where C wil go)
   {
      // to keep things simple, we  will assume sqrt(nranks) tiles in each of x and y axes.
      // if sqrt(nranks) is not an even integer, then this approach will result in some
      // ranks without tiles/work to do

      double root = sqrt(as->nranks);
      int nranks_per_axis = (int) root;

      xtiles = ytiles = nranks_per_axis;

      // set up x coords for tile boundaries
      int xlocs[xtiles+1];
      int xsize = as->global_mesh_size[0] / nranks_per_axis; // size of each tile in x

      int xval=0;
      for (int i=0; i<xtiles; i++, xval+=xsize) {
         xlocs[i] = xval;
      }
      xlocs[xtiles] = as->global_mesh_size[0];

      // set up y coords for tile boundaries
      int ylocs[ytiles+1];
      int ysize = as->global_mesh_size[1] / nranks_per_axis; // size of each tile in y

      int yval=0;
      for (int i=0; i<ytiles; i++, yval+=ysize) {
         ylocs[i] = yval;
      }
      ylocs[ytiles] = as->global_mesh_size[1];

      // now, build 2D array of tiles
      int rank=0;
      for (int j = 0; j < ytiles; j++) {  // fix me
         vector < Tile2D > tile_row;
         for (int i=0; i < xtiles; i++) {
            int width, height;
            width = xlocs[i+1]-xlocs[i];
            height = ylocs[j+1]-ylocs[j];
            Tile2D t = Tile2D(xlocs[i], ylocs[j], width, height, rank++);
            tile_row.push_back(t);
         }
         tileArray->push_back(tile_row);
      }
   }
}

void
sendStridedBuffer(double *srcBuf, 
      int srcWidth, int srcHeight, 
      int srcOffsetColumn, int srcOffsetRow, 
      int sendWidth, int sendHeight, 
      int fromRank, int toRank ) 
{
   int msgTag = 0;

   //
   // ADD YOUR CODE HERE
   // That performs sending of  data using MPI_Send(), going "fromRank" and to "toRank". The
   // data to be sent is in srcBuf, which has width srcWidth, srcHeight.
   // Your code needs to send a subregion of srcBuf, where the subregion is of size
   // sendWidth by sendHeight values, and the subregion is offset from the origin of
   // srcBuf by the values specificed by srcOffsetColumn, srcOffsetRow.
   int globalSize[2] = {srcHeight, srcWidth};
   int startOffset[2] = {srcOffsetRow, srcOffsetColumn};
   int subRegionSize[2] = {sendHeight, sendWidth};

   MPI_Datatype subRegionType;
   MPI_Type_create_subarray(2, globalSize, subRegionSize, startOffset, MPI_ORDER_C, MPI_DOUBLE, &subRegionType);
   MPI_Type_commit(&subRegionType);
   
   MPI_Send(srcBuf, 1, subRegionType, toRank, msgTag, MPI_COMM_WORLD);
   MPI_Type_free(&subRegionType);
}

void
recvStridedBuffer(double *dstBuf, 
      int dstWidth, int dstHeight, 
      int dstOffsetColumn, int dstOffsetRow, 
      int expectedWidth, int expectedHeight, 
      int fromRank, int toRank ) {

   int msgTag = 0;
   int recvSize[2];
   MPI_Status stat;

   //
   // ADD YOUR CODE HERE
   // That performs receiving of data using MPI_Recv(), coming "fromRank" and destined for
   // "toRank". The size of the data that arrives will be of size expectedWidth by expectedHeight 
   // values. This incoming data is to be placed into the subregion of dstBuf that has an origin
   // at dstOffsetColumn, dstOffsetRow, and that is expectedWidth, expectedHeight in size.
   //
   int globalSize[2] = {dstHeight, dstWidth};
   int startOffset[2] = {dstOffsetRow, dstOffsetColumn};
   int subRegionSize[2] = {expectedHeight, expectedWidth};

   MPI_Datatype subRegionType;
   MPI_Type_create_subarray(2, globalSize, subRegionSize, startOffset, MPI_ORDER_C, MPI_DOUBLE, &subRegionType);
   MPI_Type_commit(&subRegionType);
   
   MPI_Recv(dstBuf, 1, subRegionType, fromRank, msgTag, MPI_COMM_WORLD, &stat);
   MPI_Type_free(&subRegionType);

}


void do_rect_dgemm(double *A, double *B, double *C, int A_width, int A_height, int B_width, int B_height, int C_width, int C_height)
{
   // for(int i=0;i<C_height;i++){
   //    for(int j=0;j<C_width;j++){
   //       double dot = 0.0;
   //       for(int k=0;k<A_height;k++){
   //          dot += A[k*A_width+j] * B[i*B_width+k];
   //       }
   //       C[i*C_width+j] += dot;
   //    }
   // }
   std::vector<double> buf(2 * C_width * C_height);
   double *Acopy = buf.data() + 0;
   double *Bcopy = Acopy + C_width * C_height;
   for(int i=0;i<C_height;i++){
      memcpy(&Acopy[i*C_width], &A[i*C_width], sizeof(double) * C_width);
      memcpy(&Bcopy[i*C_width], &B[2*i*C_width], sizeof(double) * C_width);
   }
   int n = C_width;
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., Acopy, n, Bcopy, n, 1., C, n);    
   for(int i=0;i<C_height;i++){
      memcpy(&Acopy[i*C_width], &A[i*n+n*n], sizeof(double) * C_width);
      memcpy(&Bcopy[i*C_width], &B[2*i*n+n], sizeof(double) * C_width);
   }
   cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1., Acopy, n, Bcopy, n, 1., C, n);
}

void
mmulAllTiles(int myrank, vector < vector < Tile2D > > & AtileArray, vector < vector < Tile2D > > & BtileArray, vector < vector < Tile2D > > & CtileArray) {
   for (int row=0;row<CtileArray.size(); row++)
   {
      for (int col=0; col<CtileArray[row].size(); col++)
      {  
         Tile2D *Ct = &(CtileArray[row][col]);
         for(int k=0;k<BtileArray.size();k++)
         {
            Tile2D *Bt = &(BtileArray[k][0]);
            for(int p=0;p<AtileArray[0].size();p++)
            {
               Tile2D *At = &(AtileArray[0][p]);
               if(At->tileRank==Ct->tileRank && Bt->tileRank==Ct->tileRank && Ct->tileRank==myrank){
                  
                  do_rect_dgemm(At->A.data(), Bt->B.data(), Ct->C.data(), At->width, At->height, Bt->width, Bt->height, Ct->width, Ct->height);
               }
            }
         }
      }
   }
}

void
scatterAllTiles(int myrank, vector < vector < Tile2D > > & tileArray, double *s, int global_width, int global_height, int type)
{

#if DEBUG_TRACE
   printf(" Rank %d is entering scatterAllTiles \n", myrank);
#endif
   for (int row=0;row<tileArray.size(); row++)
   {
      for (int col=0; col<tileArray[row].size(); col++)
      {  
         Tile2D *t = &(tileArray[row][col]);

         if (myrank != 0 && myrank == t->tileRank)
         {
            int fromRank=0;

            // receive a tile's buffer 
            t->inputBuffer.resize(t->width*t->height);
            t->outputBuffer.resize(t->width*t->height);
#if DEBUG_TRACE
            printf("scatterAllTiles() receive side:: t->tileRank=%d, myrank=%d, t->inputBuffer->size()=%d, t->outputBuffersize()=%d \n", t->tileRank, myrank, t->inputBuffer.size(), t->outputBuffer.size());
#endif
            recvStridedBuffer(t->inputBuffer.data(), t->width, t->height,
                  0, 0,  // offset into the tile buffer: we want the whole thing
                  t->width, t->height, // how much data coming from this tile
                  fromRank, myrank); 
            if(type==0){
               t->A.resize(t->width*t->height);
               memcpy((void *)(t->A.data()), (void *)(t->inputBuffer.data()), sizeof(double)*t->width*t->height);
            }
            else if(type==1){
               t->B.resize(t->width*t->height);
               memcpy((void *)(t->B.data()), (void *)(t->inputBuffer.data()), sizeof(double)*t->width*t->height);
            }
            else{
               t->C.resize(t->width*t->height);
               memcpy((void *)(t->C.data()), (void *)(t->inputBuffer.data()), sizeof(double)*t->width*t->height);
            }
         }
         else if (myrank == 0)
         {
            if (t->tileRank != 0) {
#if DEBUG_TRACE
               printf("scatterAllTiles() send side: t->tileRank=%d, myrank=%d, t->inputBuffer->size()=%d \n", t->tileRank, myrank, t->inputBuffer.size());
#endif
               printf("scatterAllTiles() send side: t->tileRank=%d, myrank=%d, t->inputBuffer->size()=%d t->xloc=%d t->yloc=%d \n", t->tileRank, myrank, t->inputBuffer.size(), t->xloc, t->yloc);
               sendStridedBuffer(s, // ptr to the buffer to send
                     global_width, global_height,  // size of the src buffer
                     t->xloc, t->yloc, // offset into the send buffer
                     t->width, t->height,  // size of the buffer to send,
                     myrank, t->tileRank);   // from rank, to rank
            }
            else // rather then have rank 0 send to rank 0, just do a strided copy into a tile's input buffer
            {
               t->inputBuffer.resize(t->width*t->height);
               t->outputBuffer.resize(t->width*t->height);

               off_t s_offset=0, d_offset=0;
               double *d = t->inputBuffer.data();

               for (int j=0;j<t->height;j++, s_offset+=global_width, d_offset+=t->width)
               {
                  memcpy((void *)(d+d_offset), (void *)(s+s_offset), sizeof(double)*t->width);
               }
                if(type==0){
               t->A.resize(t->width*t->height);
               memcpy((void *)(t->A.data()), (void *)(t->inputBuffer.data()), sizeof(double)*t->width*t->height);
            }
            else if(type==1){
               t->B.resize(t->width*t->height);
               memcpy((void *)(t->B.data()), (void *)(t->inputBuffer.data()), sizeof(double)*t->width*t->height);
            }
            else{
               t->C.resize(t->width*t->height);
               memcpy((void *)(t->C.data()), (void *)(t->inputBuffer.data()), sizeof(double)*t->width*t->height);
                           }
            }
         }
      }
   } // loop over 2D array of tiles

#if DEBUG_TRACE
   MPI_Barrier(MPI_COMM_WORLD);
   if (myrank == 1){
      printf("\n\n ----- rank=%d, inside scatterAllTiles debug printing of the tile array \n", myrank);
   }
   MPI_Barrier(MPI_COMM_WORLD);
#endif
      
}

void
gatherAllTiles(int myrank, vector < vector < Tile2D > > & tileArray, double *d, int global_width, int global_height)
{

   for (int row=0;row<tileArray.size(); row++)
   {
      for (int col=0; col<tileArray[row].size(); col++)
      {  
         Tile2D *t = &(tileArray[row][col]);

#if DEBUG_TRACE
         printf("gatherAllTiles(): t->tileRank=%d, myrank=%d, t->outputBuffer->size()=%d \n", t->tileRank, myrank, t->outputBuffer.size());
#endif

         if (myrank != 0 && t->tileRank == myrank)
         {
            // send the tile's output buffer to rank 0
            sendStridedBuffer(t->C.data(), // ptr to the buffer to send
               t->width, t->height,  // size of the src buffer
               0, 0, // offset into the send buffer
               t->width, t->height,  // size of the buffer to send,
               t->tileRank, 0);   // from rank, to rank
         }
         else if (myrank == 0)
         {
            if (t->tileRank != 0) {
               // receive a tile's buffer and copy back into the output buffer d
               recvStridedBuffer(d, global_width, global_height,
                     t->xloc, t->yloc,  // offset of this tile
                     t->width, t->height, // how much data coming from this tile
                     t->tileRank, myrank);
            }
            else // copy from a tile owned by rank 0 back into the main buffer
            {
               double *s = t->C.data();
               off_t s_offset=0, d_offset=0;
               d_offset = t->yloc * global_width + t->xloc;

               for (int j=0;j<t->height;j++, s_offset+=t->width, d_offset+=global_width)
               {
                  memcpy((void *)(d+d_offset), (void *)(s+s_offset), sizeof(double)*t->width);
               }
            }
         }

      }
   } // loop over 2D array of tiles
}

bool check_accuracy(double *A, double *Anot, int nvalues)
{
  double eps = 1e-5;
  for (size_t i = 0; i < nvalues; i++) 
  {
    if (fabsf(A[i] - Anot[i]) > eps) {
       return false;
    }
  }
  return true;
}

int main(int ac, char *av[]) {

   AppState as;
   vector < vector < Tile2D > > tileArray;
   std::chrono::time_point<std::chrono::high_resolution_clock> start_time, end_time;
   std::chrono::duration<double> elapsed_scatter_time, elapsed_sobel_time, elapsed_gather_time;

   MPI_Init(&ac, &av);
  
   int myrank, nranks; 
   MPI_Comm_rank( MPI_COMM_WORLD, &myrank);
   MPI_Comm_size( MPI_COMM_WORLD, &nranks);
   as.myrank = myrank;
   as.nranks = nranks;

   if (parseArgs(ac, av, &as) != 0)
   {
      MPI_Finalize();
      return 1;
   }

   char hostname[256];
   gethostname(hostname, sizeof(hostname));

   printf("Hello world, I'm rank %d of %d total ranks running on <%s>\n", as.myrank, as.nranks, hostname);
   MPI_Barrier(MPI_COMM_WORLD);

#if DEBUG_TRACE
   if (as.myrank == 0)
      printf("\n\n ----- All ranks will computeMeshDecomposition \n");
#endif
   vector < vector < Tile2D > > AtileArray;
   vector < vector < Tile2D > > BtileArray;
   vector < vector < Tile2D > > CtileArray;
   as.decomp = as.Adecomp;
   computeMeshDecomposition(&as, &AtileArray);
   as.decomp = as.Bdecomp;
   computeMeshDecomposition(&as, &BtileArray);
   as.decomp = as.Cdecomp;
   computeMeshDecomposition(&as, &CtileArray);
   if (as.myrank == 0 && as.debug==1) // print out the AppState and tileArray
   {
      as.print();
   }

   MPI_Barrier(MPI_COMM_WORLD);

   if (as.action == MESH_LABELING_ONLY) {
      if (as.myrank==0) {
         printf("\n\n Rank 0 is writing out mesh labels to a file \n");
      }
   }
   else 
   {
      // === rank 0 loads the input file 
      if (as.myrank == 0)
      {
#if DEBUG_TRACE
         printf("\n\n Rank 0 is loading input \n");
#endif
         fill(as.A.data(), as.global_mesh_size[0]*as.global_mesh_size[1]);
         fill(as.B.data(), as.global_mesh_size[0]*as.global_mesh_size[1]);
         fill(as.C.data(), as.global_mesh_size[0]*as.global_mesh_size[1]);
         printf("Working on problem size N=%d \n", as.global_mesh_size[0]);

      }
      MPI_Barrier(MPI_COMM_WORLD);

      // ----------- scatter phase of processing

      // start the timer
      start_time = std::chrono::high_resolution_clock::now();
      
      scatterAllTiles(as.myrank, AtileArray, as.A.data(), as.global_mesh_size[0], as.global_mesh_size[1], 0);
      scatterAllTiles(as.myrank, BtileArray, as.B.data(), as.global_mesh_size[0], as.global_mesh_size[1], 1);
      scatterAllTiles(as.myrank, CtileArray, as.C.data(), as.global_mesh_size[0], as.global_mesh_size[1], 2);

      // end the timer
      MPI_Barrier(MPI_COMM_WORLD);
      end_time = std::chrono::high_resolution_clock::now();
      elapsed_scatter_time = end_time - start_time;

      // ----------- the actual processing
      MPI_Barrier(MPI_COMM_WORLD);

      // start the timer
      start_time = std::chrono::high_resolution_clock::now();

      mmulAllTiles(as.myrank, AtileArray, BtileArray, CtileArray);

      // end the timer
      MPI_Barrier(MPI_COMM_WORLD);
      end_time = std::chrono::high_resolution_clock::now();
      elapsed_sobel_time = end_time - start_time;


      // ----------- gather processing

      // create output buffer space on rank 0
      if (as.myrank == 0) {
         as.output_data_floats.resize(as.global_mesh_size[0]*as.global_mesh_size[1]);
         // initialize to a known value outside the range of expected values 
         std::fill(as.output_data_floats.begin(), as.output_data_floats.end(), -1.0);
      }

      MPI_Barrier(MPI_COMM_WORLD);

      // start the timer
      start_time = std::chrono::high_resolution_clock::now();

      gatherAllTiles(as.myrank, CtileArray, as.output_data_floats.data(), as.global_mesh_size[0], as.global_mesh_size[1]);

      // end the timer
      MPI_Barrier(MPI_COMM_WORLD);
      end_time = std::chrono::high_resolution_clock::now();
      elapsed_gather_time = end_time - start_time;
   }

   MPI_Barrier(MPI_COMM_WORLD);

   if (as.myrank == 0) {
      printf("\n\nTiming results from rank 0: \n");
      printf("\tScatter time:\t%6.4f (ms) \n", elapsed_scatter_time*1000.0);
      printf("\tMmul time:\t%6.4f (ms) \n", elapsed_sobel_time*1000.0);
      printf("\tGather time:\t%6.4f (ms) \n", elapsed_gather_time*1000.0);
      // int n=as.global_mesh_size[0];
      cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, as.A.data(), n, as.B.data(), n, 1., as.C.data(), n);
      // printArray(as.C.data(), n, n);
      // printArray(as.output_data_floats.data(), n);
      if (check_accuracy(as.C.data(), as.output_data_floats.data(), n*n) == false)
            printf(" Error: your answer is not the same as that computed by BLAS. \n");
      else
            printf(" Your answer is the same as that computed by BLAS. \n");
   }

   MPI_Finalize();
   return 0;
}
// EOF
