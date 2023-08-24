//This implementation is based off the parallel implementation on https://stackoverflow.com/questions/12763991/dijkstras-algorithm-in-cuda.
//That program is based of a variant of the solution described in the paper on Accelerating large graph algorithms on the GPU using CUDA by Parwan Harish and P.J. Narayanan
//along with corrections as desribed in CUDA Solutions for the SSSP Problem by Pedro J. Martín, Roberto Torres, and Antonio Gavilanes

#include <climits>
#include <cstddef>
#include <cstdlib>
#include <sstream>
#include <vector>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// Includes CUDA
#include <cuda_runtime.h>
// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h
// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check


#define BLOCK_SIZE 32
#define ITER 5

struct graphInfo{
    //Arrays containing information in regards to vertices, edges and weights of edges in graph
    int *vertexArr;
    int *edgeArr;
    float *weightArr;

    //Information on number of edges and vertices in graph
    int numVerts;
    int numEdges;
};

//Integer Division that rounds up to the next integer
static int integerCeil(int a, int b) 
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

void populateGraph(graphInfo *graph, int numVerts, int edgePerVert)
{
    //Allocating memory for storing information about graph
    graph->numVerts = numVerts;
    graph->numEdges = numVerts * edgePerVert;
    graph->vertexArr = (int *)malloc(graph->numVerts * sizeof(int));
    graph->edgeArr = (int *)malloc(graph->numEdges * sizeof(int));
    graph->weightArr = (float *)malloc(graph->numEdges * sizeof(float));

    srand((unsigned int)time(NULL));

    for(int i = 0; i < graph->numVerts; i++)
    {
        //Set up list of vertexes based off the number of edges that each will have
        graph->vertexArr[i] = i * edgePerVert;
    }

    int *tempEdgeArr = (int *)malloc(edgePerVert * sizeof(int));
    for(int i = 0; i < numVerts; i++)
    {
        //Initialising the edges to the maximum edge value
        for(int j = 0; j < edgePerVert; j++)
        {
            tempEdgeArr[j] = INT_MAX;
        }

        for(int j = 0; j < edgePerVert; j++)
        {
            bool next = false;
            int temp;

            while (!next) 
            {
                next = true;
                //Chooses a random vert to make this edge connected to
                temp = (rand()%graph->numVerts);

                //Makes sure this edge hasn't been assigned a vert that already been assigned
                for(int k = 0; k < edgePerVert; k++)
                {
                    if(temp == tempEdgeArr[k])
                    {
                        next = false;
                    }
                }
                //Makes sure the edge doesn't connect a vertex to itself
                if(temp == i)
                {
                    next = false;
                }
                //Adds this to the list of assigned edges so it won't be considered again the next time around
                if(next)
                {
                    tempEdgeArr[j] = temp;
                }
            }
            //Adding this edge to the list of edges
            graph->edgeArr[i * edgePerVert + j] = temp;
            //Assigning a random weight to this edge
            graph->weightArr[i * edgePerVert + j] = (float)rand()/(float)(RAND_MAX/10);
        }
    }
}

int minDist(float *shortestDist, bool *processedVerts, const int sourceVert, const int numVerts)
{
    int minIndex = sourceVert;
    float min = FLT_MAX;

    for(int i = 0; i < numVerts; i++)
    {
        if(!processedVerts[i] && shortestDist[i] <= min)
        {
            min = shortestDist[i];
            minIndex = i;
        }
    }
    return minIndex;
}

double serialDijkstra(float *graph, float *host_Distances, int sourceVert, const int numVerts)
{
    double average_time;
    struct timespec start, end;

    //List of vertices that we set to true as we process the vertices one by one
    bool *processedVerts = (bool *)malloc(numVerts * sizeof(bool));

    //Initialize the two arrays
    for(int i = 0; i < numVerts; i++)
    {
        host_Distances[i] = FLT_MAX;
        processedVerts[i] = false;
    }

    //Set the source vector to 0 as the distance to itself will always be zero  
    host_Distances[sourceVert] = 0.f;


    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }
    for(int i = 0; i < numVerts-1; i++)
    {
        //Find the mininum distance vertex from all vertices that haven't been processed yet
        int currentVert = minDist(host_Distances, processedVerts , sourceVert, numVerts);
        
        //Vertex is marked as processed
        processedVerts[currentVert] = true;

        for(int j = 0; j < numVerts; j++)
        {
            //Only update distance if it hasn't been processed, there's an edge from current vertex to query vertex and if the cost of the path from source to current vertex is smaller than current shortest distance
            if(!processedVerts[j] && graph[currentVert * numVerts + j] && host_Distances[currentVert] != FLT_MAX && (host_Distances[currentVert]+graph[currentVert*numVerts + j])< host_Distances[j])
            {
                host_Distances[j] = host_Distances[currentVert] + graph[currentVert * numVerts + j];
            }
        }
        
        
    }
    if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }

    average_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;

    return average_time;
}

//Checks if all the vertices have been processed 
bool allProcessed(bool *processedVerts, int numVerts)
{
    
    for(int i = 0; i < numVerts; i++)
    {
        if(processedVerts[i])
        {
            return false;
        }
    }

    return true;
}

//Experimental atomicMin for float values
__device__ __forceinline__ float atomicMinFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

//Initialise the arrays in the device memory with default values
__global__ void globalInitialize(bool*  device_processedVerts, float*  device_distances, float*  device_updateDistances, const int sourceVert, const int numVerts)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if(x < numVerts)
    {
        //If current vertex is the source vertex, set it as processed, set the shortest distance to source as 0
        if(sourceVert == x)
        {
            device_processedVerts[x] = true;
            device_distances[x] = 0.f;
            device_updateDistances[x] = 0.f;
        }
        //For any other current vertex, mark it as unprocessed, set the shortest distance and update distance to max
        else {
            device_processedVerts[x] = false;
            device_distances[x] = FLT_MAX;
            device_updateDistances[x] = FLT_MAX;
        }
    }
}

//Kernel that sets up the updating array that will be used in the next Kernel
__global__ void globalDijkstra_SetupUpdates(const int*  vertexArr, const int*  edgeArr, const float*  weightArr, bool*  processedVerts, float*  distances, float*  updateDistances, const int numVerts, const int numEdges)
{
    
    //Get index of current thread
    int x = blockIdx.x*blockDim.x + threadIdx.x;

    //Check if within the number of vertices available
    if(x < numVerts)
    {
        //If this is the starting vertex
        if(processedVerts[x])
        {
            //Set it as unprocessed
            processedVerts[x] = false;

            //Setting up starting Vertex
            int startEdge = vertexArr[x];
            int endEdge;


            //If there is a next vertex, set it as the destination, if not, this is the endpoint
            if((x + 1) < numVerts)
            {
                endEdge = vertexArr[x+1];
            }
            else
            {
                endEdge = numEdges;
            }

            //Loops through all known edges, compares weights and updates with the shortest distance
            for(int i = startEdge; i < endEdge; i++)
            {
                int currentIndex = edgeArr[i];
                atomicMinFloat(&updateDistances[currentIndex], distances[x] + weightArr[i]);
            }
        }
    }
}

//Kernel that updates the shortest distances using earlier calculated values
__global__ void globalDijkstra_Update(const int*  vertexArr, const int*  edgeArr, const float*  weightArr, bool*  processedVerts, float*  distances, float*  updateDistances, const int numVerts)
{
    
    //Get index of current thread
    int x = blockIdx.x * blockDim.x + threadIdx.x;


    if(x < numVerts)
    {
        //If the shortest distance in the update array is smaller, make the shortest distance the update distance value, once this is done it has been processed
        if(distances[x] > updateDistances[x])
        {
            distances[x] = updateDistances[x];
            processedVerts[x] = true;
        }

        //Set the update distance to the value in shortest distance array, hopefully being the one we've calculated in our earlier kernel
        updateDistances[x] = distances[x];
    }
}

float parallelDijkstra(graphInfo *graph, const int sourceVert, float*  host_Distances)
{
    float time;
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    int *device_vertexArr;
    int *device_edgeArr;
    float *device_weightArr;

    //creating space in device memory space for storing arrays of vertex, edge and weights
    cudaMalloc(&device_vertexArr, sizeof(int) * graph->numVerts);
    cudaMalloc(&device_edgeArr, sizeof(int) * graph->numEdges);
    cudaMalloc(&device_weightArr, sizeof(float) * graph->numEdges);

    //copy from the host memory into device memory for later use
    cudaMemcpy(device_vertexArr, graph->vertexArr,sizeof(int) * graph->numVerts, cudaMemcpyHostToDevice);
    cudaMemcpy(device_edgeArr, graph->edgeArr,sizeof(int) * graph->numEdges, cudaMemcpyHostToDevice);
    cudaMemcpy(device_weightArr, graph->weightArr,sizeof(float) * graph->numEdges, cudaMemcpyHostToDevice);

    bool *device_processedVerts;
    float *device_distances;
    float *device_updateDistances;

    //creating space in device memory space for processed vertices, shortest cost array and updating cost array
    cudaMalloc(&device_processedVerts, sizeof(bool) * graph->numVerts);
    cudaMalloc(&device_distances, sizeof(float) * graph->numVerts);
    cudaMalloc(&device_updateDistances, sizeof(float) * graph->numVerts);

    bool *host_processedVerts = (bool *)malloc(sizeof(bool) * graph->numVerts);

    //Initialize the arrays
    globalInitialize <<<integerCeil(graph->numVerts, BLOCK_SIZE), BLOCK_SIZE>>>(device_processedVerts, device_distances, device_updateDistances, sourceVert, graph->numVerts);

    //Synchronizing threads after initialization of arrays
    cudaDeviceSynchronize();

    //
    cudaMemcpy(host_processedVerts, device_processedVerts, sizeof(bool) * graph->numVerts, cudaMemcpyDeviceToHost);

    cudaEventRecord(launch_begin,0);
    while (!allProcessed(host_processedVerts, graph->numVerts)) {
        for(int i = 0; i < ITER; i++)
        {
            globalDijkstra_SetupUpdates<<<integerCeil(graph->numVerts, BLOCK_SIZE), BLOCK_SIZE>>>(device_vertexArr, device_edgeArr, device_weightArr, device_processedVerts, device_distances, device_updateDistances, graph->numVerts, graph->numEdges);
            //Synchronizing threads after setting up of update arrays
            cudaDeviceSynchronize();

            globalDijkstra_Update<<<integerCeil(graph->numVerts, BLOCK_SIZE), BLOCK_SIZE>>>(device_vertexArr, device_edgeArr, device_weightArr, device_processedVerts, device_distances, device_updateDistances, graph->numVerts);
            //Synchronizing threads after updating shortest distances
            cudaDeviceSynchronize();
        }

        cudaMemcpy(host_processedVerts, device_processedVerts, sizeof(bool) * graph->numVerts, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);

    time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    
    cudaMemcpy(host_Distances, device_distances, sizeof(float) * graph->numVerts, cudaMemcpyDeviceToHost);

    free(host_processedVerts);
    cudaFree(device_vertexArr);
    cudaFree(device_edgeArr);
    cudaFree(device_weightArr);
    cudaFree(device_processedVerts);
    cudaFree(device_distances);
    cudaFree(device_updateDistances);

    return time;
}

bool validateResults(float *sDistances, float *pDistances, int numVerts)
{
    for(int i = 0; i < numVerts; i++)
    {
        if(sDistances[i] != pDistances[i])
        {
            printf("%i is incorrect\n",i);
            printf("%f vs %f", sDistances[i], pDistances[i]);
            return false;
        }
    }

    return true;
}

int main()
{
    int numVerts = 10000;
    int edgePerVert = 5;
    int sourceVert = 0;

    graphInfo graph;

    //Populating the graph with random value
    populateGraph(&graph, numVerts, edgePerVert);

    //Converting the array of weights to a matrix
    float *weightMatrix = (float *)malloc(numVerts * numVerts * sizeof(float));

    //Initialising the matrix of weights to the maximum float value 
    for(int i = 0; i < numVerts*numVerts; i++)
    {
        weightMatrix[i] = FLT_MAX;
    }

    //Filling up the matrix with values from the list
    for(int i = 0; i < numVerts; i++)
    {
        weightMatrix[i * numVerts + i] = 0.f;
    }

    for(int i = 0; i < numVerts; i++)
    {
        for(int j = 0; j < edgePerVert; j++)
        {
            //Filling up the weight matrix
            weightMatrix[i * numVerts + graph.edgeArr[graph.vertexArr[i]+j]] = graph.weightArr[graph.vertexArr[i]+j];
            //printf("Vertex nr. %i; Edge nr. %i; Weight = %f\n", i, graph.edgeArr[graph.vertexArr[i] + j],graph.weightArr[graph.vertexArr[i] + j]);
        }
    }

    //creating space in host memory space for storing list of shortest distances calculated by serial function
    float *host_Distances_serial = (float *)malloc(numVerts * sizeof(float));

    printf("----------------------------------------------------------------------------------------------\n");
    printf("Number of Vertices: %d\n", numVerts);
    printf("Number of Edges per Vertex: %d\n", edgePerVert);
    printf("Number of Blocks: %d\n", BLOCK_SIZE);
    printf("----------------------------------------------------------------------------------------------\n");
    printf("Sequential Implementation Beginning!\n");
    //Run the serial version of dijkstra
    double sequentialTimeTaken = serialDijkstra(weightMatrix, host_Distances_serial, sourceVert, numVerts);

    // printf("\nSerial Result:\n");
    // for(int i = 0; i < numVerts; i++)
    // {
    //     printf("From Vert %i to Vert %i = %1.3f\n", sourceVert, i, host_Distances_serial[i]);
    // }

    printf("Sequential Implementation Complete!\n");
    printf("----------------------------------------------------------------------------------------------\n");

    //creating space in host memory space for storing list of shortest distances calculated by parallel function
    float *host_Distances_parallel = (float *)malloc(numVerts * sizeof(float));

    printf("Parallel Implementation Beginning!\n");
    //Run the parallel version of dijkstra
    float CUDATimeTaken = parallelDijkstra(&graph, sourceVert, host_Distances_parallel);

    // printf("\nGlobal Result:\n");
    // for(int i = 0; i < numVerts; i++)
    // {
    //     printf("From Vert %i to Vert %i = %1.3f\n", sourceVert, i, host_Distances_parallel[i]);
    // }

    printf("Parallel Implementation Complete!\n");
    printf("----------------------------------------------------------------------------------------------\n");
    printf("Validating Results!\n");

    if(validateResults(host_Distances_serial, host_Distances_parallel, numVerts))
    {
        printf("Results Correct!\n");
        printf("----------------------------------------------------------------------------------------------\n");
        printf("Time Taken for the Sequential Implementation: %.5fs\n", sequentialTimeTaken);
        printf("Time Taken for the Parallel Implementation: %.5fms\n", CUDATimeTaken/ITER);
        printf("Improvement Ratio (Serial/Parallel): %1.3fx\n", (sequentialTimeTaken*1000)/(CUDATimeTaken/ITER));
    }else {
        printf("Results Incorrect!\n");
    }
    printf("----------------------------------------------------------------------------------------------\n");


    free(host_Distances_serial);
    free(host_Distances_parallel);

    return 0;
}


