#include <mpi.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <time.h>

#define INFINITE 9999

typedef struct graphInfo_s
{
    //Information on number of edges and vertices in graph
    int numVerts;
    int numEdges;

    //Arrays containing information in regards to vertices, edges and weights of edges in graph
    int *vertexArr;
    int *edgeArr;
    float *weightArr;
} graphInfo;

typedef struct vertexInfo_s
{
    //Retains the distance information of the shortest edge 
    float val;
    //Retains the index of the shortest edge
    int vertex;
} vertexInfo;

void setBounds(int split, int numVerts, int rank, int *lowerLim, int *upperLim)
{
    //Sets the lower and upper limits for this process based off its rank
    *lowerLim = rank * split;
    *upperLim = (rank + 1) * split;

    //Changes the upper limit if its above the final value 
    if(*upperLim > numVerts)
    {
        *upperLim = numVerts;
    }
    //Changes the lower limit if the above the final value
    if(*lowerLim > numVerts)
    {
        *lowerLim = *upperLim-split;
    }
}

void parallel_minDist(float *shortestDistance, bool *processedVerts, const int sourceVert, int lowerLim, int upperLim, int rank, vertexInfo *localInfo)
{
    //Sets the minimum index of the source
    int minIndex = sourceVert;
    float min = INFINITE;

    //Finds the shortest distance that hasn't been processed yet
    for(int i = lowerLim; i < upperLim; i++)
    {
        if(!processedVerts[i] && shortestDistance[i] <= min)
        {
            //printf("Rank:%d\tLocalVector:%d\t Old Value:%f\t",rank,i, min);
            min = shortestDistance[i];
            minIndex = i;
            //printf("New Value:%f\n",min);
        }
    }

    localInfo->val = min;
    localInfo->vertex = minIndex;
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

// void setupUpdates(graphInfo *graph, bool* processedVerts, float* shortestDist, float* updateDist, int lowerLim, int upperLim, int rank)
// {
//     float localMin;
//     float * localUpdates = (float *)malloc(graph->numVerts * sizeof(float));
//     memcpy(localUpdates, updateDist, graph->numVerts * sizeof(float));
//     float globalMin;
//     for(int i = lowerLim; i < upperLim; i++)
//     {
//         if(processedVerts[i])
//         {
//             processedVerts[i] = false;
//             int startEdge = graph->vertexArr[i];
//             int endEdge;
//             if((i+1) < graph->numVerts)
//             {
//                 endEdge = graph->vertexArr[i+1];
//             }
//             else
//             {
//                 endEdge = graph->numEdges;
//             }
//             for(int j = startEdge; i < endEdge; i++)
//             {
//                 int currentIndex = graph->edgeArr[i];
//                 if(updateDist[currentIndex] > shortestDist[i] + graph->edgeArr[i])
//                 {
//                     localMin = shortestDist[i] + graph->edgeArr[i];
//                     localUpdates[currentIndex] = localMin;                   
//                 }
//             }        
//         }
//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Allreduce(localUpdates, updateDist, graph->numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//     if(rank == 0)
//     {
//         printf("\nGlobal Result:\n");
//         for(int i = 0; i < graph->numVerts; i++)
//         {
//             printf("process %d: From Vert = %1.3f\n",rank, updateDist[i]);
//         }
//     }
//     //MPI_Bcast(updateDist, graph->numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);                
//     //if(rank == 0)
//     //{
//     //    updateDist[currentIndex] = globalMin;
//     //} 
// }

// void updateArrays(graphInfo *graph, bool* processedVerts, float* shortestDist, float* updateDist, int lowerLim, int upperLim)
// {
//     for(int i = lowerLim; i < upperLim; i++)
//     {
//         if(shortestDist[i] > updateDist[i])
//         {
//             shortestDist[i] = updateDist[i];
//             processedVerts[i] = true;
//         }
//         updateDist[i] = shortestDist[i];
//     }
// }

// void parallel_dijkstra(graphInfo *graph, int lowerLim, int upperLim, float *shortestDistance, int numVerts, int rank, int sourceVert)
// {
//     bool *processedVerts = (bool *)malloc(numVerts * sizeof(bool));
//     bool *localProcessedVerts = (bool *)malloc(numVerts * sizeof(bool));
//     float *updateDistances = (float *)malloc(numVerts * sizeof(float));
//     float *localShortest = (float *)malloc(numVerts * sizeof(float));
//     float *globalUpdates = (float *)malloc(numVerts * sizeof(float));
//     float *localMin = (float *)malloc(2 * sizeof(float));
//     if(rank == 0)
//     {
//         for(int i = 0; i < numVerts; i++)
//         {
//             if(i == sourceVert)
//             {
//                 processedVerts[sourceVert] = true;
//                 localProcessedVerts[sourceVert] = true;
//                 localShortest[sourceVert] = 0.f;
//                 //shortestDistance[sourceVert] = 0.f;
//                 updateDistances[sourceVert] = 0.f;
//             }
//             else
//             {
//                 //shortestDistance[i] = FLT_MAX;
//                 localProcessedVerts[sourceVert] = true;
//                 localShortest[i] = INFINITE;
//                 updateDistances[i] = INFINITE;
//                 processedVerts[i] = false;
//             }
//         }
//         //MPI_Bcast(shortestDistance, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(localShortest, numVerts, MPI_INT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         MPI_Bcast(localProcessedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         MPI_Bcast(updateDistances, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//     }
//     else
//     {
//         //MPI_Bcast(shortestDistance, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(localShortest, numVerts, MPI_INT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         MPI_Bcast(localProcessedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         MPI_Bcast(updateDistances, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//     }
//     MPI_Barrier(MPI_COMM_WORLD);  
//     //while (!allProcessed(processedVerts, numVerts))
//     {
//         setupUpdates(graph, localProcessedVerts, localShortest, updateDistances, lowerLim, upperLim, rank);
//         updateArrays(graph, localProcessedVerts, localShortest, updateDistances, lowerLim, upperLim);
//         //MPI_Gather(&localProcessedVerts[lowerLim], upperLim-lowerLim, MPI_C_BOOL, processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         //MPI_Reduce(updateDistances, globalUpdates, numVerts, MPI_FLOAT, MPI_MIN, 0, MPI_COMM_WORLD);
//     }
//     // if(rank == 0)
//     // {
//     //     printf("\nGlobal Result:\n");
//     //     for(int i = 0; i < numVerts; i++)
//     //     {
//     //         printf("process %d: From Vert %i to Vert %i = %1.3f\n",rank, sourceVert, i, updateDistances[i]);
//     //     }
//     // }
// }

// void parallel_dijkstra(graphInfo *graph, int lowerLim, int upperLim, float *shortestDistances, int numVerts, int rank, int sourceVert)
// {
//     bool *processedVerts = (bool *)malloc(numVerts * sizeof(bool));
//     float *updateDistances = (float *)malloc(numVerts * sizeof(float));
//     float *globalUpdates = (float *)malloc(numVerts * sizeof(float));
//     bool *globalProcessedVerts = (bool *)malloc(numVerts * sizeof(bool));
//     float *globalShortest = (float *)malloc(numVerts * sizeof(float));
//     if(rank == 0)
//     {
//         for(int i = 0; i < numVerts; i++)
//         {
//             if(i == sourceVert)
//             {
//                 globalProcessedVerts[sourceVert] = true;
//                 globalShortest[sourceVert] = 0.f;
//                 globalUpdates[sourceVert] = 0.f;
//                 processedVerts[sourceVert] = true;
//                 shortestDistances[sourceVert] = 0.f;
//                 updateDistances[sourceVert] = 0.f;
//             }
//             else
//             {
//                 globalUpdates[i] = FLT_MAX;
//                 globalProcessedVerts[sourceVert] = true;
//                 globalShortest[i] = INFINITE;
//                 updateDistances[i] = INFINITE;
//                 shortestDistances[sourceVert] = INFINITE;
//                 processedVerts[i] = false;
//             }
//         }
//         MPI_Bcast(updateDistances, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(globalUpdates, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(shortestDistances, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(globalShortest, numVerts, MPI_INT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         MPI_Bcast(globalProcessedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//     }
//     else
//     {
//         MPI_Bcast(updateDistances, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(globalUpdates, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(shortestDistances, numVerts, MPI_FLOAT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(globalShortest, numVerts, MPI_INT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
//         MPI_Bcast(globalProcessedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);    
//     }
//     int currentIndex = 0;
//     while (!allProcessed(processedVerts, numVerts)){
//     for(int i = 0; i < numVerts; i++)
//     {
//         if(processedVerts[i])
//         {
//             processedVerts[i] = false;
//             int startEdge = graph->vertexArr[i];
//             int endEdge;
//             if((i+1) < numVerts)
//             {
//                 endEdge = graph->vertexArr[i+1];
//             }
//             else
//             {
//                 endEdge = graph->numEdges;
//             }          
//             for(int j = startEdge; j < endEdge; j++)
//             {
//                 int currentIndex = graph->edgeArr[j];
//                 if(updateDistances[currentIndex] > (shortestDistances[i] + graph->weightArr[j]))
//                 {
//                     updateDistances[currentIndex] = shortestDistances[i] + graph->weightArr[j];
//                 }
//             }
//         }
//         if(shortestDistances[i] > updateDistances[i])
//             {
//                 shortestDistances[i] = updateDistances[i];
//                 processedVerts[i] = true;
//             }
//         updateDistances[i] = shortestDistances[i];
//             // MPI_Barrier(MPI_COMM_WORLD);
//             // MPI_Allreduce(updateDistances, globalUpdates, numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//             // MPI_Allreduce(shortestDistances, globalShortest, numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//             // MPI_Allreduce(processedVerts, globalProcessedVerts, numVerts, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
//             // printf("THing: %f \n", updateDistances[i]);
//     }
//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Allreduce(updateDistances, globalUpdates, numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(shortestDistances, globalShortest, numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(processedVerts, globalProcessedVerts, numVerts, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
//     for(int i = lowerLim; i < upperLim; i++)
//     {
//         if(shortestDistances[i] > updateDistances[i])
//             {
//                 shortestDistances[i] = updateDistances[i];
//                 processedVerts[i] = true;
//             }
//         updateDistances[i] = updateDistances[i];
//     }
//     if(rank == 0)
//     {
//         printf("\nGlobal Result:\n");
//         for(int i = 0; i < numVerts; i++)
//         {
//             printf("process %d: From Vert %d to Vert %d = %f\n",rank, sourceVert, i, globalShortest[i]);
//         }
//     }
//     MPI_Barrier(MPI_COMM_WORLD);
//     MPI_Allreduce(updateDistances, globalUpdates, numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(shortestDistances, globalShortest, numVerts, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
//     MPI_Allreduce(processedVerts, globalProcessedVerts, numVerts, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
//     if(rank == 0)
//     {
//         printf("\nGlobal Result:\n");
//         for(int i = 0; i < numVerts; i++)
//         {
//             printf("process %d: From Vert %i to Vert %d = %d\n",rank, sourceVert, i, globalProcessedVerts[i]);
//         }
//     }
//     //}
// }

double parallel_dijkstra(float *graph, int lowerLim, int upperLim, float *shortestDistances, int numVerts, int rank, int sourceVert)
{
    //List of all vertices, keeps track for which ones have been processed
    bool *processedVerts = (bool *)malloc(numVerts * sizeof(bool));
    double start;
    double end;

    //Initializes the arrays in a single process and then broadcasts this information so processing time won't be wasted
    if(rank == 0){
        for(int i = 0; i < numVerts; i++)
        {
            shortestDistances[i] = INFINITE;
            processedVerts[i] = false;
        }
        shortestDistances[sourceVert] = 0.f;

        MPI_Bcast(shortestDistances, numVerts, MPI_FLOAT, 0 ,MPI_COMM_WORLD);
        MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0 ,MPI_COMM_WORLD);
    }
    else{
        MPI_Bcast(shortestDistances, numVerts, MPI_FLOAT, 0 ,MPI_COMM_WORLD);
        MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0 ,MPI_COMM_WORLD);
    }

    //Start counting time in this process
    double localStart = MPI_Wtime();

    for(int i = 0; i < numVerts-1; i++)
    {
        vertexInfo localInfo;
        parallel_minDist(shortestDistances,processedVerts, sourceVert, lowerLim, upperLim,rank, &localInfo);

        vertexInfo globalInfo;

        MPI_Reduce(&localInfo, &globalInfo, 1, MPI_FLOAT_INT, MPI_MINLOC, 0 ,MPI_COMM_WORLD);

        if(rank == 0){
            processedVerts[globalInfo.vertex] = true;

            for(int j = 0; j < numVerts; j++)
            {
                //Only update distance if it hasn't been processed, there's an edge from current vertex to query vertex and if the cost of the path from source to current vertex is smaller than current shortest distance
                if(!processedVerts[j] && graph[globalInfo.vertex * numVerts + j] && shortestDistances[globalInfo.vertex] <= INFINITE && (shortestDistances[globalInfo.vertex]+graph[globalInfo.vertex*numVerts + j])< shortestDistances[j])
                {
                    shortestDistances[j] = shortestDistances[globalInfo.vertex] + graph[globalInfo.vertex * numVerts + j];
                }
            }

            MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            MPI_Bcast(shortestDistances, numVerts, MPI_FLOAT, 0 , MPI_COMM_WORLD);
        }
        else{
            MPI_Bcast(processedVerts, numVerts, MPI_C_BOOL, 0, MPI_COMM_WORLD);
            MPI_Bcast(shortestDistances, numVerts, MPI_FLOAT, 0 , MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    //Stop counting time in this process
    double localEnd = MPI_Wtime(); 

    //Reduce the time from earliest second to the latest second recorded
    MPI_Reduce(&localStart, &start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&localEnd, &end, 1, MPI_DOUBLE, MPI_MAX, 0 , MPI_COMM_WORLD);

    return (end-start);
}

void populateGraph(graphInfo *graph, int numVerts, int edgePerVert);
double serialDijkstra(float *graph, float *host_Distances, int sourceVert, const int numVerts);
int minDist(float *shortestDist, bool *processedVerts, const int sourceVert, const int numVerts);
void baseDijkstra( graphInfo *graph, float *shortestDistance, int numVerts, int edgePerVert, int sourceVert, float *weightMatrix);
bool validateResults(float *sDistances, float *pDistances, int numVerts);

int main(int argc, char *argv[])
{
    int numVerts = 10000;
    int edgePerVert = 5;
    int sourceVert = 0;
    int split, lowerBound, upperBound;

    graphInfo graph;

    graph.numVerts = numVerts;
    graph.numEdges = numVerts * edgePerVert;
    graph.vertexArr = (int *)malloc(graph.numVerts * sizeof(int));
    graph.edgeArr = (int *)malloc(graph.numEdges * sizeof(int));
    graph.weightArr = (float *)malloc(graph.numEdges * sizeof(float));

    float *weightMatrix;
    weightMatrix = (float *)malloc(numVerts * numVerts * sizeof(float));

    double baseTimeTaken;
    double mpiTimeTaken;

    float *sequential_Distances = (float *)malloc(numVerts * sizeof(float));
    float *parallel_Distances = (float *)malloc(numVerts * sizeof(float));

    int num_procs, myrank;

    MPI_Init(&argc, &argv);

    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    //Setting up the program to be run 
    if(myrank == 0)
    {
        //Deciding how many sections this program will be split up into based off the number of
        split = numVerts/num_procs;
        if(numVerts%num_procs > 0)
        {
            split++;
        }

        graphInfo send;

        //Allocating memory for storing information about graph
        send.numVerts = numVerts;
        send.numEdges = numVerts * edgePerVert;
        send.vertexArr = (int *)malloc(send.numVerts * sizeof(int));
        send.edgeArr = (int *)malloc(send.numEdges * sizeof(int));
        send.weightArr = (float *)malloc(send.numEdges * sizeof(float));
        graph = send;

        //Initialize the graph in root process
        populateGraph(&send, numVerts, edgePerVert);
        baseDijkstra(&graph, sequential_Distances, numVerts, edgePerVert, sourceVert, weightMatrix);

        //Broadcast this information to the rest of the processes
        MPI_Bcast(&split, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(send.vertexArr, send.numVerts , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(send.edgeArr, send.numEdges , MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(send.weightArr, send.numEdges , MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    else
    {
        //Other processes receive this information
        MPI_Bcast(&split, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph.vertexArr, graph.numVerts, MPI_INT, 0 , MPI_COMM_WORLD);
        MPI_Bcast(graph.edgeArr, graph.numEdges, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(graph.weightArr, graph.numEdges, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    //Synchronize processes to make sure all have information on graph data
    MPI_Barrier(MPI_COMM_WORLD);

    if(myrank == 0)
    {
        printf("----------------------------------------------------------------------------------------------\n");
        printf("Number of Vertices: %d\n", numVerts);
        printf("Number of Edges per Vertex: %d\n", edgePerVert);
        printf("Number of Processes: %d\n", num_procs);
        printf("----------------------------------------------------------------------------------------------\n");
        printf("Starting Serial Dijkstra Implementation on Root Process!\n");
        //Run the serial iteration on the root process
        baseTimeTaken = serialDijkstra(weightMatrix, sequential_Distances, sourceVert, numVerts);
        printf("Serial Dijkstra Implementation on Root Process Complete!\n");
    }

    //Synchronize so the root node isn't behind
    MPI_Barrier(MPI_COMM_WORLD);

    //Set the bounds based on the ranks of the images 
    setBounds(split, numVerts, myrank, &lowerBound, &upperBound);

    if(myrank == 0)
    {
        printf("----------------------------------------------------------------------------------------------\n");
        printf("Starting Parallel Dijkstra Implementation!\n");
    }
    //Run the parallel iteration on all processes
    mpiTimeTaken = parallel_dijkstra(weightMatrix, lowerBound, upperBound, parallel_Distances, numVerts, myrank, sourceVert);
    if(myrank == 0)
    {
        printf("Parallel Dijkstra Implementation Complete!\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(myrank == 0){
        //Check results against one another to make sure the answers are correct
        printf("----------------------------------------------------------------------------------------------\n");
        if(validateResults(sequential_Distances, parallel_Distances, numVerts))
        {
            printf("Validation Successful!\n");
            printf("----------------------------------------------------------------------------------------------\n");
            printf("Base Time Taken: %fs\n", baseTimeTaken);
            printf("Parallel Time Taken: %fs\n", mpiTimeTaken);
            printf("Improvement Ratio (Serial/Parallel): %1.3fx\n", baseTimeTaken/mpiTimeTaken);
            printf("----------------------------------------------------------------------------------------------\n");
        }
        else
        {
            printf("Validation Unsuccessful!\n");
        }
    }

    MPI_Finalize();

    return 0;

}

bool validateResults(float *sDistances, float *pDistances, int numVerts)
{
    float threshold = 0.0000001;
    for(int i = 0; i < numVerts; i++)
    {
        if(abs(sDistances[i] - pDistances[i]) > threshold)
        {
            printf("%i is incorrect\n",i);
            printf("%f vs %f\n", sDistances[i], pDistances[i]);
            return false;
        }
    }

    return true;
}

void populateGraph(graphInfo *graph, int numVerts, int edgePerVert)
{
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
            graph->weightArr[i * edgePerVert + j] =  (float)rand()/(float)(RAND_MAX/10);
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
        host_Distances[i] = INFINITE;
        processedVerts[i] = false;
    }

    //Set the source vector to 0 as the distance to itself will always be zero  
    host_Distances[sourceVert] = 0.f;

    double wstart = MPI_Wtime();
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
            if(!processedVerts[j] && graph[currentVert * numVerts + j] && host_Distances[currentVert] != INFINITE && (host_Distances[currentVert]+graph[currentVert*numVerts + j])< host_Distances[j])
            {
                host_Distances[j] = host_Distances[currentVert] + graph[currentVert * numVerts + j];
            }
        }
        
        
    }

    // for(int i = 0; i < numVerts; i++){
    //     printf("Process %d\tprocessed:%d\n", i, processedVerts[i]);
    // }

    if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }

    double wend = MPI_Wtime();

    average_time = (wend - wstart);
    //(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;

    return average_time;
}

void baseDijkstra( graphInfo *graph, float *shortestDistance, int numVerts, int edgePerVert, int sourceVert, float *weightMatrix)
{

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
            weightMatrix[i * numVerts + graph->edgeArr[graph->vertexArr[i]+j]] = graph->weightArr[graph->vertexArr[i]+j];
        }
    }
}