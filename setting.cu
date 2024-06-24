#include <stdio.h>

int main()
{
    cudaDeviceProp  prop;

    int count;
    cudaGetDeviceCount( &count );

    for (int i=0; i< count; i++) {
        cudaGetDeviceProperties( &prop, i );
        printf( "   --- General Information for device %d ---\n", i );
        printf( "Name:  %s\n", prop.name );
        printf( "Compute capability:  %d.%d\n", prop.major, prop.minor );
        printf( "Clock rate:  %d\n", prop.clockRate );
        printf( "\n" );

        printf( "   --- Memory Information for device %d ---\n", i );
        printf( "Total global mem:  %ld\n", prop.totalGlobalMem );
        printf( "Total constant Mem:  %ld\n", prop.totalConstMem );
        printf( "Max mem pitch:  %ld\n", prop.memPitch );
        printf( "Texture Alignment:  %ld\n", prop.textureAlignment );
		printf( "\n" );

        printf( "   --- SM Information for device %d ---\n", i );
        printf( "SM count:  %d\n", prop.multiProcessorCount );
        printf( "Maximum resident threads per SM:  %d\n", prop.maxThreadsPerMultiProcessor );
        printf( "Maximum number of resident blocks per SM:  %d\n", prop.maxBlocksPerMultiProcessor );
        printf( "Shared mem per SM:  %ld\n", prop.sharedMemPerBlock );
        printf( "Registers per SM:  %d\n", prop.regsPerBlock );
        printf( "Threads in warp:  %d\n", prop.warpSize );
        printf( "Max threads per block:  %d\n", prop.maxThreadsPerBlock );
        printf( "Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2] );
        printf( "Max grid dimensions:  (%d, %d, %d)\n",prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2] );
        printf( "\n" );
    }

    return 0;
}