#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <OpenCL/opencl.h>

#define SHARED
#define DATA_SIZE (16)

#define STRINGFY(x) #x

const char* source_reduce_global = STRINGFY(
__kernel void kernel_reduce(__global float* input, __global float* output) {
    int global_idx = get_global_id(0);
    int local_idx = get_local_id(0);
    int block_size = get_local_size(0);
    int group_id = get_group_id(0);

    for(int i = block_size/2; i > 0; i >>= 1) {
        if(local_idx < i)
            input[global_idx] += input[global_idx + i];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if(local_idx == 0)
        output[group_id] = input[global_idx];
}
);

const char* source_reduce_shared = STRINGFY(
__kernel void kernel_reduce(__global float* input, __global float* output, __local float* sinput) {
    int global_idx = get_global_id(0);
    int local_idx = get_local_id(0);
    int block_size = get_local_size(0);
    int group_id = get_group_id(0);

    sinput[local_idx] = input[global_idx];
    barrier(CLK_GLOBAL_MEM_FENCE);

    for(int i = block_size/2; i > 0; i >>= 1) {
        if(local_idx < i)
            sinput[local_idx] += sinput[local_idx + i];

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    if(local_idx == 0)
        output[group_id] = sinput[0];
}
);

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    int err;                            // error code returned from api calls

    float data[DATA_SIZE];              // original data set given to device
    float results[DATA_SIZE];           // results returned from device

    size_t global;                      // global domain size for our calculation
    size_t local;                       // local domain size for our calculation

    cl_device_id device_id;             // compute device id 
    cl_context context;                 // compute context
    cl_command_queue commands;          // compute command queue
    cl_program program;                 // compute program
    cl_kernel kernel;                   // compute kernel

    cl_mem input;                       // device memory used for the input array
    cl_mem output;                      // device memory used for the output array

    int i = 0;
    unsigned int count = DATA_SIZE;
    for(i = 0; i < count; i++)
        data[i] = i+1; 

    err = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group!\n");
        return EXIT_FAILURE;
    }

    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
        printf("Error: Failed to create a compute context!\n");
        return EXIT_FAILURE;
    }

    commands = clCreateCommandQueue(context, device_id, 0, &err);
    if (!commands) {
        printf("Error: Failed to create a command commands!\n");
        return EXIT_FAILURE;
    }

#ifdef SHARED
    const char* source_reduce = source_reduce_shared;
#else
    const char* source_reduce = source_reduce_global;
#endif

    program = clCreateProgramWithSource(context, 1, &source_reduce, NULL, &err);
    if (!program) {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;
    }

    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];

        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    kernel = clCreateKernel(program, "kernel_reduce", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
    if (!input || !output) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    

    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    err = 0;
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
#ifdef SHARED
    err |= clSetKernelArg(kernel, 2, sizeof(cl_float)*DATA_SIZE/4, NULL);
#endif
    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        exit(1);
    }

    err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to retrieve kernel work group info! %d\n", err);
        exit(1);
    }

    global = DATA_SIZE;
    local = DATA_SIZE/4;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to execute kernel!\n");
        return EXIT_FAILURE;
    }

    clFinish(commands);

    err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Result:\n");
    for(int i = 0; i < DATA_SIZE; i++) {
        printf("%f", results[i]);
        printf( (i+1) % 4 == 0 ? "\n" : "\t" );
    }
    printf("\n");

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

