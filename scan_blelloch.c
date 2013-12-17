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
#define NUM_GROUPS (1)
#define DATA_SIZE (8)

#define STRINGFY(x) #x

const char* source_scan_blelloch_global = STRINGFY(
__kernel void kernel_scan(__global float* input, __global float* output) {
    int global_idx = get_global_id(0);
    int local_idx = get_local_id(0);
    int block_size = get_local_size(0);
    int group_id = get_group_id(0);

    output[global_idx] = input[global_idx];
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    //up-sweep phase
    int d = 1;
    for(int i = block_size >> 1; i > 0; i >>= 1) {
        if(global_idx < i) {
            int a = d*(2*global_idx + 1) - 1;
            int b = d*(2*global_idx + 2) - 1;

            output[b] += output[a];
        }

        mem_fence(CLK_GLOBAL_MEM_FENCE);

        d <<= 1;
    }

    //down-sweep phase
    if(global_idx == 0)
        output[block_size - 1] = 0;

    for(int i = 1; i < block_size; i <<= 1) {
        d >>= 1;

        if(global_idx < i) {
            int a = d*(2*global_idx + 1) - 1;
            int b = d*(2*global_idx + 2) - 1;

            float value = output[a];
            output[a] = output[b];
            output[b] += value;
        }

        mem_fence(CLK_GLOBAL_MEM_FENCE);
    }
}
);

const char* source_scan_blelloch_shared = STRINGFY(
__kernel void kernel_scan(__global float* input, __global float* output, __local float* sinput) {
    int global_idx = get_global_id(0);
    int local_idx = get_local_id(0);
    int block_size = get_local_size(0);
    int group_id = get_group_id(0);

    sinput[global_idx] = input[global_idx];
    mem_fence(CLK_GLOBAL_MEM_FENCE);

    //up-sweep phase
    int d = 1;
    for(int i = block_size >> 1; i > 0; i >>= 1) {
        if(global_idx < i) {
            int a = d*(2*global_idx + 1) - 1;
            int b = d*(2*global_idx + 2) - 1;

            sinput[b] += sinput[a];
        }

        mem_fence(CLK_GLOBAL_MEM_FENCE);

        d <<= 1;
    }

    //down-sweep phase
    if(global_idx == 0)
        sinput[block_size - 1] = 0;

    for(int i = 1; i < block_size; i <<= 1) {
        d >>= 1;

        if(global_idx < i) {
            int a = d*(2*global_idx + 1) - 1;
            int b = d*(2*global_idx + 2) - 1;

            float value = sinput[a];
            sinput[a] = sinput[b];
            sinput[b] += value;
        }

        mem_fence(CLK_GLOBAL_MEM_FENCE);
    }

    output[global_idx] = sinput[global_idx];
    mem_fence(CLK_GLOBAL_MEM_FENCE);
}
);

////////////////////////////////////////////////////////////////////////////////

int execute_kernel(cl_command_queue commands, cl_kernel kernel, cl_mem input, cl_mem output) {
    int err;

    err = 0;
    err |= clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
#ifdef SHARED
    err |= clSetKernelArg(kernel, 2, sizeof(cl_float)*DATA_SIZE/NUM_GROUPS, NULL);
#endif

    if (err != CL_SUCCESS) {
        printf("Error: Failed to set kernel arguments! %d\n", err);
        return err;
    }

    size_t global = DATA_SIZE;
    size_t local = DATA_SIZE/NUM_GROUPS;
    err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to execute kernel!\n");
        return err;
    }

    return err;
}

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
    cl_mem temporary;                   //

    int i = 0;
    for(i = 0; i < DATA_SIZE; i++)
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
    const char* source_reduce = source_scan_blelloch_shared;
#else
    const char* source_reduce = source_scan_blelloch_global;
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

    kernel = clCreateKernel(program, "kernel_scan", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Error: Failed to create compute kernel!\n");
        exit(1);
    }

    input = clCreateBuffer(context,  CL_MEM_READ_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);
    output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE, NULL, NULL);
    temporary = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * DATA_SIZE/NUM_GROUPS, NULL, NULL);
    if (!input || !output || !temporary) {
        printf("Error: Failed to allocate device memory!\n");
        exit(1);
    }    

    err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * DATA_SIZE, data, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to write to source array!\n");
        exit(1);
    }

    err = execute_kernel(commands, kernel, input, output);
    if(err != CL_SUCCESS) {
        exit(1);
    }

    clFinish(commands);

    err = clEnqueueReadBuffer(commands, output, CL_TRUE, 0, sizeof(float) * DATA_SIZE, results, 0, NULL, NULL );  
    if (err != CL_SUCCESS) {
        printf("Error: Failed to read output array! %d\n", err);
        exit(1);
    }

    printf("Result:\n");
    for(int i = 0; i < DATA_SIZE; i++) {
        printf("%f", results[i]);
        printf( (i+1) % 8 == 0 ? "\n" : "\t" );
    }
    printf("\n");

    clReleaseMemObject(input);
    clReleaseMemObject(output);
    clReleaseMemObject(temporary);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(commands);
    clReleaseContext(context);

    return 0;
}

