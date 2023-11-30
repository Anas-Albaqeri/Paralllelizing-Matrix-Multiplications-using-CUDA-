#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

// Display a 2D matrix
void displayMatrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

int main()
{
    // Set matrix dimensions
    int rowsA = 10000;
    int colsA = 30000;
    int colsB = 20000;

    // Allocate memory on the host for input and output matrices
    float *matrixA, *matrixB, *matrixC;
    matrixA = (float *)malloc(rowsA * colsA * sizeof(float));
    matrixB = (float *)malloc(colsA * colsB * sizeof(float));
    matrixC = (float *)malloc(rowsA * colsB * sizeof(float));

    // Initialize input matrices with random values
    for (int i = 0; i < rowsA * colsA; i++)
        matrixA[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < colsA * colsB; i++)
        matrixB[i] = rand() / (float)RAND_MAX;

    // Set up OpenCL

    // Get the platform and device
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Create OpenCL buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * rowsA * colsA, matrixA, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * colsA * colsB, matrixB, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * rowsA * colsB, NULL, NULL);

    // Compile the OpenCL kernel
    const char *source = "#include <cl.cl>\n"; // Add the content of optimized_matrix_mult.cl here
    cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "multiplyMatrices", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &rowsA);
    clSetKernelArg(kernel, 4, sizeof(int), &colsA);
    clSetKernelArg(kernel, 5, sizeof(int), &colsB);

    // Set global and local work sizes
    size_t globalSize[2] = {rowsA, colsB};
    size_t localSize[2] = {16, 16};

    // Execute the kernel
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);
    clFinish(queue);

    // Read the result back to the host
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * rowsA * colsB, matrixC, 0, NULL, NULL);

    // Display the result matrix
    // displayMatrix(matrixC, rowsA, colsB);

    // Release OpenCL resources
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    // Free host memory
    free(matrixA);
    free(matrixB);
    free(matrixC);

    return 0;
}
_kernel void multiplyMatrices(__global const float *inputMatrixA, __global const float *inputMatrixB, __global float *outputMatrixC, int rowsA, int colsA, int colsB)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    // Ensure that the thread is within the valid matrix dimensions
    if (row < rowsA && col < colsB)
    {
        float result = 0;

        // Perform dot product for the given element of the output matrix
        for (int i = 0; i < colsA; i++)
        {
            result += inputMatrixA[row * colsA + i] * inputMatrixB[i * colsB + col];
        }

        // Save the result to the output matrix
        outputMatrixC[row * colsB + col] = result;
    }
}
