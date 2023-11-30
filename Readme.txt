# Matrix Multiplication Program

## Standard Matrix Multiplication without Tiling

1. **Google Colab Setup:**
   - Open a new document (new notebook) in Google Colab.
   - Change the runtime to GPU (e.g., TF4).
   - Verify the presence of the nvcc compiler by running the command: `!nvcc --version`.
   - If the compiler is not installed or you encounter a "command not found" error, reinstall the compiler and required packages.
   - Once the compiler is ready, compile the main code as provided.
   - After compilation, run the script with default settings or specify user-defined variables (matrices and block sizes).
   - The total execution time is approximately 17 minutes based on my settings.

## Tiling Version of the Code

2. **Google Colab Setup:**
   - Use the same notebook opened previously or open a new one (ensure the nvcc compiler is running).
   - Run the provided scripts for the tiling version, resulting in a total program execution time of around 9 minutes.
   - Specify exact dimensions as needed.
   - The output file is labeled "matrix_multi_tiling_results"; feel free to modify labels as required.

## Sequential Code

3. **Sequential Code Execution:**
   - Ensure you wrap the sequential code with the magic cell "%%writefile".
   - Run the code with the necessary dimensions or use a script for comparisons across different dimensions.
