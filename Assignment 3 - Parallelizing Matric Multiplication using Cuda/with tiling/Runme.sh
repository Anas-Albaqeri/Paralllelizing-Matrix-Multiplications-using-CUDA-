%%bash
#!/bin/bash

# Compile the CUDA code
!nvcc -o matrix_multi_tiling matrix_multi_tiling.cu

# Specify the output file
output_file="matrix_mult_tiling_results.txt"

# Loop through matrix sizes and write results to the output file
for tile_size in 4 8 16 32 64; 
do
	for ((rowsA = 1000, colsA=3000, colsB=2000; rowsA <= 10000; rowsA += 1000, colsA = 3*rowsA, colsB = 2*rowsA))
	do
            echo "Running matrix multiplication for rowsA=$rowsA, colsA=$colsA, colsB=$colsB, tile=($tile_size, $tile_size)" >> $output_file

            # Run the compiled CUDA code with current matrix size and append results to the output file
            ./matrix_multi_tiling $rowsA $colsA $colsB $tile_size>> $output_file

            echo "-------------------------------------------------------" >> $output_file
	done
	    echo"#########################################################">> $output_file
done 