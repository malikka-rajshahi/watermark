import torch
import time
import argparse

def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory

def get_gpu_memory_usage():
    return torch.cuda.memory_allocated()

def continuous_matrix_multiply(duration, max_memory_usage=0.8):
    if not torch.cuda.is_available():
        #print("Error: GPU not available. This script requires a GPU.")
        return

    device = torch.device("cuda")
    total_memory = get_gpu_memory()
    max_memory = int(total_memory * max_memory_usage)
    
    start_time = time.time()
    matrix_size = 1000  # Starting size
    iteration = 0
    
    while time.time() - start_time < duration:
        try:
            # Create matrices
            matrix1 = torch.rand(matrix_size, matrix_size, device=device)
            matrix2 = torch.rand(matrix_size, matrix_size, device=device)
            
            # Perform matrix multiplication
            result = torch.matmul(matrix1, matrix2)
            
            # Clear variables to free up memory
            del matrix1, matrix2, result
            
            # Clear cache every other iteration
            if iteration % 2 == 0:
                torch.cuda.empty_cache()
            
            iteration += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                # Reduce matrix size and clear cache if we run out of memory
                matrix_size = int(matrix_size * 0.8)
                torch.cuda.empty_cache()
                #print(f"Reduced matrix size to {matrix_size} due to OOM error")
            else:
                raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory-Aware Continuous GPU Matrix Multiplication")
    parser.add_argument("duration", type=int, help="Duration to run in seconds")
    parser.add_argument("--max_memory", type=float, default=0.4, help="Maximum fraction of GPU memory to use (default: 0.8)")
    args = parser.parse_args()
    
    continuous_matrix_multiply(args.duration, args.max_memory)
