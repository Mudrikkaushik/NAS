import torch
import time
import psutil
import os

print("="*60)
print("GPU DIAGNOSTICS")
print("="*60)

# Check CUDA availability
print(f"\n1. CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"   Number of GPUs: {torch.cuda.device_count()}")
    
    # Memory info
    print(f"\n2. GPU Memory:")
    print(f"   Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   Allocated: {torch.cuda.memory_allocated(0) / 1e9:.4f} GB")
    print(f"   Cached: {torch.cuda.memory_reserved(0) / 1e9:.4f} GB")
    
    # Check cuDNN
    print(f"\n3. cuDNN:")
    print(f"   Enabled: {torch.backends.cudnn.enabled}")
    print(f"   Version: {torch.backends.cudnn.version()}")
    print(f"   Benchmark: {torch.backends.cudnn.benchmark}")
    print(f"   Deterministic: {torch.backends.cudnn.deterministic}")

# CPU info
print(f"\n4. CPU Info:")
print(f"   CPU Count: {psutil.cpu_count()}")
print(f"   CPU Percent: {psutil.cpu_percent()}%")

# Check DataLoader workers
print(f"\n5. DataLoader Configuration Check:")
print(f"   Recommended num_workers: {min(os.cpu_count(), 4)}")

# Test GPU vs CPU speed
print(f"\n6. Speed Test (Matrix Multiplication):")

size = 5000
iterations = 10

if torch.cuda.is_available():
    # GPU test
    device = torch.device('cuda')
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iterations):
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   GPU Time: {gpu_time:.4f}s")
    
    # Clear GPU memory
    del a, b, c
    torch.cuda.empty_cache()

# CPU test
device = torch.device('cpu')
start = time.time()
for _ in range(iterations):
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    c = torch.mm(a, b)
cpu_time = time.time() - start
print(f"   CPU Time: {cpu_time:.4f}s")

if torch.cuda.is_available():
    print(f"   Speedup: {cpu_time/gpu_time:.2f}x")

# Test CNN forward pass
print(f"\n7. CNN Forward Pass Test:")
from model_cnn import CNN

# Create a sample architecture
sample_genes = {
    'num_conv': 3,
    'conv_configs': [
        {'filters': 64, 'kernel_size': 3},
        {'filters': 128, 'kernel_size': 3},
        {'filters': 256, 'kernel_size': 3}
    ],
    'pool_type': 'max',
    'activation': 'relu',
    'fc_units': 256
}

batch_size = 256
input_tensor = torch.randn(batch_size, 3, 32, 32)

if torch.cuda.is_available():
    # GPU test
    model_gpu = CNN(sample_genes).cuda()
    input_gpu = input_tensor.cuda()
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        output = model_gpu(input_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"   GPU Forward Pass (100 iterations): {gpu_time:.4f}s")
    
    del model_gpu, input_gpu, output
    torch.cuda.empty_cache()

# CPU test
model_cpu = CNN(sample_genes).cpu()
input_cpu = input_tensor.cpu()

start = time.time()
for _ in range(100):
    output = model_cpu(input_cpu)
cpu_time = time.time() - start
print(f"   CPU Forward Pass (100 iterations): {cpu_time:.4f}s")

if torch.cuda.is_available():
    print(f"   Speedup: {cpu_time/gpu_time:.2f}x")

print("\n" + "="*60)
print("POTENTIAL ISSUES TO CHECK:")
print("="*60)
print("\n1. DataLoader num_workers=0 (default)")
print("   - This causes CPU bottleneck in data loading")
print("   - Recommendation: Set num_workers=4 and pin_memory=True")

print("\n2. cuDNN benchmark not enabled")
print("   - Can speed up training for fixed-size inputs")
print("   - Add: torch.backends.cudnn.benchmark = True")

print("\n3. Small batch size relative to GPU capacity")
print("   - Current batch_size=256 might be too small")
print("   - Try increasing to 512 or 1024 if memory allows")

print("\n4. Frequent GPU cache clearing")
print("   - torch.cuda.empty_cache() adds overhead")
print("   - Only call when necessary (memory issues)")

print("\n5. Data transfer overhead")
print("   - Moving data to GPU every batch can be slow")
print("   - Use pin_memory=True in DataLoader")

print("\n" + "="*60)
