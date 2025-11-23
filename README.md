# \# Neural Architecture Search (NAS) with Genetic Algorithm - Project Repor# Neural Architecture Search (NAS) with Genetic Algorithm - Project Report

# 

# \*\*Date:\*\* November 18, 2025  

# \*\*Project:\*\* NAS-GA-Basic Implementation  

# \*\*Location:\*\* `c:\\mudrik\\IITJ\\AI\\assign2\\nas-ga-basic`

# 

# ---

# 

# \## Project Overview

# 

# This project implements a Neural Architecture Search (NAS) system using Genetic Algorithms (GA) to automatically discover optimal CNN architectures for the CIFAR-10 dataset.

# 

# \### Key Components

# \- \*\*Dataset:\*\* CIFAR-10 (5000 training samples, 1000 validation samples)

# \- \*\*Algorithm:\*\* Genetic Algorithm with population-based evolution

# \- \*\*Framework:\*\* PyTorch with CUDA support

# \- \*\*Hardware:\*\* NVIDIA GeForce RTX 4060 Laptop GPU

# 

# \### Configuration Parameters

# \- Population Size: 10

# \- Generations: 5

# \- Mutation Rate: 0.3

# \- Crossover Rate: 0.7

# \- Batch Size: 256

# 

# ---

# 

# \## Work Completed

# 

# \### 1. Initial Setup and Execution Guidance

# \- Reviewed project structure and identified main entry point (`nas\_run.py`)

# \- Documented execution instructions

# \- Identified required dependencies: `torch` and `torchvision`

# 

# \### 2. Dependency Resolution

# 

# \#### Issue #1: ml\_dtypes Compatibility Error

# \*\*Error Message:\*\*

# ```

# AttributeError: module 'ml\_dtypes' has no attribute 'float4\_e2m1fn'. 

# Did you mean: 'float8\_e4m3fn'?

# ```

# 

# \*\*Root Cause:\*\*

# \- Version conflict between `ml\_dtypes` (v0.3.2) and other packages

# \- TensorFlow 2.16.1 required `ml\_dtypes~=0.3.1`, but newer version needed

# 

# \*\*Resolution:\*\*

# \- Upgraded `ml\_dtypes` from v0.3.2 to v0.5.4

# \- Conflict with TensorFlow noted but not critical (project uses PyTorch only)

# 

# \### 3. GPU Acceleration Implementation

# 

# \#### Issue #2: CPU-Only PyTorch Installation

# \*\*Problem:\*\*

# \- PyTorch 2.2.2 was installed without CUDA support

# \- Code was running on CPU despite having NVIDIA RTX 4060 GPU available

# \- `torch.cuda.is\_available()` returned `False`

# 

# \*\*System Verification:\*\*

# ```

# GPU: NVIDIA GeForce RTX 4060 Laptop GPU

# Driver Version: 581.57

# CUDA Version: 13.0

# ```

# 

# \*\*Resolution Steps:\*\*

# 1\. Uninstalled CPU-only versions:

# &nbsp;  - `torch 2.2.2`

# &nbsp;  - `torchvision 0.17.2`

# 

# 2\. Installed CUDA-enabled versions:

# &nbsp;  ```bash

# &nbsp;  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# &nbsp;  ```

# &nbsp;  - `torch 2.5.1+cu121`

# &nbsp;  - `torchvision 0.20.1+cu121`

# 

# 3\. Verified GPU detection:

# &nbsp;  - CUDA available: ✅ True

# &nbsp;  - GPU name: NVIDIA GeForce RTX 4060 Laptop GPU

# 

# \### 4. Code Bug Fixes

# 

# \#### Issue #3: Directory Creation Error

# \*\*Error Message:\*\*

# ```

# FileExistsError: \[WinError 183] Cannot create a file when that file already exists: 

# 'C:\\\\mudrik\\\\IITJ\\\\AI\\\\assign2\\\\nas-ga-basic\\\\outputs\\\\run\_1'

# ```

# 

# \*\*Root Cause:\*\*

# \- Script counted directories with "log" in name: `all\_logs = \[i for i in os.listdir(...) if 'log' in i]`

# \- But directories were named `run\_1`, `run\_2`, etc. (no "log" in name)

# \- This caused incorrect run number calculation and attempted to recreate existing directory

# 

# \*\*Code Changes Made:\*\*

# ```python

# \# BEFORE (Buggy)

# all\_logs = \[i for i in os.listdir(os.path.join(parent, 'outputs')) if 'log' in i]

# os.mkdir(os.path.join(parent, 'outputs', f'run\_{len(all\_logs)+1}'))

# 

# \# AFTER (Fixed)

# all\_runs = \[i for i in os.listdir(os.path.join(parent, 'outputs')) if 'run\_' in i]

# run\_num = len(all\_runs) + 1

# run\_dir = os.path.join(parent, 'outputs', f'run\_{run\_num}')

# if not os.path.exists(run\_dir):

# &nbsp;   os.mkdir(run\_dir)

# ```

# 

# \*\*Additional Fixes:\*\*

# \- Updated all references from `all\_logs` to `run\_num` throughout the script

# \- Changed file path references to use `run\_dir` variable for consistency

# 

# ---

# 

# \## Execution Results

# 

# \### Run 1 (CPU - Incomplete)

# \- \*\*Device:\*\* CPU

# \- \*\*Status:\*\* Interrupted during Generation 1

# \- \*\*Location:\*\* `outputs/run\_1/`

# \- \*\*Progress:\*\* Evaluating architecture 2/10 when stopped

# 

# \### Run 2 (GPU - In Progress)

# \- \*\*Device:\*\* CUDA (NVIDIA GeForce RTX 4060)

# \- \*\*Status:\*\* Running successfully

# \- \*\*Location:\*\* `outputs/run\_2/`

# \- \*\*GPU Metrics:\*\*

# &nbsp; - Utilization: 29%

# &nbsp; - Memory Usage: 1213 MiB / 8188 MiB

# &nbsp; - Power Draw: 52W / 93W

# &nbsp; - Temperature: 67°C

# 

# ---

# 

# \## Performance Improvements

# 

# \### GPU vs CPU Comparison

# \- \*\*Before:\*\* Running on CPU (slow, estimated 15-30 minutes)

# \- \*\*After:\*\* Running on GPU (5-20x faster expected)

# \- \*\*GPU Acceleration:\*\* Successfully utilizing CUDA cores for neural network training

# 

# ---

# 

# \## Technical Stack

# 

# \### Dependencies

# ```

# torch==2.5.1+cu121

# torchvision==0.20.1+cu121

# ml\_dtypes==0.5.4

# numpy==1.26.4

# pillow==10.2.0

# ```

# 

# \### Python Environment

# \- Python 3.11

# \- Windows OS

# \- CUDA 12.1 (compatible with CUDA 13.0 driver)

# 

# ---

# 

# \## Known Issues and Warnings

# 

# \### Dependency Conflicts (Non-Critical)

# 1\. \*\*facenet-pytorch 2.6.0\*\* requires:

# &nbsp;  - `torch<2.3.0,>=2.2.0` (we have 2.5.1+cu121)

# &nbsp;  - `torchvision<0.18.0,>=0.17.0` (we have 0.20.1+cu121)

# &nbsp;  

# 2\. \*\*open-webui 0.6.15\*\* requires:

# &nbsp;  - `pillow==11.2.1` (we have 10.2.0)

# 

# \*\*Impact:\*\* These conflicts don't affect the NAS project as it doesn't use facenet-pytorch or open-webui.

# 

# ---

# 

# \## Files Modified

# 

# 1\. \*\*nas\_run.py\*\*

# &nbsp;  - Fixed directory creation logic (lines 14-18)

# &nbsp;  - Updated run number tracking (line 50)

# &nbsp;  - Updated output path handling (line 64)

# 

# ---

# 

# \## Output Structure

# 

# ```

# outputs/

# ├── run\_1/

# │   └── nas\_run.log (CPU run - incomplete)

# └── run\_2/

# &nbsp;   ├── nas\_run.log (GPU run - in progress)

# &nbsp;   └── best\_arch.pkl (will be created upon completion)

# ```

# 

# ---

# 

# \## Lessons Learned

# 

# 1\. \*\*Always verify CUDA installation\*\* when working with PyTorch on GPU systems

# 2\. \*\*Check PyPI index URLs\*\* - default pip install may not include CUDA support

# 3\. \*\*Test directory logic\*\* with edge cases to avoid file system errors

# 4\. \*\*Monitor GPU utilization\*\* using `nvidia-smi` to confirm GPU usage

# 5\. \*\*Version conflicts\*\* in dependency resolver may not always be critical

# 

# ---

# 

# \## Next Steps

# 

# 1\. Monitor `run\_2` completion

# 2\. Analyze best architecture found by genetic algorithm

# 3\. Evaluate final model performance on full test set

# 4\. Consider tuning hyperparameters (population size, generations, mutation rate)

# 5\. Potentially increase training samples for better architecture search

# 

# ---

# 

# \## Conclusion

# 

# Successfully configured and deployed a Neural Architecture Search system with GPU acceleration. Resolved multiple technical issues including dependency conflicts, GPU detection, and code bugs. The system is now running efficiently on NVIDIA RTX 4060 GPU with proper CUDA support.

# 

# \*\*Status:\*\* ✅ Operational and Running on GPU

# t

# 

# \*\*Date:\*\* November 18, 2025  

# \*\*Project:\*\* NAS-GA-Basic Implementation  

# \*\*Location:\*\* `c:\\mudrik\\IITJ\\AI\\assign2\\nas-ga-basic`

# 

# ---

# 

# \## Project Overview

# 

# This project implements a Neural Architecture Search (NAS) system using Genetic Algorithms (GA) to automatically discover optimal CNN architectures for the CIFAR-10 dataset.

# 

# \### Key Components

# \- \*\*Dataset:\*\* CIFAR-10 (5000 training samples, 1000 validation samples)

# \- \*\*Algorithm:\*\* Genetic Algorithm with population-based evolution

# \- \*\*Framework:\*\* PyTorch with CUDA support

# \- \*\*Hardware:\*\* NVIDIA GeForce RTX 4060 Laptop GPU

# 

# \### Configuration Parameters

# \- Population Size: 10

# \- Generations: 5

# \- Mutation Rate: 0.3

# \- Crossover Rate: 0.7

# \- Batch Size: 256

# 

# ---

# 

# \## Work Completed

# 

# \### 1. Initial Setup and Execution Guidance

# \- Reviewed project structure and identified main entry point (`nas\_run.py`)

# \- Documented execution instructions

# \- Identified required dependencies: `torch` and `torchvision`

# 

# \### 2. Dependency Resolution

# 

# \#### Issue #1: ml\_dtypes Compatibility Error

# \*\*Error Message:\*\*

# ```

# AttributeError: module 'ml\_dtypes' has no attribute 'float4\_e2m1fn'. 

# Did you mean: 'float8\_e4m3fn'?

# ```

# 

# \*\*Root Cause:\*\*

# \- Version conflict between `ml\_dtypes` (v0.3.2) and other packages

# \- TensorFlow 2.16.1 required `ml\_dtypes~=0.3.1`, but newer version needed

# 

# \*\*Resolution:\*\*

# \- Upgraded `ml\_dtypes` from v0.3.2 to v0.5.4

# \- Conflict with TensorFlow noted but not critical (project uses PyTorch only)

# 

# \### 3. GPU Acceleration Implementation

# 

# \#### Issue #2: CPU-Only PyTorch Installation

# \*\*Problem:\*\*

# \- PyTorch 2.2.2 was installed without CUDA support

# \- Code was running on CPU despite having NVIDIA RTX 4060 GPU available

# \- `torch.cuda.is\_available()` returned `False`

# 

# \*\*System Verification:\*\*

# ```

# GPU: NVIDIA GeForce RTX 4060 Laptop GPU

# Driver Version: 581.57

# CUDA Version: 13.0

# ```

# 

# \*\*Resolution Steps:\*\*

# 1\. Uninstalled CPU-only versions:

# &nbsp;  - `torch 2.2.2`

# &nbsp;  - `torchvision 0.17.2`

# 

# 2\. Installed CUDA-enabled versions:

# &nbsp;  ```bash

# &nbsp;  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# &nbsp;  ```

# &nbsp;  - `torch 2.5.1+cu121`

# &nbsp;  - `torchvision 0.20.1+cu121`

# 

# 3\. Verified GPU detection:

# &nbsp;  - CUDA available: ✅ True

# &nbsp;  - GPU name: NVIDIA GeForce RTX 4060 Laptop GPU

# 

# \### 4. Code Bug Fixes

# 

# \#### Issue #3: Directory Creation Error

# \*\*Error Message:\*\*

# ```

# FileExistsError: \[WinError 183] Cannot create a file when that file already exists: 

# 'C:\\\\mudrik\\\\IITJ\\\\AI\\\\assign2\\\\nas-ga-basic\\\\outputs\\\\run\_1'

# ```

# 

# \*\*Root Cause:\*\*

# \- Script counted directories with "log" in name: `all\_logs = \[i for i in os.listdir(...) if 'log' in i]`

# \- But directories were named `run\_1`, `run\_2`, etc. (no "log" in name)

# \- This caused incorrect run number calculation and attempted to recreate existing directory

# 

# \*\*Code Changes Made:\*\*

# ```python

# \# BEFORE (Buggy)

# all\_logs = \[i for i in os.listdir(os.path.join(parent, 'outputs')) if 'log' in i]

# os.mkdir(os.path.join(parent, 'outputs', f'run\_{len(all\_logs)+1}'))

# 

# \# AFTER (Fixed)

# all\_runs = \[i for i in os.listdir(os.path.join(parent, 'outputs')) if 'run\_' in i]

# run\_num = len(all\_runs) + 1

# run\_dir = os.path.join(parent, 'outputs', f'run\_{run\_num}')

# if not os.path.exists(run\_dir):

# &nbsp;   os.mkdir(run\_dir)

# ```

# 

# \*\*Additional Fixes:\*\*

# \- Updated all references from `all\_logs` to `run\_num` throughout the script

# \- Changed file path references to use `run\_dir` variable for consistency

# 

# ---

# 

# \## Execution Results

# 

# \### Run 1 (CPU - Incomplete)

# \- \*\*Device:\*\* CPU

# \- \*\*Status:\*\* Interrupted during Generation 1

# \- \*\*Location:\*\* `outputs/run\_1/`

# \- \*\*Progress:\*\* Evaluating architecture 2/10 when stopped

# 

# \### Run 2 (GPU - In Progress)

# \- \*\*Device:\*\* CUDA (NVIDIA GeForce RTX 4060)

# \- \*\*Status:\*\* Running successfully

# \- \*\*Location:\*\* `outputs/run\_2/`

# \- \*\*GPU Metrics:\*\*

# &nbsp; - Utilization: 29%

# &nbsp; - Memory Usage: 1213 MiB / 8188 MiB

# &nbsp; - Power Draw: 52W / 93W

# &nbsp; - Temperature: 67°C

# 

# ---

# 

# \## Performance Improvements

# 

# \### GPU vs CPU Comparison

# \- \*\*Before:\*\* Running on CPU (slow, estimated 15-30 minutes)

# \- \*\*After:\*\* Running on GPU (5-20x faster expected)

# \- \*\*GPU Acceleration:\*\* Successfully utilizing CUDA cores for neural network training

# 

# ---

# 

# \## Technical Stack

# 

# \### Dependencies

# ```

# torch==2.5.1+cu121

# torchvision==0.20.1+cu121

# ml\_dtypes==0.5.4

# numpy==1.26.4

# pillow==10.2.0

# ```

# 

# \### Python Environment

# \- Python 3.11

# \- Windows OS

# \- CUDA 12.1 (compatible with CUDA 13.0 driver)

# 

# ---

# 

# \## Known Issues and Warnings

# 

# \### Dependency Conflicts (Non-Critical)

# 1\. \*\*facenet-pytorch 2.6.0\*\* requires:

# &nbsp;  - `torch<2.3.0,>=2.2.0` (we have 2.5.1+cu121)

# &nbsp;  - `torchvision<0.18.0,>=0.17.0` (we have 0.20.1+cu121)

# &nbsp;  

# 2\. \*\*open-webui 0.6.15\*\* requires:

# &nbsp;  - `pillow==11.2.1` (we have 10.2.0)

# 

# \*\*Impact:\*\* These conflicts don't affect the NAS project as it doesn't use facenet-pytorch or open-webui.

# 

# ---

# 

# \## Files Modified

# 

# 1\. \*\*nas\_run.py\*\*

# &nbsp;  - Fixed directory creation logic (lines 14-18)

# &nbsp;  - Updated run number tracking (line 50)

# &nbsp;  - Updated output path handling (line 64)

# 

# ---

# 

# \## Output Structure

# 

# ```

# outputs/

# ├── run\_1/

# │   └── nas\_run.log (CPU run - incomplete)

# └── run\_2/

# &nbsp;   ├── nas\_run.log (GPU run - in progress)

# &nbsp;   └── best\_arch.pkl (will be created upon completion)

# ```

# 

# ---

# 

# \## Lessons Learned

# 

# 1\. \*\*Always verify CUDA installation\*\* when working with PyTorch on GPU systems

# 2\. \*\*Check PyPI index URLs\*\* - default pip install may not include CUDA support

# 3\. \*\*Test directory logic\*\* with edge cases to avoid file system errors

# 4\. \*\*Monitor GPU utilization\*\* using `nvidia-smi` to confirm GPU usage

# 5\. \*\*Version conflicts\*\* in dependency resolver may not always be critical

# 

# ---

# 

# \## Next Steps

# 

# 1\. Monitor `run\_2` completion

# 2\. Analyze best architecture found by genetic algorithm

# 3\. Evaluate final model performance on full test set

# 4\. Consider tuning hyperparameters (population size, generations, mutation rate)

# 5\. Potentially increase training samples for better architecture search

# 

# ---

# 

# \## Conclusion

# 

# Successfully configured and deployed a Neural Architecture Search system with GPU acceleration. Resolved multiple technical issues including dependency conflicts, GPU detection, and code bugs. The system is now running efficiently on NVIDIA RTX 4060 GPU with proper CUDA support.

# 

# \*\*Status:\*\* ✅ Operational and Running on GPU



