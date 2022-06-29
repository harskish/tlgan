## Setup
1. Install anaconda or miniconda
2. Install git, then clone respository: `git clone https://github.com/harskish/tlgan/`
3. Create environment: `conda create -n tlgan python=3.9`
4. Activate environment: `conda activate tlgan`
5. Install dependencies 
    - On NVIDIA Ampere GPUs (3000 series) or newer:<br> `conda env update -f env_cu11.yml --prune`
    - On older GPUs:<br> `conda env update -f env_cu10.yml --prune`
6. Setup submodules: `git submodule update --init --recursive`

### CUDA setup
The networks (based on StyleGAN2) contain custom CUDA kernels for improved performance.
1. Install CUDA toolkit (match the version in env_cuXX.yml)
2. On Windows: install and open 'x64 Native Tools Command Prompt for VS 2019'
    - Visual Studio 2019 Community Edition contains the required tools

### Interactive viewers (optional)
The interactive viewers (<i>visualize.py</i> and <i>grid_viz.py</i>) benefit in performance from having access to a version of PyCUDA compiled with OpenGL support

#### Windows
Install the included dependencies:<br/> 
`pip install bin/cuXXX/*`

#### Linux
1. Install CUDA toolkit (match the version in env_cuXX.yml)
2. Download pycuda sources from: https://pypi.org/project/pycuda/#files
3. Extract files: `tar -xzf pycuda-VERSION.tar.gz`
4. Configure: `python configure.py --cuda-enable-gl --cuda-root=/path/to/cuda`
5. Compile and install: `make install`