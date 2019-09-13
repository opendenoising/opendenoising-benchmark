# Installation guide.

Here we present the steps to install the OpenDenoising benchmark.

## 1. Creating a virtual environment

In order to use the present benchmark, we recommend creating a virtual environment for it. On a computer having Python3
installed, a GPU CUDA-compatible, CUDA installed:

* Install virtualenv:
```sh
sudo apt install virtualenv
```

* Create a virtual environment anywhere with VENV-NAME the name given to the environment:
```sh
virtualenv --system-site-packages -p python3 ~/virtualenvironments/VENV_NAME
```

* Activate the venv:
```sh
source ~/virtualenvironments/VENV_NAME/bin/activate
```
If the last command succeeded, the command line should now be preceded by (VENV_NAME).
The virtual environment can be exited using:
```sh
deactivate
```

## 2. Python package requirements

Here is a list of required packages to run OpenDenoising benchmark:

**Requirements for GPU Users**

```
Keras==2.2.4
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
keras2onnx==1.5.0
matplotlib==3.1.0
numpy==1.16.4
onnx==1.5.0
onnx-tf==1.3.0
onnxruntime-gpu==0.5.0
opencv-python==4.1.0.25
pandas==0.24.2
Pillow==6.0.0
PyGithub==1.43.8
scikit-image==0.15.0
scipy==1.3.0
seaborn==0.9.0
six==1.12.0
tensorboard==1.14.0
tensorflow-gpu==1.14.0
tf2onnx==1.5.1
torch==1.2.0
torchvision==0.4.0
tqdm==4.32.2
```

To install them, you can simply go to the project's root, and run the following command,

```
pip install -r requirements_gpu.txt
```

**Requirements for CPU Users**

```
Keras==2.2.4
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
keras2onnx==1.5.0
matplotlib==3.1.0
numpy==1.16.4
onnx==1.5.0
onnx-tf==1.3.0
onnxruntime==0.5.0
opencv-python==4.1.0.25
pandas==0.24.2
Pillow==6.0.0
PyGithub==1.43.8
scikit-image==0.15.0
scipy==1.3.0
seaborn==0.9.0
six==1.12.0
tensorboard==1.14.0
tensorflow==1.14.0
tf2onnx==1.5.1
torch==1.2.0
torchvision==0.4.0
tqdm==4.32.2
```

To install them, you can simply go to the project's root, and run the following command,

```
pip install -r requirements_cpu.txt
```

We recommend you to use a Virtual Environment to run the benchmark.

__Note:__ If you want to run Matlab code in the benchmark, you need to have a Matlab of version at least 2018b, with a valid license.
You need to install [Matlab's Python Engine](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html).

## 3. [Optional] Matlab dependencies

Our Matlab support covers Matlab Deep Learning Toolbox (training and inference) and Matconvnet (only inference). Here
we detail the steps for installing Matlab's dependencies.

__Warning for Matlab users:__

If you will use Matlab Deep Learning toolbox with recent GPU cards (such as RTX 2080 ti), you should add the Following
lines to your startup script:

```Matlab
warning off parallel:gpu:device:DeviceLibsNeedsRecompiling
try
    gpuArray.eye(2)^2;
catch ME
end
try
    nnet.internal.cnngpu.reluForward(1);
catch ME
end
```

otherwise, when you run a MatlabModel you can run into errors. For more informations, [take a look on this post](https://fr.mathworks.com/matlabcentral/answers/439616-does-matlab-2018b-support-nvidia-geforce-2080-ti-rtx-for-creating-training-implementing-deep-learnin).
You should also add "./OpenDenoising/data/" to Matlab's Path by using [Set Path](https://fr.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html).

### 3.1. Adding the Benchmark to matlab's path.

Let "PATH_TO_BENCHMARK" denote the path to the OpenDenoising folder in your computer. To add it to Matlab's main path,
you need to modify the file "pathdef.m". If you are on Windows, all you have to do is use "set path" tool on Matlab's
main window. However if you are using Linux and you do not have the rights to modify it, you can run the following commands
on the terminal,

```sh
sudo nano /usr/local/MATLAB/R2018b/toolbox/local/pathdef.m
```

This will open nano on the needed file with the right permissions. You need to write the following line before the default
entries,

```sh
'PATH_TO_BENCHMARK/data:', ...
```

__Remark:__ If you are using any third-party software that depends on Matlab (such as BM3D), you also need to include it to the
pathdef.

### 3.2. Installing Matlab's Python engine

Open an terminal, then, go to matlab engine setup folder,

```
cd /usr/local/MATLAB/R2018b/extern/engines/python
```

Following [matlab's instructions](https://fr.mathworks.com/help/matlab/matlab_external/install-matlab-engine-api-for-python-in-nondefault-locations.html),
install the engine on your venv folder,

```
sudo $VENVROOT/bin/python setup.py install --prefix="$VENVROOT/"
```

Notice that, since we are running the sudo command, the command line will ["ignore"](https://stackoverflow.com/questions/15441440/sudo-python-runs-old-python-version)
all your aliases, so you need to specify the path to your venv python. Equally, the --prefix option specify where matlab
will output its files, so that you can run its engine. To test if your installation was succesfull, you can try to execute the following
python script:

```
import matlab.engine
eng = matlab.engine.start_matlab()
x = 4.0
eng.workspace['y'] = x
a = eng.eval('sqrt(y)')
print(a)
```

### 3.3. Matconvnet installation

__Remark:__ be sure to add Matconvnet to Matlab default path.

#### 3.3.1. Setting up multiple CUDA versions

If you will use [Matconvnet toolbox](http://www.vlfeat.org/matconvnet/), you need to install gcc-6 by running

```
sudo apt install gcc-6 g++-6
```

before compiling the library on Matlab. Moreover, since the toolbox requires CUDA 9.1 (which is a different version from
  Tensorflow's requirement), you need to install multiple CUDA's on your system (which are independent from each other).
Assuming you already have on your system a CUDA version different from 9.1, you need to follow these steps,

* Download CUDA Toolkit 9.1 from NVIDIA's [website](https://developer.nvidia.com/cuda-91-download-archive), then execute
it using the '--override' option, as follows:

```sh
./cuda_9.1.85_387.26_linux.run --override
```

The override option is needed, so that the installer won't fail because of driver version
(if you have a newer version of CUDA, it is likely that you have a more recent driver). Once you run the previous line,
the installer will ask you the following questions,

```
You are attempting to install on an unsupported configuration. Do you wish to continue?
> y
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 387.26?
> n
Install the CUDA 9.1 Toolkit?
> y
Install the CUDA 9.1 Samples?
> y
Enter CUDA Samples Location
> Default location
Enter Toolkit Location
> Default location
Do you want to install a symbolic link at /usr/local/cuda?
> n
```

By doing this, CUDA 9.1 will be installed on /usr/local/cuda-9.1. The crucial part of having two CUDAs installed,
without messing your previous installation, is to not create the symbolic link between cuda-9.1 folder and CUDA folder.
Moreover, such choice does not stop you from using CUDA-9.1 in Matconvnet.

* Add the different CUDA paths to LD_LIBRARY_PATH:

```sh
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/cuda-9.1/lib64:\$LD_LIBRARY_PATH
```

At the end of this process, your LD_LIBRARY_PATH should contain the following line as substring:

```
/usr/local/cuda/lib64:/usr/local/cuda-10.1/lib64:/usr/local/cuda-9.1/lib64
```

#### 3.3.2. Compiling Matconvnet library

Go to the directory where you extracted matconvnet files, then, after lauching matlab, use the following commands,

```
cd matlab
CudaPath = "/usr/local/cuda-9.1";
vl_compilenn('EnableGpu', true, 'CudaRoot', CudaPath, 'EnableCudnn', true)
```

vl_compilenn is a matlab function that will compilate matconvnet library. Here's what each option means,

```
EnableGpu: enables GPU usage by matconvnet.
CudaRoot: indicates the path to Cuda's root folder.
EnableCudnn: enables matconvnet to use cudnn acceleration.
```

__obs (27.06.19):__ For matlab 2018b users, matconvnet compiling script happens to have a bug, which can be easily corrected by replacing **line 620** by,

```Matlab
args = horzcat({'-outdir', mex_dir}, ...
flags.base, flags.mexlink, ...
'-R2018a',...
{['LDFLAGS=$LDFLAGS ' strjoin(flags.mexlink_ldflags)]}, ...
{['LDOPTIMFLAGS=$LDOPTIMFLAGS ' strjoin(flags.mexlink_ldoptimflags)]}, ...
{['LINKLIBS=' strjoin(flags.mexlink_linklibs) ' $LINKLIBS']}, ...
objs);
```

and **line 359** to:

```
flags.mexlink = {'-lmwblas'};
```

For more informations, consult [this github page](https://github.com/vlfeat/matconvnet/issues/1143). After compiling the
libary, you should consider adding Matconvnet to Matlab's path by using [Set Path](https://fr.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html).

## 4. Check Driver requirements

* [Tensorflow requirements](https://www.tensorflow.org/install/source#tested_build_configurations)
* [Pytorch requirements](https://pytorch.org/get-started/locally/)
* [Matlab requirements](https://fr.mathworks.com/help/parallel-computing/gpu-support-by-release.html)
* [OnnxRuntime requirements](https://github.com/microsoft/onnxruntime)

|    Framework    | Cuda Version | Gcc Compiler |
|:---------------:|:------------:|:------------:|
| Tensorflow 1.14 |     10.0     |       7      |
|   Matlab 2018b  |      9.1     |       6      |
|     Pytorch     |     10.0     |       -      |
|   Onnxruntime   |     10.0     |       -      |
