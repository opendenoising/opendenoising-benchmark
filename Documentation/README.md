# OpenDenoising documentation

In this folder you will find the source files for the OpenDenoising benchmark documentation. You can either access the documentation
by the [ReadTheDocs website](https://opendenoising-docs.readthedocs.io/en/latest/), or by compiling it yourself.

## Compiling the documentation

To compile the documentation, you need to first install the OpenDenoising benchmark project. To do so, follow these steps,

* Clone the OpenDenoising benchmark repository on a folder of your choice,

```sh
$ git clone https://github.com/opendenoising/benchmark
```

* If you do not have virtualenv installed, use the following command on your terminal,

```sh
$ sudo apt install virtualenv
```

* Create a virtual environment on a folder of your choice with VENV-NAME the name given to the environment:

```sh
$ virtualenv --system-site-packages -p python3 ~/virtualenvironments/VENV_NAME
```

* Activate the venv:

```sh
$ source ~/virtualenvironments/VENV_NAME/bin/activate
```

* On the project's root folder, run (GPU users),

```sh
$ pip install -r requirements_gpu.txt
```

* or (CPU users),

```sh
$ pip install -r requirements_cpu.txt
```

* Additionally, go to "./Documentation/" folder and run,

```sh
$ pip install -r docs_requirements.txt
```

This should install all the pre-requisites for compiling the documentation using Sphinx. Now, being on "./Documentation"
folder, run,

```sh
$ make html
```

to run the process of compilation.