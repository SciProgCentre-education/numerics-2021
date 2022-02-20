# MIPT numerical methods seminars


If you want to run `python` jupyter notebooks with `C/C++/CUDA` extensions on
[Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)
or [Datalore](https://datalore.jetbrains.com/notebooks)
clone `NOA`,  executing inside your notebook:

```python
!git clone https://github.com/grinisrit/noa.git
noa_location = 'noa'
```

Also, make sure that `ninja` and `g++-9` or higher are available. The following commands will do that for you:
```python
!pip install Ninja
!add-apt-repository ppa:ubuntu-toolchain-r/test -y
!apt update
!apt upgrade -y
!apt install gcc-9 g++-9
!update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 --slave /usr/bin/g++ g++ /usr/bin/g++-9
```

Finally, for `GPU` development install `CUDA` 11.2 or higher if it's not available: 
```python
!sudo apt-get update && sudo apt-get install cuda-nvcc-11-2 -y
```
Check installation:

```
!gcc --version
!g++ --version
!nvcc --version
```