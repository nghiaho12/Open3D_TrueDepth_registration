![](http://nghiaho.com/wp-content/uploads/2020/10/animated_mesh.png)

# Open3D_TrueDepth_registration
Automatic point cloud registration. Currently, I've only tested it on data from Apple's TrueDepth camera. Adapating it to other system should not be hard.

# Dataset
Download the test dataset and extract it somewhere
```
curl -O http://nghiaho.com/uploads/box_can.zip
```

# Git submodule
Call the following to pull in the pybind11 submodule.
```
git submodule update --init --recursive
```

# Python libraries
You'll need the following Python libraries installed
- Open3D
- Numpy
- Scipy
- OpenCV

All the above can be installing using pip
```
pip install open3d
pip install numpy
pip install scipy
pip install opencv-python
```

# Compiling pose_graph.cpp
You'll need the following libraries installed
- Eigen (http://eigen.tuxfamily.org/index.php?title=Main_Page)
- Ceres Solver (http://ceres-solver.org/)

On Ubuntu you can try
```
sudo apt-get install libeigen3-dev
sudo apt-get install libceres-dev
```

Compile the C++ pose graph file.
```
cd cpp
mkdir build
cd build
cmake ..
make
make install (do not run as sudo!)
```

# Running
Go back to the root folder and run

```
python3 run.py [path to test dataset]
```

For all available options
```
$ python3 run.py -h
```

# Useful options

## Tuning for your scenario
You'll want to adjust the following so your object is segmented out from the background
- --min_depth
- --max_depth

## Meshing
You can enable mesh reconstruction with --mesh 1. If you only expect a single mesh you can also use --keep_largest_mesh 1. This is also useful for removing noise.

