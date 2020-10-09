# Open3D_TrueDepth_registration
Automatic point cloud registration from Apple's TrueDepth camera system.

Call the following to pull in the pybind11 submodule.
```
git submodule update --init --recursive
```

## Compiling pose_graph.cpp
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

## Running
Go back to the root folder and run

```
python3 run.py [folder]
```

For all available options
```
$ python3 run.py -h
```

You'll most likely need to adjust the following options for your setup
- --min_depth (min depth distance (meters))
- --max_depth (max depth distance (meters))