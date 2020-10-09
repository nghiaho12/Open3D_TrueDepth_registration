#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <map>
#include <array>
#include <fstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace py = pybind11;

constexpr int POSE_DIM = 7; // quaternion + translation
constexpr int MATHC_DIM = 8;

struct CostFunc {
    CostFunc(
        const std::array<double, 3> &pt1,
        const std::array<double, 3> &pt2) :
        pt1(pt1),
        pt2(pt2) {
    }

    template <typename T>
    bool operator()(const T* const rot1,
                    const T* const trans1,
                    const T* const rot2,
                    const T* const trans2,
                    T* residuals) const {
        T pt1_[3], pt1__[3];
        T pt2_[3], pt2__[3];

        pt1_[0] = T(pt1[0]);
        pt1_[1] = T(pt1[1]);
        pt1_[2] = T(pt1[2]);

        pt2_[0] = T(pt2[0]);
        pt2_[1] = T(pt2[1]);
        pt2_[2] = T(pt2[2]);

        ceres::UnitQuaternionRotatePoint(rot1, pt1_, pt1__);
        ceres::UnitQuaternionRotatePoint(rot2, pt2_, pt2__);

        pt1__[0] += trans1[0];
        pt1__[1] += trans1[1];
        pt1__[2] += trans1[2];

        pt2__[0] += trans2[0];
        pt2__[1] += trans2[1];
        pt2__[2] += trans2[2];

        residuals[0] = pt1__[0] - pt2__[0];
        residuals[1] = pt1__[1] - pt2__[1];
        residuals[2] = pt1__[2] - pt2__[2];

        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(
        const std::array<double, 3> &pt1,
        const std::array<double, 3> &pt2) {
        return new ceres::AutoDiffCostFunction<CostFunc,
                                               3,  // residual
                                               4,  // rot1
                                               3,  // trans1
                                               4,  // rot2
                                               3  // trans2
                                               >(new CostFunc(pt1, pt2));
    }

    const std::array<double, 3> pt1;
    const std::array<double, 3> pt2;
};

void run_pose_graph_optimization(std::vector<double> &poses, const std::vector<double> &matches) {
    ceres::Problem problem;
    ceres::QuaternionParameterization *quaternion_parameterisation = new ceres::QuaternionParameterization;

    for (size_t i=0; i < matches.size(); i+=MATHC_DIM) {
        const auto &m = matches.data() + i;

        int id1 = static_cast<int>(m[0]);
        int id2 = static_cast<int>(m[1]);

        std::array<double, 3> pt1{m[2], m[3], m[4]};
        std::array<double, 3> pt2{m[5], m[6], m[7]};

        ceres::CostFunction *cost = CostFunc::Create(pt1, pt2);

        problem.AddResidualBlock(cost,
            nullptr,
            poses.data() + id1*POSE_DIM,    // quaternion
            poses.data() + id1*POSE_DIM+4,  // translation
            poses.data() + id2*POSE_DIM,    // quaternion
            poses.data() + id2*POSE_DIM+4); // translation

        problem.SetParameterization(poses.data() + id1*POSE_DIM, quaternion_parameterisation);
        problem.SetParameterization(poses.data() + id2*POSE_DIM, quaternion_parameterisation);
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    options.minimizer_progress_to_stdout = true;

    //options.num_threads = 16;
    //options.minimizer_progress_to_stdout = true;
    //options.max_num_iterations = args.params.maxIterations;
    //options.function_tolerance = args.params.funcTol;

    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

// wrap C++ function with NumPy array IO
py::array pose_graph(
    py::array_t<double, py::array::c_style | py::array::forcecast> poses,
    py::array_t<double, py::array::c_style | py::array::forcecast> matches) {

    // check input dimensions
    if (poses.ndim() != 2) {
        throw std::runtime_error("poses should be 2-D NumPy array");
    }
    if (poses.shape()[1] != POSE_DIM) {
        throw std::runtime_error("poses should have size [N,7]");
    }

    if (matches.ndim() != 2) {
        throw std::runtime_error("matches should be 2-D NumPy array");
    }
    if (matches.shape()[1] != 8) {
        throw std::runtime_error("matches should have size [N,8]");
    }

    std::vector<double> poses_vec(poses.size());
    std::vector<double> matches_vec(matches.size());

    std::memcpy(poses_vec.data(), poses.data(), poses.size()*sizeof(double));
    std::memcpy(matches_vec.data(), matches.data(), matches.size()*sizeof(double));

    run_pose_graph_optimization(poses_vec, matches_vec);

    ssize_t ndim = 2;
    std::vector<ssize_t> shape{poses.shape()[0], poses.shape()[1]};
    std::vector<ssize_t> strides{sizeof(double)*POSE_DIM, sizeof(double)};

    // return 2-D NumPy array
    return py::array(py::buffer_info(
        poses_vec.data(),                           /* data as contiguous array  */
        sizeof(double),                          /* size of one scalar        */
        py::format_descriptor<double>::format(), /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
    ));
}

PYBIND11_MODULE(pose_graph, m)
{
    m.doc() = "Pose graph optimization";
    m.def("pose_graph", &pose_graph, "Optimize camera pose given 3D-3D matches");
}
