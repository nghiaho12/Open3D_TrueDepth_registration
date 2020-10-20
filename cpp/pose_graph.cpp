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
constexpr int MATCH_DIM = 8;
constexpr int ODOMETRY_DIM = 9;

// Cost function for pose and 3D correspondence.
struct PoseAndMatchesCost {
    PoseAndMatchesCost(
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

    // Factory to hide the construction of the PoseAndMatchesCosttion object from
    // the client code.
    static ceres::CostFunction* Create(
        const std::array<double, 3> &pt1,
        const std::array<double, 3> &pt2) {
        return new ceres::AutoDiffCostFunction<PoseAndMatchesCost,
            3,  // residual
            4,  // rot1
            3,  // trans1
            4,  // rot2
            3  // trans2
            >(new PoseAndMatchesCost(pt1, pt2));
    }

    const std::array<double, 3> pt1;
    const std::array<double, 3> pt2;
};

template <typename T>
void MyRotationMatrixToQuaternion(const T* R, T* angle_axis) {
  ceres::RotationMatrixToQuaternion(ceres::RowMajorAdapter3x3(R), angle_axis);
}

template <typename T>
inline void MyRotationMatrixToAngleAxis(const T* R, T* angle_axis) {
  ceres::RotationMatrixToAngleAxis(ceres::RowMajorAdapter3x3(R), angle_axis);
}


// Cost function for pose and odometry between poses.
struct PoseOdometryCost {
    PoseOdometryCost(
        const std::array<double, 4> &odometry_rot,
        const std::array<double, 3> &odometry_trans) :
        odometry_rot(odometry_rot),
        odometry_trans(odometry_trans) {
    }

    template <typename T>
    bool operator()(const T* const rot1,
                    const T* const trans1,
                    const T* const rot2,
                    const T* const trans2,
                    T* residuals) const {
        T R1[9];
        T R2[9];
        T R_odometry[9];

        T odometry_rot_[4];
        T odometry_trans_[3];

        for (int i=0; i < 4; i++) {
            odometry_rot_[i] = T(odometry_rot[i]);
        }

        for (int i=0; i < 3; i++) {
            odometry_trans_[i] = T(odometry_trans[i]);
        }

        // Rotation seems easier to work with in matrix form
        ceres::QuaternionToRotation(rot1, R1);
        ceres::QuaternionToRotation(rot2, R2);
        ceres::QuaternionToRotation(odometry_rot_, R_odometry);

        // rotation
        T R3[9];

        // R3 = R1*R_odometry
        for (int i=0; i < 3; i++) {
            const T *R1_row = R1 + i*3;
            for (int j=0; j < 3; j++) {
                // row i * col(j)
                R3[i*3 + j]  = R1_row[0]*R_odometry[0*3 + j];
                R3[i*3 + j] += R1_row[1]*R_odometry[1*3 + j];
                R3[i*3 + j] += R1_row[2]*R_odometry[2*3 + j];
            }
        }

        T R4[9];

        // R4 = R3.transposose() * R2
        for (int i=0; i < 3; i++) {
            for (int j=0; j < 3; j++) {
                // col i * col(j)
                R4[i*3 + j]  = R3[0*3 + i]*R2[0*3 + j];
                R4[i*3 + j] += R3[1*3 + i]*R2[1*3 + j];
                R4[i*3 + j] += R3[2*3 + i]*R2[2*3 + j];
            }
        }

        T q[4];
        T axis_angle[3];
        T trans2_pred[3];
        //MyRotationMatrixToQuaternion(R3, q);
        //MyRotationMatrixToQuaternion(R4, q);
        MyRotationMatrixToAngleAxis(R4, axis_angle);

        // translation
        // R1*odometry_trans + t1
        ceres::UnitQuaternionRotatePoint(rot1, odometry_trans_, trans2_pred);
        trans2_pred[0] += trans1[0];
        trans2_pred[1] += trans1[1];
        trans2_pred[2] += trans1[2];

        // residual is target - predicted

        // translation residual
        residuals[0] = trans2[0] - trans2_pred[0];
        residuals[1] = trans2[1] - trans2_pred[1];
        residuals[2] = trans2[2] - trans2_pred[2];

        residuals[3] = axis_angle[0];
        residuals[4] = axis_angle[1];
        residuals[5] = axis_angle[2];

        //residuals[3] = 1.0 - q[0];
        //residuals[4] = 0.0 - q[1];
        //residuals[5] = 0.0 - q[2];
        //residuals[6] = 0.0 - q[3];

        return true;
    }

    // Factory to hide the construction of the PoseAndMatchesCosttion object from
    // the client code.
    static ceres::CostFunction* Create(
        const std::array<double, 4> &odometry_rot,
        const std::array<double, 3> &odometry_trans) {
        return new ceres::AutoDiffCostFunction<PoseOdometryCost,
            6,  // residual
            4,  // rot1
            3,  // trans1
            4,  // rot2
            3  // trans2
            >(new PoseOdometryCost(odometry_rot, odometry_trans));
    }

    const std::array<double, 4> odometry_rot;
    const std::array<double, 3> odometry_trans;
};

void cpp_optimize_pose_graph_with_matches(std::vector<double> &poses, const std::vector<double> &matches) {
    ceres::Problem problem;
    ceres::QuaternionParameterization *quaternion_parameterisation = new ceres::QuaternionParameterization;

    for (size_t i=0; i < matches.size(); i+=MATCH_DIM) {
        const auto &m = matches.data() + i;

        int id1 = static_cast<int>(m[0]);
        int id2 = static_cast<int>(m[1]);

        std::array<double, 3> pt1{m[2], m[3], m[4]};
        std::array<double, 3> pt2{m[5], m[6], m[7]};

        ceres::CostFunction *cost = PoseAndMatchesCost::Create(pt1, pt2);

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

void cpp_optimize_pose_graph_with_odometry(std::vector<double> &poses, const std::vector<double> &odometry) {
    ceres::Problem problem;
    ceres::QuaternionParameterization *quaternion_parameterisation = new ceres::QuaternionParameterization;

    for (size_t i=0; i < odometry.size(); i+=ODOMETRY_DIM) {
        const auto &m = odometry.data() + i;

        int id1 = static_cast<int>(m[0]);
        int id2 = static_cast<int>(m[1]);

        std::array<double, 4> odometry_rot{m[2], m[3], m[4], m[5]};
        std::array<double, 3> odometry_trans{m[6], m[7], m[8]};

        ceres::CostFunction *cost = PoseOdometryCost::Create(odometry_rot, odometry_trans);

        problem.AddResidualBlock(cost,
            nullptr,
            poses.data() + id1*POSE_DIM,    // quaternion
            poses.data() + id1*POSE_DIM+4,  // translation
            poses.data() + id2*POSE_DIM,    // quaternion
            poses.data() + id2*POSE_DIM+4); // translation

        problem.SetParameterization(poses.data() + id1*POSE_DIM, quaternion_parameterisation);
        problem.SetParameterization(poses.data() + id2*POSE_DIM, quaternion_parameterisation);

        if (i == 0) {
            // Fixed params for the first pose
            problem.SetParameterBlockConstant(poses.data());
            problem.SetParameterBlockConstant(poses.data()+4);
        }
    }

    ceres::Solver::Options options;
    ceres::Solver::Summary summary;

    options.minimizer_progress_to_stdout = true;

    Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

// wrap C++ function with NumPy array IO
py::array py_optimize_pose_graph_with_matches(
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

    cpp_optimize_pose_graph_with_matches(poses_vec, matches_vec);

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

py::array py_optimize_pose_graph_with_odometry(
    py::array_t<double, py::array::c_style | py::array::forcecast> poses,
    py::array_t<double, py::array::c_style | py::array::forcecast> odometry) {

    // check input dimensions
    if (poses.ndim() != 2) {
        throw std::runtime_error("poses should be 2-D NumPy array");
    }
    if (poses.shape()[1] != POSE_DIM) {
        throw std::runtime_error("poses should have size [N,7]");
    }

    if (odometry.ndim() != 2) {
        throw std::runtime_error("odometry should be 2-D NumPy array");
    }
    if (odometry.shape()[1] != ODOMETRY_DIM) {
        throw std::runtime_error("odometry should have size [N,9]");
    }

    std::vector<double> poses_vec(poses.size());
    std::vector<double> odometry_vec(odometry.size());

    std::memcpy(poses_vec.data(), poses.data(), poses.size()*sizeof(double));
    std::memcpy(odometry_vec.data(), odometry.data(), odometry.size()*sizeof(double));

    cpp_optimize_pose_graph_with_odometry(poses_vec, odometry_vec);

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
    m.def("optimize_pose_graph_with_matches", &py_optimize_pose_graph_with_matches, "Optimize camera pose given 3D-3D matches");
    m.def("optimize_pose_graph_with_odometry", &py_optimize_pose_graph_with_odometry, "Optimize camera pose only");
}
