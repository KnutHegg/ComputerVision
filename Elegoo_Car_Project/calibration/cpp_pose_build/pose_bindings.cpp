#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

#include <cmath>
#include <array>
#include <limits>
#include <vector>

namespace py = pybind11;

static inline double reproj_error_px(
    const std::array<cv::Point3d, 4>& obj_pts,
    const std::array<cv::Point2d, 4>& img_pts,
    const cv::Mat& rvec,
    const cv::Mat& tvec,
    const cv::Mat& K,
    const cv::Mat& dist
) {
    std::vector<cv::Point3d> obj(obj_pts.begin(), obj_pts.end());
    std::vector<cv::Point2d> proj;
    cv::projectPoints(obj, rvec, tvec, K, dist, proj);

    double err = 0.0;
    for (int i = 0; i < 4; i++) {
        const double dx = proj[i].x - img_pts[i].x;
        const double dy = proj[i].y - img_pts[i].y;
        err += std::sqrt(dx * dx + dy * dy);
    }
    return err / 4.0;
}

static inline double yaw_deg_from_rvec(const cv::Mat& rvec) {
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    const double yaw = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    constexpr double kPi = 3.14159265358979323846;
    return yaw * 180.0 / kPi;
}

py::tuple estimate_pose_square(
    py::array_t<double, py::array::c_style | py::array::forcecast> corners_4x2,
    py::array_t<double, py::array::c_style | py::array::forcecast> K_3x3,
    py::array_t<double, py::array::c_style | py::array::forcecast> dist_1xN,
    double tag_size_m
) {
    if (corners_4x2.ndim() != 2 || corners_4x2.shape(0) != 4 || corners_4x2.shape(1) != 2) {
        throw std::runtime_error("corners must be shape (4,2)");
    }
    if (K_3x3.ndim() != 2 || K_3x3.shape(0) != 3 || K_3x3.shape(1) != 3) {
        throw std::runtime_error("K must be shape (3,3)");
    }
    if (dist_1xN.ndim() != 2 || dist_1xN.shape(0) != 1) {
        throw std::runtime_error("dist must be shape (1,N)");
    }

    cv::Mat K(3, 3, CV_64F, (void*)K_3x3.data());
    cv::Mat dist(1, (int)dist_1xN.shape(1), CV_64F, (void*)dist_1xN.data());

    const double half = tag_size_m / 2.0;

    const std::array<cv::Point3d, 4> obj_base = {
        cv::Point3d(-half, -half, 0.0),
        cv::Point3d( half, -half, 0.0),
        cv::Point3d( half,  half, 0.0),
        cv::Point3d(-half,  half, 0.0),
    };

    std::array<cv::Point2d, 4> c;
    for (int i = 0; i < 4; i++) {
        c[i] = cv::Point2d(corners_4x2.at(i, 0), corners_4x2.at(i, 1));
    }

    // 4 rotations + reversed rotations
    const int orders[8][4] = {
        {0,1,2,3},
        {1,2,3,0},
        {2,3,0,1},
        {3,0,1,2},
        {0,3,2,1},
        {3,2,1,0},
        {2,1,0,3},
        {1,0,3,2},
    };

    bool found = false;
    double best_err = std::numeric_limits<double>::infinity();
    cv::Mat best_rvec, best_tvec;

    for (int k = 0; k < 8; k++) {
        std::array<cv::Point2d, 4> img_pts = {
            c[orders[k][0]],
            c[orders[k][1]],
            c[orders[k][2]],
            c[orders[k][3]],
        };

        std::vector<cv::Point3d> obj(obj_base.begin(), obj_base.end());
        std::vector<cv::Point2d> img(img_pts.begin(), img_pts.end());

        cv::Mat rvec, tvec;
        bool ok = cv::solvePnP(obj, img, K, dist, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
        if (!ok) continue;

        double err = reproj_error_px(obj_base, img_pts, rvec, tvec, K, dist);
        if (tvec.at<double>(2, 0) <= 0) err += 1e6;

        if (err < best_err) {
            best_err = err;
            best_rvec = rvec;
            best_tvec = tvec;
            found = true;
        }
    }

    if (!found || best_tvec.empty()) {
        return py::make_tuple(false, 0.0, 0.0, 0.0, best_err);
    }

    const double x_m = best_tvec.at<double>(0, 0);
    const double z_m = best_tvec.at<double>(2, 0);
    const double yaw_deg = yaw_deg_from_rvec(best_rvec);

    return py::make_tuple(true, x_m, z_m, yaw_deg, best_err);
}

PYBIND11_MODULE(cpp_pose, m) {
    m.doc() = "C++ pose estimation for planar square (pybind11 + OpenCV)";
    m.def("estimate_pose_square", &estimate_pose_square,
          py::arg("corners_4x2"),
          py::arg("K_3x3"),
          py::arg("dist_1xN"),
          py::arg("tag_size_m"));
}