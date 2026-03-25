#pragma once

#include <utility>
#include <vector>
#include <complex>
#include <Eigen/Dense>
#include <string>
#include <opencv2/opencv.hpp>


Eigen::Vector2d get_S(double a, double h, double a0, double x0);
std::pair<double, double> get_a0S(double h);
double get_a0F(double h);

Eigen::Vector3d MF_hexagonal_FS(
    int Np,
    double spacing,
    double z,
    double epsb,
    double mub
);

Eigen::Vector3d MF_hexagonal(
    int Np,
    double spacing,
    double z,
    const std::vector<double>& ds,
    const std::vector<std::complex<double>>& epss,
    const std::vector<std::complex<double>>& mus
);


void add_axes_and_labels(cv::Mat& img,
                         float xmin, float xmax,
                         float ymin, float ymax,
                         const std::string& xlabel,
                         const std::string& ylabel
);

cv::Mat make_colorbar(int height,
                      float vmin, float vmax,
                      float linthresh,
                      float linscale,
                      float base
);
                      
void Grid_Sqr_2D_sym_hor(
    int Np,
    double spacing,
    double z,
    double epsb,
    double mub,
    double fr,
    double fp,
    double fgamma,
    double epsd,
    double rd,
    double density,
    double lattice,
    double magnetoelastic,
    double nb,
    int size,
    const std::string& clrmp
);

struct SpinPhononPhotonResult {
    double d;
    double b0_ev;
    double b0_over_hbarOmega;
    double meff;
    double w0_evnm3;
};

SpinPhononPhotonResult SpinPhononPhotonParams(
    double fr,
    double fp,
    double fgamma,
    double eps_d,
    double radius,
    double density,
    double lattice,
    double magnetoelastic,
    double nb
);

void Make_Fig2();
void Make_Fig3();
