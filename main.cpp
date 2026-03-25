#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include "unconventional_figures.h"
#include "multilayer_greens_function.h"
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    // // Figure 1b: Square symmeric grid 
    // std::cout << "Running Figure 1b: Square symmetric grid" << std::endl;

    // // ===== Parameters (same as Julia test block style) =====
    // int Np = 50;
    // double spacing = 100.0;   // nm
    // double z = 100.0;        // nm
    // double rd = 100;

    // double eps_b = 1.0;
    // double mu_b  = 1.0;

    // double fr = 100;
    // double fp = 1;
    // double fgamma = 0.01;

    // double eps_d = 1;
    // double density = 6.4;
    // double lattice = 1;
    // double magnetoelastic = 1e6;
    // double nb = 1;

    // int size = 20;
    // std::string cmap = "jet";  // not used anymore (OpenCV)

    // // ===== Call your function =====
    // Grid_Sqr_2D_sym_hor(Np,spacing,z,eps_b,mu_b,fr,fp,fgamma,eps_d,rd,
    //     density,lattice,magnetoelastic,nb,size,cmap);
    // std::cout << "Done." << std::endl;

    // // Figure 2
    // Make_Fig2();

    // Figure 3
    Make_Fig3();

}