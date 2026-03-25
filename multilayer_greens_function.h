#pragma once

#include <array>
#include <complex>
#include <vector>
#include <utility>
#include <Eigen/Dense>
#include <string>

namespace mlgf {

using cd = std::complex<double>;
using Vec2c = Eigen::Matrix<cd, 2, 1>;
using Mat2c = Eigen::Matrix<cd, 2, 2>;
using Mat3d = Eigen::Matrix3d;
using Mat3c = Eigen::Matrix<cd, 3, 3>;

struct LayerAtFreq0 {
    cd eps;
    cd mu;
    cd k;
};

struct LayerAtkRho0 {
    cd eps;
    cd mu;
    cd k;
    cd kz;
};

struct ABTriplet {
    Vec2c s_ab;
    Vec2c p_ab;
    Vec2c p_az;
};

struct ABplMatrix {
    Vec2c a00;
    Vec2c a01;
    Vec2c a10;
    Vec2c a11;
};

struct EscIntegrands {
    cd Ip1;
    cd Ip2;
    cd Ip5;
    cd Ip9;
    cd Ip11;
};

double omega_eV_to_k0_nm(double omega, double c = 197.326980459302466);

Mat3d Rotate_ESC(double theta, const Mat3d& esc);
Mat3d Rotate_ESCv2(double theta, const Mat3d& esc);
std::array<Mat3d, 3> Rotate_ESC_3quarter(const Mat3d& esc);

Mat3d ElectroS_Coupling(
    double L,
    double rho,
    double phi,
    double z,
    double zp,
    const std::vector<double>& ds,
    const std::vector<cd>& epss,
    const std::vector<cd>& mus
);

Eigen::Vector3d MF_Coupling(
    int Np,
    double spacing,
    double z,
    const std::vector<double>& ds,
    const std::vector<cd>& epss,
    const std::vector<cd>& mus
);

Eigen::Vector3d MF_Coupling_FS(
    int Np,
    double spacing,
    double z,
    double epsb,
    double mub
);

struct LayerAtFreq {
    cd eps;
    cd mu;
    cd k;
};

struct LayerAtkRho {
    cd eps;
    cd mu;
    cd k;
    cd kz;
};

cd kl(cd k0, cd eps, cd mu);
cd kz(cd k, cd krho);

Eigen::Matrix<std::complex<double>, 3, 3> FreeSpaceGF(
    const Eigen::Vector3d& r,
    const Eigen::Vector3d& rp,
    std::complex<double> kB
);

Eigen::Matrix<std::complex<double>, 3, 3> MultiLayerGF(
    const Eigen::Vector3d& r,
    const Eigen::Vector3d& rp,
    std::complex<double> k0,
    const std::vector<double>& ds,
    const std::vector<std::complex<double>>& epss,
    const std::vector<std::complex<double>>& mus,
    const std::string& pol
);
} // namespace mlgf