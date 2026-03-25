#include "unconventional_figures.h"
#include "multilayer_greens_function.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>
#include <iostream>
#include "matplotlibcpp.h"
#include <opencv2/opencv.hpp>

namespace plt = matplotlibcpp;

namespace {
constexpr double pi = 3.14159265358979323846;
constexpr double c0 = 2.99792458e8;
constexpr double hbar = 1.0545718170e-34;
constexpr double eps0 = 1e7 / (4 * pi * c0 * c0);
constexpr double debye = 1e-21 / c0;
constexpr double ev_to_J = 1.60217662e-19;
}

static double eV2kBT(double E_eV) {
    constexpr double kB = 1.380649 * 6.241509e-5; // eV/K
    return E_eV / kB;
}

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
) {
    double r = radius * 1e-9;
    double Omega = 2.0 * pi * fr * c0 * 100.0;
    double hbarOmega = hbar * Omega;

    double V = 4.0 * pi * std::pow(r, 3) / 3.0;
    double oeps_d = 1.0 / (eps_d + 2.0);
    double f0 = std::sqrt(1.0 + oeps_d - fgamma / 4.0);

    double d = std::sqrt(9.0 * eps0 * V * hbarOmega * fp / (2.0 * f0)) * oeps_d / debye;

    double a = lattice * 1e-9;
    double rho = density * std::sqrt(1e3);
    double b0 = magnetoelastic * std::sqrt(2.0 * V * hbar / rho / Omega) / a;
    double b0_ev = b0 / ev_to_J;

    double mu0 = 4.0 * pi * 1e-7;
    double meff = 3.0 * magnetoelastic * std::sqrt(fp) * oeps_d /
                  (std::sqrt(2.0 * mu0) * Omega * a * std::sqrt(rho) * nb * std::sqrt(f0));

    double num = 9.0 * V * V * magnetoelastic * magnetoelastic * fp;
    double den = 2.0 * Omega * Omega * a * a * rho * f0 * std::pow(eps_d + 2.0, 2);
    double w0 = num / den;
    double w0_evnm3 = w0 * 1e27 / ev_to_J;

    return {d, b0_ev, b0 / hbarOmega, meff, w0_evnm3};
}

namespace {

std::vector<double> linspace(double start, double end, int n) {
    std::vector<double> out(n);
    if (n == 1) {
        out[0] = start;
        return out;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = start + (end - start) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return out;
}

int argmin(const std::vector<double>& v) {
    return static_cast<int>(std::distance(v.begin(), std::min_element(v.begin(), v.end())));
}

} // namespace

Eigen::Vector2d get_S(double a, double h, double a0, double x0) {
    Eigen::Vector2d sol(0.0, 0.0);

    if (a >= a0) {
        auto fx = [](double x) {
            return x;
        };

        auto ftot = [h](double bL, double x) {
            return std::tanh(4.0 * bL * h * x * x * x) / h;
        };

        auto sec = [&](double bL, double x) {
            return std::abs(fx(x) - ftot(bL, x));
        };

        std::vector<double> m1 = linspace(1e-3, x0, 2000);
        std::vector<double> m2 = linspace(x0, 2.0001, 2000);

        std::vector<double> fs1(m1.size());
        std::vector<double> fs2(m2.size());

        for (std::size_t i = 0; i < m1.size(); ++i) {
            fs1[i] = sec(a, m1[i]);
        }
        for (std::size_t i = 0; i < m2.size(); ++i) {
            fs2[i] = sec(a, m2[i]);
        }

        sol(0) = m1[argmin(fs1)];
        sol(1) = m2[argmin(fs2)];
    }

    return sol;
}

std::pair<double, double> get_a0S(double h) {
    // For a given h, the critical value of a=βΛx is evaluated so that
    // a double-degenerate solution is obtained.

    const int Na = 1000;
    std::vector<double> a = linspace(0.0, 6.0, Na);

    auto sec = [h](double aval, double Sx) {
        return std::abs(Sx - std::tanh(4.0 * aval * h * Sx * Sx * Sx) / h);
    };

    auto dsec = [](double x, double xpr, double S, double Spr) {
        return (S - Spr) / (x - xpr);
    };

    const int Nm = 1000;
    std::vector<double> m = linspace(0.0, 2.0, Nm);
    std::vector<double> ds(Nm - 1, std::numeric_limits<double>::quiet_NaN());

    double asol = 0.0;
    double msol = 0.0;

    int ia = 0;
    while (ia < Na) {
        int isol = 0;

        std::vector<double> sc(Nm);
        for (int i = 0; i < Nm; ++i) {
            sc[i] = sec(a[ia], m[i]);
        }

        ds[0] = dsec(m[1], m[0], sc[1], sc[0]);

        int im = 2;
        while (im < Nm) {
            ds[im - 1] = dsec(m[im], m[im - 1], sc[im], sc[im - 1]);

            double min_sc = std::min({sc[im], sc[im - 1], sc[im - 2]});
            if ((ds[im - 2] * ds[im - 1]) < 0.0 && min_sc < 1e-2) {
                isol += 1;
                if (isol == 3) {
                    msol = m[im];
                    asol = a[ia];
                    ia = Na;   // break outer loop
                    break;
                }
            }
            im += 1;
        }

        ia += 1;
    }

    return {asol, msol};
}

double get_a0F(double h) {
    // For a given h, the critical value of a=βΛx is evaluated so that
    // the free energy has a minimum lower than 0.

    const int Na = 1000;
    std::vector<double> a = linspace(0.0, 6.0, Na);

    auto Free_Energy = [h](double betaL, double Sx) {
        return 3.0 * betaL * h * h * std::pow(Sx, 4)
             - std::log(std::cosh(4.0 * betaL * h * std::pow(Sx, 3)));
    };

    auto dsec = [](double x, double xpr, double F, double Fpr) {
        return (F - Fpr) / (x - xpr);
    };

    const int Nm = 500;
    std::vector<double> m = linspace(0.0, 2.0, Nm);
    std::vector<double> ds(Nm - 1, std::numeric_limits<double>::quiet_NaN());

    double asol = 0.0;

    int ia = 0;
    while (ia < Na) {
        std::vector<double> fe(Nm);
        for (int i = 0; i < Nm; ++i) {
            fe[i] = Free_Energy(a[ia], m[i]);
        }

        ds[0] = dsec(m[1], m[0], fe[1], fe[0]);

        for (int im = 2; im < Nm; ++im) {
            ds[im - 1] = dsec(m[im], m[im - 1], fe[im], fe[im - 1]);

            double min_fe = std::min(fe[im], fe[im - 1]);
            if ((ds[im - 2] * ds[im - 1]) < 0.0 && min_fe < 0.0) {
                asol = a[ia];
                ia = Na;   // break outer loop
                break;
            }
        }

        ia += 1;
    }

    return asol;

    
}

Eigen::Vector3d MF_hexagonal_FS(
    int Np,
    double spacing,
    double z,
    double epsb,
    double mub
) {
    constexpr double pi = 3.14159265358979323846;

    double a = spacing;
    Eigen::Matrix3d MF = Eigen::Matrix3d::Zero();

    double theta1 = 30.0 * pi / 180.0;
    double theta2 = theta1 + 120.0 * pi / 180.0;

    Eigen::Vector3d r(0.0, 0.0, z);

    for (int ix = -Np; ix <= Np; ++ix) {
        for (int iy = -Np; iy <= Np; ++iy) {
            double xp = ix * a * std::cos(theta1) + iy * a * std::cos(theta2);
            double yp = ix * a * std::sin(theta1) + iy * a * std::sin(theta2);

            Eigen::Vector3d rp(xp, yp, z);
            Eigen::Vector3d rho_f = r - rp;

            Eigen::Matrix3d ECT = Eigen::Matrix3d::Zero();

            if (!(ix == 0 && iy == 0)) {
                double n_rho_f = rho_f.norm();
                Eigen::Vector3d e_rho_f = rho_f / n_rho_f;
                Eigen::Matrix3d lambda0 =
                    3.0 * (e_rho_f * e_rho_f.transpose()) - Eigen::Matrix3d::Identity();

                ECT = lambda0 / (4.0 * pi * std::pow(n_rho_f, 3) * epsb * mub);
            }

            MF += ECT;
        }
    }

    return Eigen::Vector3d(MF(0, 0), MF(1, 1), MF(2, 2));
}

Eigen::Vector3d MF_hexagonal(
    int Np,
    double spacing,
    double z,
    const std::vector<double>& ds,
    const std::vector<std::complex<double>>& epss,
    const std::vector<std::complex<double>>& mus
) {
    constexpr double pi = 3.14159265358979323846;

    double a = spacing;
    Eigen::Matrix3d MF = Eigen::Matrix3d::Zero();

    double theta1 = 30.0 * pi / 180.0;
    double theta2 = theta1 + 120.0 * pi / 180.0;

    Eigen::Vector3d r(0.0, 0.0, z);

    auto add_site = [&](int ix, int iy) {
        double xp = ix * a * std::cos(theta1) + iy * a * std::cos(theta2);
        double yp = ix * a * std::sin(theta1) + iy * a * std::sin(theta2);

        Eigen::Vector3d rp(xp, yp, z);

        double dx = r(0) - rp(0);
        double dy = r(1) - rp(1);
        double rho = std::sqrt(dx * dx + dy * dy);
        double phi = std::atan2(dy, dx);
        double L = rho;

        std::vector<double> ds_scaled(ds.size());
        for (std::size_t k = 0; k < ds.size(); ++k) {
            ds_scaled[k] = ds[k] / L;
        }

        Eigen::Matrix3d ECT = mlgf::ElectroS_Coupling(
            L,
            rho / L,
            phi,
            r(2) / L,
            rp(2) / L,
            ds_scaled,
            epss,
            mus
        );

        MF += ECT;
    };

    for (int ix = -Np; ix <= -1; ++ix) {
        for (int iy = -Np; iy <= Np; ++iy) {
            add_site(ix, iy);
        }
    }

    for (int ix = 1; ix <= Np; ++ix) {
        for (int iy = -Np; iy <= Np; ++iy) {
            add_site(ix, iy);
        }
    }

    {
        int ix = 0;
        for (int iy = -Np; iy <= -1; ++iy) {
            add_site(ix, iy);
        }
        for (int iy = 1; iy <= Np; ++iy) {
            add_site(ix, iy);
        }
    }

    return Eigen::Vector3d(MF(0, 0), MF(1, 1), MF(2, 2));
}

static float symlog_transform(float x, float linthresh, float linscale, float base) {
    float ax = std::abs(x);
    float s = (x >= 0.0f) ? 1.0f : -1.0f;

    if (ax <= linthresh) {
        return s * linscale * (ax / linthresh);
    }

    return s * (linscale + std::log(ax / linthresh) / std::log(base));
}


static cv::Mat make_heatmap_symlog(
    const std::vector<float>& data,
    int N,
    float vmin,
    float vmax,
    float linthresh = 1e-3f,
    float linscale = 1.0f,
    float base = 10.0f,
    int upsample = 8
) {
    cv::Mat raw(N, N, CV_32F);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            raw.at<float>(j, i) = data[j * N + i];
        }
    }

    // Clip to plotting range
    cv::Mat clipped = raw.clone();
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float& val = clipped.at<float>(j, i);
            if (val < vmin) val = vmin;
            if (val > vmax) val = vmax;
        }
    }

    // Use symmetric max for normalization
    float vmax_abs = std::max(std::abs(vmin), std::abs(vmax));
    float tmax = symlog_transform(vmax_abs, linthresh, linscale, base);

    cv::Mat normalized(N, N, CV_8U);
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float val = clipped.at<float>(j, i);
            float t = symlog_transform(val, linthresh, linscale, base);

            // map [-tmax, +tmax] -> [0,255]
            float u = 0.5f * (t / tmax + 1.0f);
            if (u < 0.0f) u = 0.0f;
            if (u > 1.0f) u = 1.0f;

            normalized.at<unsigned char>(j, i) =
                static_cast<unsigned char>(255.0f * u);
        }
    }

    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_JET);

    cv::Mat enlarged;
    cv::resize(
        colored,
        enlarged,
        cv::Size(N * upsample, N * upsample),
        0, 0,
        cv::INTER_NEAREST
    );

    return enlarged;
}

void add_axes_and_labels(cv::Mat& img,
                        float xmin, float xmax,
                        float ymin, float ymax,
                        const std::string& xlabel,
                        const std::string& ylabel)
{
    int w = img.cols;
    int h = img.rows;

    // margins
    int left = 80;
    int bottom = 60;

    // expand canvas
    cv::Mat canvas(h + bottom, w + left, img.type(), cv::Scalar(255,255,255));

    // copy original image
    img.copyTo(canvas(cv::Rect(left, 0, w, h)));

    // axes
    cv::line(canvas, {left, h}, {left + w, h}, {0,0,0}, 2); // x-axis
    cv::line(canvas, {left, 0}, {left, h}, {0,0,0}, 2);     // y-axis

    // ticks (simple: 5 ticks)
    for (int i = 0; i <= 5; ++i) {
        float tx = xmin + i*(xmax-xmin)/5.0f;
        int px = left + i*w/5;
        cv::line(canvas, {px, h}, {px, h+5}, {0,0,0}, 1);

        char buf[50];
        snprintf(buf, sizeof(buf), "%.2f", tx);
        cv::putText(canvas, buf, {px-20, h+25},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {0,0,0}, 1);
    }

    for (int i = 0; i <= 5; ++i) {
        float ty = ymin + i*(ymax-ymin)/5.0f;
        int py = h - i*h/5;
        cv::line(canvas, {left-5, py}, {left, py}, {0,0,0}, 1);

        char buf[50];
        snprintf(buf, sizeof(buf), "%.2f", ty);
        cv::putText(canvas, buf, {5, py+5},
                    cv::FONT_HERSHEY_SIMPLEX, 0.4, {0,0,0}, 1);
    }

    // labels
    cv::putText(canvas, xlabel, {left + w/2 - 40, h + 50},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);

    cv::putText(canvas, ylabel, {5, 20},
                cv::FONT_HERSHEY_SIMPLEX, 0.6, {0,0,0}, 2);

    img = canvas;
}

cv::Mat make_colorbar(int height,
                             float vmin, float vmax,
                             float linthresh, float linscale, float base)
{
    int bar_width = 50;
    int left_margin = 70;

    // Leave some padding at top/bottom for labels, but keep total height fixed
    int top_pad = 20;
    int bottom_pad = 20;
    int bar_height = height - top_pad - bottom_pad;

    cv::Mat bar(bar_height, bar_width, CV_8U);

    float vmax_abs = std::max(std::abs(vmin), std::abs(vmax));
    float tmax = symlog_transform(vmax_abs, linthresh, linscale, base);
    if (std::abs(tmax) < 1e-20f) {
        tmax = 1.0f;
    }

    for (int j = 0; j < bar_height; ++j) {
        float y = 1.0f - static_cast<float>(j) / static_cast<float>(bar_height - 1);
        float val = vmin + y * (vmax - vmin);

        float t = symlog_transform(val, linthresh, linscale, base);
        float u = 0.5f * (t / tmax + 1.0f);
        u = std::clamp(u, 0.0f, 1.0f);

        unsigned char c = static_cast<unsigned char>(255.0f * u);
        for (int i = 0; i < bar_width; ++i) {
            bar.at<unsigned char>(j, i) = c;
        }
    }

    cv::Mat colored;
    cv::applyColorMap(bar, colored, cv::COLORMAP_JET);

    // Final canvas has EXACTLY the requested height
    cv::Mat canvas(height, bar_width + left_margin, CV_8UC3, cv::Scalar(255, 255, 255));
    colored.copyTo(canvas(cv::Rect(left_margin, top_pad, bar_width, bar_height)));

    char buf[64];
    snprintf(buf, sizeof(buf), "%.2e", vmax);
    cv::putText(canvas, buf, {5, top_pad + 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    snprintf(buf, sizeof(buf), "%.2e", 0.0);
    cv::putText(canvas, buf, {15, height / 2},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    snprintf(buf, sizeof(buf), "%.2e", vmin);
    cv::putText(canvas, buf, {5, height - 10},
                cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 0, 0), 1);

    return canvas;
}

void add_title(cv::Mat& img, const std::string& title) {
    cv::putText(
        img,
        title,
        cv::Point(15, 30),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(255, 255, 255),
        2,
        cv::LINE_AA
    );
}

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
) {
    constexpr double pi = 3.14159265358979323846;

    double a = spacing;

    std::vector<double> xs(2 * Np + 1), ys(2 * Np + 1);
    for (int i = 0; i < 2 * Np + 1; ++i) {
        xs[i] = -Np * a + i * a;
        ys[i] = -Np * a + i * a;
    }

    int size_point = 10;
    if (size >= 0) {
        size_point = size;
    }
    (void)size_point;

    auto params = SpinPhononPhotonParams(
        fr, fp, fgamma, epsd, rd, density, lattice, magnetoelastic, nb
    );
    double w0_meVnm3 = params.w0_evnm3 * 1000.0;

    int N = 2 * Np + 1;
    std::vector<std::vector<Eigen::Matrix3d>> ECT(
        N, std::vector<Eigen::Matrix3d>(N, Eigen::Matrix3d::Zero())
    );
    Eigen::Matrix3d MFC = Eigen::Matrix3d::Zero();

    Eigen::Vector3d r(0.0, 0.0, z);

    for (int ix = Np + 1; ix < N; ++ix) {
        {
            Eigen::Vector3d rp(xs[ix], ys[Np], z);

            Eigen::Vector3d rho_f = r - rp;
            double n_rho_f = rho_f.norm();
            Eigen::Vector3d e_rho_f = rho_f / n_rho_f;
            Eigen::Matrix3d lambda0 =
                3.0 * (e_rho_f * e_rho_f.transpose()) - Eigen::Matrix3d::Identity();

            ECT[ix][Np] = lambda0 / (4.0 * pi * std::pow(n_rho_f, 3) * epsb * mub);

            auto rots = mlgf::Rotate_ESC_3quarter(ECT[ix][Np]);
            ECT[Np][ix]       = rots[0];
            ECT[N - 1 - ix][Np] = rots[1];
            ECT[Np][N - 1 - ix] = rots[2];

            MFC += ECT[ix][Np] + ECT[Np][ix] + ECT[N - 1 - ix][Np] + ECT[Np][N - 1 - ix];
        }

        for (int iy = Np + 1; iy < ix; ++iy) {
            Eigen::Vector3d rp(xs[ix], ys[iy], z);

            Eigen::Vector3d rho_f = r - rp;
            double n_rho_f = rho_f.norm();
            Eigen::Vector3d e_rho_f = rho_f / n_rho_f;
            Eigen::Matrix3d lambda0 =
                3.0 * (e_rho_f * e_rho_f.transpose()) - Eigen::Matrix3d::Identity();

            ECT[ix][iy] = lambda0 / (4.0 * pi * std::pow(n_rho_f, 3) * epsb * mub);

            auto rots1 = mlgf::Rotate_ESC_3quarter(ECT[ix][iy]);
            ECT[N - 1 - iy][ix]         = rots1[0];
            ECT[N - 1 - ix][N - 1 - iy] = rots1[1];
            ECT[iy][N - 1 - ix]         = rots1[2];

            MFC += ECT[ix][iy] + ECT[N - 1 - iy][ix] + ECT[N - 1 - ix][N - 1 - iy] + ECT[iy][N - 1 - ix];

            double theta = std::acos(-2.0 * rp(1) / (rp(0) + rp(1) * rp(1) / rp(0)));
            ECT[iy][ix] = mlgf::Rotate_ESC(theta, ECT[ix][iy]);

            auto rots2 = mlgf::Rotate_ESC_3quarter(ECT[iy][ix]);
            ECT[N - 1 - ix][iy]         = rots2[0];
            ECT[N - 1 - iy][N - 1 - ix] = rots2[1];
            ECT[ix][N - 1 - iy]         = rots2[2];

            MFC += ECT[iy][ix] + ECT[N - 1 - ix][iy] + ECT[N - 1 - iy][N - 1 - ix] + ECT[ix][N - 1 - iy];
        }

        {
            Eigen::Vector3d rp(xs[ix], ys[ix], z);

            Eigen::Vector3d rho_f = r - rp;
            double n_rho_f = rho_f.norm();
            Eigen::Vector3d e_rho_f = rho_f / n_rho_f;
            Eigen::Matrix3d lambda0 =
                3.0 * (e_rho_f * e_rho_f.transpose()) - Eigen::Matrix3d::Identity();

            ECT[ix][ix] = lambda0 / (4.0 * pi * std::pow(n_rho_f, 3) * epsb * mub);

            auto rots = mlgf::Rotate_ESC_3quarter(ECT[ix][ix]);
            ECT[N - 1 - ix][ix]         = rots[0];
            ECT[N - 1 - ix][N - 1 - ix] = rots[1];
            ECT[ix][N - 1 - ix]         = rots[2];

            MFC += ECT[ix][ix] + ECT[N - 1 - ix][ix] + ECT[N - 1 - ix][N - 1 - ix] + ECT[ix][N - 1 - ix];
        }
    }

    std::vector<float> dat_xx(N * N);
    std::vector<float> dat_yy(N * N);
    std::vector<float> dat_xy(N * N);
    std::vector<float> dat_zz(N * N);

    double dmax = 0.0;
    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            Eigen::Matrix3d M = ECT[i][j] * w0_meVnm3;

            dat_xx[j * N + i] = static_cast<float>(M(0, 0));
            dat_yy[j * N + i] = static_cast<float>(M(1, 1));
            dat_xy[j * N + i] = static_cast<float>(M(0, 1));
            dat_zz[j * N + i] = static_cast<float>(M(2, 2));

            dmax = std::max(dmax, std::abs(M(0, 0)));
            dmax = std::max(dmax, std::abs(M(1, 1)));
            dmax = std::max(dmax, std::abs(M(0, 1)));
            dmax = std::max(dmax, std::abs(M(2, 2)));
        }
    }

    // float plot_max = static_cast<float>(dmax);
    // float plot_min = -plot_max;

    float plot_max = 1.0f;
    float plot_min = -plot_max;

    float linthresh = 1e-3f;
    float linscale  = 1.0f;
    float base      = 10.0f;

    cv::Mat img_xx = make_heatmap_symlog(dat_xx, N, plot_min, plot_max, linthresh, linscale, base);
    cv::Mat img_yy = make_heatmap_symlog(dat_yy, N, plot_min, plot_max, linthresh, linscale, base);
    cv::Mat img_xy = make_heatmap_symlog(dat_xy, N, plot_min, plot_max, linthresh, linscale, base);
    cv::Mat img_zz = make_heatmap_symlog(dat_zz, N, plot_min, plot_max, linthresh, linscale, base);

    add_title(img_xx, "xx");
    add_title(img_yy, "yy");
    add_title(img_xy, "xy");
    add_title(img_zz, "zz");

    float xmin = xs.front()/1000.0f;
    float xmax = xs.back()/1000.0f;
    float ymin = ys.front()/1000.0f;
    float ymax = ys.back()/1000.0f;

    // add axes
    add_axes_and_labels(img_xx, xmin, xmax, ymin, ymax, "x (um)", "y (um)");
    add_axes_and_labels(img_yy, xmin, xmax, ymin, ymax, "x (um)", "y (um)");
    add_axes_and_labels(img_xy, xmin, xmax, ymin, ymax, "x (um)", "y (um)");
    add_axes_and_labels(img_zz, xmin, xmax, ymin, ymax, "x (um)", "y (um)");
    cv::Mat row;
    cv::hconcat(std::vector<cv::Mat>{img_xx, img_yy, img_xy, img_zz}, row);

    cv::Mat colorbar = make_colorbar(row.rows,
                                    plot_min, plot_max,
                                    linthresh, linscale, base);

    cv::Mat final_img;
    cv::hconcat(row, colorbar, final_img);

    cv::imwrite("Fig1b.png", final_img);

    std::cout << "Saved Fig1b.png\n";

}
static std::vector<double> linspace_local(double start, double end, int n) {
    std::vector<double> out(n);
    if (n == 1) {
        out[0] = start;
        return out;
    }
    for (int i = 0; i < n; ++i) {
        out[i] = start + (end - start) * static_cast<double>(i) / static_cast<double>(n - 1);
    }
    return out;
}

static double eV2kBT_local(double E_eV) {
    constexpr double kB = 1.380649 * 6.241509e-5; // eV/K
    return E_eV / kB;
}

static cv::Mat fig_to_mat_rgb() {
    namespace plt = matplotlibcpp;

    plt::save(".__tmp_panel.png");

    cv::Mat bgr = cv::imread(".__tmp_panel.png", cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("Failed to read temporary panel image.");
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

static cv::Mat add_panel_title(const cv::Mat& img, const std::string& title) {
    cv::Mat out = img.clone();
    cv::putText(
        out,
        title,
        cv::Point(20, 35),
        cv::FONT_HERSHEY_SIMPLEX,
        1.0,
        cv::Scalar(0, 0, 0),
        2,
        cv::LINE_AA
    );
    return out;
}

void Make_Fig2() {
    namespace plt = matplotlibcpp;

    // -------------------------------
    // Panel 1
    // -------------------------------
    std::vector<double> m = linspace_local(-1.05, 1.05, 5000);

    double a4 = 0.2;
    double a3 = 0.504249509916091;
    double a2 = 0.688801373948792;
    double a1 = 1.0;
    (void)a2;

    std::vector<double> tm1(m.size()), tm3(m.size()), tm4(m.size());
    for (std::size_t i = 0; i < m.size(); ++i) {
        tm1[i] = std::tanh(4.0 * a1 * std::pow(m[i], 3));
        tm3[i] = std::tanh(4.0 * a3 * std::pow(m[i], 3));
        tm4[i] = std::tanh(4.0 * a4 * std::pow(m[i], 3));
    }

    plt::figure();
    plt::figure_size(700, 300);
    plt::plot(m, m, "--");
    plt::plot(m, tm1);
    plt::plot(m, tm3);
    plt::plot(m, tm4);
    plt::xlabel("<Sx>");
    plt::ylabel("tanh(4 beta Lambda_x <Sx>^3)");
    cv::Mat panel1 = add_panel_title(fig_to_mat_rgb(), "Panel 1");
    plt::close();

    // -------------------------------
    // Panel 2
    // -------------------------------
    plt::figure();
    plt::figure_size(700, 300);

    {
        double h = 1.0;
        auto [a0, m0] = get_a0S(h);
        double a0F = get_a0F(h);

        std::vector<double> a = linspace_local(a0, a0F, 30);
        std::vector<double> x(a.size()), y1(a.size()), y2(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            Eigen::Vector2d s = get_S(a[i], h, a0, m0);
            x[i] = a0 / a[i];
            y1[i] = s(0);
            y2[i] = s(1);
        }
        plt::plot(x, y1, {{"color", "black"}, {"linestyle", ":"}});
        plt::plot(x, y2, {{"color", "black"}});

        a = linspace_local(a0F, 1.5, 20);
        x.resize(a.size()); y1.resize(a.size()); y2.resize(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            Eigen::Vector2d s = get_S(a[i], h, a0, m0);
            x[i] = a0 / a[i];
            y1[i] = s(0);
            y2[i] = s(1);
        }
        plt::plot(x, y1, {{"color", "black"}, {"linestyle", ":"}});
        plt::plot(x, y2, {{"color", "black"}});

        a = linspace_local(1.5, 500.0, 400);
        x.resize(a.size()); y1.resize(a.size()); y2.resize(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            Eigen::Vector2d s = get_S(a[i], h, a0, m0);
            x[i] = a0 / a[i];
            y1[i] = s(0);
            y2[i] = s(1);
        }
        plt::plot(x, y1, {{"color", "black"}, {"linestyle", ":"}});
        plt::plot(x, y2, {{"color", "black"}});
    }

    {
        double h = std::sqrt(2.0);
        auto [a0, m0] = get_a0S(h);
        double a0F = get_a0F(h);

        std::vector<double> a = linspace_local(a0, a0F, 30);
        std::vector<double> x(a.size()), y1(a.size()), y2(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            Eigen::Vector2d s = get_S(a[i], h, a0, m0);
            x[i] = a0 / a[i];
            y1[i] = s(0);
            y2[i] = s(1);
        }
        plt::plot(x, y1, {{"color", "#E69F00"}, {"linestyle", ":"}});
        plt::plot(x, y2, {{"color", "#E69F00"}});

        a = linspace_local(a0F, 1.5, 20);
        x.resize(a.size()); y1.resize(a.size()); y2.resize(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            Eigen::Vector2d s = get_S(a[i], h, a0, m0);
            x[i] = a0 / a[i];
            y1[i] = s(0);
            y2[i] = s(1);
        }
        plt::plot(x, y1, {{"color", "#E69F00"}, {"linestyle", ":"}});
        plt::plot(x, y2, {{"color", "#E69F00"}});

        a = linspace_local(1.5, 500.0, 400);
        x.resize(a.size()); y1.resize(a.size()); y2.resize(a.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            Eigen::Vector2d s = get_S(a[i], h, a0, m0);
            x[i] = a0 / a[i];
            y1[i] = s(0);
            y2[i] = s(1);
        }
        plt::plot(x, y1, {{"color", "#E69F00"}, {"linestyle", ":"}});
        plt::plot(x, y2, {{"color", "#E69F00"}});
    }

    plt::ylabel("|<Sx>|");
    plt::xlabel("T/Tc");
    cv::Mat panel2 = add_panel_title(fig_to_mat_rgb(), "Panel 2");
    plt::close();

    // -------------------------------
    // Panel 3
    // -------------------------------
    int Np = 100; // start smaller than 600 for speed
    std::vector<double> a = linspace_local(300.0, 1000.0, 20);

    std::vector<Eigen::Vector3d> MF_hex(a.size());
    std::vector<Eigen::Vector3d> MF_sqr(a.size());

    for (std::size_t i = 0; i < a.size(); ++i) {
        MF_hex[i] = MF_hexagonal_FS(Np, a[i], 100.0, 1.0, 1.0);
        MF_sqr[i] = mlgf::MF_Coupling_FS(Np, a[i], 100.0, 1.0, 1.0);
    }

    double fr = 100.0;
    double fp = 1.0;
    double fgamma = 0.01;
    double density = 6.4;
    double lattice = 1.0;
    double magnetoelastic = 1e6;
    double epsd = 1.0;
    double nb = 1.0;
    double rd = 100.0;

    auto params = SpinPhononPhotonParams(
        fr, fp, fgamma, epsd, rd, density, lattice, magnetoelastic, nb
    );
    double w0_evnm3 = params.w0_evnm3;

    std::vector<double> Tc_sqr_3D(a.size()), Tc_sqr_1D(a.size());
    std::vector<double> Tc_hex_3D(a.size()), Tc_hex_1D(a.size());
    printf("MF_sqr[0] = (%.3e, %.3e, %.3e)\n", MF_sqr[0](0), MF_sqr[0](1), MF_sqr[0](2));   
    printf("MF_hex[0] = (%.3e, %.3e, %.3e)\n", MF_hex[0](0), MF_hex[0](1), MF_hex[0](2));   
    
    for (std::size_t i = 0; i < a.size(); ++i) {
        double Lx = MF_sqr[i](0) * w0_evnm3;
        double Ly = MF_sqr[i](1) * w0_evnm3;

        double h1D = std::sqrt(Lx / Lx);
        double h3D = std::sqrt(1.0 + Lx / Ly);

        auto [aS3D, mtmp1] = get_a0S(h3D);
        auto [aS1D, mtmp2] = get_a0S(h1D);
        (void)mtmp1;
        (void)mtmp2;

        Tc_sqr_3D[i] = eV2kBT_local(Lx / aS3D);
        Tc_sqr_1D[i] = eV2kBT_local(Lx / aS1D);
    }

    for (std::size_t i = 0; i < a.size(); ++i) {
        double Lx = MF_hex[i](0) * w0_evnm3;
        double Ly = MF_hex[i](1) * w0_evnm3;

        double h1D = std::sqrt(Lx / Lx);
        double h3D = std::sqrt(1.0 + Lx / Ly);

        auto [aS3D, mtmp1] = get_a0S(h3D);
        auto [aS1D, mtmp2] = get_a0S(h1D);
        (void)mtmp1;
        (void)mtmp2;

        Tc_hex_3D[i] = eV2kBT_local(Lx / aS3D);
        Tc_hex_1D[i] = eV2kBT_local(Lx / aS1D);
    }

    plt::figure();
    plt::figure_size(700, 300);
    plt::plot(a, Tc_sqr_3D, {{"color", "black"}});
    plt::plot(a, Tc_sqr_1D, {{"color", "black"}, {"linestyle", "--"}});
    plt::plot(a, Tc_hex_3D, {{"color", "#E69F00"}});
    plt::plot(a, Tc_hex_1D, {{"color", "#E69F00"}, {"linestyle", "--"}});
    plt::ylabel("Tc (K)");
    plt::xlabel("a (nm)");
    cv::Mat panel3 = add_panel_title(fig_to_mat_rgb(), "Panel 3");
    plt::close();

    // -------------------------------
    // Combine panels into one figure
    // -------------------------------
    int target_h = std::max({panel1.rows, panel2.rows, panel3.rows});

    auto resize_to_h = [target_h](const cv::Mat& img) {
        if (img.rows == target_h) return img;
        int new_w = static_cast<int>(img.cols * (static_cast<double>(target_h) / img.rows));
        cv::Mat out;
        cv::resize(img, out, cv::Size(new_w, target_h));
        return out;
    };

    panel1 = resize_to_h(panel1);
    panel2 = resize_to_h(panel2);
    panel3 = resize_to_h(panel3);

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{panel1, panel2, panel3}, combined);

    cv::cvtColor(combined, combined, cv::COLOR_RGB2BGR);
    cv::imwrite("Fig2.png", combined);

    std::cout << "Saved Fig2.png\n";
}

void Make_Fig3() {
    namespace plt = matplotlibcpp;
    using cd = std::complex<double>;

    auto linspace_local = [](double start, double end, int n) {
        std::vector<double> out(n);
        if (n == 1) {
            out[0] = start;
            return out;
        }
        for (int i = 0; i < n; ++i) {
            out[i] = start + (end - start) * static_cast<double>(i) / static_cast<double>(n - 1);
        }
        return out;
    };

    auto eV2kBT_local = [](double E_eV) {
        constexpr double kB = 1.380649 * 6.241509e-5; // eV/K
        return E_eV / kB;
    };

    double S = 1.0;
    double fr = 100.0;
    double fp = 1.0;
    double fgamma = 0.01;
    double density = 6.4;
    double lattice = 1.0;
    double magnetoelastic = 1e6;
    double epsd = 1.0;
    double mud = 1.0;
    double nb = 1.0;
    double rd = 100.0;
    (void)S;
    (void)mud;

    auto params = SpinPhononPhotonParams(
        fr, fp, fgamma, epsd, rd, density, lattice, magnetoelastic, nb
    );
    double w0_evnm3 = params.w0_evnm3;

    int Np = 50;
    double a = 4.0 * rd;

    std::vector<cd> mus = {cd(1.0, 0.0), cd(1.0, 0.0)};
    std::vector<double> ds = {0.0};

    plt::figure();
    plt::figure_size(700, 450);

    auto add_branch = [&](const std::vector<double>& zvals,
                          const std::vector<cd>& epss,
                          const std::string& color) {
        std::vector<Eigen::Vector3d> MF_SC(zvals.size());
        std::vector<Eigen::Vector3d> MF_FS(zvals.size());
        printf("Adding branch with epss = (%.3e, %.3e)\n", epss[0].real(), epss[1].real());
        for (std::size_t i = 0; i < zvals.size(); ++i) {
            MF_SC[i] = MF_hexagonal(Np, a, zvals[i], ds, epss, mus);
            MF_FS[i] = MF_hexagonal_FS(Np, a, zvals[i], 1.0, 1.0);
        }
        printf("MF caclulated for branch with epss = (%.3e, %.3e)\n", epss[0].real(), epss[1].real());

        std::vector<double> Tc(zvals.size());

        for (std::size_t i = 0; i < zvals.size(); ++i) {
            Eigen::Vector3d total = (MF_FS[i] + MF_SC[i]) * w0_evnm3;

            double Lx = total(0);
            double Ly = total(1);
            double Lz = total(2);
            (void)Ly;
            (void)Lz;

            double h1D = std::sqrt(Lx / Lx);
            auto [aS1D, mtmp] = get_a0S(h1D);
            (void)mtmp;

            Tc[i] = eV2kBT_local(Lx / aS1D);
        }

        plt::plot(zvals, Tc, {{"color", color}});
    };

    // first z range: 0..120
    {
        std::vector<double> z2 = linspace_local(0.0, 120.0, 15);

        add_branch(z2, {cd(1.0, 0.0), cd(2.5, 0.0)}, "#E69F00");
        add_branch(z2, {cd(1.0, 0.0), cd(10.0, 0.0)}, "#56B4E9");
        add_branch(z2, {cd(1.0, 0.0), cd(-1e12, 0.0)}, "black");
    }

    // second z range: 120..500
    {
        std::vector<double> z2 = linspace_local(120.0, 500.0, 15);

        add_branch(z2, {cd(1.0, 0.0), cd(2.5, 0.0)}, "#E69F00");
        add_branch(z2, {cd(1.0, 0.0), cd(10.0, 0.0)}, "#56B4E9");
        add_branch(z2, {cd(1.0, 0.0), cd(-1e12, 0.0)}, "black");
    }

    plt::ylabel("Tc (K)");
    plt::xlabel("d (nm)");
    plt::save("Fig3b.png");

    std::cout << "Saved Fig3b.png\n";
}