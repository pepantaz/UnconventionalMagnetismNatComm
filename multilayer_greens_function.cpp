#include "multilayer_greens_function.h"
#include <boost/math/special_functions/bessel.hpp>

#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace mlgf {

static constexpr double PI = 3.14159265358979323846;
static const cd I(0.0, 1.0);

struct LayerAtFreqDyn {
    cd eps;
    cd mu;
    cd k;
};

struct LayerAtkRhoDyn {
    cd eps;
    cd mu;
    cd k;
    cd kz;
};

static cd kl_dyn(cd k0, cd eps, cd mu) {
    return k0 * std::sqrt(cd(eps * mu));
}

// ensure imag(kz) >= 0
static cd kz_dyn(cd k, cd krho) {
    cd v = std::sqrt(cd(k * k - krho * krho));
    if (std::imag(v) < 0.0) {
        v = -v;
    }
    return v;
}

cd kl(cd k0, cd eps, cd mu) {
    return kl_dyn(k0, eps, mu);
}

cd kz(cd k, cd krho) {
    return kz_dyn(k, krho);
}

static LayerAtFreqDyn make_layer_dyn(cd k0, cd eps, cd mu) {
    return {eps, mu, kl_dyn(k0, eps, mu)};
}

static LayerAtkRhoDyn make_layer_krho_dyn(cd krho, const LayerAtFreqDyn& lf) {
    return {lf.eps, lf.mu, lf.k, kz_dyn(lf.k, krho)};
}

Eigen::Matrix<std::complex<double>, 3, 3> FreeSpaceGF(
    const Eigen::Vector3d& r,
    const Eigen::Vector3d& rp,
    cd kB
) {
    Eigen::Vector3d R = r - rp;
    double aR = R.norm();

    Eigen::Matrix<cd, 3, 3> tens =
        (cd(1.0, 0.0) + (I * kB * aR - cd(1.0, 0.0)) / (kB * kB * aR * aR))
            * Eigen::Matrix<cd, 3, 3>::Identity()
        + ((cd(3.0, 0.0) - cd(3.0, 0.0) * I * kB * aR - kB * kB * aR * aR)
            / (kB * kB * std::pow(aR, 4)))
            * (R.cast<cd>() * R.cast<cd>().transpose());

    return tens * std::exp(I * kB * aR) / (4.0 * PI * aR);
}

static cd fresnel_s_dyn(cd k1z, cd mu1, cd k2z, cd mu2) {
    return (mu2 * k1z - mu1 * k2z) / (mu2 * k1z + mu1 * k2z);
}

static cd fresnel_p_dyn(cd k1z, cd eps1, cd k2z, cd eps2) {
    return (eps2 * k1z - eps1 * k2z) / (eps2 * k1z + eps1 * k2z);
}

static std::pair<cd, cd> fresnel_dyn(
    cd k1z, cd eps1, cd mu1,
    cd k2z, cd eps2, cd mu2
) {
    return {
        fresnel_s_dyn(k1z, mu1, k2z, mu2),
        fresnel_p_dyn(k1z, eps1, k2z, eps2)
    };
}

static std::pair<cd, cd> fresnel_dyn(const LayerAtkRhoDyn& l1, const LayerAtkRhoDyn& l2) {
    return fresnel_dyn(l1.kz, l1.eps, l1.mu, l2.kz, l2.eps, l2.mu);
}

static cd amplRatP_dyn(cd F, cd BoA, double dp, cd kpz, cd kcz) {
    return (F * std::exp(I * dp * (kpz + kcz))
          + BoA * std::exp(-I * dp * (kpz - kcz)))
         / (std::exp(I * dp * (kpz - kcz))
          + F * BoA * std::exp(-I * dp * (kpz + kcz)));
}

static cd amplRatM_dyn(cd F, cd AoB, double dc, cd knz, cd kcz) {
    return (F * std::exp(-I * dc * (knz + kcz))
          + AoB * std::exp(I * dc * (knz - kcz)))
         / (std::exp(-I * dc * (knz - kcz))
          + F * AoB * std::exp(I * dc * (knz + kcz)));
}

static std::pair<cd, cd> AmplRatPl_dyn(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRhoDyn>& layers,
    std::size_t lprime
) {
    cd BoAs(0.0, 0.0), BoAp(0.0, 0.0);

    if (lprime >= 1) {
        auto [fs, fp] = fresnel_dyn(layers[1], layers[0]);
        cd expkzd = std::exp(I * 2.0 * ds[0] * layers[1].kz);
        BoAs = fs * expkzd;
        BoAp = fp * expkzd;
    }

    for (std::size_t i = 2; i <= lprime; ++i) {
        auto [Fs, Fp] = fresnel_dyn(layers[i], layers[i - 1]);
        BoAs = amplRatP_dyn(Fs, BoAs, ds[i - 1], layers[i - 1].kz, layers[i].kz);
        BoAp = amplRatP_dyn(Fp, BoAp, ds[i - 1], layers[i - 1].kz, layers[i].kz);
    }

    return {BoAs, BoAp};
}

static std::pair<cd, cd> AmplRatMn_dyn(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRhoDyn>& layers,
    std::size_t lprime
) {
    cd AoBs(0.0, 0.0), AoBp(0.0, 0.0);
    std::size_t N = layers.size();

    for (std::size_t ii = N - 1; ii > lprime; --ii) {
        std::size_t i = ii - 1;
        auto [Fs, Fp] = fresnel_dyn(layers[i], layers[i + 1]);
        AoBs = amplRatM_dyn(Fs, AoBs, ds[i], layers[i + 1].kz, layers[i].kz);
        AoBp = amplRatM_dyn(Fp, AoBp, ds[i], layers[i + 1].kz, layers[i].kz);
    }

    return {AoBs, AoBp};
}

static ABTriplet AmplLprPl_dyn(
    const LayerAtkRhoDyn& layer,
    const std::pair<cd, cd>& AoBM,
    const std::pair<cd, cd>& BoAP,
    double zpr
) {
    cd eip = std::exp(I * layer.kz * zpr);
    cd eim = std::exp(-I * layer.kz * zpr);
    cd AoBMs = AoBM.first;
    cd AoBMp = AoBM.second;
    cd BoAPs = BoAP.first;
    cd BoAPp = BoAP.second;

    cd As_ab = AoBMs * (BoAPs * eim + eip) / (cd(1.0, 0.0) - BoAPs * AoBMs);
    cd Bs_ab = BoAPs * (AoBMs * eip + eim) / (cd(1.0, 0.0) - BoAPs * AoBMs);

    cd Ap_ab = AoBMp * (BoAPp * eim - eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_ab = -BoAPp * (AoBMp * eip - eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    cd Ap_az = AoBMp * (BoAPp * eim + eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_az = BoAPp * (AoBMp * eip + eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    return {
        Vec2c(As_ab, Bs_ab),
        Vec2c(Ap_ab, Bp_ab),
        Vec2c(Ap_az, Bp_az)
    };
}

static ABTriplet AmplLprMn_dyn(
    const LayerAtkRhoDyn& layer,
    const std::pair<cd, cd>& AoBM,
    const std::pair<cd, cd>& BoAP,
    double zpr
) {
    cd eip = std::exp(I * layer.kz * zpr);
    cd eim = std::exp(-I * layer.kz * zpr);
    cd AoBMs = AoBM.first;
    cd AoBMp = AoBM.second;
    cd BoAPs = BoAP.first;
    cd BoAPp = BoAP.second;

    cd As_ab = AoBMs * (BoAPs * eim + eip) / (cd(1.0, 0.0) - BoAPs * AoBMs);
    cd Bs_ab = BoAPs * (AoBMs * eip + eim) / (cd(1.0, 0.0) - BoAPs * AoBMs);

    cd Ap_ab = -AoBMp * (BoAPp * eim - eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_ab = BoAPp * (AoBMp * eip - eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    cd Ap_az = AoBMp * (BoAPp * eim + eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_az = BoAPp * (AoBMp * eip + eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    return {
        Vec2c(As_ab, Bs_ab),
        Vec2c(Ap_ab, Bp_ab),
        Vec2c(Ap_az, Bp_az)
    };
}

static ABplMatrix make_ABpl(const Vec2c& p_ab, const Vec2c& p_az) {
    return {p_ab, p_ab, p_az, p_az};
}

static Vec2c matvec2(const Mat2c& M, const Vec2c& v) {
    return M * v;
}

static ABplMatrix apply_Mp_gamma(
    const Mat2c& Mp,
    cd gamma_pab,
    cd gamma_paz,
    const ABplMatrix& ABpl
) {
    return {
        gamma_pab * (Mp * ABpl.a00),
        gamma_pab * (Mp * ABpl.a01),
        gamma_paz * (Mp * ABpl.a10),
        gamma_paz * (Mp * ABpl.a11)
    };
}

static std::tuple<cd, cd, cd> gamma_sp_dyn(
    const LayerAtkRhoDyn& l1,
    const LayerAtkRhoDyn& l2
) {
    cd gamma_s = (l2.mu * l1.kz + l1.mu * l2.kz) / (2.0 * l2.mu * l2.kz);
    cd gamma_pab = l1.mu * l2.kz * (l2.eps * l1.kz + l1.eps * l2.kz)
        / (2.0 * l2.mu * l1.kz * l2.eps * l2.kz);
    cd gamma_paz = l1.mu * (l2.eps * l1.kz + l1.eps * l2.kz)
        / (2.0 * l2.mu * l2.eps * l2.kz);
    return {gamma_s, gamma_pab, gamma_paz};
}

static std::pair<Vec2c, ABplMatrix> AmplLPl_dyn(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRhoDyn>& layers,
    std::size_t l,
    std::size_t lpr,
    const ABTriplet& ABs
) {
    Vec2c ABsl = ABs.s_ab;
    ABplMatrix ABpl = make_ABpl(ABs.p_ab, ABs.p_az);

    for (std::size_t ii = lpr; ii > l; --ii) {
        std::size_t i = ii - 1;
        double dcur = ds[i];
        const auto& lcur = layers[i];
        const auto& lnxt = layers[i + 1];

        auto [Fs, Fp] = fresnel_dyn(lcur, lnxt);
        auto [gamma_s, gamma_pab, gamma_paz] = gamma_sp_dyn(lcur, lnxt);

        cd epsum = std::exp(I * dcur * (lnxt.kz + lcur.kz));
        cd emsum = std::exp(-I * dcur * (lnxt.kz + lcur.kz));
        cd epdif = std::exp(I * dcur * (lnxt.kz - lcur.kz));
        cd emdif = std::exp(-I * dcur * (lnxt.kz - lcur.kz));

        Mat2c Ms, Mp;
        Ms << epdif, Fs * epsum,
              Fs * emsum, emdif;
        Mp << epdif, Fp * epsum,
              Fp * emsum, emdif;

        ABsl = gamma_s * (Ms * ABsl);
        ABpl = apply_Mp_gamma(Mp, gamma_pab, gamma_paz, ABpl);
    }

    return {ABsl, ABpl};
}

static std::pair<Vec2c, ABplMatrix> AmplLMn_dyn(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRhoDyn>& layers,
    std::size_t l,
    std::size_t lpr,
    const ABTriplet& ABs
) {
    Vec2c ABsl = ABs.s_ab;
    ABplMatrix ABpl = make_ABpl(ABs.p_ab, ABs.p_az);

    for (std::size_t i = lpr + 1; i <= l; ++i) {
        double dprv = ds[i - 1];
        const auto& lcur = layers[i];
        const auto& lprv = layers[i - 1];

        auto [Fs, Fp] = fresnel_dyn(lcur, lprv);
        auto [gamma_s, gamma_pab, gamma_paz] = gamma_sp_dyn(lcur, lprv);

        cd epsum = std::exp(I * dprv * (lprv.kz + lcur.kz));
        cd emsum = std::exp(-I * dprv * (lprv.kz + lcur.kz));
        cd epdif = std::exp(I * dprv * (lprv.kz - lcur.kz));
        cd emdif = std::exp(-I * dprv * (lprv.kz - lcur.kz));

        Mat2c Ms, Mp;
        Ms << epdif, Fs * epsum,
              Fs * emsum, emdif;
        Mp << epdif, Fp * epsum,
              Fp * emsum, emdif;

        ABsl = gamma_s * (Ms * ABsl);
        ABpl = apply_Mp_gamma(Mp, gamma_pab, gamma_paz, ABpl);
    }

    return {ABsl, ABpl};
}
// ---------- small helpers ----------

template <typename T>
static T sqr(const T& x) {
    return x * x;
}

double omega_eV_to_k0_nm(double omega, double c) {
    return omega / c;
}

static std::size_t layer_index_desc(const std::vector<double>& ds, double z) {
    std::size_t idx = 0;
    while (idx < ds.size() && ds[idx] > z) {
        ++idx;
    }
    return idx; // 0-based layer index
}

static cd kl0(const cd&, const cd&) {
    return cd(0.0, 0.0);
}

// Julia: kz0(k,kρ) = 1im*kρ
static cd kz0(const cd&, const cd& krho) {
    return I * krho;
}

static LayerAtFreq0 make_layer0(const cd& eps, const cd& mu) {
    return {eps, mu, kl0(eps, mu)};
}

static LayerAtkRho0 make_layer_krho0(const cd& krho, const LayerAtFreq0& lf) {
    return {lf.eps, lf.mu, lf.k, kz0(lf.k, krho)};
}

// Julia's electrostatic p reflection coefficient branch
static cd esc_fp(const cd& eps_i, const cd& eps_j) {
    if (std::abs(eps_i) > std::abs(eps_j)) {
        return (eps_j / eps_i - cd(1.0, 0.0)) / (cd(1.0, 0.0) + eps_j / eps_i);
    } else {
        return (cd(1.0, 0.0) - eps_i / eps_j) / (eps_i / eps_j + cd(1.0, 0.0));
    }
}

// Julia:
// amplRatP(F,BoA,dp,kpz,kcz)
static cd amplRatP_single(cd F, cd BoA, double dp, cd kpz, cd kcz) {
    return (F * std::exp(I * dp * (kpz + kcz)) +
            BoA * std::exp(-I * dp * (kpz - kcz))) /
           (std::exp(I * dp * (kpz - kcz)) +
            F * BoA * std::exp(-I * dp * (kpz + kcz)));
}

// Julia:
// amplRatM(F,AoB,dc,knz,kcz)
static cd amplRatM_single(cd F, cd AoB, double dc, cd knz, cd kcz) {
    return (F * std::exp(-I * dc * (knz + kcz)) +
            AoB * std::exp(I * dc * (knz - kcz))) /
           (std::exp(-I * dc * (knz - kcz)) +
            F * AoB * std::exp(I * dc * (knz + kcz)));
}

static std::pair<cd, cd> AmplRatPl_ESC(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRho0>& layers,
    std::size_t lprime
) {
    cd BoAs(0.0, 0.0), BoAp(0.0, 0.0);

    if (lprime >= 1) {
        cd fs = cd(0.0, 0.0);
        cd fp = esc_fp(layers[1].eps, layers[0].eps);
        cd expkzd = std::exp(I * 2.0 * ds[0] * layers[1].kz);
        BoAs = fs * expkzd;
        BoAp = fp * expkzd;
    }

    for (std::size_t i = 2; i <= lprime; ++i) {
        cd fs = cd(0.0, 0.0);
        cd fp = esc_fp(layers[i].eps, layers[i - 1].eps);
        BoAs = amplRatP_single(fs, BoAs, ds[i - 1], layers[i - 1].kz, layers[i].kz);
        BoAp = amplRatP_single(fp, BoAp, ds[i - 1], layers[i - 1].kz, layers[i].kz);
    }

    return {BoAs, BoAp};
}

static std::pair<cd, cd> AmplRatMn_ESC(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRho0>& layers,
    std::size_t lprime
) {
    cd AoBs(0.0, 0.0), AoBp(0.0, 0.0);
    const std::size_t N = layers.size();

    for (std::size_t ii = N - 1; ii > lprime; --ii) {
        std::size_t i = ii - 1;
        cd fs = cd(0.0, 0.0);
        cd fp = esc_fp(layers[i].eps, layers[i + 1].eps);
        AoBs = amplRatM_single(fs, AoBs, ds[i], layers[i + 1].kz, layers[i].kz);
        AoBp = amplRatM_single(fp, AoBp, ds[i], layers[i + 1].kz, layers[i].kz);
    }

    return {AoBs, AoBp};
}

// Julia: AmplLprPl
static ABTriplet AmplLprPl(
    const LayerAtkRho0& layer,
    const std::pair<cd, cd>& AoBM,
    const std::pair<cd, cd>& BoAP,
    double zpr
) {
    cd eip = std::exp(I * layer.kz * zpr);
    cd eim = std::exp(-I * layer.kz * zpr);

    cd AoBMs = AoBM.first;
    cd AoBMp = AoBM.second;
    cd BoAPs = BoAP.first;
    cd BoAPp = BoAP.second;

    cd As_ab = AoBMs * (BoAPs * eim + eip) / (cd(1.0, 0.0) - BoAPs * AoBMs);
    cd Bs_ab = BoAPs * (AoBMs * eip + eim) / (cd(1.0, 0.0) - BoAPs * AoBMs);

    cd Ap_ab = AoBMp * (BoAPp * eim - eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_ab = -BoAPp * (AoBMp * eip - eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    cd Ap_az = AoBMp * (BoAPp * eim + eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_az = BoAPp * (AoBMp * eip + eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    return {
        Vec2c(As_ab, Bs_ab),
        Vec2c(Ap_ab, Bp_ab),
        Vec2c(Ap_az, Bp_az)
    };
}

// Julia: AmplLprMn
static ABTriplet AmplLprMn(
    const LayerAtkRho0& layer,
    const std::pair<cd, cd>& AoBM,
    const std::pair<cd, cd>& BoAP,
    double zpr
) {
    cd eip = std::exp(I * layer.kz * zpr);
    cd eim = std::exp(-I * layer.kz * zpr);

    cd AoBMs = AoBM.first;
    cd AoBMp = AoBM.second;
    cd BoAPs = BoAP.first;
    cd BoAPp = BoAP.second;

    cd As_ab = AoBMs * (BoAPs * eim + eip) / (cd(1.0, 0.0) - BoAPs * AoBMs);
    cd Bs_ab = BoAPs * (AoBMs * eip + eim) / (cd(1.0, 0.0) - BoAPs * AoBMs);

    cd Ap_ab = -AoBMp * (BoAPp * eim - eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_ab = BoAPp * (AoBMp * eip - eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    cd Ap_az = AoBMp * (BoAPp * eim + eip) / (cd(1.0, 0.0) - BoAPp * AoBMp);
    cd Bp_az = BoAPp * (AoBMp * eip + eim) / (cd(1.0, 0.0) - BoAPp * AoBMp);

    return {
        Vec2c(As_ab, Bs_ab),
        Vec2c(Ap_ab, Bp_ab),
        Vec2c(Ap_az, Bp_az)
    };
}

// Julia: γsp
static std::tuple<cd, cd, cd> gamma_sp(
    const LayerAtkRho0& l1,
    const LayerAtkRho0& l2
) {
    cd gamma_s = (l2.mu * l1.kz + l1.mu * l2.kz) / (2.0 * l2.mu * l2.kz);
    cd gamma_pab = l1.mu * l2.kz * (l2.eps * l1.kz + l1.eps * l2.kz) /
                   (2.0 * l2.mu * l1.kz * l2.eps * l2.kz);
    cd gamma_paz = l1.mu * (l2.eps * l1.kz + l1.eps * l2.kz) /
                   (2.0 * l2.mu * l2.eps * l2.kz);
    return {gamma_s, gamma_pab, gamma_paz};
}



// Julia: AmplLPl
static std::pair<Vec2c, ABplMatrix> AmplLPl(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRho0>& layers,
    std::size_t l,
    std::size_t lpr,
    const ABTriplet& ABs
) {
    Vec2c ABsl = ABs.s_ab;
    ABplMatrix ABpl = make_ABpl(ABs.p_ab, ABs.p_az);

    for (std::size_t ii = lpr; ii > l; --ii) {
        std::size_t i = ii - 1;

        double dcur = ds[i];
        const auto& lcur = layers[i];
        const auto& lnxt = layers[i + 1];

        cd Fs = cd(0.0, 0.0);
        cd Fp = esc_fp(lcur.eps, lnxt.eps);

        auto [gamma_s, gamma_pab, gamma_paz] = gamma_sp(lcur, lnxt);

        cd epsum = std::exp(I * dcur * (lnxt.kz + lcur.kz));
        cd emsum = std::exp(-I * dcur * (lnxt.kz + lcur.kz));
        cd epdif = std::exp(I * dcur * (lnxt.kz - lcur.kz));
        cd emdif = std::exp(-I * dcur * (lnxt.kz - lcur.kz));

        Mat2c Ms, Mp;
        Ms << epdif, Fs * epsum,
              Fs * emsum, emdif;
        Mp << epdif, Fp * epsum,
              Fp * emsum, emdif;

        ABsl = gamma_s * (Ms * ABsl);
        ABpl = apply_Mp_gamma(Mp, gamma_pab, gamma_paz, ABpl);
    }

    return {ABsl, ABpl};
}

// Julia: AmplLMn
static std::pair<Vec2c, ABplMatrix> AmplLMn(
    const std::vector<double>& ds,
    const std::vector<LayerAtkRho0>& layers,
    std::size_t l,
    std::size_t lpr,
    const ABTriplet& ABs
) {
    Vec2c ABsl = ABs.s_ab;
    ABplMatrix ABpl = make_ABpl(ABs.p_ab, ABs.p_az);

    for (std::size_t i = lpr + 1; i <= l; ++i) {
        double dprv = ds[i - 1];
        const auto& lcur = layers[i];
        const auto& lprv = layers[i - 1];

        cd Fs = cd(0.0, 0.0);
        cd Fp = esc_fp(lcur.eps, lprv.eps);

        auto [gamma_s, gamma_pab, gamma_paz] = gamma_sp(lcur, lprv);

        cd epsum = std::exp(I * dprv * (lprv.kz + lcur.kz));
        cd emsum = std::exp(-I * dprv * (lprv.kz + lcur.kz));
        cd epdif = std::exp(I * dprv * (lprv.kz - lcur.kz));
        cd emdif = std::exp(-I * dprv * (lprv.kz - lcur.kz));

        Mat2c Ms, Mp;
        Ms << epdif, Fs * epsum,
              Fs * emsum, emdif;
        Mp << epdif, Fp * epsum,
              Fp * emsum, emdif;

        ABsl = gamma_s * (Ms * ABsl);
        ABpl = apply_Mp_gamma(Mp, gamma_pab, gamma_paz, ABpl);
    }

    return {ABsl, ABpl};
}

// For the ESC branch we only need H1(i x) and H2(-i x) along the vertical contours.
// Using K_n(x):
// H_n^(1)(i x)  = (2/(pi i)) exp(-i n pi/2) K_n(x)
// H_n^(2)(-i x) = -(2/(pi i)) exp(+i n pi/2) K_n(x)
static cd hankel1_on_pos_imag_axis(int n, double x) {
    double K = boost::math::cyl_bessel_k(static_cast<double>(n), x);
    cd phase = std::exp(-I * (static_cast<double>(n) * PI / 2.0));
    return (2.0 / (PI * I)) * phase * K;
}

static cd hankel2_on_neg_imag_axis(int n, double x) {
    double K = boost::math::cyl_bessel_k(static_cast<double>(n), x);
    cd phase = std::exp(I * (static_cast<double>(n) * PI / 2.0));
    return -(2.0 / (PI * I)) * phase * K;
}

static EscIntegrands Fintegrands_ESC(
    cd krho,
    double rho,
    double z,
    const LayerAtkRho0& layer,
    const Vec2c&,
    const ABplMatrix& ABpl,
    bool plus_case,
    bool use_h1
) {
    cd krhor = krho * rho;
    cd klz = layer.kz;
    cd epsmu = layer.eps * layer.mu;

    cd epiklzz = std::exp(I * klz * z);
    cd emiklzz = std::exp(-I * klz * z);

    auto mix = [&](const Vec2c& AB, bool plus_sign) -> cd {
        if (plus_sign) {
            return AB(0) * epiklzz + AB(1) * emiklzz;
        }
        return AB(0) * epiklzz - AB(1) * emiklzz;
    };

    // Here krho is purely imaginary on the contours, so x = |Im(krho*rho)|
    double x = std::abs(std::imag(krhor));

    cd b0, b2;
    if (x == 0.0) {
        // limit at zero; this branch is only touched at t=0
        b0 = cd(1.0, 0.0);
        b2 = cd(0.0, 0.0);
    } else {
        if (use_h1) {
            b0 = hankel1_on_pos_imag_axis(0, x);
            b2 = hankel1_on_pos_imag_axis(2, x);
        } else {
            b0 = hankel2_on_neg_imag_axis(0, x);
            b2 = hankel2_on_neg_imag_axis(2, x);
        }
    }

    cd b1_over_rho = krho * 0.5 * (b0 + b2);
    cd b1 = rho * b1_over_rho;
    cd b0krho = krho * b0;

    cd fplxx = plus_case ? mix(ABpl.a00, false) : -mix(ABpl.a00, false);
    cd fplxz = mix(ABpl.a01, false);
    cd fplzx = plus_case ? mix(ABpl.a10, true) : -mix(ABpl.a10, true);
    cd fplzz = mix(ABpl.a11, true);

    cd klz_over_epsmu = klz / epsmu;
    cd krho2_over_epsmu = krho * krho / epsmu;
    cd minus_i_krho2_over_epsmu = -I * krho2_over_epsmu;

    EscIntegrands out;
    out.Ip1  = klz_over_epsmu * b0krho * fplxx;
    out.Ip2  = klz_over_epsmu * b1_over_rho * fplxx;
    out.Ip5  = minus_i_krho2_over_epsmu * b1 * fplxz;
    out.Ip9  = minus_i_krho2_over_epsmu * b1 * fplzx;
    out.Ip11 = (krho2_over_epsmu / klz) * b0krho * fplzz;
    return out;
}

static Mat3d integs2G_ESC(
    cd Ip1, cd Ip2, cd Ip5, cd Ip9, cd Ip11,
    double phi, double L
) {
    double sphi = std::sin(phi);
    double cphi = std::cos(phi);
    double c2phi = std::cos(2.0 * phi);

    cd Fxx = Ip1 * cphi * cphi - Ip2 * c2phi;
    cd Fxy = (Ip1 - 2.0 * Ip2) * sphi * cphi;
    cd Fxz = Ip5 * cphi;
    cd Fyx = Fxy;
    cd Fyy = Ip1 * sphi * sphi + Ip2 * c2phi;
    cd Fyz = Ip5 * sphi;
    cd Fzx = Ip9 * cphi;
    cd Fzy = Ip9 * sphi;
    cd Fzz = Ip11;

    Mat3d out;
    out <<
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fxx),
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fxy),
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fxz),

        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fyx),
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fyy),
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fyz),

        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fzx),
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fzy),
        (-1.0 / (4.0 * PI * std::pow(L, 3))) * std::imag(Fzz);

    return out;
}

// simple Simpson on one finite interval
static EscIntegrands simpson_integrate_chunk(
    const std::function<EscIntegrands(double)>& f,
    double a,
    double b,
    int n
) {
    if (n % 2 != 0) {
        ++n;
    }
    double h = (b - a) / n;

    EscIntegrands sum{};
    auto add_scaled = [](EscIntegrands& acc, const EscIntegrands& v, double s) {
        acc.Ip1  += s * v.Ip1;
        acc.Ip2  += s * v.Ip2;
        acc.Ip5  += s * v.Ip5;
        acc.Ip9  += s * v.Ip9;
        acc.Ip11 += s * v.Ip11;
    };

    add_scaled(sum, f(a), 1.0);
    add_scaled(sum, f(b), 1.0);

    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        add_scaled(sum, f(x), (i % 2 == 0) ? 2.0 : 4.0);
    }

    add_scaled(sum, sum, 0.0); // no-op, keeps structure obvious

    sum.Ip1  *= h / 3.0;
    sum.Ip2  *= h / 3.0;
    sum.Ip5  *= h / 3.0;
    sum.Ip9  *= h / 3.0;
    sum.Ip11 *= h / 3.0;
    return sum;
}

static EscIntegrands integrate_to_infinity(
    const std::function<EscIntegrands(double)>& f,
    double chunk = 2.0,
    int n_per_chunk = 200,
    int max_chunks = 200,
    double abs_tol = 1e-10
) {
    EscIntegrands total{};
    double a = 1e-16;

    auto norm5 = [](const EscIntegrands& v) {
        return std::abs(v.Ip1) + std::abs(v.Ip2) + std::abs(v.Ip5) +
               std::abs(v.Ip9) + std::abs(v.Ip11);
    };

    for (int c = 0; c < max_chunks; ++c) {
        double b = a + chunk;
        EscIntegrands part = simpson_integrate_chunk(f, a, b, n_per_chunk);

        total.Ip1  += part.Ip1;
        total.Ip2  += part.Ip2;
        total.Ip5  += part.Ip5;
        total.Ip9  += part.Ip9;
        total.Ip11 += part.Ip11;

        if (norm5(part) < abs_tol) {
            break;
        }
        a = b;
    }

    return total;
}

Mat3d ElectroS_Coupling(
    double L,
    double rho,
    double phi,
    double z,
    double zp,
    const std::vector<double>& ds,
    const std::vector<cd>& epss,
    const std::vector<cd>& mus
) {
    if (epss.size() != mus.size()) {
        throw std::runtime_error("epss and mus must have same size");
    }
    if (epss.size() != ds.size() + 1) {
        throw std::runtime_error("Need number of layers = ds.size() + 1");
    }

    std::size_t l = layer_index_desc(ds, z);
    std::size_t lp = layer_index_desc(ds, zp);

    std::vector<LayerAtFreq0> layers0;
    layers0.reserve(epss.size());
    for (std::size_t i = 0; i < epss.size(); ++i) {
        layers0.push_back(make_layer0(epss[i], mus[i]));
    }

    auto f_pos = [&](double t) -> EscIntegrands {
        cd krho = cd(0.0, t); // +i t
        std::vector<LayerAtkRho0> layers;
        layers.reserve(layers0.size());
        for (const auto& lf : layers0) {
            layers.push_back(make_layer_krho0(krho, lf));
        }

        auto BoAP = AmplRatPl_ESC(ds, layers, lp);
        auto AoBM = AmplRatMn_ESC(ds, layers, lp);

        ABTriplet ABs;
        Vec2c ABsl;
        ABplMatrix ABpl;

        if (z >= zp) {
            ABs = AmplLprPl(layers[lp], AoBM, BoAP, zp);
            auto result = AmplLPl(ds, layers, l, lp, ABs);
            ABsl = result.first;
            ABpl = result.second;
            auto v = Fintegrands_ESC(krho, rho, z, layers[l], ABsl, ABpl, true, true);
            // contour derivative dz/dt = +i
            v.Ip1  *= I;
            v.Ip2  *= I;
            v.Ip5  *= I;
            v.Ip9  *= I;
            v.Ip11 *= I;
            return v;
        } else {
            ABs = AmplLprMn(layers[lp], AoBM, BoAP, zp);
            auto result = AmplLMn(ds, layers, l, lp, ABs);
            ABsl = result.first;
            ABpl = result.second;
            auto v = Fintegrands_ESC(krho, rho, z, layers[l], ABsl, ABpl, false, true);
            v.Ip1  *= I;
            v.Ip2  *= I;
            v.Ip5  *= I;
            v.Ip9  *= I;
            v.Ip11 *= I;
            return v;
        }
    };

    auto f_neg = [&](double t) -> EscIntegrands {
        cd krho = cd(0.0, -t); // -i t
        std::vector<LayerAtkRho0> layers;
        layers.reserve(layers0.size());
        for (const auto& lf : layers0) {
            layers.push_back(make_layer_krho0(krho, lf));
        }

        auto BoAP = AmplRatPl_ESC(ds, layers, lp);
        auto AoBM = AmplRatMn_ESC(ds, layers, lp);

        ABTriplet ABs;
        Vec2c ABsl;
        ABplMatrix ABpl;

        if (z >= zp) {
            ABs = AmplLprPl(layers[lp], AoBM, BoAP, zp);
            auto result = AmplLPl(ds, layers, l, lp, ABs);
            ABsl = result.first;
            ABpl = result.second;
            auto v = Fintegrands_ESC(krho, rho, z, layers[l], ABsl, ABpl, true, false);
            // contour derivative dz/dt = -i
            v.Ip1  *= -I;
            v.Ip2  *= -I;
            v.Ip5  *= -I;
            v.Ip9  *= -I;
            v.Ip11 *= -I;
            return v;
        } else {
            ABs = AmplLprMn(layers[lp], AoBM, BoAP, zp);
            auto result = AmplLMn(ds, layers, l, lp, ABs);
            ABsl = result.first;
            ABpl = result.second;
            auto v = Fintegrands_ESC(krho, rho, z, layers[l], ABsl, ABpl, false, false);
            v.Ip1  *= -I;
            v.Ip2  *= -I;
            v.Ip5  *= -I;
            v.Ip9  *= -I;
            v.Ip11 *= -I;
            return v;
        }
    };

    EscIntegrands ints_pos = integrate_to_infinity(f_pos);
    EscIntegrands ints_neg = integrate_to_infinity(f_neg);

    EscIntegrands integs;
    integs.Ip1  = 0.5 * (ints_pos.Ip1  + ints_neg.Ip1);
    integs.Ip2  = 0.5 * (ints_pos.Ip2  + ints_neg.Ip2);
    integs.Ip5  = 0.5 * (ints_pos.Ip5  + ints_neg.Ip5);
    integs.Ip9  = 0.5 * (ints_pos.Ip9  + ints_neg.Ip9);
    integs.Ip11 = 0.5 * (ints_pos.Ip11 + ints_neg.Ip11);

    return integs2G_ESC(integs.Ip1, integs.Ip2, integs.Ip5, integs.Ip9, integs.Ip11, phi, L);
}

Mat3d Rotate_ESC(double theta, const Mat3d& esc) {
    double c = std::cos(theta);
    double s = std::sin(theta);

    Mat3d R, Ri;
    R <<  c, -s, 0.0,
          s,  c, 0.0,
        0.0, 0.0, 1.0;

    Ri <<  c,  s, 0.0,
          -s,  c, 0.0,
         0.0, 0.0, 1.0;

    return Ri * (esc * R);
}

Mat3d Rotate_ESCv2(double theta, const Mat3d& esc) {
    return Rotate_ESC(theta, esc);
}

std::array<Mat3d, 3> Rotate_ESC_3quarter(const Mat3d& esc) {
    Mat3d Rpi2, Ripi2, Rpi, Ripi, R3pi2, Ri3pi2;

    Rpi2 << 0.0, -1.0, 0.0,
            1.0,  0.0, 0.0,
            0.0,  0.0, 1.0;

    Ripi2 << 0.0,  1.0, 0.0,
            -1.0,  0.0, 0.0,
             0.0,  0.0, 1.0;

    Rpi << -1.0,  0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0,  0.0, 1.0;

    Ripi << -1.0,  0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0,  0.0, 1.0;

    R3pi2 <<  0.0,  1.0, 0.0,
             -1.0,  0.0, 0.0,
              0.0,  0.0, 1.0;

    Ri3pi2 <<  0.0, -1.0, 0.0,
               1.0,  0.0, 0.0,
               0.0,  0.0, 1.0;

    return {
        Ripi2 * (esc * Rpi2),
        Ripi  * (esc * Rpi),
        Ri3pi2 * (esc * R3pi2)
    };
}

static Mat3d FreeSpaceCoupling_local(const Eigen::Vector3d& r,
                                     const Eigen::Vector3d& rp,
                                     double eps_b,
                                     double mu_b) {
    Eigen::Vector3d rho_f = r - rp;
    double n_rho_f = rho_f.norm();
    Eigen::Vector3d e_rho_f = rho_f / n_rho_f;
    Mat3d lambda0 = 3.0 * (e_rho_f * e_rho_f.transpose()) - Mat3d::Identity();
    return lambda0 / (4.0 * PI * std::pow(n_rho_f, 3) * eps_b * mu_b);
}

Eigen::Vector3d MF_Coupling(
    int Np,
    double spacing,
    double z,
    const std::vector<double>& ds,
    const std::vector<cd>& epss,
    const std::vector<cd>& mus
) {
    double a = spacing;

    std::vector<double> xs(2 * Np + 1), ys(2 * Np + 1);
    for (int i = 0; i < 2 * Np + 1; ++i) {
        xs[i] = -Np * a + i * a;
        ys[i] = -Np * a + i * a;
    }

    Mat3d MFC = Mat3d::Zero();
    Eigen::Vector3d r(0.0, 0.0, z);

    for (int ix = Np + 1; ix < 2 * Np + 1; ++ix) {
        {
            Eigen::Vector3d rp(xs[ix], ys[Np], z);
            double dx = r(0) - rp(0);
            double dy = r(1) - rp(1);
            double rho = std::sqrt(dx * dx + dy * dy);
            double phi = std::atan2(dy, dx);
            double L = rho;

            Mat3d ECT = ElectroS_Coupling(L, rho / L, phi, r(2) / L, rp(2) / L,
                                          [&]() {
                                              std::vector<double> ds_scaled(ds.size());
                                              for (std::size_t k = 0; k < ds.size(); ++k) ds_scaled[k] = ds[k] / L;
                                              return ds_scaled;
                                          }(),
                                          epss, mus);

            Mat3d block;
            block << 2.0 * (ECT(0,0) + ECT(1,1)), 2.0 * (ECT(0,1) - ECT(1,0)), 0.0,
                     2.0 * (ECT(1,0) - ECT(0,1)), 2.0 * (ECT(0,0) + ECT(1,1)), 0.0,
                     0.0,                            0.0,                          4.0 * ECT(2,2);
            MFC += block;
        }

        for (int iy = Np + 1; iy < ix; ++iy) {
            Eigen::Vector3d rp(xs[ix], ys[iy], z);
            double dx = r(0) - rp(0);
            double dy = r(1) - rp(1);
            double rho = std::sqrt(dx * dx + dy * dy);
            double phi = std::atan2(dy, dx);
            double L = rho;

            std::vector<double> ds_scaled(ds.size());
            for (std::size_t k = 0; k < ds.size(); ++k) ds_scaled[k] = ds[k] / L;

            Mat3d ECT = ElectroS_Coupling(L, rho / L, phi, r(2) / L, rp(2) / L,
                                          ds_scaled, epss, mus);

            Mat3d block1;
            block1 << 2.0 * (ECT(0,0) + ECT(1,1)), 2.0 * (ECT(0,1) - ECT(1,0)), 0.0,
                      2.0 * (ECT(1,0) - ECT(0,1)), 2.0 * (ECT(0,0) + ECT(1,1)), 0.0,
                      0.0,                           0.0,                          4.0 * ECT(2,2);
            MFC += block1;

            double theta = std::acos(-2.0 * rp(1) / (rp(0) + rp(1) * rp(1) / rp(0)));
            Mat3d ECTr = Rotate_ESCv2(theta, ECT);

            Mat3d block2;
            block2 << 2.0 * (ECTr(0,0) + ECTr(1,1)), 2.0 * (ECTr(0,1) - ECTr(1,0)), 0.0,
                      2.0 * (ECTr(1,0) - ECTr(0,1)), 2.0 * (ECTr(0,0) + ECTr(1,1)), 0.0,
                      0.0,                             0.0,                            4.0 * ECTr(2,2);
            MFC += block2;
        }

        {
            Eigen::Vector3d rp(xs[ix], ys[ix], z);
            double dx = r(0) - rp(0);
            double dy = r(1) - rp(1);
            double rho = std::sqrt(dx * dx + dy * dy);
            double phi = std::atan2(dy, dx);
            double L = rho;

            std::vector<double> ds_scaled(ds.size());
            for (std::size_t k = 0; k < ds.size(); ++k) ds_scaled[k] = ds[k] / L;

            Mat3d ECT = ElectroS_Coupling(L, rho / L, phi, r(2) / L, rp(2) / L,
                                          ds_scaled, epss, mus);

            Mat3d block;
            block << 2.0 * (ECT(0,0) + ECT(1,1)), 2.0 * (ECT(0,1) - ECT(1,0)), 0.0,
                     2.0 * (ECT(1,0) - ECT(0,1)), 2.0 * (ECT(0,0) + ECT(1,1)), 0.0,
                     0.0,                            0.0,                          4.0 * ECT(2,2);
            MFC += block;
        }
    }

    return Eigen::Vector3d(MFC(0,0), MFC(1,1), MFC(2,2));
}

Eigen::Vector3d MF_Coupling_FS(
    int Np,
    double spacing,
    double z,
    double epsb,
    double mub
) {
    double a = spacing;

    std::vector<double> xs(2 * Np + 1), ys(2 * Np + 1);
    for (int i = 0; i < 2 * Np + 1; ++i) {
        xs[i] = -Np * a + i * a;
        ys[i] = -Np * a + i * a;
    }

    Mat3d MFC = Mat3d::Zero();
    Eigen::Vector3d r(0.0, 0.0, z);

    for (int ix = Np + 1; ix < 2 * Np + 1; ++ix) {
        {
            Eigen::Vector3d rp(xs[ix], ys[Np], z);
            Mat3d ECT = FreeSpaceCoupling_local(r, rp, epsb, mub);

            Mat3d block;
            block << 2.0 * (ECT(0,0) + ECT(1,1)), 2.0 * (ECT(0,1) - ECT(1,0)), 0.0,
                     2.0 * (ECT(1,0) - ECT(0,1)), 2.0 * (ECT(0,0) + ECT(1,1)), 0.0,
                     0.0,                            0.0,                          4.0 * ECT(2,2);
            MFC += block;
        }

        for (int iy = Np + 1; iy < ix; ++iy) {
            Eigen::Vector3d rp(xs[ix], ys[iy], z);
            Mat3d ECT = FreeSpaceCoupling_local(r, rp, epsb, mub);

            Mat3d block1;
            block1 << 2.0 * (ECT(0,0) + ECT(1,1)), 2.0 * (ECT(0,1) - ECT(1,0)), 0.0,
                      2.0 * (ECT(1,0) - ECT(0,1)), 2.0 * (ECT(0,0) + ECT(1,1)), 0.0,
                      0.0,                           0.0,                          4.0 * ECT(2,2);
            MFC += block1;

            double theta = std::acos(-2.0 * rp(1) / (rp(0) + rp(1) * rp(1) / rp(0)));
            Mat3d ECTr = Rotate_ESCv2(theta, ECT);

            Mat3d block2;
            block2 << 2.0 * (ECTr(0,0) + ECTr(1,1)), 2.0 * (ECTr(0,1) - ECTr(1,0)), 0.0,
                      2.0 * (ECTr(1,0) - ECTr(0,1)), 2.0 * (ECTr(0,0) + ECTr(1,1)), 0.0,
                      0.0,                             0.0,                            4.0 * ECTr(2,2);
            MFC += block2;
        }

        {
            Eigen::Vector3d rp(xs[ix], ys[ix], z);
            Mat3d ECT = FreeSpaceCoupling_local(r, rp, epsb, mub);

            Mat3d block;
            block << 2.0 * (ECT(0,0) + ECT(1,1)), 2.0 * (ECT(0,1) - ECT(1,0)), 0.0,
                     2.0 * (ECT(1,0) - ECT(0,1)), 2.0 * (ECT(0,0) + ECT(1,1)), 0.0,
                     0.0,                            0.0,                          4.0 * ECT(2,2);
            MFC += block;
        }
    }

    return Eigen::Vector3d(MFC(0,0), MFC(1,1), MFC(2,2));
}

struct DynIntegrands {
    cd Is1;
    cd Is2;
    cd Ip1;
    cd Ip2;
    cd Ip5;
    cd Ip9;
    cd Ip11;
};

static cd besselJ_int(int n, double x) {
    return cd(boost::math::cyl_bessel_j(static_cast<double>(n), x), 0.0);
}

// H_n^(1)(i x)  = (2/(pi i)) exp(-i n pi/2) K_n(x)
// H_n^(2)(-i x) = -(2/(pi i)) exp(+i n pi/2) K_n(x)
static cd hankel1_on_pos_imag_axis_dyn(int n, double x) {
    double K = boost::math::cyl_bessel_k(static_cast<double>(n), x);
    cd phase = std::exp(-I * (static_cast<double>(n) * PI / 2.0));
    return (2.0 / (PI * I)) * phase * K;
}

static cd hankel2_on_neg_imag_axis_dyn(int n, double x) {
    double K = boost::math::cyl_bessel_k(static_cast<double>(n), x);
    cd phase = std::exp(I * (static_cast<double>(n) * PI / 2.0));
    return -(2.0 / (PI * I)) * phase * K;
}

static DynIntegrands Fintegrands_dyn(
    cd krho,
    double rho,
    double z,
    const LayerAtkRhoDyn& layer,
    const Vec2c& ABsl,
    const ABplMatrix& ABpl,
    bool plus_case,
    int bessel_mode
) {
    // bessel_mode:
    // 0 -> J_n on real axis
    // 1 -> H1 on +imag vertical contour
    // 2 -> H2 on -imag vertical contour

    cd krhor = krho * rho;
    cd kl = layer.k;
    cd klz = layer.kz;

    cd epiklzz = std::exp(I * klz * z);
    cd emiklzz = std::exp(-I * klz * z);

    auto mix = [&](const Vec2c& AB, bool plus_sign) -> cd {
        if (plus_sign) {
            return AB(0) * epiklzz + AB(1) * emiklzz;
        }
        return AB(0) * epiklzz - AB(1) * emiklzz;
    };

    cd b0, b2;

    if (bessel_mode == 0) {
        // real-axis branch
        double x = std::real(krhor);
        if (std::abs(x) < 1e-15) {
            b0 = cd(1.0, 0.0);
            b2 = cd(0.0, 0.0);
        } else {
            b0 = besselJ_int(0, x);
            b2 = besselJ_int(2, x);
        }
    } else if (bessel_mode == 1) {
        double x = std::abs(std::imag(krhor));
        if (x < 1e-15) {
            b0 = cd(1.0, 0.0);
            b2 = cd(0.0, 0.0);
        } else {
            b0 = hankel1_on_pos_imag_axis_dyn(0, x);
            b2 = hankel1_on_pos_imag_axis_dyn(2, x);
        }
    } else {
        double x = std::abs(std::imag(krhor));
        if (x < 1e-15) {
            b0 = cd(1.0, 0.0);
            b2 = cd(0.0, 0.0);
        } else {
            b0 = hankel2_on_neg_imag_axis_dyn(0, x);
            b2 = hankel2_on_neg_imag_axis_dyn(2, x);
        }
    }

    cd b1_over_rho = krho * 0.5 * (b0 + b2);
    cd b1 = rho * b1_over_rho;
    cd b0krho = krho * b0;

    // s polarization
    cd fsl = mix(ABsl, true);
    cd Is1 = b0krho / klz * fsl;
    cd Is2 = b1_over_rho / klz * fsl;

    // p polarization
    cd fplxx = plus_case ? mix(ABpl.a00, false) : -mix(ABpl.a00, false);
    cd fplxz = mix(ABpl.a01, false);
    cd fplzx = plus_case ? mix(ABpl.a10, true) : -mix(ABpl.a10, true);
    cd fplzz = mix(ABpl.a11, true);

    cd klz_over_kl2 = klz / (kl * kl);
    cd krho2_over_kl2 = (krho * krho) / (kl * kl);
    cd minus_i_krho2_over_kl2 = -I * krho2_over_kl2;

    DynIntegrands out;
    out.Is1  = Is1;
    out.Is2  = Is2;
    out.Ip1  = klz_over_kl2 * b0krho * fplxx;
    out.Ip2  = klz_over_kl2 * b1_over_rho * fplxx;
    out.Ip5  = minus_i_krho2_over_kl2 * b1 * fplxz;
    out.Ip9  = minus_i_krho2_over_kl2 * b1 * fplzx;
    out.Ip11 = (krho2_over_kl2 / klz) * b0krho * fplzz;
    return out;
}

static Eigen::Matrix<cd, 3, 3> integs2G_dyn(
    cd Is1, cd Is2, cd Ip1, cd Ip2, cd Ip5, cd Ip9, cd Ip11,
    double phi,
    const std::string& pol
) {
    double sphi = std::sin(phi);
    double cphi = std::cos(phi);
    double c2phi = std::cos(2.0 * phi);

    cd Fxx, Fxy, Fxz, Fyx, Fyy, Fyz, Fzx, Fzy, Fzz;

    if (pol == "P") {
        Fxx = Ip1 * cphi * cphi - Ip2 * c2phi;
        Fxy = (Ip1 - 2.0 * Ip2) * sphi * cphi;
        Fxz = Ip5 * cphi;
        Fyx = Fxy;
        Fyy = Ip1 * sphi * sphi + Ip2 * c2phi;
        Fyz = Ip5 * sphi;
        Fzx = Ip9 * cphi;
        Fzy = Ip9 * sphi;
        Fzz = Ip11;
    } else if (pol == "S") {
        Fxx = Is1 * sphi * sphi + Is2 * c2phi;
        Fxy = (-Is1 + 2.0 * Is2) * sphi * cphi;
        Fxz = cd(0.0, 0.0);
        Fyx = Fxy;
        Fyy = Is1 * cphi * cphi - Is2 * c2phi;
        Fyz = cd(0.0, 0.0);
        Fzx = cd(0.0, 0.0);
        Fzy = cd(0.0, 0.0);
        Fzz = cd(0.0, 0.0);
    } else {
        Fxx = Is1 * sphi * sphi + Is2 * c2phi + Ip1 * cphi * cphi - Ip2 * c2phi;
        Fxy = (-Is1 + 2.0 * Is2 + Ip1 - 2.0 * Ip2) * sphi * cphi;
        Fxz = Ip5 * cphi;
        Fyx = Fxy;
        Fyy = Is1 * cphi * cphi - Is2 * c2phi + Ip1 * sphi * sphi + Ip2 * c2phi;
        Fyz = Ip5 * sphi;
        Fzx = Ip9 * cphi;
        Fzy = Ip9 * sphi;
        Fzz = Ip11;
    }

    Eigen::Matrix<cd, 3, 3> F;
    F << Fxx, Fxy, Fxz,
         Fyx, Fyy, Fyz,
         Fzx, Fzy, Fzz;

    return (I / (4.0 * PI)) * F;
}

struct DynIntContext {
    std::function<DynIntegrands(cd,int)> eval;
    double klmax;
    double rho;
    double phi;
    double dz;
};

static DynIntContext MultiLayerGF_integrand_dyn(
    const Eigen::Vector3d& r,
    const Eigen::Vector3d& rp,
    cd k0,
    const std::vector<double>& ds,
    const std::vector<cd>& epss,
    const std::vector<cd>& mus
) {
    double dx = r(0) - rp(0);
    double dy = r(1) - rp(1);
    double rho = std::sqrt(dx * dx + dy * dy);
    double phi = std::atan2(dy, dx);
    double z = r(2);
    double zp = rp(2);

    std::size_t l = layer_index_desc(ds, z);
    std::size_t lp = layer_index_desc(ds, zp);

    std::vector<LayerAtFreqDyn> layers_freq;
    layers_freq.reserve(epss.size());
    for (std::size_t i = 0; i < epss.size(); ++i) {
        layers_freq.push_back(make_layer_dyn(k0, epss[i], mus[i]));
    }

    double klmax = 0.0;
    for (const auto& lf : layers_freq) {
        klmax = std::max(klmax, std::real(lf.k));
    }

    auto eval = [=](cd krho, int bessel_mode) -> DynIntegrands {
        std::vector<LayerAtkRhoDyn> layers;
        layers.reserve(layers_freq.size());
        for (const auto& lf : layers_freq) {
            layers.push_back(make_layer_krho_dyn(krho, lf));
        }

        auto BoAP = AmplRatPl_dyn(ds, layers, lp);
        auto AoBM = AmplRatMn_dyn(ds, layers, lp);

        ABTriplet ABs;
        Vec2c ABsl;
        ABplMatrix ABpl;

        if (z >= zp) {
            ABs = AmplLprPl_dyn(layers[lp], AoBM, BoAP, zp);
            auto result = AmplLPl_dyn(ds, layers, l, lp, ABs);
            ABsl = result.first;
            ABpl = result.second;
            return Fintegrands_dyn(krho, rho, z, layers[l], ABsl, ABpl, true, bessel_mode);
        } else {
            ABs = AmplLprMn_dyn(layers[lp], AoBM, BoAP, zp);
            auto result = AmplLMn_dyn(ds, layers, l, lp, ABs);
            ABsl = result.first;
            ABpl = result.second;
            return Fintegrands_dyn(krho, rho, z, layers[l], ABsl, ABpl, false, bessel_mode);
        }
    };

    return {eval, klmax, rho, phi, z - zp};
}

template <typename T>
static void add_dyn_scaled(DynIntegrands& acc, const DynIntegrands& v, T s) {
    acc.Is1  += s * v.Is1;
    acc.Is2  += s * v.Is2;
    acc.Ip1  += s * v.Ip1;
    acc.Ip2  += s * v.Ip2;
    acc.Ip5  += s * v.Ip5;
    acc.Ip9  += s * v.Ip9;
    acc.Ip11 += s * v.Ip11;
}

static DynIntegrands simpson_integrate_real(
    const std::function<DynIntegrands(double)>& f,
    double a,
    double b,
    int n
) {
    if (n % 2 != 0) ++n;
    double h = (b - a) / n;

    DynIntegrands sum{};
    add_dyn_scaled(sum, f(a), 1.0);
    add_dyn_scaled(sum, f(b), 1.0);

    for (int i = 1; i < n; ++i) {
        double x = a + i * h;
        add_dyn_scaled(sum, f(x), (i % 2 == 0) ? 2.0 : 4.0);
    }

    sum.Is1  *= h / 3.0;
    sum.Is2  *= h / 3.0;
    sum.Ip1  *= h / 3.0;
    sum.Ip2  *= h / 3.0;
    sum.Ip5  *= h / 3.0;
    sum.Ip9  *= h / 3.0;
    sum.Ip11 *= h / 3.0;
    return sum;
}

static DynIntegrands integrate_vertical_tail(
    const std::function<DynIntegrands(double)>& f,
    double chunk = 1.0,
    int n_per_chunk = 300,
    int max_chunks = 300,
    double abs_tol = 1e-11,
    double start = 1e-8
) {
    DynIntegrands total{};
    double a = start;

    auto norm7 = [](const DynIntegrands& v) {
        return std::abs(v.Is1) + std::abs(v.Is2) + std::abs(v.Ip1) +
               std::abs(v.Ip2) + std::abs(v.Ip5) + std::abs(v.Ip9) +
               std::abs(v.Ip11);
    };

    for (int c = 0; c < max_chunks; ++c) {
        double b = a + chunk;
        DynIntegrands part = simpson_integrate_real(f, a, b, n_per_chunk);

        total.Is1  += part.Is1;
        total.Is2  += part.Is2;
        total.Ip1  += part.Ip1;
        total.Ip2  += part.Ip2;
        total.Ip5  += part.Ip5;
        total.Ip9  += part.Ip9;
        total.Ip11 += part.Ip11;

        if (norm7(part) < abs_tol) break;
        a = b;
    }

    return total;
}

Eigen::Matrix<cd, 3, 3> MultiLayerGF(
    const Eigen::Vector3d& r,
    const Eigen::Vector3d& rp,
    cd k0,
    const std::vector<double>& ds,
    const std::vector<cd>& epss,
    const std::vector<cd>& mus,
    const std::string& pol
) {
    auto ctx = MultiLayerGF_integrand_dyn(r, rp, k0, ds, epss, mus);

    double krho_end = ctx.klmax + std::real(k0);

    // Real-axis contribution: J_n
    auto f_real = [&](double x) -> DynIntegrands {
        return ctx.eval(cd(x, 0.0), 0);
    };
    DynIntegrands ints1 = simpson_integrate_real(f_real, 1e-16, krho_end, 800);

    // Vertical contour from krho_end + i t : H1 branch
    auto f_pos = [&](double t) -> DynIntegrands {
        cd krho(krho_end, t);
        DynIntegrands v = ctx.eval(krho, 1);
        v.Is1  *= I;
        v.Is2  *= I;
        v.Ip1  *= I;
        v.Ip2  *= I;
        v.Ip5  *= I;
        v.Ip9  *= I;
        v.Ip11 *= I;
        return v;
    };

    // Vertical contour from krho_end - i t : H2 branch
    auto f_neg = [&](double t) -> DynIntegrands {
        cd krho(krho_end, -t);
        DynIntegrands v = ctx.eval(krho, 2);
        v.Is1  *= -I;
        v.Is2  *= -I;
        v.Ip1  *= -I;
        v.Ip2  *= -I;
        v.Ip5  *= -I;
        v.Ip9  *= -I;
        v.Ip11 *= -I;
        return v;
    };

    DynIntegrands ints2a = integrate_vertical_tail(f_pos, 1.0, 300, 300, 1e-11);
    DynIntegrands ints2b = integrate_vertical_tail(f_neg, 1.0, 300, 300, 1e-11);

    DynIntegrands ints2;
    ints2.Is1  = 0.5 * (ints2a.Is1  + ints2b.Is1);
    ints2.Is2  = 0.5 * (ints2a.Is2  + ints2b.Is2);
    ints2.Ip1  = 0.5 * (ints2a.Ip1  + ints2b.Ip1);
    ints2.Ip2  = 0.5 * (ints2a.Ip2  + ints2b.Ip2);
    ints2.Ip5  = 0.5 * (ints2a.Ip5  + ints2b.Ip5);
    ints2.Ip9  = 0.5 * (ints2a.Ip9  + ints2b.Ip9);
    ints2.Ip11 = 0.5 * (ints2a.Ip11 + ints2b.Ip11);

    DynIntegrands integs;
    integs.Is1  = ints1.Is1  + ints2.Is1;
    integs.Is2  = ints1.Is2  + ints2.Is2;
    integs.Ip1  = ints1.Ip1  + ints2.Ip1;
    integs.Ip2  = ints1.Ip2  + ints2.Ip2;
    integs.Ip5  = ints1.Ip5  + ints2.Ip5;
    integs.Ip9  = ints1.Ip9  + ints2.Ip9;
    integs.Ip11 = ints1.Ip11 + ints2.Ip11;

    return integs2G_dyn(
        integs.Is1, integs.Is2,
        integs.Ip1, integs.Ip2, integs.Ip5, integs.Ip9, integs.Ip11,
        ctx.phi,
        pol
    );
}

} // namespace mlgf