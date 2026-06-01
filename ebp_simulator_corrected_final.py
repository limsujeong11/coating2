"""
Spin Coating Thin-Film Simulator — EBP + Meyerhofer + Eddy-Viscosity Correction
=================================================================================
Subject 1: Reconstructing the Emslie-Bonner-Peck Theory
SKKU Fluid Mechanics Term Project 2026 Spring

Core physics:
  - EBP thin-film PDE (laminar baseline)
  - Meyerhofer concentration-coupled viscosity
  - Transition/turbulent eddy-viscosity correction (engineering level only)
  - Correct Re_rot = rho*omega*r^2/eta  (no h in denominator)
  - Meyerhofer Lambda analysis (rotation vs evaporation regime)

NOT a DNS/LES/RANS solver. Eddy-viscosity is an engineering correction.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EBP Spin Coating Simulator",
    page_icon="🔬", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-title{font-family:'IBM Plex Mono',monospace;font-size:1.85rem;font-weight:700;
  color:#0f172a;border-bottom:3px solid #0ea5e9;padding-bottom:.45rem;margin-bottom:.1rem;}
.subtitle{font-size:.85rem;color:#64748b;margin-bottom:1.2rem;line-height:1.5;}
.sec{font-weight:700;color:#0f172a;font-size:.72rem;text-transform:uppercase;
  letter-spacing:.07em;margin:1rem 0 .35rem;padding-top:.55rem;border-top:1px solid #e2e8f0;}
.mc{border-radius:10px;padding:.85rem 1rem;margin:.2rem 0;}
.mc-b{background:linear-gradient(135deg,#f0f9ff,#dbeafe);border:1px solid #93c5fd;}
.mc-g{background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:1px solid #86efac;}
.mc-y{background:linear-gradient(135deg,#fffbeb,#fef3c7);border:1px solid #fcd34d;}
.mc-r{background:linear-gradient(135deg,#fff1f2,#ffe4e6);border:1px solid #fca5a5;}
.mc-p{background:linear-gradient(135deg,#faf5ff,#ede9fe);border:1px solid #c4b5fd;}
.ml{font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.05em;}
.ml-b{color:#1d4ed8;} .ml-g{color:#15803d;} .ml-y{color:#b45309;}
.ml-r{color:#b91c1c;} .ml-p{color:#6d28d9;}
.mv{font-size:1.4rem;font-weight:800;}
.mv-b{color:#1e3a8a;} .mv-g{color:#14532d;} .mv-y{color:#78350f;}
.mv-r{color:#7f1d1d;} .mv-p{color:#4c1d95;}
.mu{font-size:.73rem;color:#64748b;}
.info{background:#f0f9ff;border-left:4px solid #0ea5e9;padding:.6rem .9rem;
  border-radius:0 8px 8px 0;font-size:.82rem;margin:.45rem 0;}
.ok  {background:#f0fdf4;border-left:4px solid #22c55e;padding:.6rem .9rem;
  border-radius:0 8px 8px 0;font-size:.82rem;margin:.45rem 0;}
.warn{background:#fffbeb;border-left:4px solid #f59e0b;padding:.6rem .9rem;
  border-radius:0 8px 8px 0;font-size:.82rem;margin:.45rem 0;}
.fail{background:#fff1f2;border-left:4px solid #ef4444;padding:.6rem .9rem;
  border-radius:0 8px 8px 0;font-size:.82rem;margin:.45rem 0;}
.eq  {background:#f8fafc;border:1px solid #cbd5e1;border-radius:8px;
  padding:.7rem 1.1rem;font-family:'IBM Plex Mono',monospace;
  font-size:.77rem;color:#1e40af;margin:.5rem 0;white-space:pre-line;line-height:1.7;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔬 EBP Spin Coating Simulator</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">'
    'Emslie-Bonner-Peck thin-film theory &nbsp;·&nbsp; Meyerhofer viscosity model '
    '&nbsp;·&nbsp; Rotation/evaporation regime analysis &nbsp;·&nbsp; '
    'Eddy-viscosity boundary-layer correction (engineering level) &nbsp;|&nbsp; '
    'SKKU Fluid Mechanics 2026</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Process Settings")

    st.markdown('<div class="sec">Photoresist</div>', unsafe_allow_html=True)
    pr_preset = st.selectbox("PR type", [
        "Custom",
        "i-line PR (~20 mPa·s, 500 nm)",
        "KrF DUV PR (~10 mPa·s, 300 nm)",
        "ArF DUV PR (~5 mPa·s, 150 nm)",
        "EUV MO-PR (~3 mPa·s, 80 nm)",
    ])
    PRESETS = {
        "i-line PR (~20 mPa·s, 500 nm)":  dict(eta0=20, rho=1060,h0=500, n=2.0,E=8),
        "KrF DUV PR (~10 mPa·s, 300 nm)": dict(eta0=10, rho=1050,h0=300, n=2.2,E=12),
        "ArF DUV PR (~5 mPa·s, 150 nm)":  dict(eta0=5,  rho=1030,h0=150, n=2.5,E=15),
        "EUV MO-PR (~3 mPa·s, 80 nm)":    dict(eta0=3,  rho=1020,h0=80,  n=3.0,E=20),
    }
    P = PRESETS.get(pr_preset)
    def pv(k,d): return P[k] if P else d

    st.markdown('<div class="sec">4-Phase Spin Recipe</div>', unsafe_allow_html=True)
    omega_rpm = st.slider("Target ω (rpm)", 500, 8000, 3000, 100)
    omega_max = omega_rpm * 2*np.pi/60
    c1, c2 = st.columns(2)
    with c1:
        t_d  = st.slider("Dispense (s)", 0.5, 3.0, 1.0, 0.5)
        om_d_rpm = st.slider("Dispense ω (rpm)", 100, 600, 300, 50)
        t_up = st.slider("Ramp-up (s)", 0.5, 5.0, 2.0, 0.5)
    with c2:
        t_st = st.slider("Steady (s)", 5.0, 60.0, 20.0, 5.0)
        t_dn = st.slider("Ramp-down (s)", 0.5, 5.0, 2.0, 0.5)
    t_end = t_d + t_up + t_st + t_dn
    st.markdown(f'<div class="info">Total: <b>{t_end:.1f} s</b></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sec">PR Properties</div>', unsafe_allow_html=True)
    h0_nm    = st.slider("h₀ (nm)",    50, 2000, pv('h0',500), 10)
    h0       = h0_nm * 1e-9
    eta0_mPas= st.slider("η₀ (mPa·s)", 1, 200, pv('eta0',20), 1)
    eta0     = eta0_mPas * 1e-3
    rho      = st.number_input("ρ (kg/m³)", 800, 1800, pv('rho',1060), 10)
    n_visc   = st.slider("Meyerhofer n", 1.0, 4.0, float(pv('n',2.0)), 0.1,
                          help="η(t)=η₀·(h₀/h̄)ⁿ. n≈2 for most PR solutions.")
    E_nm_s   = st.slider("E (nm/s)", 0, 50, pv('E',10), 1,
                          help="Uniform solvent evaporation flux.")
    E        = E_nm_s * 1e-9
    gamma_mNm= st.slider("γ surface tension (mN/m)", 10, 70, 30, 1,
                          help="Used for Capillary number Ca.")
    gamma_SI = gamma_mNm * 1e-3   # [N/m]
    eta_gel_ratio = st.slider("Gelation η/η₀", 10, 100, 50, 5,
                               help="Meyerhofer gelation: when η/η₀ ≥ this ratio.")

    st.markdown('<div class="sec">Wafer</div>', unsafe_allow_html=True)
    wafer = st.selectbox("Diameter", ["300 mm","200 mm","150 mm"], index=0)
    R     = {"150 mm":0.075,"200 mm":0.100,"300 mm":0.150}[wafer]

    st.markdown('<div class="sec">Boundary-Layer Transition Correction</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="eq">Re_rot = ρ·ω·r²/η_lam  (no h in denominator)\n'
        'Laminar  : Re_rot < 1×10⁵\n'
        'Transition: 1×10⁵ ≤ Re < Re_turb\n'
        'Turbulent : Re ≥ Re_turb\n'
        'Engineering eddy-viscosity correction only.</div>',
        unsafe_allow_html=True)
    turb_on = st.checkbox("Enable eddy-viscosity correction", value=True)
    Re_turb = st.select_slider("Re_turb", [100000,200000,300000,500000,1000000],
                                value=300000,
                                help="Transition onset. Default 3×10⁵ (Dorfman 1963).")
    C_mix   = st.slider("C_mix", 0.001, 0.15, 0.015, 0.001, format="%.3f")
    beta_exp= st.slider("β exponent", 0.3, 1.0, 0.70, 0.05)

    st.markdown('<div class="sec">Grid</div>', unsafe_allow_html=True)
    Nr = st.select_slider("Nr", [40,60,80,100,120], value=80)

# ─────────────────────────────────────────────────────────────────────────────
# PHYSICS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def omega_profile(t_now, omega_max, om_d_rads, t_d, t_up, t_st, t_dn):
    """4-phase industrial spin recipe → ω(t) [rad/s]."""
    if t_now < t_d:
        return om_d_rads
    elif t_now < t_d + t_up:
        f = (t_now - t_d) / max(t_up, 1e-9)
        return om_d_rads + (omega_max - om_d_rads) * f
    elif t_now < t_d + t_up + t_st:
        return omega_max
    else:
        elapsed = t_now - (t_d + t_up + t_st)
        return omega_max * max(0.0, 1.0 - elapsed / max(t_dn, 1e-9))


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def compute_flow_regime_and_eta_eff(r, eta_lam, ow, rho_f,
                                     Re_lam=1e5, Re_turb=3e5,
                                     C_mix=0.015, beta_exp=0.70,
                                     turbulent=True):
    """
    Re_rot(r) = rho*omega*r^2 / eta_lam   [dimensionless, no h]

    Regime:  0=laminar, 1=transition, 2=turbulent
    eta_eff = eta_lam * [1 + C_mix * S(Re) * (Re/Re_turb - 1)^beta]
    where S = smoothstep((Re - Re_lam)/(Re_turb - Re_lam))

    This is an engineering-level eddy-viscosity correction.
    NOT a full turbulence solver.
    """
    eta_eff = np.full_like(r, eta_lam, dtype=float)
    if (not turbulent) or ow < 1.0:
        return eta_eff, np.zeros_like(r), np.zeros_like(r, dtype=int)

    Re_rot = rho_f * ow * r**2 / eta_lam      # correct: no h

    regime = np.zeros_like(r, dtype=int)
    regime[(Re_rot >= Re_lam) & (Re_rot < Re_turb)] = 1
    regime[Re_rot >= Re_turb] = 2

    x      = (Re_rot - Re_lam) / max(Re_turb - Re_lam, 1.0)
    S      = smoothstep(x)
    excess = np.maximum(Re_rot / max(Re_turb, 1.0) - 1.0, 0.0)
    eta_eff = eta_lam * (1.0 + C_mix * S * excess**beta_exp)

    return np.maximum(eta_eff, eta_lam), Re_rot, regime


def compute_sublayer_indicator(Re_rot, R, C_delta=62.0):
    """Laminar sublayer thickness (diagnostic only, NOT coupled into PDE)."""
    Re_safe = np.maximum(Re_rot, 1.0)
    return np.clip(R * C_delta * Re_safe**(-7.0/8.0), 0.0, 0.05*R)


@st.cache_data
def solve_ebp(omega_max, om_d_rads,
              h0, eta0, rho, E, R, n_visc,
              t_end, Nr,
              t_d, t_up, t_st, t_dn,
              Re_turb, C_mix, beta_exp,
              turbulent, eta_gel_ratio):
    """
    Transition-corrected EBP solver.

    Governing PDE:
      ∂h/∂t = −(ρω²/3η_eff)·(1/r)·∂(r²h³)/∂r − E   [m/s]

    η_eff: Meyerhofer + eddy-viscosity correction
    Re_rot = ρωr²/η_lam  (no h)

    Returns 9 outputs: r_mm, t, h[nm], omega, Re, eta_eff[mPa·s], regime, Lambda, eta_lam[mPa·s]
    """
    Re_lam_fixed = 1.0e5
    r    = np.linspace(0, R, Nr)
    dr   = r[1] - r[0]
    r_p  = r + dr / 2
    dtmax = t_end / 300.0

    h    = np.full(Nr, h0, dtype=float)
    t    = 0.0
    snap_dt   = t_end / 150
    next_snap = 0.0

    t_s=[]; h_s=[]; ow_s=[]; Re_s=[]; eta_s=[]; reg_s=[]
    # Meyerhofer Lambda series (scalar per snapshot)
    lambda_s=[]; eta_lam_s=[]

    while t < t_end - 1e-12:
        ow = omega_profile(t, omega_max, om_d_rads, t_d, t_up, t_st, t_dn)

        h_avg   = max(float(np.mean(h)), 1e-18)
        eta_lam = max(eta0 * (h0 / h_avg)**n_visc, eta0)

        eta_eff, Re_rot, regime = compute_flow_regime_and_eta_eff(
            r=r, eta_lam=eta_lam, ow=ow, rho_f=rho,
            Re_lam=Re_lam_fixed, Re_turb=Re_turb,
            C_mix=C_mix, beta_exp=beta_exp, turbulent=turbulent)

        eta_face = 2.0*eta_eff[:-1]*eta_eff[1:] / np.maximum(eta_eff[:-1]+eta_eff[1:], 1e-30)
        coeff_f  = rho * ow**2 / (3.0 * np.maximum(eta_face, 1e-30))
        coeff_n  = rho * ow**2 / (3.0 * np.maximum(eta_eff,  1e-30))

        h_max  = max(float(np.max(h)), 1e-18)
        D_max  = max(float(np.max(coeff_f)) * h_max**3 * R, 1e-30)
        dt     = min(0.45*dr**2/(2.0*D_max), dtmax, t_end - t)

        h3f  = ((h[:-1]+h[1:])/2)**3
        Q    = coeff_f * r_p[:-1]**2 * h3f

        div          = np.empty(Nr)
        div[1:Nr-1]  = (Q[1:]-Q[:-1]) / (r[1:Nr-1]*dr)
        div[0]       = 2.0 * coeff_n[0] * h[0]**3
        div[Nr-1]    = div[Nr-2]

        h    = np.maximum(h - dt*(div+E), 0.0)
        h[-1] = max(0.0, 2.0*h[-2]-h[-3])
        t   += dt

        if t >= next_snap or t >= t_end-1e-12:
            t_s.append(t); h_s.append(h.copy()); ow_s.append(ow)
            Re_s.append(Re_rot.copy()); eta_s.append(eta_eff.copy())
            reg_s.append(regime.copy())
            # Meyerhofer Lambda: spin vs evaporation thinning rate
            eta_eff_mean = float(np.mean(eta_eff))
            if E > 0 and eta_eff_mean > 0 and ow > 1e-3:
                spin_rate = 2*rho*ow**2*h_avg**3/(3*eta_eff_mean)
                lam_val   = spin_rate / E
            else:
                lam_val = 1e6 if E <= 0 else 0.0
            lambda_s.append(lam_val)
            eta_lam_s.append(eta_lam)
            next_snap += snap_dt

    return (r*1e3,
            np.array(t_s),
            np.array(h_s)*1e9,
            np.array(ow_s),
            np.array(Re_s),
            np.array(eta_s)*1e3,
            np.array(reg_s, dtype=int),
            np.array(lambda_s),
            np.array(eta_lam_s)*1e3)   # mPa·s


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
om_d_rads = om_d_rpm * 2*np.pi/60

with st.spinner("Solving EBP PDE..."):
    (r_mm, t_s, h_s, ow_s, Re_s, eta_s, reg_s,
     lambda_s, eta_lam_s) = solve_ebp(
        omega_max=omega_max, om_d_rads=om_d_rads,
        h0=h0, eta0=eta0, rho=rho, E=E, R=R, n_visc=n_visc,
        t_end=t_end, Nr=Nr,
        t_d=t_d, t_up=t_up, t_st=t_st, t_dn=t_dn,
        Re_turb=Re_turb, C_mix=C_mix, beta_exp=beta_exp,
        turbulent=turb_on, eta_gel_ratio=eta_gel_ratio)

# Laminar baseline (for comparison)
with st.spinner("Solving laminar baseline..."):
    (r_mm, t_l, h_l, ow_l, Re_l, eta_l, reg_l,
     lambda_l, eta_lam_l) = solve_ebp(
        omega_max=omega_max, om_d_rads=om_d_rads,
        h0=h0, eta0=eta0, rho=rho, E=E, R=R, n_visc=n_visc,
        t_end=t_end, Nr=Nr,
        t_d=t_d, t_up=t_up, t_st=t_st, t_dn=t_dn,
        Re_turb=Re_turb, C_mix=0.0, beta_exp=beta_exp,
        turbulent=False, eta_gel_ratio=eta_gel_ratio)

# ─── derived metrics ──────────────────────────────────────────────────────────
INNER_FRAC = 0.85
n_inner = max(2, int(INNER_FRAC*Nr))

def inner_count(n_points, inner_frac=INNER_FRAC):
    """Number of radial nodes used for inner-wafer WIWNU/mean statistics."""
    return min(n_points, max(2, int(inner_frac*n_points)))

def wiwnu(hf, inner_frac=INNER_FRAC):
    """Within-inner-wafer non-uniformity, reported as ±%.
    Uses the inner 85% of whatever radial grid is passed, so it is safe for grid sweeps.
    """
    n_in = inner_count(len(hf), inner_frac)
    hi = hf[:n_in]
    hi = hi[hi > 0]
    if hi.size < 2 or hi.mean() < 1e-3:
        return 0.0
    return (hi.max() - hi.min()) / hi.mean() * 100 / 2

def edge_bead(hf, inner_frac=INNER_FRAC):
    """Simple edge-bead indicator [same unit as hf].
    Defined as max(edge region) - mean(inner region). Safe for any grid size.
    """
    n_in = inner_count(len(hf), inner_frac)
    inner = hf[:n_in]
    inner = inner[inner > 0]
    if inner.size == 0:
        return 0.0
    edge = hf[n_in:]
    hb = edge.max() if edge.size > 0 else inner.mean()
    return max(0.0, hb - inner.mean())

# Gelation: eta_lam/eta0 >= eta_gel_ratio
def t_gel_fn(t_arr, eta_lam_arr_mPas, eta0_mPas, ratio):
    """Primary: viscosity-ratio criterion (Meyerhofer).
    Returns the first actual solver snapshot time where η_lam/η0 reaches the threshold.
    If the threshold is never reached, returns the final simulated time.
    """
    t_arr = np.asarray(t_arr, dtype=float)
    eta_lam_arr_mPas = np.asarray(eta_lam_arr_mPas, dtype=float)
    if t_arr.size == 0 or eta_lam_arr_mPas.size == 0:
        return 0.0
    ratios = eta_lam_arr_mPas / max(float(eta0_mPas), 1e-30)
    hit = np.where(ratios >= ratio)[0]
    return float(t_arr[hit[0]]) if hit.size > 0 else float(t_arr[-1])

h_final  = h_s[-1]
unif     = wiwnu(h_final)
ebead    = edge_bead(h_final)
h_mean   = h_final[:n_inner][h_final[:n_inner]>0].mean() if h_final[:n_inner].max()>0 else 0
t_gel    = t_gel_fn(t_s, eta_lam_s, eta0_mPas, eta_gel_ratio)

reg_final   = reg_s[-1]
lam_mask    = reg_final < 1
trans_mask  = reg_final == 1
turb_mask   = reg_final >= 2
frac_lam    = float(lam_mask.mean())*100
frac_trans  = float(trans_mask.mean())*100
frac_turb   = float(turb_mask.mean())*100
r_trans_onset = float(r_mm[np.argmax(trans_mask)]) if trans_mask.any() else float(r_mm[-1])
r_turb_onset  = float(r_mm[np.argmax(turb_mask)])  if turb_mask.any()  else float(r_mm[-1])

# Meyerhofer regime durations
lambda_arr = lambda_s
t_arr      = t_s
t_rot_dominated = float(np.sum(lambda_arr > 3.0) / len(lambda_arr) * t_end)
t_evap_dominated= float(np.sum(lambda_arr < 1/3) / len(lambda_arr) * t_end)
# transition time where Lambda crosses 1 (descending)
crossing = np.where(np.diff(np.sign(lambda_arr - 1.0)) < 0)[0]
t_meyer_cross = float(t_arr[crossing[0]]) if len(crossing)>0 else t_end
final_regime_str = ("Rotation-dominated" if lambda_arr[-1]>3
                    else "Evaporation-dominated" if lambda_arr[-1]<1/3
                    else "Transition")

# Dimensionless numbers (at final state, key radii)
r_positions = {"center":0, "mid":Nr//2, "rim":Nr-2}
eta0_SI = eta0
Re_rim   = rho*omega_max*(R)**2/max(eta0,1e-30)
Ca_rim   = eta0*omega_max*R / max(gamma_SI,1e-30)
epsilon  = h_mean*1e-9 / R if h_mean > 0 else h0/R
Ev_final = 1.0/max(lambda_arr[-1],1e-30) if lambda_arr[-1]>0 else 0.0

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Process Results",
    "🌊 Meyerhofer Analysis",
    "🔄 BL / Transition",
    "🔬 Validation",
    "🎨 Design Explorer",
    "🌀 Animation",
])

# ─────────────────────────────────────────────────────────────────────────────
# helper
def mcard(col, label, val, unit, kind="b"):
    col.markdown(
        f'<div class="mc mc-{kind}"><div class="ml ml-{kind}">{label}</div>'
        f'<div class="mv mv-{kind}">{val}</div>'
        f'<div class="mu">{unit}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PROCESS RESULTS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Process Output Summary")

    cols = st.columns(7)
    kind_u = "g" if unif<=2.0 else "r"
    mcard(cols[0], "Mean h̄ final",  f"{h_mean:.1f}",  "nm")
    mcard(cols[1], "WIWNU",         f"±{unif:.2f}",   "%",     kind_u)
    mcard(cols[2], "Edge bead",     f"{ebead:.1f}",   "nm",    "y")
    mcard(cols[3], "t_gel (η-ratio)",f"{t_gel:.1f}",  "s",     "p")
    mcard(cols[4], "Lam zone",      f"{frac_lam:.0f}","% wafer","b")
    mcard(cols[5], "Trans zone",    f"{frac_trans:.0f}","% wafer","y")
    mcard(cols[6], "Turb zone",     f"{frac_turb:.0f}","% wafer","p")

    spec_ok = unif <= 2.0
    sbake_ok= t_gel > t_end + 5.0
    c1,c2 = st.columns(2)
    c1.markdown(
        f'<div class="{"ok" if spec_ok else "fail"}">'
        f'WIWNU ±2% spec: {"PASS ✓" if spec_ok else "FAIL ✗"}  ({unif:.3f}%)</div>',
        unsafe_allow_html=True)
    c2.markdown(
        f'<div class="{"ok" if sbake_ok else "warn"}">'
        f'Soft-bake window: {"OK ✓" if sbake_ok else "Short ⚠"}  '
        f'(t_gel = {t_gel:.1f} s, need > {t_end+5:.0f} s)</div>',
        unsafe_allow_html=True)

    # ── Assumptions / BCs expandable ─────────────────────────────────────────
    with st.expander("📋 Model Assumptions and Boundary Conditions"):
        st.markdown("""
<div class="eq">Governing PDE (EBP, Emslie-Bonner-Peck 1958):
  ∂h/∂t = −(ρω²/3η_eff) · (1/r) · ∂(r²h³)/∂r − E   [m/s]

Meyerhofer viscosity (concentration-coupled):
  η_lam(t) = η₀ · (h₀/h̄(t))ⁿ   [Pa·s]

Corrected Re (no h in denominator):
  Re_rot(r,t) = ρ·ω(t)·r² / η_lam(t)   [–]

Eddy-viscosity correction (engineering level only):
  η_eff = η_lam · [1 + C_mix · S(Re) · (Re/Re_turb−1)^β]
  S = smoothstep blending in [Re_lam=1×10⁵, Re_turb]</div>

**Physical assumptions:**
- Newtonian photoresist baseline (laminar EBP is exact for Newtonian fluid)
- Incompressible fluid, constant density
- Axisymmetric radial spreading (∂/∂θ = 0)
- Thin-film / lubrication approximation: h ≪ R, Re·(h/R) ≪ 1
- No-slip BC at wafer surface: v_r(z=0) = 0
- Stress-free free surface: ∂v_r/∂z|_(z=h) = 0 (no air friction)
- Uniform solvent evaporation rate E [m/s]
- Edge bead shown as radial h(r) indicator, not full free-surface instability
- Turbulence/transition: engineering eddy-viscosity correction only (NOT DNS/LES/RANS)
""", unsafe_allow_html=True)

    # ── Dimensionless numbers panel ───────────────────────────────────────────
    with st.expander("📐 Dimensionless Numbers"):
        re_vals = {pos: float(Re_s[-1][idx]) for pos, idx in r_positions.items()}
        st.markdown(f"""
| Number | Definition | Value | Significance |
|--------|-----------|-------|--------------|
| Re_rot (center) | ρωr²/η_lam | {re_vals['center']:.2e} | Rotating-disk inertia vs viscous |
| Re_rot (mid)    | ρωr²/η_lam | {re_vals['mid']:.2e}    | Transition starts here if > 1×10⁵ |
| Re_rot (rim)    | ρωr²/η_lam | {re_vals['rim']:.2e}    | Max Re — turbulence onset |
| Re_rot (wafer, max) | ρωR²/η₀ | {Re_rim:.2e} | Initial rim Re |
| Ca (rim)        | η₀·ωR/γ    | {Ca_rim:.3e} | Ca≫1 → surface tension negligible |
| ε = h̄/R         | h̄/R        | {epsilon:.2e} | ε≪1 → thin-film approx valid |
| Λ (final)       | spin/evap rate | {lambda_arr[-1]:.3f} | Meyerhofer regime indicator |
| Ev = 1/Λ (final)| evap/spin rate | {Ev_final:.3f} | >1 → evaporation dominated |

**Thin-film validity check:**
- ε = h̄/R = {epsilon:.2e} {"✓ ≪ 1, lubrication approximation valid" if epsilon < 0.01 else "⚠ check lubrication assumption"}
- Ca = {Ca_rim:.1e} {"✓ ≫ 1, surface tension negligible in bulk" if Ca_rim > 10 else "⚠ surface tension may matter"}
""")

    st.markdown("---")
    # ── Radial profile plot ───────────────────────────────────────────────────
    st.markdown('<div class="sec">Final Radial Thickness h(r) — Regime Zones</div>',
                unsafe_allow_html=True)
    RCOL = {0:'#dbeafe', 1:'#fef3c7', 2:'#ede9fe'}
    RLAB = {0:'Laminar', 1:'Transitional', 2:'Turbulent/Corrected'}
    fig1, ax1 = plt.subplots(figsize=(12,5)); ax1.set_facecolor('#f8fafc')
    prev_r, prev_reg = r_mm[0], reg_final[0]
    for i in range(1, Nr):
        if reg_final[i] != prev_reg or i == Nr-1:
            ax1.axvspan(prev_r, r_mm[i], color=RCOL[prev_reg], alpha=0.35)
            prev_r = r_mm[i]; prev_reg = reg_final[i]
    ax1.plot(r_mm, h_l[-1],  color='#0369a1', lw=2, ls='--', alpha=0.7,
             label='Laminar EBP (baseline)')
    ax1.plot(r_mm, h_final, color='#0f172a', lw=2.5,
             label='Corrected model (eddy-viscosity)')
    ax1.axhline(h_mean, color='#0ea5e9', ls=':', lw=1.5,
                label=f'h̄ = {h_mean:.1f} nm')
    if h_mean > 0:
        ax1.axhspan(h_mean*0.98, h_mean*1.02, color='#0ea5e9', alpha=0.09,
                    label='±2% WIWNU band')
    if trans_mask.any():
        ax1.axvline(r_trans_onset, color='#f59e0b', lw=1.5, ls='-.',
                    label=f'Transition onset r={r_trans_onset:.1f}mm')
    if turb_mask.any():
        ax1.axvline(r_turb_onset, color='#8b5cf6', lw=1.5, ls=':',
                    label=f'Turbulent onset r={r_turb_onset:.1f}mm')
    ax1.axvline(r_mm[n_inner], color='#94a3b8', lw=1, ls='--', alpha=0.6,
                label='EBR boundary (inner 85%)')
    from matplotlib.patches import Patch
    hs, ls2 = ax1.get_legend_handles_labels()
    hs += [Patch(color=RCOL[k], alpha=0.5, label=RLAB[k]) for k in [0,1,2]]
    ax1.legend(handles=hs, fontsize=8, ncol=3)
    ax1.set_xlabel('r  [mm]', fontsize=11); ax1.set_ylabel('h  [nm]', fontsize=11)
    ax1.set_title(f'h(r, t={t_end:.1f}s) | ω={omega_rpm}rpm | η₀={eta0_mPas}mPa·s | {wafer}')
    ax1.grid(alpha=0.2); st.pyplot(fig1); plt.close()

    # ── h̄(t) + spin recipe ───────────────────────────────────────────────────
    st.markdown('<div class="sec">h̄(t) and Spin Recipe</div>', unsafe_allow_html=True)
    fig2, axes = plt.subplots(1,2,figsize=(13,4.5))
    for ax in axes: ax.set_facecolor('#f8fafc')
    axes[0].plot(t_l, h_l.mean(axis=1), color='#0369a1', lw=1.8, ls='--',
                 label='Laminar baseline')
    axes[0].plot(t_s, h_s.mean(axis=1), color='#0f172a', lw=2.2,
                 label='Corrected model')
    axes[0].axhline(h_mean, color='#0ea5e9', ls=':', lw=1.2, alpha=0.6)
    axes[0].set_xlabel('t  [s]'); axes[0].set_ylabel('h̄  [nm]')
    axes[0].set_title('Mean film thickness h̄(t)'); axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)
    t_pl = np.linspace(0, t_end, 800)
    rpm_pl = np.array([omega_profile(tt, omega_max, om_d_rads, t_d, t_up, t_st, t_dn)
                       for tt in t_pl])*60/(2*np.pi)
    axes[1].fill_between(t_pl, rpm_pl, alpha=0.15, color='#0ea5e9')
    axes[1].plot(t_pl, rpm_pl, color='#0369a1', lw=2.2)
    axes[1].axvspan(0,       t_d,           alpha=.10, color='#94a3b8', label='Dispense')
    axes[1].axvspan(t_d,     t_d+t_up,      alpha=.08, color='#f59e0b', label='Ramp-up')
    axes[1].axvspan(t_d+t_up,t_d+t_up+t_st,alpha=.06, color='#22c55e', label='Steady')
    axes[1].axvspan(t_d+t_up+t_st,t_end,   alpha=.08, color='#ef4444', label='Ramp-down')
    axes[1].set_xlabel('t  [s]'); axes[1].set_ylabel('ω  [rpm]')
    axes[1].set_title('4-phase spin recipe ω(t)'); axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2); st.pyplot(fig2); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MEYERHOFER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Meyerhofer Regime Analysis")
    st.markdown("""
<div class="eq">Meyerhofer thinning ratio Λ(t):

  Spin thinning rate  = 2ρω(t)²h̄(t)³ / (3η_eff_mean)   [m/s]
  Evaporation rate    = E                                 [m/s]
  Λ(t) = spin_rate / E

  Λ > 3   → Rotation-dominated  (EBP classical regime)
  1/3 ≤ Λ ≤ 3 → Transition regime
  Λ < 1/3 → Evaporation-dominated  (Meyerhofer late stage)</div>
""", unsafe_allow_html=True)

    # Metric cards
    c1,c2,c3,c4 = st.columns(4)
    mcard(c1, "Rot-dominated",   f"{t_rot_dominated:.1f}", "s",  "b")
    mcard(c2, "Evap-dominated",  f"{t_evap_dominated:.1f}","s",  "y")
    mcard(c3, "Meyerhofer τ",    f"{t_meyer_cross:.1f}",   "s",  "g")
    mcard(c4, "Final regime",    final_regime_str,          "",   "p")

    # Lambda(t) plot
    fig_m, ax_m = plt.subplots(figsize=(12, 4.5)); ax_m.set_facecolor('#f8fafc')
    lambda_plot = np.maximum(lambda_s, 1e-4)
    ax_m.semilogy(t_s, lambda_plot, color='#0f172a', lw=2.5, label='Λ(t) = spin/evap rate')
    ax_m.axhline(3.0, color='#0ea5e9', lw=1.5, ls='--', label='Λ=3 (rotation limit)')
    ax_m.axhline(1.0, color='#64748b', lw=1.2, ls=':', label='Λ=1')
    ax_m.axhline(1/3, color='#f59e0b', lw=1.5, ls='--', label='Λ=1/3 (evaporation limit)')
    ax_m.axhspan(3.0,  max(lambda_plot.max(),3.1), color='#dbeafe', alpha=0.3,
                label='Rotation-dominated')
    ax_m.axhspan(1/3, 3.0, color='#f0fdf4', alpha=0.3, label='Transition')
    ax_m.axhspan(1e-4, 1/3, color='#fef3c7', alpha=0.3, label='Evaporation-dominated')
    # vertical markers
    ax_m.axvline(t_d+t_up, color='#475569', lw=1.2, ls=':', label='End ramp-up')
    ax_m.axvline(t_d+t_up+t_st, color='#f59e0b', lw=1.2, ls=':', label='Start ramp-down')
    if len(crossing) > 0:
        ax_m.axvline(t_arr[crossing[0]], color='#ef4444', lw=2, ls='-.',
                     label=f'Meyerhofer τ = {t_meyer_cross:.1f} s')
    ax_m.set_xlabel('t  [s]', fontsize=11); ax_m.set_ylabel('Λ = spin/evap rate  [–]', fontsize=11)
    ax_m.set_title('Meyerhofer regime indicator Λ(t) — log scale')
    ax_m.legend(fontsize=8, ncol=3); ax_m.grid(alpha=0.2, which='both')
    st.pyplot(fig_m); plt.close()

    # Viscosity growth
    st.markdown('<div class="sec">Meyerhofer Viscosity Growth η(t)/η₀</div>',
                unsafe_allow_html=True)
    fig_eta, ax_eta = plt.subplots(figsize=(12, 3.5)); ax_eta.set_facecolor('#f8fafc')
    eta_ratio = eta_lam_s / eta0_mPas
    ax_eta.plot(t_s, eta_ratio, color='#8b5cf6', lw=2.5, label='η_lam(t)/η₀')
    ax_eta.axhline(eta_gel_ratio, color='#ef4444', lw=2, ls='--',
                   label=f'Gelation criterion η/η₀ = {eta_gel_ratio}')
    ax_eta.axvline(t_gel, color='#dc2626', lw=1.5, ls='-.',
                   label=f't_gel = {t_gel:.1f} s')
    ax_eta.set_xlabel('t  [s]'); ax_eta.set_ylabel('η/η₀  [–]')
    ax_eta.set_title('Viscosity ratio η_lam(t)/η₀ — Meyerhofer model')
    ax_eta.legend(fontsize=9); ax_eta.grid(alpha=0.2)
    st.pyplot(fig_eta); plt.close()

    st.markdown("""
<div class="info">
<b>Physical interpretation:</b> In the early stage (Λ≫1), centrifugal thinning dominates 
and h̄ ~ t⁻¹/² (EBP classical result). As the film thins, η increases (Meyerhofer model), 
reducing the centrifugal rate. Eventually Λ<1/3 and the film thins primarily by evaporation. 
Gelation (η/η₀ ≥ gel_ratio) marks the end of the soft-bake window.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BOUNDARY LAYER / TRANSITION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Boundary-Layer / Transition Analysis")
    st.markdown("""
<div class="info"><b>Note:</b> The eddy-viscosity correction is an engineering-level approximation
based on the rotating-disk Reynolds-number criterion. It is NOT a full turbulence simulation
(no DNS/LES/RANS). The laminar EBP PDE is the physical baseline; this correction modifies η_eff
near the wafer rim where Re_rot exceeds Re_turb.</div>
""", unsafe_allow_html=True)

    if not turb_on:
        st.info("Enable eddy-viscosity correction in the sidebar to see transition analysis.")
    else:
        # Re_rot space-time map
        fig3, axes3 = plt.subplots(1,2, figsize=(14,5))
        for ax in axes3: ax.set_facecolor('#f8fafc')

        Ns = len(t_s)
        idx_s = np.unique(np.linspace(0, Ns-1, 8, dtype=int))
        cm3 = plt.cm.plasma
        for k, ii in enumerate(idx_s):
            axes3[0].plot(r_mm, Re_s[ii], color=cm3(k/(len(idx_s)-1)),
                          lw=1.6, label=f't={t_s[ii]:.1f}s')
        axes3[0].axhline(1e5,    color='#0ea5e9', lw=2, ls='--', label='Re_lam=1×10⁵')
        axes3[0].axhline(Re_turb,color='#8b5cf6', lw=2, ls='-.',
                          label=f'Re_turb={Re_turb:.0e}')
        axes3[0].axhspan(1e5, Re_turb, color='#fef3c7', alpha=0.25, label='Transition')
        axes3[0].set_yscale('log')
        axes3[0].set_xlabel('r  [mm]'); axes3[0].set_ylabel('Re_rot  [–]')
        axes3[0].set_title('Re_rot(r,t) — log scale')
        axes3[0].legend(fontsize=7, ncol=2); axes3[0].grid(alpha=0.2, which='both')

        Re_log = np.log10(np.maximum(Re_s, 1.0))
        im3 = axes3[1].imshow(Re_log, origin='lower', aspect='auto',
            extent=[r_mm[0],r_mm[-1],t_s[0],t_s[-1]], cmap='plasma')
        axes3[1].contour(r_mm, t_s, Re_s, levels=[1e5],
                          colors='cyan', linewidths=1.8)
        axes3[1].contour(r_mm, t_s, Re_s, levels=[Re_turb],
                          colors='white', linewidths=1.8, linestyles='--')
        axes3[1].set_xlabel('r  [mm]'); axes3[1].set_ylabel('t  [s]')
        axes3[1].set_title('Re_rot(r,t) map — cyan=Re_lam, white=Re_turb')
        plt.colorbar(im3, ax=axes3[1], label='log₁₀(Re_rot)')
        st.pyplot(fig3); plt.close()

        # Regime fraction vs time
        fig4, axes4 = plt.subplots(1,2, figsize=(13,4))
        for ax in axes4: ax.set_facecolor('#f8fafc')
        fl = (reg_s==0).mean(axis=1)*100
        ft = (reg_s==1).mean(axis=1)*100
        fu = (reg_s>=2).mean(axis=1)*100
        axes4[0].fill_between(t_s, 0, fl, color='#dbeafe', alpha=0.85, label='Laminar')
        axes4[0].fill_between(t_s, fl, fl+ft, color='#fef3c7', alpha=0.85, label='Transitional')
        axes4[0].fill_between(t_s, fl+ft, 100, color='#ede9fe', alpha=0.85,
                               label='Corrected (turbulent)')
        axes4[0].set_xlabel('t  [s]'); axes4[0].set_ylabel('%')
        axes4[0].set_title('Regime fraction vs time')
        axes4[0].legend(fontsize=9); axes4[0].grid(alpha=0.2); axes4[0].set_ylim(0,100)

        # eta_eff profiles
        for k, ii in enumerate(idx_s):
            axes4[1].plot(r_mm, eta_s[ii], color=cm3(k/(len(idx_s)-1)),
                          lw=1.5, label=f't={t_s[ii]:.1f}s')
        axes4[1].axhline(eta0_mPas, color='#64748b', lw=1.5, ls=':',
                          label=f'η₀={eta0_mPas}mPa·s')
        axes4[1].set_xlabel('r  [mm]'); axes4[1].set_ylabel('η_eff  [mPa·s]')
        axes4[1].set_title('η_eff(r,t) — eddy-viscosity correction')
        axes4[1].legend(fontsize=7, ncol=2); axes4[1].grid(alpha=0.2)
        st.pyplot(fig4); plt.close()

        # Sublayer indicator
        st.markdown('<div class="sec">Laminar Sublayer Thickness δ_sub(r) — Diagnostic Only</div>',
                    unsafe_allow_html=True)
        st.markdown("""
<div class="warn">This sublayer indicator is a DIAGNOSTIC visualization only.
It is NOT coupled into the PDE or h(t) computation. It estimates δ_sub = R·62·Re⁻⁷/⁸
from rotating-disk boundary-layer theory (Cebeci & Bradshaw 1977).</div>
""", unsafe_allow_html=True)
        fig5, ax5 = plt.subplots(figsize=(11, 3.5)); ax5.set_facecolor('#f8fafc')
        for k, ii in enumerate(idx_s):
            delta_sub = compute_sublayer_indicator(Re_s[ii], R)*1e9
            ax5.plot(r_mm, delta_sub, color=cm3(k/(len(idx_s)-1)),
                     lw=1.5, label=f't={t_s[ii]:.1f}s')
        ax5.plot(r_mm, h_s[-1], color='#0f172a', lw=2.5, ls='--', label='h(r, t_end) [nm]')
        ax5.set_xlabel('r  [mm]'); ax5.set_ylabel('δ_sub  [nm]')
        ax5.set_title('Laminar sublayer indicator δ_sub(r) vs film thickness h(r) — diagnostic only')
        ax5.legend(fontsize=8, ncol=3); ax5.grid(alpha=0.2)
        st.pyplot(fig5); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Solver Validation")
    st.markdown("""
<div class="eq">EBP analytical solution (Emslie, Bonner & Peck 1958):
  E=0, n_visc=0 (η=const), no ramp, laminar only

  h(t) = h₀ / √(1 + 4ρω²h₀²t / 3η₀)

  Derived by: EBP PDE → E=0, η=const → spatially uniform h →
  separable ODE → integrate from h(0)=h₀.
  Comparison at r=0 (wafer center), t ≥ 3 s.</div>
""", unsafe_allow_html=True)

    with st.spinner("Running analytical validation..."):
        (_, t_v, h_v, _, _, _, _, _, _) = solve_ebp(
            omega_max=omega_max, om_d_rads=0.0,
            h0=h0, eta0=eta0, rho=rho, E=0.0, R=R, n_visc=0.0,
            t_end=30, Nr=Nr,
            t_d=0, t_up=0, t_st=30, t_dn=0,
            Re_turb=int(1e10), C_mix=0.0, beta_exp=beta_exp,
            turbulent=False, eta_gel_ratio=eta_gel_ratio)

    t_an = np.linspace(0, 30, 500)
    tau  = 4*rho*omega_max**2*h0**2/(3*eta0)
    h_an = h0*1e9 / np.sqrt(1+tau*t_an)
    hc   = h_v[:,0]

    fig_v, axv = plt.subplots(1,2, figsize=(13,4.5))
    for ax in axv: ax.set_facecolor('#f8fafc')
    axv[0].plot(t_an, h_an, 'k--', lw=2.5, label='Analytical h₀/√(1+τt)')
    axv[0].plot(t_v, hc, color='#0369a1', lw=2, label='Numerical h(r=0)')
    axv[0].set_xlabel('t  [s]'); axv[0].set_ylabel('h(r=0)  [nm]')
    axv[0].set_title('A. Laminar EBP validation (E=0, η=const)')
    axv[0].legend(fontsize=9); axv[0].grid(alpha=0.2)
    hc_i = np.interp(t_an, t_v, hc)
    mask = t_an >= 3.0
    errs = np.abs(hc_i[mask]-h_an[mask])/h_an[mask]*100
    axv[1].plot(t_an[mask], errs, color='#ef4444', lw=1.8)
    axv[1].axhline(2.0, color='#22c55e', lw=1.5, ls='--', label='2% spec')
    axv[1].set_xlabel('t  [s]'); axv[1].set_ylabel('Rel. error  [%]')
    axv[1].set_title(f'Error at r=0 (t≥3s) — mean={errs.mean():.3f}%')
    axv[1].legend(fontsize=9); axv[1].grid(alpha=0.2)
    st.pyplot(fig_v); plt.close()

    vA_ok = errs.mean() < 2.0
    st.markdown(
        f'<div class="{"ok" if vA_ok else "fail"}">A. Analytical validation: '
        f'mean err={errs.mean():.3f}% | max={errs.max():.3f}% | '
        f'{"PASS ✓" if vA_ok else "FAIL ✗"}</div>', unsafe_allow_html=True)

    # Limit check B: C_mix=0 → laminar
    st.markdown("**B. C_mix = 0 limit: eddy-viscosity correction must vanish**")
    with st.spinner("Running C_mix=0 check..."):
        (_, t_c0, h_c0, _, _, _, _, _, _) = solve_ebp(
            omega_max=omega_max, om_d_rads=0.0,
            h0=h0, eta0=eta0, rho=rho, E=0.0, R=R, n_visc=0.0,
            t_end=30, Nr=Nr,
            t_d=0, t_up=0, t_st=30, t_dn=0,
            Re_turb=Re_turb, C_mix=0.0, beta_exp=beta_exp,
            turbulent=True, eta_gel_ratio=eta_gel_ratio)
    diff_c0 = float(np.max(np.abs(h_c0[:,0]-np.interp(t_c0, t_v, hc))))
    vB_ok = diff_c0 < 0.1
    st.markdown(
        f'<div class="{"ok" if vB_ok else "fail"}">B. C_mix=0 limit: '
        f'max|h_turb(C_mix=0) − h_lam| = {diff_c0:.4f} nm | '
        f'{"PASS ✓" if vB_ok else "FAIL ✗"}</div>', unsafe_allow_html=True)

    # Grid convergence check C
    st.markdown("**C. Grid convergence — Nr = 40 / 80 / 120**")
    with st.spinner("Running grid convergence..."):
        gc_rows = []
        h_ref = None
        for nr_test in [40, 80, 120]:
            (_, _, h_gc, _, _, _, _, _, _) = solve_ebp(
                omega_max=omega_max, om_d_rads=om_d_rads,
                h0=h0, eta0=eta0, rho=rho, E=E, R=R, n_visc=n_visc,
                t_end=t_end, Nr=nr_test,
                t_d=t_d, t_up=t_up, t_st=t_st, t_dn=t_dn,
                Re_turb=Re_turb, C_mix=C_mix, beta_exp=beta_exp,
                turbulent=turb_on, eta_gel_ratio=eta_gel_ratio)
            hf_gc = h_gc[-1]
            n_in_gc = inner_count(len(hf_gc))
            hi_gc = hf_gc[:n_in_gc]; hi_gc = hi_gc[hi_gc>0]
            hm_gc = hi_gc.mean() if hi_gc.size>0 else 0
            wu_gc = wiwnu(hf_gc)
            eb_gc = edge_bead(hf_gc)
            if h_ref is None: h_ref = hm_gc
            rel = abs(hm_gc-h_ref)/max(h_ref,1e-3)*100
            gc_rows.append({"Nr": nr_test, "h̄_final (nm)": f"{hm_gc:.2f}",
                             "WIWNU ±%": f"{wu_gc:.3f}",
                             "Edge bead (nm)": f"{eb_gc:.1f}",
                             "Δh̄ vs Nr=40 (%)": f"{rel:.3f}"})
    st.dataframe(pd.DataFrame(gc_rows), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DESIGN EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Design Explorer — ±2% WIWNU Challenge Mode")
    st.markdown(
        "Find (ω, η₀) combinations satisfying ±2% WIWNU spec. "
        "Green region = PASS. Red contour = ±2% boundary.")

    c_a, c_b = st.columns(2)
    with c_a:
        sw_r = st.slider("ω range (rpm)", 500, 8000, (1000, 6000), 500)
        n_om = st.select_slider("ω points", [3,5,7,10], value=5)
    with c_b:
        se_r = st.slider("η₀ range (mPa·s)", 1, 200, (5,100), 5)
        n_et = st.select_slider("η₀ points", [3,5,7,10], value=5)

    if st.button("Run Design Sweep", type="primary"):
        om_arr  = np.linspace(sw_r[0], sw_r[1], n_om)*2*np.pi/60
        et_arr  = np.linspace(se_r[0], se_r[1], n_et)*1e-3
        om_rpm_ = np.linspace(sw_r[0], sw_r[1], n_om)
        et_mPas_= np.linspace(se_r[0], se_r[1], n_et)

        RES=np.zeros((n_om,n_et)); HF=np.zeros((n_om,n_et))
        EB=np.zeros((n_om,n_et)); TG=np.zeros((n_om,n_et))
        prog = st.progress(0)

        for i,om in enumerate(om_arr):
            for j,et in enumerate(et_arr):
                (_, t_sw, h_sw, _, _, _, _, _, eta_lam_sw) = solve_ebp(
                    omega_max=om, om_d_rads=min(om_d_rads, om*0.1),
                    h0=h0, eta0=float(et), rho=rho, E=E, R=R, n_visc=n_visc,
                    t_end=t_end, Nr=Nr,
                    t_d=t_d, t_up=t_up, t_st=t_st, t_dn=t_dn,
                    Re_turb=Re_turb, C_mix=C_mix, beta_exp=beta_exp,
                    turbulent=turb_on, eta_gel_ratio=eta_gel_ratio)
                hf_ = h_sw[-1]
                n_in_ = inner_count(len(hf_))
                hi_ = hf_[:n_in_]; hi_ = hi_[hi_>0]
                RES[i,j] = (hi_.max()-hi_.min())/hi_.mean()*100/2 if hi_.size>1 else 0
                HF[i,j]  = hi_.mean() if hi_.size>0 else 0
                EB[i,j]  = edge_bead(hf_)
                # Corrected: use the full sweep time history, not only the final time.
                TG[i,j]  = t_gel_fn(t_sw, eta_lam_sw, float(et)*1e3, eta_gel_ratio)
                prog.progress((i*n_et+j+1)/(n_om*n_et))

        # Heatmap
        fig_d, axes_d = plt.subplots(1,3, figsize=(16,5.5))
        for ax in axes_d: ax.set_facecolor('#f8fafc')
        for ax, data, title, cm_, lab in [
            (axes_d[0], RES, 'WIWNU ±%',         'RdYlGn_r', 'WIWNU ±%'),
            (axes_d[1], HF,  'Mean h̄ final [nm]','Blues',    'h̄  [nm]'),
            (axes_d[2], EB,  'Edge bead [nm]',    'Oranges',  'nm'),
        ]:
            im = ax.imshow(data.T, origin='lower', aspect='auto', cmap=cm_,
                           extent=[om_rpm_[0],om_rpm_[-1],et_mPas_[0],et_mPas_[-1]])
            if title.startswith('WIWNU') and RES.max()>2.0:
                CS = ax.contour(om_rpm_, et_mPas_, data.T, levels=[2.0],
                                colors=['#dc2626'], linewidths=2.5)
                ax.clabel(CS, fmt='±2%', fontsize=10)
                ax.contourf(om_rpm_, et_mPas_, data.T, levels=[0,2.0],
                            colors=['#bbf7d0'], alpha=0.4)
            plt.colorbar(im, ax=ax, label=lab)
            ax.set_xlabel('ω  [rpm]'); ax.set_ylabel('η₀  [mPa·s]')
            ax.set_title(title)
        st.pyplot(fig_d); plt.close()

        # Best combination
        pass_mask = RES <= 2.0
        if pass_mask.any():
            best_i, best_j = np.unravel_index(np.argmin(RES), RES.shape)
            st.markdown(f"""
<div class="ok">
<b>Recommended fab process condition:</b><br>
ω = <b>{om_rpm_[best_i]:.0f} rpm</b> &nbsp;|&nbsp;
η₀ = <b>{et_mPas_[best_j]:.0f} mPa·s</b><br>
Expected h̄_final = <b>{HF[best_i,best_j]:.1f} nm</b> &nbsp;|&nbsp;
WIWNU = <b>±{RES[best_i,best_j]:.3f}%</b> &nbsp;|&nbsp;
Edge bead = <b>{EB[best_i,best_j]:.1f} nm</b> &nbsp;|&nbsp;
t_gel = <b>{TG[best_i,best_j]:.1f} s</b>
</div>
""", unsafe_allow_html=True)

        # Full results table
        rows = []
        for i in range(n_om):
            for j in range(n_et):
                rows.append({
                    "ω (rpm)": int(om_rpm_[i]),
                    "η₀ (mPa·s)": f"{et_mPas_[j]:.0f}",
                    "WIWNU ±%": f"{RES[i,j]:.3f}",
                    "h̄ (nm)": f"{HF[i,j]:.1f}",
                    "Edge bead (nm)": f"{EB[i,j]:.1f}",
                    "t_gel (s)": f"{TG[i,j]:.1f}",
                    "Spec ✓": "✓" if RES[i,j]<=2.0 else "✗",
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

        # Process recommendations
        st.markdown("---")
        st.markdown("### 🏭 Process-Design Recommendations")
        n_pass = pass_mask.sum()
        best_unif_i, best_unif_j = np.unravel_index(np.argmin(RES), RES.shape)
        recs = []
        if n_pass == 0:
            recs.append("**No combinations met ±2% WIWNU.** "
                        "Try increasing ω (better centrifugal leveling) "
                        "or reducing η₀ (more uniform spreading before gelation).")
        if RES.min() > 1.5:
            recs.append("Best WIWNU is marginal. Consider reducing evaporation rate E "
                        "(humidity control) to extend the rotation-dominated phase.")
        if EB.max() > 50:
            recs.append("Edge bead exceeds 50 nm in some conditions. "
                        "Optimize ramp-down profile or apply EBR (Edge Bead Removal).")
        if frac_turb > 20:
            recs.append(f"Turbulent correction zone covers {frac_turb:.0f}% of wafer. "
                        "Rim uniformity is sensitive to high ω. "
                        "Consider ω < 5000 rpm for uniformity-critical applications.")
        if t_gel < t_end:
            recs.append(f"Gelation (η/η₀={eta_gel_ratio}) occurs at t={t_gel:.1f}s, "
                        f"before spin end ({t_end:.1f}s). "
                        "Reduce E or shorten steady spin time.")
        if not recs:
            recs.append("Process window is satisfactory. "
                        "Fine-tune ω and η₀ within the green region for target thickness.")
        for r_ in recs:
            st.markdown(f'<div class="info">• {r_}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — ANIMATION
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Live Spin Coating Animation")
    st.markdown(
        "Blue = laminar zone | Orange = transitional | Purple = turbulent/corrected. "
        "Arrow opacity ∝ centrifugal flux. Phase badge shows Meyerhofer regime.")

    ANIM = r"""
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:sans-serif;background:transparent}
.wrap{display:flex;flex-direction:column;align-items:center;gap:11px;padding:.8rem 0}
.stage{position:relative;width:360px;height:360px}
canvas{position:absolute;top:0;left:0}
.ctrl{display:flex;flex-direction:column;gap:6px;width:360px}
.row{display:flex;align-items:center;gap:7px}
.row label{font-size:11px;color:#475569;width:155px;flex-shrink:0}
.row input{flex:1}
.row span{font-size:11px;font-weight:700;color:#0c4a6e;min-width:62px;text-align:right}
.btns{display:flex;gap:8px;justify-content:center}
button{font-size:11px;padding:5px 14px;border-radius:7px;
  border:1px solid #bae6fd;background:transparent;color:#0c4a6e;cursor:pointer}
button:hover{background:#e0f2fe}
.mets{display:grid;grid-template-columns:repeat(5,1fr);gap:5px;width:360px}
.mc{background:#f0f9ff;border:1px solid #bae6fd;border-radius:7px;padding:5px 7px}
.ml{font-size:9px;color:#0369a1;font-weight:700;text-transform:uppercase}
.mv{font-size:12px;font-weight:800;color:#0c4a6e}
.badge{font-size:9px;padding:2px 6px;border-radius:99px;display:inline-block;font-weight:700}
.lam{background:#dbeafe;color:#1e40af}
.tr{background:#fef9c3;color:#78350f}
.tur{background:#ede9fe;color:#4c1d95}
.rot{background:#dbeafe;color:#1e40af}
.evp{background:#fef9c3;color:#78350f}
.gel{background:#dcfce7;color:#166534}
</style>
<div class="wrap">
<div class="stage">
  <canvas id="bg" width="360" height="360"></canvas>
  <canvas id="fl" width="360" height="360"></canvas>
  <canvas id="tp" width="360" height="360"></canvas>
</div>
<div class="mets">
  <div class="mc"><div class="ml">h center</div><div class="mv" id="mh">-</div></div>
  <div class="mc"><div class="ml">ω</div><div class="mv" id="mw">-</div></div>
  <div class="mc"><div class="ml">Λ (Meyer.)</div><div class="mv" id="mlam">-</div></div>
  <div class="mc"><div class="ml">Phase</div><div class="mv"><span id="mph" class="badge rot">-</span></div></div>
  <div class="mc"><div class="ml">Turb %</div><div class="mv" id="mtr">-</div></div>
</div>
<div class="ctrl">
  <div class="row"><label>ω (rpm)</label>
    <input type="range" id="srpm" min="500" max="8000" step="100" value="3000">
    <span id="s0">3000</span></div>
  <div class="row"><label>η₀ (mPa·s)</label>
    <input type="range" id="seta" min="1" max="100" step="1" value="20">
    <span id="s1">20</span></div>
  <div class="row"><label>E (nm/s)</label>
    <input type="range" id="sevp" min="0" max="50" step="1" value="10">
    <span id="s2">10</span></div>
  <div class="row"><label>Re_turb (×10⁴)</label>
    <input type="range" id="src" min="5" max="100" step="5" value="30">
    <span id="s3">30×10⁴</span></div>
  <div class="row"><label>C_mix</label>
    <input type="range" id="sct" min="0" max="0.15" step="0.001" value="0.015">
    <span id="s4">0.015</span></div>
</div>
<div class="btns">
  <button id="btnPl">Pause</button>
  <button id="btnRs">Reset</button>
</div>
</div>
<script>
const CX=180,CY=180,RMAX=162,NR=72,R_phys=0.150,rho=1060;
const Re_lam_fixed=1e5;
const [bgX,flX,tpX]=['bg','fl','tp'].map(id=>document.getElementById(id).getContext('2d'));
let rpm=3000,eta=20,evp=10,ReTurb=3e5,Cmix=0.015;
let angle=0,tSim=0,h0nm=500;
const hProf=new Float64Array(NR).fill(1.0);
let running=true,lastTs=null;

function getP(){
  rpm=+document.getElementById('srpm').value;
  eta=+document.getElementById('seta').value;
  evp=+document.getElementById('sevp').value;
  ReTurb=+document.getElementById('src').value*1e4;
  Cmix=+document.getElementById('sct').value;
}
function reset(){tSim=0;angle=0;hProf.fill(1.0);lastTs=null;}

function smoothstep(x){x=Math.max(0,Math.min(1,x));return x*x*(3-2*x);}

function getRegime(i){
  const om=rpm*2*Math.PI/60;
  const etaP=eta*1e-3;
  const rP=(i+.5)/NR*R_phys;
  // Re_rot: NO h in denominator
  const Re=rho*om*rP*rP/etaP;
  if(Re<Re_lam_fixed)return 0;
  if(Re<ReTurb)return 1;
  return 2;
}

function drawWafer(){
  bgX.clearRect(0,0,360,360);
  bgX.beginPath();bgX.arc(CX,CY,RMAX,0,Math.PI*2);
  bgX.fillStyle='#b0b4bf';bgX.fill();
  for(let i=1;i<=5;i++){
    bgX.beginPath();bgX.arc(CX,CY,RMAX*i/5,0,Math.PI*2);
    bgX.strokeStyle='rgba(70,75,90,.12)';bgX.lineWidth=.5;bgX.stroke();
  }
  bgX.beginPath();bgX.arc(CX,CY,RMAX,0,Math.PI*2);
  bgX.strokeStyle='rgba(50,55,70,.45)';bgX.lineWidth=1.6;bgX.stroke();
  bgX.beginPath();bgX.arc(CX,CY-RMAX,5.5,0,Math.PI*2);
  bgX.fillStyle='rgba(30,35,50,.65)';bgX.fill();
}

function drawFilm(ang){
  flX.clearRect(0,0,360,360);
  flX.save();flX.translate(CX,CY);flX.rotate(ang);
  const maxH=Math.max(...hProf,1e-6);
  const COLS=[
    n=>`rgba(${20+Math.round(20*(1-n))},${100+Math.round(80*(1-n))},${200+Math.round(40*n)},${.22+.55*n})`,
    n=>`rgba(${200+Math.round(30*n)},${160+Math.round(30*(1-n))},10,${.22+.55*n})`,
    n=>`rgba(${120+Math.round(60*n)},20,${180+Math.round(50*n)},${.22+.55*n})`,
  ];
  for(let i=NR-1;i>=0;i--){
    const r0=RMAX*i/NR,r1=RMAX*(i+1)/NR;
    const norm=Math.min(hProf[i]/maxH,1.0);
    flX.beginPath();flX.arc(0,0,r1,0,Math.PI*2);flX.arc(0,0,r0,0,Math.PI*2,true);
    flX.fillStyle=COLS[getRegime(i)](norm);flX.fill();
  }
  for(let l=0;l<18;l++){
    const a=l*Math.PI*2/18;
    flX.beginPath();flX.moveTo(Math.cos(a)*6,Math.sin(a)*6);
    flX.lineTo(Math.cos(a)*RMAX*.9,Math.sin(a)*RMAX*.9);
    flX.strokeStyle='rgba(255,255,255,.04)';flX.lineWidth=.8;flX.stroke();
  }
  flX.restore();
}

function drawTop(ang){
  tpX.clearRect(0,0,360,360);
  const ACOLS=['rgba(56,189,248,','rgba(251,191,36,','rgba(167,139,250,'];
  const nArr=12;
  tpX.save();tpX.translate(CX,CY);
  for(let i=0;i<nArr;i++){
    const a=ang*2+i*Math.PI*2/nArr;
    const rFrac=0.6;
    const reg=getRegime(Math.round(rFrac*NR));
    const avgH=hProf.reduce((x,y)=>x+y)/NR;
    const alpha=Math.min(.85,.2+avgH*.7);
    const col=ACOLS[reg]+alpha+')';
    const r0=RMAX*.09,r1=RMAX*.83;
    const dx=Math.cos(a),dy=Math.sin(a);
    tpX.beginPath();tpX.moveTo(dx*r0,dy*r0);tpX.lineTo(dx*r1,dy*r1);
    tpX.strokeStyle=col;tpX.lineWidth=1.5;tpX.stroke();
    const hx=dx*r1,hy=dy*r1,px=-dy*5,py=dx*5;
    tpX.beginPath();tpX.moveTo(hx,hy);
    tpX.lineTo(hx-dx*10+px,hy-dy*10+py);tpX.lineTo(hx-dx*10-px,hy-dy*10-py);
    tpX.closePath();tpX.fillStyle=col;tpX.fill();
  }
  tpX.restore();
  if(evp>0){
    const nd=Math.min(Math.round(evp*1.8),50);
    for(let d=0;d<nd;d++){
      const r=RMAX*(.08+(d%NR)/NR*.85);
      const a=(d*137.5+tSim*50)*(Math.PI/180);
      const yO=((tSim*evp*4+d*19)%38);
      tpX.beginPath();
      tpX.arc(CX+Math.cos(a)*r,CY+Math.sin(a)*r-yO,1.7,0,Math.PI*2);
      tpX.fillStyle=`rgba(125,211,252,${(.6*(1-yO/38)).toFixed(2)})`;tpX.fill();
    }
  }
  tpX.save();tpX.translate(CX,CY);
  tpX.beginPath();tpX.arc(0,0,19,0,Math.PI*2);
  tpX.fillStyle='rgba(180,188,205,.93)';tpX.fill();
  tpX.strokeStyle='rgba(110,118,138,.65)';tpX.lineWidth=1.2;tpX.stroke();
  tpX.beginPath();tpX.moveTo(Math.cos(ang*3)*15,Math.sin(ang*3)*15);tpX.lineTo(0,0);
  tpX.strokeStyle='rgba(50,58,78,.5)';tpX.lineWidth=2.5;tpX.stroke();
  tpX.restore();
}

function updatePhysics(dtSec){
  const om=rpm*2*Math.PI/60;
  const etaP=eta*1e-3;
  const h0m=h0nm*1e-9;
  const dr=1.0/NR;
  const hMax=Math.max(...hProf,1e-8);
  const coeff0=rho*om*om/(3*etaP);
  const Deff=coeff0*(hMax*h0m)**3*R_phys;
  const dtCFL=Deff>0?0.45*dr*dr/(2*Deff):0.01;
  const dt=Math.min(dtCFL,20/300,dtSec);
  const nSub=Math.max(1,Math.ceil(dtSec/dt));
  const dts=dtSec/nSub;
  for(let s=0;s<nSub;s++){
    const havg=hProf.reduce((a,b)=>a+b)/NR;
    const etaFac=Math.max(1.0,havg>0?Math.pow(1/havg,2):1);
    const etaLam=etaFac*etaP;
    const hn=new Float64Array(NR);
    for(let i=1;i<NR-1;i++){
      const ri=(i+.5)/NR,rp=(i+1)/NR,rm=i/NR;
      const rFace=rp*R_phys;
      // Re_rot: NO h in denominator
      const ReF=rho*om*rFace*rFace/etaLam;
      let etaE=etaFac;
      if(ReF>Re_lam_fixed){
        const x=(ReF-Re_lam_fixed)/Math.max(ReTurb-Re_lam_fixed,1);
        const S=smoothstep(x);
        const exc=Math.max(ReF/ReTurb-1,0);
        etaE=etaFac*(1+Cmix*S*Math.pow(exc,0.7));
      }
      const c=rho*om*om/(3*etaE*etaP);
      const h3p=((hProf[i]+hProf[i+1])/2)**3;
      const h3m=((hProf[i-1]+hProf[i])/2)**3;
      hn[i]=Math.max(0,hProf[i]-dts*(c*(rp*rp*h3p-rm*rm*h3m)/(ri*dr)+evp*1e-9/h0m));
    }
    hn[0]=Math.max(0,hProf[0]-dts*(coeff0*(0.5/NR)**2*hProf[0]**3/(dr**2)+evp*1e-9/h0m));
    hn[NR-1]=Math.max(0,2*hn[NR-2]-hn[NR-3]);
    hProf.set(hn);
  }
  angle+=om*dtSec;tSim+=dtSec;
}

function getPhase(){
  const om=rpm*2*Math.PI/60;
  const etaP=eta*1e-3;
  const havg=hProf.reduce((a,b)=>a+b)/NR*h0nm*1e-9;
  if(havg*1e9<h0nm*.02)return['Gelated','gel'];
  const spinR=2*rho*om*om*havg**3/(3*etaP);
  const evpR=evp*1e-9;
  const lam=evpR>0?spinR/evpR:1e6;
  if(lam>3)return['Rot-dominated','rot'];
  if(lam<1/3)return['Evap-dominated','evp'];
  return['Transition','tr'];
}

function loop(ts){
  if(!running){requestAnimationFrame(loop);return;}
  if(lastTs===null)lastTs=ts;
  const dt=Math.min((ts-lastTs)/1000,.04);lastTs=ts;
  getP();updatePhysics(dt);
  drawWafer();drawFilm(angle);drawTop(angle);
  const hc=hProf[0]*h0nm;
  document.getElementById('mh').textContent=hc>=1000?(hc/1000).toFixed(2)+'μm':hc.toFixed(1)+'nm';
  document.getElementById('mw').textContent=rpm+' rpm';
  const [lbl,cls]=getPhase();
  const b=document.getElementById('mph');b.textContent=lbl;b.className='badge '+cls;
  const om2=rpm*2*Math.PI/60;const etaP2=eta*1e-3;
  let nTur=0;
  for(let i=0;i<NR;i++) if(getRegime(i)>=2) nTur++;
  document.getElementById('mtr').textContent=Math.round(nTur/NR*100)+'%';
  const havg2=hProf.reduce((a,b)=>a+b)/NR*h0nm*1e-9;
  const spinR2=2*rho*om2*om2*havg2**3/(3*etaP2);
  const lam2=evp>0?spinR2/(evp*1e-9):1e6;
  document.getElementById('mlam').textContent=lam2>1000?'>1k':lam2.toFixed(2);
  requestAnimationFrame(loop);
}
document.getElementById('btnPl').addEventListener('click',function(){
  running=!running;this.textContent=running?'Pause':'Play';if(running)lastTs=null;});
document.getElementById('btnRs').addEventListener('click',()=>reset());
[['srpm','s0',v=>Math.round(v)+''],
 ['seta','s1',v=>Math.round(v)+''],
 ['sevp','s2',v=>Math.round(v)+''],
 ['src', 's3',v=>(v/10).toFixed(0)+'×10⁴'],
 ['sct', 's4',v=>parseFloat(v).toFixed(3)]].forEach(([id,oid,fmt])=>{
  document.getElementById(id).addEventListener('input',function(){
    document.getElementById(oid).textContent=fmt(this.value);});
});
drawWafer();drawFilm(0);drawTop(0);
requestAnimationFrame(loop);
</script>
"""
    st.components.v1.html(ANIM, height=700, scrolling=False)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='font-size:.82rem;color:#475569;background:#f8fafc;border-radius:8px;padding:.8rem 1.2rem;'>
This simulator is based on the EBP thin-film equation (Emslie, Bonner & Peck 1958)
with Meyerhofer viscosity growth (Meyerhofer 1978).
The boundary-layer transition correction is an engineering-level eddy-viscosity approximation
and is <b>not</b> a full turbulence simulation (no DNS/LES/RANS).
<br><br>
AI tools were used for code structuring, debugging assistance, and explanation drafting.
Physical assumptions, governing equations, validation, numerical behavior, and interpretation
were reviewed by the author before submission.
</div>
""", unsafe_allow_html=True)
