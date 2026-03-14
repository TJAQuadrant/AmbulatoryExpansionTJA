"""
Ambulatory TJA Risk Calculator
================================
Usage:  streamlit run calculator.py
Requires in same directory:
    research_model_fitted.joblib
    research_isotonic_calibrator.joblib
    research_model_imputer_fills.json
    research_features.json
    winsorization_caps.json  (optional)
    style.css
    reftable.html
    .streamlit/config.toml
"""
import os, json
import numpy as np
import joblib
import streamlit as st

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
THRESHOLD = 0.0570
AUC       = 0.740
N_TRAIN   = 309_529

st.set_page_config(page_title="Ambulatory TJA Risk Calculator",
                   layout="wide", initial_sidebar_state="collapsed")

with open(os.path.join(MODEL_DIR, "style.css"), encoding="utf-8") as _f:
    st.markdown(f"<style>{_f.read()}</style>", unsafe_allow_html=True)

# ── Artifacts ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model      = joblib.load(os.path.join(MODEL_DIR, "research_model_fitted.joblib"))
    calibrator = joblib.load(os.path.join(MODEL_DIR, "research_isotonic_calibrator.joblib"))
    with open(os.path.join(MODEL_DIR, "research_model_imputer_fills.json")) as f:
        fills = json.load(f)
    with open(os.path.join(MODEL_DIR, "research_features.json")) as f:
        feat_order = json.load(f)
    wins_path = os.path.join(MODEL_DIR, "winsorization_caps.json")
    wins = json.load(open(wins_path)) if os.path.exists(wins_path) else {
        "prior_hospitalizations_1yr": 3, "prior_ed_visits_1yr": 2
    }
    return model, calibrator, fills, feat_order, wins

try:
    model, calibrator, imputer_fills, feature_order, wins_caps = load_artifacts()
    ARTIFACTS_OK = True
except Exception as e:
    ARTIFACTS_OK = False
    ARTIFACT_ERR = str(e)

# ── Keys / Labels ──────────────────────────────────────────────────────────────
COMORBIDITY_KEYS = [
    "has_hypertension","has_heart_disease","has_copd","has_diabetes",
    "has_anemia","has_sleep_apnea","has_liver_disease",
    "has_thyroid_disease","has_recent_ablation",
    "has_thrombocytopenia","has_leukopenia",
]
COMORB_LABELS = {
    "has_hypertension":    "Hypertension",
    "has_heart_disease":   "Heart disease (CAD / HF / valvular)",
    "has_copd":            "COPD",
    "has_diabetes":        "Diabetes mellitus",
    "has_anemia":          "Anemia",
    "has_sleep_apnea":     "Obstructive sleep apnea",
    "has_liver_disease":   "Liver disease",
    "has_thyroid_disease": "Thyroid disease",
    "has_recent_ablation": "Recent cardiac ablation",
    "has_thrombocytopenia":"Thrombocytopenia",
    "has_leukopenia":      "Leukopenia",
}
EXCLUSION_KEYS = [
    "excl_insulin","excl_arrhythmia","excl_chest_pain","excl_anticoagulant",
    "excl_mi_stroke","excl_pacemaker","excl_aicd","excl_home_o2",
    "excl_organ_failure","excl_dialysis","excl_transplant","excl_substance",
    "excl_stent_single","excl_stent_multiple","excl_ablation","excl_cirrhosis",
]
EXCL_LABELS = {
    "excl_insulin":       "Insulin-dependent diabetes",
    "excl_arrhythmia":    "Arrhythmia requiring management",
    "excl_chest_pain":    "Recurrent chest pain / unstable angina",
    "excl_anticoagulant": "Anticoagulation therapy",
    "excl_mi_stroke":     "Recent MI or stroke (<6 months)",
    "excl_pacemaker":     "Pacemaker dependent",
    "excl_aicd":          "AICD / implantable defibrillator",
    "excl_home_o2":       "Home oxygen dependency",
    "excl_organ_failure": "Active organ failure",
    "excl_dialysis":      "Active dialysis",
    "excl_transplant":    "Solid organ transplant (liver/heart/lung)",
    "excl_substance":     "Active substance use disorder",
    "excl_stent_single":  "Single coronary stent <6 months",
    "excl_stent_multiple":"Multiple coronary stents <12 months",
    "excl_ablation":      "Cardiac ablation <3 months",
    "excl_cirrhosis":     "Cirrhosis",
}
EXCL_TO_MODEL = {
    "excl_insulin":       "cosi_exclude_insulin",
    "excl_arrhythmia":    "cosi_exclude_arrhythmia",
    "excl_chest_pain":    "cosi_exclude_recurrent_chest_pain",
    "excl_anticoagulant": "has_anticoagulant",
    "excl_mi_stroke":     "cosi_exclude_recent_mi_stroke",
    "excl_pacemaker":     "cosi_exclude_pacemaker",
    "excl_aicd":          "cosi_exclude_aicd",
    "excl_home_o2":       "cosi_exclude_home_oxygen",
    "excl_organ_failure": "cosi_exclude_organ_failure",
    "excl_dialysis":      "cosi_exclude_dialysis",
    "excl_transplant":    "cosi_exclude_organ_transplant_liver_heart_lung",
    "excl_substance":     "cosi_exclude_active_illegal_drug_use",
    "excl_stent_single":  "cosi_exclude_recent_stent_single",
    "excl_stent_multiple":"cosi_exclude_recent_stent_multiple",
    "excl_ablation":      "cosi_exclude_recent_ablation",
    "excl_cirrhosis":     "cosi_exclude_cirrhosis",
}

DEFAULT_INPUTS = dict(
    age=65, bmi=28.0, systolic_bp=125, diastolic_bp=78,
    sex="Female", procedure="TKA", asa_lbl="II",
    hgb_sel="Normal (>=11 g/dL)", cr_sel="<1.2 mg/dL",
    alb_sel=">=3.5 g/dL", inr_sel="<1.2",
    wbc_sel="Normal (4-12 x10^3)", ed_sel="None",
    hospitalizations=0,
)
for _k in COMORBIDITY_KEYS:
    DEFAULT_INPUTS[_k] = False
for _k in EXCLUSION_KEYS:
    DEFAULT_INPUTS[_k] = False

# ── Session state init ─────────────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "needs_recalc" not in st.session_state:
    st.session_state.needs_recalc = False

def mark_dirty():
    st.session_state.needs_recalc = True

def do_reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# ── Scoring ────────────────────────────────────────────────────────────────────
def score(inp):
    f = {}
    f["age_at_surgery"] = float(inp["age"])
    f["bmi"]            = float(inp["bmi"])
    f["asa_proxy"]      = float(inp["asa"])
    # surgery_year removed -- prospectively invalid feature (rank 2 SHAP artifact
    # of secular training trends; hardcoding any year biases all predictions)
    f["sex_encoded"]    = 1.0 if inp["sex"] == "M" else 0.0
    f["proc_THA"]       = 1.0 if inp["procedure"] == "THA" else 0.0
    f["proc_TKA"]       = 1.0 if inp["procedure"] == "TKA" else 0.0
    f["systolic_bp"]    = float(inp["systolic_bp"])
    f["diastolic_bp"]   = float(inp["diastolic_bp"])

    for k in COMORBIDITY_KEYS:
        f[k] = float(inp.get(k, False))
    f["has_kidney_transplant"] = 0.0
    for ui_k, m_k in EXCL_TO_MODEL.items():
        f[m_k] = float(inp.get(ui_k, False))

    hgb_v = {"normal":13.5,"mild":10.5,"moderate":8.0,"missing":np.nan}
    cr_v  = {"normal":0.9,"elevated":1.35,"high":1.75,"very":2.5}
    alb_v = {"normal":4.0,"borderline":3.25,"low":2.7}
    inr_v = {"normal":1.05,"elevated":1.6,"high":2.5}
    wbc_v = {"normal":7.0,"abnormal":13.5,"missing":np.nan}

    f["hemoglobin"] = hgb_v.get(inp["hgb_cat"], np.nan) if inp.get("has_anemia")        else np.nan
    f["creatinine"] = cr_v[inp["cr_cat"]]
    f["albumin"]    = alb_v.get(inp["alb_cat"], np.nan) if inp.get("has_liver_disease")  else np.nan
    f["inr"]        = inr_v.get(inp["inr_cat"], np.nan) if inp.get("excl_anticoagulant") else np.nan
    f["wbc"]        = wbc_v.get(inp["wbc_cat"], np.nan) if inp.get("has_leukopenia")     else np.nan
    f["platelets"]  = np.nan
    f["glucose"]    = np.nan
    f["hba1c"]      = np.nan

    f["hemoglobin_missing"]        = float(inp.get("has_anemia", False) and np.isnan(f["hemoglobin"]))
    f["wbc_missing"]               = float(inp["wbc_cat"] == "missing" or np.isnan(f.get("wbc", np.nan)))
    f["creatinine_missing"]        = 0.0
    # platelets and glucose: not collected in calculator but treat as not-missing
    # (imputer fills median value; flagging as missing would bias healthy patients upward)
    f["platelets_missing"]         = 0.0
    f["glucose_missing"]           = 0.0
    f["hba1c_indicated_missing"]   = float(inp.get("has_diabetes", False))
    f["inr_indicated_missing"]     = float(inp.get("excl_anticoagulant", False) and np.isnan(f["inr"]))
    f["albumin_indicated_missing"] = float(inp.get("has_liver_disease", False) and np.isnan(f["albumin"]))

    hosp = min(int(inp.get("hospitalizations", 0)), wins_caps.get("prior_hospitalizations_1yr", 3))
    ed   = min({"none":0,"1":1,"2plus":2}[inp["ed_cat"]], wins_caps.get("prior_ed_visits_1yr", 2))
    f["prior_hospitalizations_1yr"] = float(hosp)
    f["prior_ed_visits_1yr"]        = float(ed)
    f["any_prior_hosp_1yr"]         = float(hosp > 0)
    f["any_prior_ed_1yr"]           = float(ed > 0)

    f["cosi_bmi_eligible"]     = float(inp["bmi"] <= 40)
    f["cosi_bmi_40_44"]        = float(40 < inp["bmi"] <= 44)
    f["cosi_asa_eligible"]     = float(inp["asa"] <= 2)
    f["cosi_bp_eligible"]      = float(f["systolic_bp"] <= 160 and f["diastolic_bp"] <= 90)
    f["cosi_age_45plus"]       = float(inp["age"] >= 45)
    f["cosi_glucose_eligible"] = 1.0
    f["cosi_creatinine_ok"]    = float(cr_v[inp["cr_cat"]] <= 1.2)
    f["cosi_hba1c_ok"]         = 1.0
    f["cosi_platelets_ok"]     = 1.0
    f["cosi_wbc_ok"]           = float(inp["wbc_cat"] == "normal")
    f["cosi_inr_ok"]           = float(not inp.get("excl_anticoagulant", False))

    pos = ["cosi_bmi_eligible","cosi_asa_eligible","cosi_bp_eligible",
           "cosi_glucose_eligible","cosi_creatinine_ok","cosi_hba1c_ok",
           "cosi_platelets_ok","cosi_wbc_ok","cosi_inr_ok"]
    excl_model_keys = [v for k,v in EXCL_TO_MODEL.items() if k != "excl_anticoagulant"]
    f["n_cosi_positive_violations"] = float(sum(1 - f[c] for c in pos))
    f["n_cosi_absolute_exclusions"] = float(sum(f.get(v,0) for v in excl_model_keys))
    f["n_cosi_total_violations"]    = f["n_cosi_positive_violations"] + f["n_cosi_absolute_exclusions"]
    f["cosi_eligible"]              = float(f["n_cosi_total_violations"] == 0)
    f["cosi_inelig_x_prior_hosp"]   = (1 - f["cosi_eligible"]) * f["any_prior_hosp_1yr"]
    f["cosi_inelig_x_prior_ed"]     = (1 - f["cosi_eligible"]) * f["any_prior_ed_1yr"]
    f["comorbidity_count"]          = float(sum(f.get(k,0) for k in COMORBIDITY_KEYS))

    row = []
    for feat in feature_order:
        val = f.get(feat, np.nan)
        if isinstance(val, float) and np.isnan(val):
            val = float(imputer_fills.get(feat, 0.0))
        row.append(float(val))

    X        = np.array(row, dtype=np.float32).reshape(1, -1)
    raw_prob = float(model.predict_proba(X)[0][1])
    cal_prob = float(calibrator.predict([raw_prob])[0])
    return raw_prob, cal_prob, f


def make_gauge(prob, threshold=THRESHOLD):
    """SVG semicircle gauge: 0% on left, 30% on right, threshold marked."""
    MAX_DISP = 0.30
    W, H = 340, 195
    cx, cy, r_out, r_in = 170, 175, 150, 95

    def arc_pt(pct, radius):
        angle = np.pi * (1 - min(pct, MAX_DISP) / MAX_DISP)
        return cx + radius * np.cos(angle), cy - radius * np.sin(angle)

    def arc_path(p0, p1, r, large=0):
        x0, y0 = arc_pt(p0, r)
        x1, y1 = arc_pt(p1, r)
        return f"M {x0:.1f} {y0:.1f} A {r} {r} 0 {large} 0 {x1:.1f} {y1:.1f}"

    # Color zones: green 0->threshold, amber threshold->threshold*1.5, red rest
    zones = [
        (0,           threshold,         "#1a4731", "#22c55e"),
        (threshold,   threshold * 1.75,  "#3b2700", "#f59e0b"),
        (threshold * 1.75, MAX_DISP,     "#3b0a0a", "#ef4444"),
    ]

    needle_pct = min(prob, MAX_DISP)
    angle = np.pi * (1 - needle_pct / MAX_DISP)
    nx = cx + 128 * np.cos(angle)
    ny = cy - 128 * np.sin(angle)

    # threshold marker
    tx0, ty0 = arc_pt(threshold, r_out + 8)
    tx1, ty1 = arc_pt(threshold, r_in - 8)

    ticks = [(0, "0%"), (0.05, "5%"), (0.10, "10%"), (0.15, "15%"), (0.20, "20%"), (0.25, "25%"), (0.30, "30%")]

    svg = f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:340px;display:block;margin:0 auto;">'
    svg += f'<rect width="{W}" height="{H}" fill="#0b0e14" rx="10"/>'

    # Background arc
    x0b, y0b = arc_pt(0, r_out);  x1b, y1b = arc_pt(MAX_DISP, r_out)
    x0i, y0i = arc_pt(0, r_in);   x1i, y1i = arc_pt(MAX_DISP, r_in)
    svg += f'<path d="M {x0b:.1f} {y0b:.1f} A {r_out} {r_out} 0 0 0 {x1b:.1f} {y1b:.1f} L {x1i:.1f} {y1i:.1f} A {r_in} {r_in} 0 0 1 {x0i:.1f} {y0i:.1f} Z" fill="#131822"/>'

    # Colored zones
    for z0, z1, fill_bg, fill_arc in zones:
        xo0,yo0 = arc_pt(z0, r_out); xo1,yo1 = arc_pt(z1, r_out)
        xi0,yi0 = arc_pt(z0, r_in);  xi1,yi1 = arc_pt(z1, r_in)
        large = 1 if (z1-z0)/MAX_DISP > 0.5 else 0
        svg += f'<path d="M {xo0:.1f} {yo0:.1f} A {r_out} {r_out} 0 {large} 0 {xo1:.1f} {yo1:.1f} L {xi1:.1f} {yi1:.1f} A {r_in} {r_in} 0 {large} 1 {xi0:.1f} {yi0:.1f} Z" fill="{fill_bg}" opacity="0.7"/>'
        svg += f'<path d="M {xo0:.1f} {yo0:.1f} A {r_out} {r_out} 0 {large} 0 {xo1:.1f} {yo1:.1f}" fill="none" stroke="{fill_arc}" stroke-width="4" opacity="0.9"/>'

    # Tick marks and labels
    for pct, lbl in ticks:
        tx, ty = arc_pt(pct, r_out + 16)
        ix, iy = arc_pt(pct, r_out + 2)
        ox, oy = arc_pt(pct, r_out - 4)
        svg += f'<line x1="{ix:.1f}" y1="{iy:.1f}" x2="{ox:.1f}" y2="{oy:.1f}" stroke="#334155" stroke-width="1.5"/>'
        svg += f'<text x="{tx:.1f}" y="{ty:.1f}" fill="#64748b" font-size="10" text-anchor="middle" dominant-baseline="middle" font-family="Courier New">{lbl}</text>'

    # Threshold dashed line
    svg += f'<line x1="{tx0:.1f}" y1="{ty0:.1f}" x2="{tx1:.1f}" y2="{ty1:.1f}" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4,3"/>'
    ttx, tty = arc_pt(threshold, r_out + 30)
    svg += f'<text x="{ttx:.1f}" y="{tty:.1f}" fill="#f59e0b" font-size="10" text-anchor="middle" font-family="Courier New">Threshold</text>'
    svg += f'<text x="{ttx:.1f}" y="{tty+13:.1f}" fill="#f59e0b" font-size="10" text-anchor="middle" font-family="Courier New">{100*threshold:.2f}%</text>'

    # Needle
    color = "#22c55e" if prob <= threshold else ("#f59e0b" if prob <= threshold * 1.75 else "#ef4444")
    svg += f'<line x1="{cx}" y1="{cy}" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="3.5" stroke-linecap="round"/>'
    svg += f'<circle cx="{cx}" cy="{cy}" r="8" fill="{color}"/>'
    svg += f'<circle cx="{cx}" cy="{cy}" r="4" fill="#0b0e14"/>'

    # Center readout
    pct_str  = f"{100*prob:.1f}%"
    delta    = abs(prob - threshold) * 100
    dir_str  = f"{'below' if prob <= threshold else 'above'} threshold"
    svg += f'<text x="{cx}" y="{cy-28}" fill="{color}" font-size="28" font-weight="700" text-anchor="middle" font-family="Courier New">{pct_str}</text>'
    svg += f'<text x="{cx}" y="{cy-8}" fill="#64748b" font-size="11" text-anchor="middle" font-family="Courier New">{delta:.1f}% {dir_str}</text>'

    svg += "</svg>"
    return svg


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="app-hdr">
    <div class="app-hdr-title">Ambulatory TJA Risk Calculator</div>
    <div class="app-hdr-sub">Research Model &middot; AUC {AUC} &middot; n={N_TRAIN:,} &middot; Isotonic Calibration &middot; {100*THRESHOLD:.2f}% Validated Threshold</div>
</div>
""", unsafe_allow_html=True)

if not ARTIFACTS_OK:
    st.error(f"Could not load model artifacts: {ARTIFACT_ERR}")
    st.stop()

col_form, col_result = st.columns([11, 6], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# FORM COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_form:
    st.markdown('<div style="padding:16px 6px 0 6px;">', unsafe_allow_html=True)
    st.markdown('<div class="form-card">', unsafe_allow_html=True)

    # ── Demographics & Procedure ──────────────────────────────────────────────
    st.markdown("""<div class="s-hdr" style="margin-top:0;">
        <div class="dot dot-blue"></div>
        <span class="s-title">Demographics &amp; Procedure</span>
    </div>""", unsafe_allow_html=True)

    dc1, dc2, dc3, dc4 = st.columns(4)
    with dc1: age          = st.number_input("Age (years)",         min_value=18,   max_value=100,  value=65,   step=1,   on_change=mark_dirty)
    with dc2: bmi          = st.number_input("BMI (kg/m²)",         min_value=15.0, max_value=80.0, value=28.0, step=0.1, on_change=mark_dirty)
    with dc3: systolic_bp  = st.number_input("Systolic BP (mmHg)",  min_value=60,   max_value=250,  value=125,  step=1,   on_change=mark_dirty)
    with dc4: diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=40,   max_value=150,  value=78,   step=1,   on_change=mark_dirty)

    pc1, pc2, pc3 = st.columns(3)
    with pc1: sex       = st.radio("Sex",       ["Female","Male"],     horizontal=True, on_change=mark_dirty)
    with pc2: procedure = st.radio("Procedure", ["TKA","THA"],         horizontal=True, on_change=mark_dirty)
    with pc3: asa_lbl   = st.radio("ASA Class", ["I","II","III","IV"], horizontal=True, index=1, on_change=mark_dirty)
    asa = ["I","II","III","IV"].index(asa_lbl) + 1

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Laboratory Values ─────────────────────────────────────────────────────
    st.markdown("""<div class="s-hdr">
        <div class="dot dot-amber"></div>
        <span class="s-title">Laboratory Values</span>
        <span class="s-note">&nbsp;— hemoglobin rank 3, creatinine rank 5, WBC rank 13</span>
    </div>""", unsafe_allow_html=True)

    has_anemia_now   = st.session_state.get("cb_has_anemia",        False)
    has_liver_now    = st.session_state.get("cb_has_liver_disease",  False)
    has_anticoag_now = st.session_state.get("cb_excl_anticoagulant", False)
    has_leuko_now    = st.session_state.get("cb_has_leukopenia",     False)

    lc1, lc2 = st.columns([1, 3])
    with lc1: st.markdown('<span class="lab-lbl">Hemoglobin</span>', unsafe_allow_html=True)
    with lc2:
        if has_anemia_now:
            hgb_sel = st.radio("hgb", ["Normal (>=11 g/dL)","Mild anemia (9-11)","Mod-severe (<9)","Not ordered"],
                               horizontal=True, label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Anemia" in Comorbidities to activate</span>', unsafe_allow_html=True)
            hgb_sel = "Normal (>=11 g/dL)"
    hgb_cat = {"Normal (>=11 g/dL)":"normal","Mild anemia (9-11)":"mild","Mod-severe (<9)":"moderate","Not ordered":"missing"}[hgb_sel]

    lc1, lc2 = st.columns([1, 3])
    with lc1: st.markdown('<span class="lab-lbl">Creatinine</span>', unsafe_allow_html=True)
    with lc2:
        cr_sel = st.radio("cr", ["<1.2 mg/dL","1.2-1.5","1.5-2.0",">2.0"], horizontal=True, label_visibility="collapsed", on_change=mark_dirty)
    cr_cat = {"<1.2 mg/dL":"normal","1.2-1.5":"elevated","1.5-2.0":"high",">2.0":"very"}[cr_sel]

    lc1, lc2 = st.columns([1, 3])
    with lc1: st.markdown('<span class="lab-lbl">Albumin</span>', unsafe_allow_html=True)
    with lc2:
        if has_liver_now:
            alb_sel = st.radio("alb", [">=3.5 g/dL","3.0-3.5","<3.0"], horizontal=True, label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Liver disease" in Comorbidities to activate</span>', unsafe_allow_html=True)
            alb_sel = ">=3.5 g/dL"
    alb_cat = {">=3.5 g/dL":"normal","3.0-3.5":"borderline","<3.0":"low"}[alb_sel]

    lc1, lc2 = st.columns([1, 3])
    with lc1: st.markdown('<span class="lab-lbl">INR</span>', unsafe_allow_html=True)
    with lc2:
        if has_anticoag_now:
            inr_sel = st.radio("inr", ["<1.2","1.2-2.0",">2.0"], horizontal=True, label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Anticoagulation therapy" in Eligibility Flags to activate</span>', unsafe_allow_html=True)
            inr_sel = "<1.2"
    inr_cat = {"<1.2":"normal","1.2-2.0":"elevated",">2.0":"high"}[inr_sel]

    lc1, lc2 = st.columns([1, 3])
    with lc1: st.markdown('<span class="lab-lbl">WBC (pre-op)</span>', unsafe_allow_html=True)
    with lc2:
        if has_leuko_now:
            wbc_sel = st.radio("wbc", ["Normal (4-12 x10^3)","Abnormal","Not ordered"], horizontal=True, label_visibility="collapsed", on_change=mark_dirty)
        else:
            st.markdown('<span class="lab-gate">Enable "Leukopenia" in Comorbidities to activate</span>', unsafe_allow_html=True)
            wbc_sel = "Normal (4-12 x10^3)"
    wbc_cat = {"Normal (4-12 x10^3)":"normal","Abnormal":"abnormal","Not ordered":"missing"}[wbc_sel]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Comorbidities ─────────────────────────────────────────────────────────
    cc_count  = sum(1 for k in COMORBIDITY_KEYS if st.session_state.get(f"cb_{k}", False))
    badge_cls = "badge-hi" if cc_count >= 3 else "badge"
    st.markdown(f"""<div class="s-hdr">
        <div class="dot dot-teal"></div>
        <span class="s-title">Comorbidities</span>
        <span class="s-note">&nbsp;-- total count is the #1 predictor (SHAP 0.263)</span>
        <span class="{badge_cls}">{cc_count} active</span>
    </div>""", unsafe_allow_html=True)

    cmorb_items = list(COMORB_LABELS.items())
    cm1, cm2 = st.columns(2)
    comorbidity_vals = {}
    with cm1:
        for key, label in cmorb_items[:6]:
            comorbidity_vals[key] = st.checkbox(label, value=False, key=f"cb_{key}", on_change=mark_dirty)
    with cm2:
        for key, label in cmorb_items[6:]:
            comorbidity_vals[key] = st.checkbox(label, value=False, key=f"cb_{key}", on_change=mark_dirty)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Healthcare Utilization ────────────────────────────────────────────────
    st.markdown("""<div class="s-hdr">
        <div class="dot dot-muted"></div>
        <span class="s-title">Healthcare Utilization (prior 12 months)</span>
        <span class="s-note">&nbsp;-- ED visits rank 6, hospitalizations rank 8</span>
    </div>""", unsafe_allow_html=True)

    uc1, uc2 = st.columns(2)
    with uc1: hospitalizations = st.number_input("Inpatient hospitalizations", min_value=0, max_value=20, value=0, step=1, on_change=mark_dirty)
    with uc2: ed_sel = st.radio("ED visits (prior 12 months)", ["None","1","2+"], horizontal=True, on_change=mark_dirty)
    ed_cat = {"None":"none","1":"1","2+":"2plus"}[ed_sel]

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── High-Risk Eligibility Flags ───────────────────────────────────────────
    st.markdown("""<div class="s-hdr">
        <div class="dot dot-rose"></div>
        <span class="s-title">High-Risk Eligibility Flags</span>
        <span class="s-note">&nbsp;-- absolute and conditional exclusion criteria</span>
    </div>""", unsafe_allow_html=True)

    excl_items = list(EXCL_LABELS.items())
    fe1, fe2 = st.columns(2)
    exclusion_vals = {}
    with fe1:
        for key, label in excl_items[:8]:
            exclusion_vals[key] = st.checkbox(label, value=False, key=f"cb_{key}", on_change=mark_dirty)
    with fe2:
        for key, label in excl_items[8:]:
            exclusion_vals[key] = st.checkbox(label, value=False, key=f"cb_{key}", on_change=mark_dirty)

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Action Buttons ────────────────────────────────────────────────────────
    btn1, btn2, btn3 = st.columns([3, 2, 2])
    with btn1:
        calc_clicked = st.button("Calculate Risk", type="primary", use_container_width=True)
    with btn2:
        if st.button("Reset All", use_container_width=True):
            do_reset()
    with btn3:
        if st.session_state.needs_recalc and st.session_state.result is not None:
            st.markdown('<div class="stale-warn">Inputs changed — recalculate</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close form-card

    # ── Reference Table ───────────────────────────────────────────────────────
    with st.expander("Reference: Published Eligibility Criteria & Model Findings  --  ranked by predictive importance"):
        with open(os.path.join(MODEL_DIR, "reftable.html"), encoding="utf-8") as _rf:
            st.markdown(_rf.read(), unsafe_allow_html=True)

    st.markdown(f"""
    <div class="footer-txt" style="margin-top:14px;">
        <span class="footer-em">Validated Research Model.</span>
        {N_TRAIN:,} patients across multiple institutions.
        Comorbidity burden and laboratory values account for 93.5% of predictive weight.
        Threshold derived from observed outcomes in eligibility-criteria-met patients.
        For research and clinical decision support only -- not a substitute for clinical judgment.
    </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# RESULT COLUMN
# ══════════════════════════════════════════════════════════════════════════════
with col_result:
    st.markdown('<div style="padding:16px 0 0 0;">', unsafe_allow_html=True)

    inp = dict(
        age=age, bmi=bmi,
        sex="M" if sex == "Male" else "F",
        procedure=procedure, asa=asa,
        systolic_bp=systolic_bp, diastolic_bp=diastolic_bp,
        hgb_cat=hgb_cat, cr_cat=cr_cat,
        alb_cat=alb_cat, inr_cat=inr_cat,
        wbc_cat=wbc_cat, ed_cat=ed_cat,
        hospitalizations=hospitalizations,
    )
    inp.update(comorbidity_vals)
    inp.update(exclusion_vals)

    if calc_clicked:
        try:
            raw_prob, cal_prob, feats = score(inp)
            st.session_state.result = dict(
                cal_prob=cal_prob, raw_prob=raw_prob, feats=feats
            )
            st.session_state.needs_recalc = False
        except Exception as e:
            st.error(f"Scoring error: {e}")

    res = st.session_state.result

    if res is None:
        st.markdown("""
        <div class="res-card" style="text-align:center;padding:40px 20px;">
            <div style="font-size:18px;color:#475569;margin-bottom:8px;">No result yet</div>
            <div style="font-size:14px;color:#334155;">Complete the form and press<br><strong style="color:#4f90f6;">Calculate Risk</strong></div>
        </div>""", unsafe_allow_html=True)
    else:
        cal_prob     = res["cal_prob"]
        feats        = res["feats"]
        below        = cal_prob <= THRESHOLD
        active_flags = [EXCL_LABELS[k] for k in EXCLUSION_KEYS if exclusion_vals.get(k, False)]

        # Gauge
        st.markdown(make_gauge(cal_prob), unsafe_allow_html=True)
        st.markdown('<div style="height:8px;"></div>', unsafe_allow_html=True)

        # Verdict
        if below and not active_flags:
            v_bg,v_bd,v_tc = "#0f2a1a","#166534","#4ade80"
            v_title = "Ambulatory Transfer Candidate"
            v_body  = f"Predicted risk is below the validated ambulatory benchmark. No high-risk flags identified."
        elif below and active_flags:
            v_bg,v_bd,v_tc = "#0d1f18","#065f46","#34d399"
            v_title = "Candidate -- Clinical Clearance Advised"
            v_body  = f"Risk is below the ambulatory benchmark despite high-risk flag(s). Specialist clearance recommended before proceeding."
        elif exclusion_vals.get("excl_mi_stroke", False):
            v_bg,v_bd,v_tc = "#1f0a0e","#991b1b","#f87171"
            v_title = "Inpatient Setting Recommended"
            v_body  = f"Recent MI or stroke carries high independent risk regardless of overall score. Inpatient monitoring warranted."
        else:
            v_bg,v_bd,v_tc = "#1f0a0e","#991b1b","#f87171"
            v_title = "Inpatient Setting Recommended"
            v_body  = f"Predicted risk exceeds the validated ambulatory benchmark of {100*THRESHOLD:.2f}%."

        st.markdown(f"""
        <div class="verdict" style="background:{v_bg};border:1px solid {v_bd};">
            <div class="v-title" style="color:{v_tc};">{v_title}</div>
            <div class="v-body" style="color:#94a3b8;">{v_body}</div>
        </div>""", unsafe_allow_html=True)

        if active_flags:
            chips = "".join(f'<span class="flag-chip">{fl}</span>' for fl in active_flags)
            st.markdown(f"""
            <div class="info-card">
                <div class="info-ttl">Active High-Risk Flags ({len(active_flags)})</div>
                {chips}
            </div>""", unsafe_allow_html=True)

        comorb_n = int(feats.get("comorbidity_count", 0))
        viols_n  = int(feats.get("n_cosi_total_violations", 0))
        st.markdown(f"""
        <div class="info-card">
            <div class="info-ttl">Key Model Inputs</div>
            <div style="font-size:15px;color:#94a3b8;line-height:2.1;">
                <span style="color:#e2e8f0;">Comorbidity count:</span> {comorb_n}<br>
                <span style="color:#e2e8f0;">Criterion violations:</span> {viols_n}<br>
                <span style="color:#e2e8f0;">Prior hospitalizations:</span> {int(feats.get("prior_hospitalizations_1yr",0))}<br>
                <span style="color:#e2e8f0;">Prior ED visits:</span> {int(feats.get("prior_ed_visits_1yr",0))}<br>
                <span style="color:#e2e8f0;">Procedure:</span> {"THA" if feats.get("proc_THA") else "TKA"}
            </div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.needs_recalc:
            st.markdown('<div class="stale-warn" style="margin-top:4px;">Inputs changed since last calculation</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
