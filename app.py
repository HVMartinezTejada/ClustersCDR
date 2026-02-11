import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.graph_objects as go

from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpBinary,
    LpStatus, value, PULP_CBC_CMD
)

# =========================
# Config
# =========================
st.set_page_config(
    page_title="CDR Oriente Antioqueño – Optimizador Territorial (V2)",
    layout="wide"
)

OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

DEFAULT_MUNICIPIOS = [
    "La Ceja", "La Unión", "El Retiro", "Rionegro", "El Carmen de Viboral",
    "Abejorral", "Guarne", "Marinilla", "El Santuario", "El Peñol"
]

# Preset RSU (t/año) – el de tu imagen
PRESET_RSU = {
    "La Ceja": 22300,
    "La Unión": 6534,
    "El Retiro": 7178,
    "Rionegro": 63344,
    "El Carmen de Viboral": 9000,
    "Abejorral": 6080,
    "Guarne": 16914,
    "Marinilla": 19790,
    "El Santuario": 10793,
    "El Peñol": 6311,
}

# =========================
# Helpers (geo + distances)
# =========================
@st.cache_data(show_spinner=False)
def geocode_municipio(nombre: str):
    """Geocodifica municipio usando Nominatim (OSM). Retorna (lat, lon)."""
    q = f"{nombre}, Antioquia, Colombia"
    params = {"q": q, "format": "json", "limit": 1}
    headers = {"User-Agent": "cdroptimizer-streamlit/2.0 (contact: hv)"}
    r = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    lat = float(data[0]["lat"])
    lon = float(data[0]["lon"])
    return lat, lon

@st.cache_data(show_spinner=False)
def build_distance_matrix_osrm(coords_dict: dict):
    """
    coords_dict: {municipio: (lat, lon)}
    Retorna dfD (km)
    """
    names = list(coords_dict.keys())
    coord_str = ";".join([f"{coords_dict[n][1]},{coords_dict[n][0]}" for n in names])  # lon,lat
    url = OSRM_TABLE_URL + coord_str
    params = {"annotations": "distance"}  # metros
    r = requests.get(url, params=params, timeout=90)
    r.raise_for_status()
    js = r.json()
    dist_m = np.array(js["distances"], dtype=float)
    dist_km = dist_m / 1000.0
    dfD = pd.DataFrame(dist_km, index=names, columns=names)
    return dfD

def candidate_ranking_single_plant(dfD: pd.DataFrame, w: pd.Series):
    """Ranking simple para k=1: sum_i w_i * d_i,j."""
    I = list(dfD.index)
    w = w.reindex(I).fillna(0.0).astype(float)
    scores = {}
    for j in dfD.columns:
        scores[j] = float((w * dfD[j]).sum())
    df_rank = pd.DataFrame({"costo_logistico_wdist": scores}).sort_values("costo_logistico_wdist")
    df_rank.index.name = "candidato_planta"
    return df_rank.reset_index()

# =========================
# V2 Solver (capacitado + equidad + “utilización” vía max unused capacity)
# =========================
def solve_facility_location_v2(
    dfD: pd.DataFrame,
    w: pd.Series,
    k: int,
    alpha: float,
    gamma: float,
    cap_t_anio: float,
):
    """
    Modelo V2:
      min  alpha * sum_i sum_j w_i * d_ij * x_ij
         + (1-alpha) * z
         + gamma * u
    s.a.
      - asignación: sum_j x_ij = 1
      - link: x_ij <= y_j
      - exactamente k plantas: sum_j y_j = k
      - capacidad por planta: sum_i w_i x_ij <= cap * y_j
      - equidad: z >= d_ij * x_ij
      - utilización proxy: u >= cap*y_j - load_j  (max capacidad no usada)
    """
    I = list(dfD.index)
    J = list(dfD.columns)

    w = w.reindex(I).fillna(0.0).astype(float)
    if (w < 0).any():
        raise ValueError("Hay RSU negativos. Corrige entradas.")
    if w.sum() <= 0:
        raise ValueError("La suma total de RSU debe ser > 0 para optimizar.")
    if cap_t_anio <= 0:
        raise ValueError("La capacidad anual por planta debe ser > 0.")
    if w.sum() > cap_t_anio * k:
        raise ValueError(
            f"Demanda total RSU ({w.sum():,.0f} t/año) excede capacidad total ({cap_t_anio*k:,.0f} t/año). "
            "Aumenta k o la capacidad por planta."
        )

    prob = LpProblem("CDR_Oriente_Antioquia_V2", LpMinimize)

    x = LpVariable.dicts("x", (I, J), lowBound=0, upBound=1, cat=LpBinary)
    y = LpVariable.dicts("y", J, lowBound=0, upBound=1, cat=LpBinary)
    z = LpVariable("z_maxdist", lowBound=0)
    u = LpVariable("u_max_unused_capacity", lowBound=0)

    # Load per facility (linear expression)
    load = {j: lpSum(w[i] * x[i][j] for i in I) for j in J}

    # Objective
    prob += (
        alpha * lpSum(w[i] * dfD.loc[i, j] * x[i][j] for i in I for j in J)
        + (1 - alpha) * z
        + gamma * u
    )

    # Assignment
    for i in I:
        prob += lpSum(x[i][j] for j in J) == 1, f"assign_{i}"

    # Open link
    for i in I:
        for j in J:
            prob += x[i][j] <= y[j], f"open_link_{i}_{j}"

    # Exactly k
    prob += lpSum(y[j] for j in J) == int(k), "k_facilities"

    # Capacity
    for j in J:
        prob += load[j] <= cap_t_anio * y[j], f"cap_{j}"

    # Equity
    for i in I:
        for j in J:
            prob += z >= dfD.loc[i, j] * x[i][j], f"maxdist_{i}_{j}"

    # Utilization proxy: maximize worst utilization <=> minimize worst unused
    for j in J:
        prob += u >= cap_t_anio * y[j] - load[j], f"max_unused_{j}"

    # Solve (CBC)
    prob.solve(PULP_CBC_CMD(msg=False))

    status = LpStatus.get(prob.status, str(prob.status))

    opened = [j for j in J if value(y[j]) > 0.5]

    assign = []
    for i in I:
        for j in J:
            if value(x[i][j]) > 0.5:
                assign.append((i, j, float(dfD.loc[i, j]), float(w[i])))
                break

    df_assign = pd.DataFrame(assign, columns=["municipio", "planta_asignada", "dist_km", "rsu_t_anio"])
    df_assign["costo_wdist"] = df_assign["rsu_t_anio"] * df_assign["dist_km"]

    # Loads per plant (for utilization)
    loads = df_assign.groupby("planta_asignada")["rsu_t_anio"].sum().reindex(opened).fillna(0.0)
    df_plants = pd.DataFrame({
        "planta": loads.index,
        "carga_t_anio": loads.values,
    })
    df_plants["cap_t_anio"] = cap_t_anio
    df_plants["utilizacion_%"] = np.where(
        df_plants["cap_t_anio"] > 0,
        100 * df_plants["carga_t_anio"] / df_plants["cap_t_anio"],
        0.0
    )
    df_plants["cap_no_usada_t_anio"] = df_plants["cap_t_anio"] - df_plants["carga_t_anio"]
    df_plants["cap_no_usada_t_anio"] = df_plants["cap_no_usada_t_anio"].clip(lower=0)

    total_w = df_assign["rsu_t_anio"].sum()
    wavg_dist = (df_assign["rsu_t_anio"] * df_assign["dist_km"]).sum() / total_w
    max_dist = df_assign["dist_km"].max()
    total_cost = df_assign["costo_wdist"].sum()

    return {
        "status": status,
        "opened": opened,
        "objective": value(prob.objective),
        "total_cost_wdist": float(total_cost),
        "wavg_dist_km": float(wavg_dist),
        "max_dist_km": float(max_dist),
        "z_maxdist_km": float(value(z)),
        "u_max_unused_t_anio": float(value(u)),
        "df_assign": df_assign,
        "df_plants": df_plants
    }

# =========================
# Viz: radial “cluster plot”
# =========================
def radial_cluster_plot(df_assign: pd.DataFrame, opened: list):
    """
    Diagrama radial:
      - Cada planta: centro (r=0)
      - Municipios asignados: r = dist_km, ángulo distribuido
    """
    if df_assign.empty or not opened:
        return None

    fig = go.Figure()
    for plant in opened:
        sub = df_assign[df_assign["planta_asignada"] == plant].copy()
        sub = sub.sort_values("dist_km", ascending=True)

        # Planta como “centro”
        fig.add_trace(go.Scatterpolar(
            r=[0],
            theta=[0],
            mode="markers+text",
            text=[f"PLANTA: {plant}"],
            textposition="top center",
            name=f"Planta {plant}"
        ))

        # Municipios alrededor: ángulos equiespaciados
        n = len(sub)
        if n == 0:
            continue
        thetas = np.linspace(10, 350, n)  # evita 0 para no chocar
        fig.add_trace(go.Scatterpolar(
            r=sub["dist_km"].values,
            theta=thetas,
            mode="markers+text",
            text=[f"{m} ({d:.1f} km)" for m, d in zip(sub["municipio"], sub["dist_km"])],
            textposition="top center",
            name=f"Aportantes → {plant}"
        ))

    fig.update_layout(
        title="Clúster radial (distancia vial a planta asignada)",
        showlegend=True,
        polar=dict(
            radialaxis=dict(visible=True, title="Distancia (km)")
        ),
        margin=dict(l=20, r=20, t=60, b=20),
        height=520
    )
    return fig

# =========================
# State: preset restore
# =========================
def make_preset_df(municipios: list):
    rows = []
    for m in municipios:
        rows.append({
            "municipio": m,
            "rsu_t_anio": float(PRESET_RSU.get(m, 0.0))
        })
    return pd.DataFrame(rows)

if "rsu_df" not in st.session_state:
    st.session_state.rsu_df = make_preset_df(DEFAULT_MUNICIPIOS)

# =========================
# UI
# =========================
st.title("📍 Optimizador Territorial CDR – Oriente Antioqueño (V2)")
st.caption(
    "V2 = distancias viales reales (OSRM) + optimización (costo logístico + equidad + capacidad/utilización)."
)

with st.sidebar:
    st.header("1) Municipios del clúster")
    municipios = st.multiselect(
        "Selecciona municipios (candidatos y aportantes):",
        options=DEFAULT_MUNICIPIOS,
        default=DEFAULT_MUNICIPIOS
    )

    c_restore, _ = st.columns([1, 1])
    with c_restore:
        if st.button("🔄 Restaurar preset", use_container_width=True):
            st.session_state.rsu_df = make_preset_df(DEFAULT_MUNICIPIOS)

    st.header("2) RSU (t/año) – Preset editable")
    st.caption("Puedes editar manualmente o subir un CSV con columnas: municipio, rsu_t_anio")

    uploaded = st.file_uploader("Subir CSV RSU", type=["csv"])

    if uploaded is not None:
        df_rsu_up = pd.read_csv(uploaded)
        df_rsu_up.columns = [c.strip().lower() for c in df_rsu_up.columns]
        if not {"municipio", "rsu_t_anio"}.issubset(set(df_rsu_up.columns)):
            st.error("El CSV debe tener columnas: municipio, rsu_t_anio")
        else:
            # merge con preset actual para mantener municipios
            st.session_state.rsu_df = df_rsu_up[["municipio", "rsu_t_anio"]].copy()

    # Base df para editor
    base_df = st.session_state.rsu_df.copy()

    # Asegurar que estén SOLO los municipios seleccionados (pero conservando valores)
    base_df = base_df.drop_duplicates(subset=["municipio"], keep="last")
    base_df = base_df[base_df["municipio"].isin(municipios)].copy()

    # si agregaron municipios nuevos y no existen en df, los añadimos con preset (o cero)
    missing = [m for m in municipios if m not in set(base_df["municipio"])]
    if missing:
        base_df = pd.concat([base_df, make_preset_df(missing)], ignore_index=True)

    base_df = base_df.set_index("municipio").reindex(municipios).fillna(0.0).reset_index()

    st.write("Editar RSU aquí:")
    df_rsu = st.data_editor(
        base_df,
        use_container_width=True,
        num_rows="fixed"
    )

    # Persistir cambios (solo a municipios seleccionados)
    st.session_state.rsu_df = df_rsu.copy()

    st.header("3) Capacidad por planta")
    st.caption("Restricción operativa: cada planta tiene un máximo anual.")
    cap_t_h = st.number_input("Capacidad (t/h) por planta", min_value=0.1, value=12.0, step=0.5)
    horas_anio = st.number_input("Horas operativas por año (h/año)", min_value=1, value=6000, step=250)
    cap_t_anio = float(cap_t_h) * float(horas_anio)

    st.info(f"Capacidad anual por planta ≈ **{cap_t_anio:,.0f} t/año**")

    st.header("4) Optimización")
    k = st.slider("Número de plantas (k)", min_value=1, max_value=max(1, len(municipios)), value=1)

    alpha = st.slider("α (mezcla costo vs equidad)", 0.0, 1.0, 0.75, 0.05)
    st.caption(f"Interpretación: **{alpha*100:.0f}% costo / {(1-alpha)*100:.0f}% equidad**")

    gamma = st.slider("γ (penaliza subutilización)", 0.0, 5.0, 0.50, 0.05)
    st.caption("γ alto empuja a que la(s) planta(s) abierta(s) no queden “vacías” (mejor uso de capacidad).")

    run = st.button("🚀 Calcular distancias + Optimizar (V2)", type="primary", use_container_width=True)

# =========================
# Run
# =========================
if run:
    if len(municipios) < 2:
        st.error("Selecciona al menos 2 municipios.")
        st.stop()

    # RSU series
    w = df_rsu.set_index("municipio")["rsu_t_anio"].astype(float)

    # Pre-check capacidad total
    if w.sum() <= 0:
        st.error("La suma total de RSU debe ser > 0 (revisa la tabla de entradas).")
        st.stop()

    if w.sum() > cap_t_anio * k:
        st.error(
            f"RSU total ({w.sum():,.0f} t/año) excede la capacidad total ({cap_t_anio*k:,.0f} t/año). "
            "Aumenta k o la capacidad por planta (t/h u horas/año)."
        )
        st.stop()

    # Geocode
    with st.spinner("Geocodificando municipios (OSM/Nominatim)..."):
        coords = {}
        failed = []
        for m in municipios:
            c = geocode_municipio(m)
            if c is None:
                failed.append(m)
            else:
                coords[m] = c

    if failed:
        st.error(
            f"No pude geocodificar: {failed}. "
            "Prueba a renombrar (ej. 'El Peñol' -> 'Peñol') o intenta de nuevo."
        )
        st.stop()

    # Distances
    with st.spinner("Calculando matriz de distancias viales (OSRM, por carretera)..."):
        dfD = build_distance_matrix_osrm(coords)

    st.subheader("1) dfD – Matriz de distancias viales (km) todos vs todos")
    st.dataframe(dfD.style.format("{:.1f}"), use_container_width=True)

    # Ranking single-plant (simple)
    st.subheader("2) Ranking simple (k=1) por costo logístico ponderado (RSU·km)")
    df_rank = candidate_ranking_single_plant(dfD, w)
    st.dataframe(df_rank, use_container_width=True)

    # Solve k=1 baseline and chosen k (comparison)
    st.subheader("3) Comparación: 1 planta vs k plantas")
    try:
        sol_1 = solve_facility_location_v2(dfD, w, k=1, alpha=alpha, gamma=gamma, cap_t_anio=cap_t_anio)
        sol_k = solve_facility_location_v2(dfD, w, k=k, alpha=alpha, gamma=gamma, cap_t_anio=cap_t_anio)
    except Exception as e:
        st.error(str(e))
        st.stop()

    cA, cB = st.columns(2)

    with cA:
        st.markdown("### Caso A: **1 planta**")
        st.write("**Planta(s):**", ", ".join(sol_1["opened"]) if sol_1["opened"] else "—")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("WAvg dist (km)", f"{sol_1['wavg_dist_km']:.1f}")
        m2.metric("Max dist (km)", f"{sol_1['max_dist_km']:.1f}")
        m3.metric("Costo (t·km/año)", f"{sol_1['total_cost_wdist']:.0f}")
        m4.metric("Max cap no usada (t/año)", f"{sol_1['u_max_unused_t_anio']:.0f}")

    with cB:
        st.markdown(f"### Caso B: **k = {k} plantas**")
        st.write("**Planta(s):**", ", ".join(sol_k["opened"]) if sol_k["opened"] else "—")
        n1, n2, n3, n4 = st.columns(4)
        n1.metric("WAvg dist (km)", f"{sol_k['wavg_dist_km']:.1f}")
        n2.metric("Max dist (km)", f"{sol_k['max_dist_km']:.1f}")
        n3.metric("Costo (t·km/año)", f"{sol_k['total_cost_wdist']:.0f}")
        n4.metric("Max cap no usada (t/año)", f"{sol_k['u_max_unused_t_anio']:.0f}")

    st.info("Tip: si al subir k baja mucho Max dist pero sube el costo, juega con α. Si la utilización cae, sube γ o revisa si k es demasiado alto para la demanda.")

    # Show solution details (k)
    st.subheader("4) Solución detallada (k plantas)")
    df_assign = sol_k["df_assign"].copy()
    df_assign["%RSU"] = 100 * df_assign["rsu_t_anio"] / df_assign["rsu_t_anio"].sum()
    st.dataframe(df_assign.sort_values("rsu_t_anio", ascending=False), use_container_width=True)

    # Plant utilization table
    st.subheader("5) Carga / utilización por planta (capacidad y uso)")
    df_plants = sol_k["df_plants"].copy()
    st.dataframe(df_plants.sort_values("utilizacion_%", ascending=True), use_container_width=True)

    # Heuristic ratio (%RSU/km)
    st.subheader("6) %RSU / Distancia_a_Planta (heurística de priorización)")
    eps = 1e-6
    df_ratio = df_assign.copy()
    df_ratio["ratio_%RSU_por_km"] = (df_ratio["%RSU"] / (df_ratio["dist_km"] + eps))
    df_ratio = df_ratio.sort_values("ratio_%RSU_por_km", ascending=False)
    st.dataframe(df_ratio[["municipio","planta_asignada","dist_km","%RSU","ratio_%RSU_por_km"]], use_container_width=True)

    # Radial cluster plot (replace bar charts)
    st.subheader("7) Visual: clúster circular (radial) por planta asignada")
    fig_radial = radial_cluster_plot(df_assign, sol_k["opened"])
    if fig_radial is not None:
        st.plotly_chart(fig_radial, use_container_width=True)
    else:
        st.warning("No fue posible construir el gráfico radial con la solución actual.")

else:
    st.warning("Configura municipios + RSU (preset editable) + capacidad + (α, γ, k) y presiona **Calcular distancias + Optimizar (V2)**.")
