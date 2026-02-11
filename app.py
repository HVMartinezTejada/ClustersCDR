import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum,
    LpBinary, LpStatus, value
)

# =========================
# Config
# =========================
st.set_page_config(page_title="CDR Oriente Antioqueño – Optimizador Territorial", layout="wide")
OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

DEFAULT_MUNICIPIOS = [
    "La Ceja", "La Unión", "El Retiro", "Rionegro", "El Carmen de Viboral",
    "Abejorral", "Guarne", "Marinilla", "El Santuario", "El Peñol"
]

# -------------------------
# Preset RSU (t/año)
# Ajusta estos valores con tus cifras base (PGIRS / línea base).
# -------------------------
DEFAULT_RSU_PRESET = {
    "La Ceja": 30000.0,
    "La Unión": 8000.0,
    "El Retiro": 12000.0,
    "Rionegro": 60000.0,
    "El Carmen de Viboral": 18000.0,
    "Abejorral": 6000.0,
    "Guarne": 16000.0,
    "Marinilla": 14000.0,
    "El Santuario": 9000.0,
    "El Peñol": 5000.0,
}

# =========================
# Helpers (geo + distances)
# =========================
@st.cache_data(show_spinner=False)
def geocode_municipio(nombre: str):
    """Geocodifica municipio usando Nominatim (OSM). Retorna (lat, lon)."""
    q = f"{nombre}, Antioquia, Colombia"
    params = {"q": q, "format": "json", "limit": 1}
    headers = {"User-Agent": "cdroptimizer-streamlit/1.0 (contact: user)"}
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
    r = requests.get(url, params=params, timeout=60)
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
# Solver V2 (capacidad + gamma + unmet)
# =========================
def solve_facility_location_v2(
    dfD: pd.DataFrame,
    w: pd.Series,
    k: int,
    alpha: float,
    gamma: float,
    cap_per_plant_t_anio: float,
    allow_unmet: bool,
    penalty_unmet: float,
):
    """
    Objetivo:
      alpha * sum_i sum_j served_i * d_ij * x_ij
    + (1-alpha) * z
    + gamma * u
    + (allow_unmet) penalty_unmet * unmet

    Donde:
      served_i <= w_i
      sum_j x_ij == 1
      x_ij <= y_j
      sum_j y_j == k

      Capacidad por planta:
        sum_i served_i * x_ij <= cap * y_j

      Equidad:
        z >= d_ij * x_ij  (max-dist)

      Subutilización:
        u >= cap*y_j - load_j  para todo j
        load_j = sum_i served_i * x_ij

      unmet = sum_i (w_i - served_i)
    """
    I = list(dfD.index)
    J = list(dfD.columns)

    w = w.reindex(I).fillna(0.0).astype(float)
    if (w < 0).any():
        raise ValueError("Hay RSU negativos. Corrige entradas.")
    if w.sum() <= 0:
        raise ValueError("La suma total de RSU debe ser > 0 para optimizar.")

    k = int(k)
    if k < 1 or k > len(J):
        raise ValueError("k fuera de rango.")

    if cap_per_plant_t_anio <= 0:
        raise ValueError("La capacidad por planta (t/año) debe ser > 0.")

    prob = LpProblem("CDR_Oriente_Antioquia_V2", LpMinimize)

    x = LpVariable.dicts("x", (I, J), lowBound=0, upBound=1, cat=LpBinary)
    y = LpVariable.dicts("y", J, lowBound=0, upBound=1, cat=LpBinary)

    served = LpVariable.dicts("served", I, lowBound=0)  # t/año servidas por municipio
    z = LpVariable("z_maxdist", lowBound=0)             # km
    u = LpVariable("u_max_unused_capacity", lowBound=0) # t/año (capacidad no usada máxima)

    # Variables auxiliares
    load = LpVariable.dicts("load", J, lowBound=0)      # t/año atendidas por planta

    # Definir served bounds
    for i in I:
        prob += served[i] <= float(w[i]), f"served_leq_w_{i}"

    # Asignación + apertura
    for i in I:
        prob += lpSum(x[i][j] for j in J) == 1, f"assign_{i}"
    for i in I:
        for j in J:
            prob += x[i][j] <= y[j], f"open_link_{i}_{j}"

    prob += lpSum(y[j] for j in J) == k, "k_facilities"

    # Cargar por planta: load_j = sum_i served_i * x_ij
    # OJO: served_i * x_ij es bilineal en teoría, pero aquí served_i es continuo y x_ij binario,
    # PuLP no soporta bilineal directamente. Solución práctica: linearizar con "served_ij".
    served_ij = LpVariable.dicts("served_ij", (I, J), lowBound=0)

    for i in I:
        for j in J:
            # served_ij <= served_i
            prob += served_ij[i][j] <= served[i], f"servedij_le_served_{i}_{j}"
            # served_ij <= w_i * x_ij
            prob += served_ij[i][j] <= float(w[i]) * x[i][j], f"servedij_le_wx_{i}_{j}"
            # served_ij >= served_i - w_i*(1-x_ij)
            prob += served_ij[i][j] >= served[i] - float(w[i]) * (1 - x[i][j]), f"servedij_ge_served_minus_{i}_{j}"

    for j in J:
        prob += load[j] == lpSum(served_ij[i][j] for i in I), f"load_def_{j}"
        # Capacidad
        prob += load[j] <= cap_per_plant_t_anio * y[j], f"cap_{j}"
        # Subutilización (capacidad no usada)
        prob += u >= cap_per_plant_t_anio * y[j] - load[j], f"unused_{j}"

    # Equidad (max distance)
    for i in I:
        for j in J:
            prob += z >= dfD.loc[i, j] * x[i][j], f"maxdist_{i}_{j}"

    # Unmet
    unmet = lpSum(float(w[i]) - served[i] for i in I)

    # Objetivo principal: costo logístico ponderado (t·km/año) usando served_ij
    cost_wdist = lpSum(dfD.loc[i, j] * served_ij[i][j] for i in I for j in J)

    obj = alpha * cost_wdist + (1 - alpha) * z + gamma * u
    if allow_unmet:
        obj += penalty_unmet * unmet
    else:
        # Si no permites unmet, fuerza served_i = w_i (es decir, servir todo)
        for i in I:
            prob += served[i] == float(w[i]), f"force_full_service_{i}"

    prob += obj

    # Solve
    prob.solve()
    status = LpStatus.get(prob.status, str(prob.status))

    # Extract opened + assignment
    opened = [j for j in J if value(y[j]) > 0.5]

    assign = []
    for i in I:
        chosen_j = None
        for j in J:
            if value(x[i][j]) > 0.5:
                chosen_j = j
                break
        if chosen_j is None:
            chosen_j = J[0]
        dist_km = float(dfD.loc[i, chosen_j])
        rsu_i = float(w[i])
        served_i = float(value(served[i]))
        assign.append((i, chosen_j, dist_km, rsu_i, served_i))

    df_assign = pd.DataFrame(assign, columns=["municipio", "planta_asignada", "dist_km", "rsu_t_anio", "served_t_anio"])
    df_assign["costo_wdist"] = df_assign["served_t_anio"] * df_assign["dist_km"]

    # KPIs
    total_w = df_assign["rsu_t_anio"].sum()
    total_served = df_assign["served_t_anio"].sum()
    unmet_t = total_w - total_served

    # promedio ponderado por lo servido (más coherente cuando hay unmet)
    if total_served > 0:
        wavg_dist = (df_assign["served_t_anio"] * df_assign["dist_km"]).sum() / total_served
    else:
        wavg_dist = np.nan

    max_dist = df_assign["dist_km"].max() if len(df_assign) else np.nan
    total_cost = df_assign["costo_wdist"].sum()

    # Utilización por planta
    plant_load = df_assign.groupby("planta_asignada")["served_t_anio"].sum().reindex(opened).fillna(0.0)
    plant_util = (plant_load / cap_per_plant_t_anio * 100.0).replace([np.inf, -np.inf], np.nan)

    util_avg = float(np.nanmean(plant_util.values)) if len(plant_util) else np.nan
    util_min = float(np.nanmin(plant_util.values)) if len(plant_util) else np.nan
    util_max = float(np.nanmax(plant_util.values)) if len(plant_util) else np.nan

    df_plants = pd.DataFrame({
        "planta": list(plant_load.index),
        "carga_t_anio": plant_load.values,
        "capacidad_t_anio": cap_per_plant_t_anio,
        "utilizacion_%": plant_util.values
    }).sort_values("utilizacion_%", ascending=False)

    return {
        "status": status,
        "opened": opened,
        "objective": float(value(prob.objective)) if prob.objective is not None else np.nan,
        "total_cost_wdist": float(total_cost),
        "wavg_dist_km": float(wavg_dist) if wavg_dist == wavg_dist else np.nan,
        "max_dist_km": float(max_dist) if max_dist == max_dist else np.nan,
        "z_maxdist_km": float(value(z)) if z is not None else np.nan,
        "u_max_unused_cap_t_anio": float(value(u)) if u is not None else np.nan,
        "unmet_t_anio": float(unmet_t),
        "served_total_t_anio": float(total_served),
        "demand_total_t_anio": float(total_w),
        "util_avg_%": util_avg,
        "util_min_%": util_min,
        "util_max_%": util_max,
        "df_assign": df_assign,
        "df_plants": df_plants,
    }

# =========================
# UI
# =========================
st.title("📍 Optimizador Territorial CDR – Oriente Antioqueño (distancias viales reales)")
st.caption("Calcula matriz vial (km) y ubica 1 o k plantas minimizando costo logístico, equidad territorial y (V2) subutilización/capacidad.")

# Sidebar
with st.sidebar:
    st.header("1) Municipios del clúster")
    municipios = st.multiselect(
        "Selecciona municipios (candidatos y aportantes):",
        options=DEFAULT_MUNICIPIOS,
        default=DEFAULT_MUNICIPIOS
    )

    # Preset + restore
    st.header("2) RSU (t/año) – Preset editable")
    if "df_rsu_state" not in st.session_state:
        # Inicializa con preset (solo municipios seleccionados)
        init = []
        for m in DEFAULT_MUNICIPIOS:
            init.append([m, float(DEFAULT_RSU_PRESET.get(m, 0.0))])
        st.session_state.df_rsu_state = pd.DataFrame(init, columns=["municipio", "rsu_t_anio"])

    colA, colB = st.columns(2)
    with colA:
        restore = st.button("↩️ Restaurar preset", use_container_width=True)
    with colB:
        st.write("")  # spacer

    if restore:
        init = []
        for m in DEFAULT_MUNICIPIOS:
            init.append([m, float(DEFAULT_RSU_PRESET.get(m, 0.0))])
        st.session_state.df_rsu_state = pd.DataFrame(init, columns=["municipio", "rsu_t_anio"])

    uploaded = st.file_uploader("Subir CSV RSU (opcional)", type=["csv"])
    if uploaded is not None:
        df_rsu_up = pd.read_csv(uploaded)
        df_rsu_up.columns = [c.strip().lower() for c in df_rsu_up.columns]
        if not {"municipio", "rsu_t_anio"}.issubset(set(df_rsu_up.columns)):
            st.error("El CSV debe tener columnas: municipio, rsu_t_anio")
        else:
            st.session_state.df_rsu_state = df_rsu_up.rename(columns={"rsu_t_anio": "rsu_t_anio"}).copy()

    df_rsu = st.session_state.df_rsu_state.copy()
    df_rsu = df_rsu[df_rsu["municipio"].isin(municipios)].copy()
    df_rsu = df_rsu.drop_duplicates(subset=["municipio"], keep="last")
    df_rsu = df_rsu.set_index("municipio").reindex(municipios).fillna(0.0).reset_index()

    st.write("Editar RSU aquí (t/año):")
    df_rsu = st.data_editor(df_rsu, use_container_width=True, num_rows="fixed")

    # Persistir cambios
    # (guardamos solo lo visible; para simplicidad)
    st.session_state.df_rsu_state = pd.concat([
        st.session_state.df_rsu_state[~st.session_state.df_rsu_state["municipio"].isin(municipios)],
        df_rsu
    ], ignore_index=True)

    st.header("3) Parámetros del modelo")
    k = st.slider("Número de plantas (k)", min_value=1, max_value=max(1, len(municipios)), value=1)

    alpha = st.slider("Peso costo logístico (α)", 0.0, 1.0, 0.75, 0.05)
    st.caption(f"Interpretación: **{int(alpha*100)}% costo** / **{int((1-alpha)*100)}% equidad**")

    gamma = st.slider("Peso anti-subutilización (γ)", 0.0, 3.0, 0.5, 0.1)
    st.caption("γ alto penaliza plantas poco cargadas (prefiere menos plantas o asignación más concentrada).")

    st.header("4) Capacidad por planta")
    cap_tph = st.number_input("Capacidad nominal (t/h)", min_value=0.1, value=12.0, step=0.5)
    horas_dia = st.number_input("Horas operación/día", min_value=1.0, value=16.0, step=1.0)
    dias_anio = st.number_input("Días operación/año", min_value=1.0, value=300.0, step=5.0)
    cap_per_plant = float(cap_tph * horas_dia * dias_anio)
    st.info(f"Capacidad por planta ≈ **{cap_per_plant:,.0f} t/año**")

    st.header("5) ¿Permitir demanda > capacidad?")
    allow_unmet = st.toggle("Permitir RSU no servido (correr igual)", value=True)
    penalty_unmet = st.number_input(
        "Penalización por RSU no servido (peso)",
        min_value=0.0, value=500.0, step=50.0,
        help="Más alto = el modelo intentará servir más RSU aunque aumenten distancias."
    )

    run = st.button("🚀 Calcular distancias + Optimizar", type="primary", use_container_width=True)

    st.header("6) Explorador (barrido k, α, γ)")
    do_sweep = st.toggle("Activar modo barrido", value=False)
    if do_sweep:
        k_min = st.number_input("k mínimo", min_value=1, max_value=max(1, len(municipios)), value=1, step=1)
        k_max = st.number_input("k máximo", min_value=1, max_value=max(1, len(municipios)), value=min(3, len(municipios)), step=1)
        alpha_min = st.slider("α mínimo", 0.0, 1.0, 0.5, 0.05)
        alpha_max = st.slider("α máximo", 0.0, 1.0, 0.9, 0.05)
        alpha_step = st.selectbox("Paso α", options=[0.05, 0.1, 0.2], index=1)
        gamma_min = st.slider("γ mínimo", 0.0, 3.0, 0.0, 0.1)
        gamma_max = st.slider("γ máximo", 0.0, 3.0, 1.5, 0.1)
        gamma_step = st.selectbox("Paso γ", options=[0.1, 0.25, 0.5], index=1)
        run_sweep = st.button("🧭 Correr barrido", use_container_width=True)
    else:
        run_sweep = False

# =========================
# Main run
# =========================
def compute_coords(municipios):
    coords = {}
    failed = []
    for m in municipios:
        c = geocode_municipio(m)
        if c is None:
            failed.append(m)
        else:
            coords[m] = c
    return coords, failed

def ensure_inputs(municipios, df_rsu):
    if len(municipios) < 2:
        st.error("Selecciona al menos 2 municipios.")
        st.stop()
    w = df_rsu.set_index("municipio")["rsu_t_anio"].astype(float)
    if w.sum() <= 0:
        st.error("La suma total de RSU debe ser > 0 (t/año).")
        st.stop()
    return w

if run or run_sweep:
    w = ensure_inputs(municipios, df_rsu)

    with st.spinner("Geocodificando municipios (OSM/Nominatim)..."):
        coords, failed = compute_coords(municipios)
    if failed:
        st.error(f"No pude geocodificar: {failed}. Prueba renombrar (ej. 'El Peñol' -> 'Peñol') o intenta de nuevo.")
        st.stop()

    with st.spinner("Calculando matriz de distancias viales (OSRM, por carretera)..."):
        dfD = build_distance_matrix_osrm(coords)

    st.subheader("1) dfD – Matriz de distancias viales (km) todos vs todos")
    st.dataframe(dfD.style.format("{:.1f}"), use_container_width=True)

    # Ranking k=1
    st.subheader("2) Ranking de candidatos (k=1) por costo logístico ponderado")
    df_rank = candidate_ranking_single_plant(dfD, w)
    st.dataframe(df_rank, use_container_width=True)

    # ---------- Barrido ----------
    if run_sweep:
        st.subheader("🧭 Mapa conceptual de decisiones (barrido k, α, γ)")
        k_vals = list(range(int(k_min), int(k_max) + 1))
        alpha_vals = np.round(np.arange(alpha_min, alpha_max + 1e-9, alpha_step), 3).tolist()
        gamma_vals = np.round(np.arange(gamma_min, gamma_max + 1e-9, gamma_step), 3).tolist()

        rows = []
        with st.spinner("Corriendo barrido... (esto puede tardar un poco)"):
            for kk in k_vals:
                for aa in alpha_vals:
                    for gg in gamma_vals:
                        sol = solve_facility_location_v2(
                            dfD=dfD, w=w, k=kk, alpha=float(aa), gamma=float(gg),
                            cap_per_plant_t_anio=cap_per_plant,
                            allow_unmet=allow_unmet,
                            penalty_unmet=float(penalty_unmet),
                        )
                        rows.append({
                            "k": kk,
                            "alpha": float(aa),
                            "gamma": float(gg),
                            "status": sol["status"],
                            "opened": ", ".join(sol["opened"]),
                            "costo_logistico_tkm_anio": sol["total_cost_wdist"],
                            "dist_prom_pond_km": sol["wavg_dist_km"],
                            "dist_max_km": sol["max_dist_km"],
                            "util_avg_%": sol["util_avg_%"],
                            "util_min_%": sol["util_min_%"],
                            "unmet_t_anio": sol["unmet_t_anio"],
                        })

        df_sweep = pd.DataFrame(rows)
        st.dataframe(df_sweep, use_container_width=True)

        metric = st.selectbox(
            "Métrica a visualizar",
            options=[
                "costo_logistico_tkm_anio",
                "dist_prom_pond_km",
                "dist_max_km",
                "util_avg_%",
                "util_min_%",
                "unmet_t_anio",
            ],
            index=0
        )

        gamma_slice = st.selectbox("Fijar γ (rebanada para mapa k vs α)", options=sorted(df_sweep["gamma"].unique()), index=0)
        df_slice = df_sweep[df_sweep["gamma"] == gamma_slice].copy()

        # Heatmap k vs alpha
        pivot = df_slice.pivot_table(index="k", columns="alpha", values=metric, aggfunc="mean")
        pivot = pivot.sort_index().sort_index(axis=1)

        fig_hm = px.imshow(
            pivot,
            text_auto=".2f",
            aspect="auto",
            labels=dict(x="α", y="k", color=metric),
            title=f"Mapa conceptual: k vs α (γ fijo = {gamma_slice}) — métrica: {metric}"
        )
        st.plotly_chart(fig_hm, use_container_width=True)

        st.info("Tip: usa el mapa para ver regiones de decisión: centralización (k bajo, γ alto) vs distribución (k alto, α bajo).")

    # ---------- Corrida puntual ----------
    if run:
        st.subheader("3) Solución optimizada (corrida puntual)")

        sol = solve_facility_location_v2(
            dfD=dfD, w=w, k=k, alpha=alpha, gamma=gamma,
            cap_per_plant_t_anio=cap_per_plant,
            allow_unmet=allow_unmet,
            penalty_unmet=float(penalty_unmet),
        )

        # KPIs
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Estado solver", sol["status"])
        c2.metric("Costo logístico (t·km/año)", f"{sol['total_cost_wdist']:.0f}")
        c3.metric("Distancia prom. ponderada (km)", f"{sol['wavg_dist_km']:.1f}" if sol["wavg_dist_km"] == sol["wavg_dist_km"] else "—")
        c4.metric("Distancia máxima (km)", f"{sol['max_dist_km']:.1f}" if sol["max_dist_km"] == sol["max_dist_km"] else "—")
        c5.metric("RSU no servido (t/año)", f"{sol['unmet_t_anio']:.0f}")

        st.write("**Plantas seleccionadas:**", ", ".join(sol["opened"]) if sol["opened"] else "—")

        # Nota de capacidad
        demand = sol["demand_total_t_anio"]
        served = sol["served_total_t_anio"]
        cap_total = cap_per_plant * int(k)
        if demand > cap_total and allow_unmet:
            st.warning(
                f"Demanda total RSU (**{demand:,.0f} t/año**) excede capacidad total (**{cap_total:,.0f} t/año**). "
                f"Se optimizó sirviendo **{served:,.0f} t/año** y dejando **{(demand-served):,.0f} t/año** sin atender. "
                f"Sugerencia: aumenta **k** o la **capacidad por planta**."
            )

        # Asignaciones
        df_assign = sol["df_assign"].copy()
        df_assign["%RSU"] = 100 * df_assign["rsu_t_anio"] / max(1e-9, df_assign["rsu_t_anio"].sum())
        df_assign["%Servido"] = 100 * df_assign["served_t_anio"] / max(1e-9, df_assign["served_t_anio"].sum()) if df_assign["served_t_anio"].sum() > 0 else 0.0

        st.subheader("4) Asignación municipios → planta (con RSU servida)")
        st.dataframe(df_assign.sort_values("rsu_t_anio", ascending=False), use_container_width=True)

        st.subheader("5) Carga / utilización por planta")
        st.dataframe(sol["df_plants"], use_container_width=True)

        # Heurística %RSU/dist
        st.subheader("6) %RSU / Distancia_a_Planta (heurística)")
        eps = 1e-6
        df_ratio = df_assign.copy()
        df_ratio["ratio_%RSU_por_km"] = (df_ratio["%RSU"] / (df_ratio["dist_km"] + eps))
        df_ratio = df_ratio.sort_values("ratio_%RSU_por_km", ascending=False)
        st.dataframe(df_ratio[["municipio","planta_asignada","dist_km","%RSU","ratio_%RSU_por_km"]], use_container_width=True)

        # Visualización circular (approx) usando dist_km como radio y plantas como centros
        st.subheader("7) Visual tipo ‘radio de acción’ (aprox. por distancias asignadas)")
        # Para un primer V2: usamos un scatter con “planta_asignada” como color y dist_km como eje radial (simple y útil)
        fig_sc = px.scatter(
            df_assign,
            x="dist_km",
            y="served_t_anio",
            color="planta_asignada",
            hover_data=["municipio", "rsu_t_anio", "served_t_anio"],
            title="Municipios (distancia a planta) vs RSU servida — color = planta asignada"
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.info("Siguiente mejora natural: convertir esto en un diagrama circular por planta (polos) con anillos de distancia (10, 20, 30 km).")

else:
    st.warning("Configura municipios + RSU (preset editable) y presiona **Calcular distancias + Optimizar**.")
