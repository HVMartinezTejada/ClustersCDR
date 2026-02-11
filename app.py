import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import pydeck as pdk
from pulp import (
    LpProblem, LpMinimize, LpVariable, lpSum, LpBinary,
    LpStatus, value
)

# =========================
# Config
# =========================
st.set_page_config(
    page_title="CDR Oriente Antioqueño – Optimizador Territorial",
    layout="wide"
)

OSRM_TABLE_URL = "https://router.project-osrm.org/table/v1/driving/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

DEFAULT_MUNICIPIOS = [
    "La Ceja", "La Unión", "El Retiro", "Rionegro", "El Carmen de Viboral",
    "Abejorral", "Guarne", "Marinilla", "El Santuario", "El Peñol"
]

# Preset RSU (t/año) - como tu tabla
DEFAULT_RSU_PRESET = {
    "La Ceja": 22300,
    "La Unión": 6534,
    "El Retiro": 7178,
    "Rionegro": 63344,
    "El Carmen de Viboral": 9000,
    "Abejorral": 6080,
    "Guarne": 16914,
    "Marinilla": 19790,
    "El Santuario": 10793,
    "El Peñol": 6311
}

# =========================
# Helpers (geo + distances)
# =========================
@st.cache_data(show_spinner=False)
def geocode_municipio(nombre: str):
    """Geocodifica municipio usando Nominatim (OSM). Retorna (lat, lon)."""
    q = f"{nombre}, Antioquia, Colombia"
    params = {"q": q, "format": "json", "limit": 1}
    headers = {"User-Agent": "cdroptimizer-streamlit/1.0"}
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
    coord_str = ";".join([f"{coords_dict[n][1]},{coords_dict[n][0]}" for n in names])
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

def solve_facility_location(dfD: pd.DataFrame, w: pd.Series, k: int, alpha: float, cap_t_anio: float):
    """
    Optimiza:
      min alpha * sum_i sum_j w_i * d_ij * x_ij + (1-alpha) * z
    s.a.
      - cada municipio i asignado a 1 planta j
      - x_ij <= y_j
      - sum_j y_j == k
      - z >= d_ij * x_ij (equidad: minimiza la peor distancia)
      - capacidad por planta: sum_i w_i * x_ij <= cap_t_anio * y_j

    Retorna: dict resultados + df asignaciones
    """
    I = list(dfD.index)
    J = list(dfD.columns)

    w = w.reindex(I).fillna(0.0).astype(float)

    if (w < 0).any():
        raise ValueError("Hay RSU negativos. Corrige entradas.")
    if w.sum() <= 0:
        raise ValueError("La suma total de RSU debe ser > 0 para optimizar.")
    if cap_t_anio <= 0:
        raise ValueError("La capacidad por planta (t/año) debe ser > 0.")

    prob = LpProblem("CDR_Oriente_Antioquia", LpMinimize)

    x = LpVariable.dicts("x", (I, J), lowBound=0, upBound=1, cat=LpBinary)
    y = LpVariable.dicts("y", J, lowBound=0, upBound=1, cat=LpBinary)
    z = LpVariable("z_maxdist", lowBound=0)

    prob += alpha * lpSum(w[i] * dfD.loc[i, j] * x[i][j] for i in I for j in J) + (1 - alpha) * z

    # Asignación 1 a 1
    for i in I:
        prob += lpSum(x[i][j] for j in J) == 1, f"assign_{i}"

    # No asignar a plantas no abiertas
    for i in I:
        for j in J:
            prob += x[i][j] <= y[j], f"open_link_{i}_{j}"

    # Exactamente k plantas
    prob += lpSum(y[j] for j in J) == int(k), "k_facilities"

    # Equidad: z >= d_ij * x_ij
    for i in I:
        for j in J:
            prob += z >= dfD.loc[i, j] * x[i][j], f"maxdist_{i}_{j}"

    # Capacidad por planta: sum_i w_i x_ij <= cap_t_anio * y_j
    for j in J:
        prob += lpSum(w[i] * x[i][j] for i in I) <= cap_t_anio * y[j], f"cap_{j}"

    prob.solve()
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

    total_w = df_assign["rsu_t_anio"].sum()
    wavg_dist = (df_assign["rsu_t_anio"] * df_assign["dist_km"]).sum() / total_w
    max_dist = df_assign["dist_km"].max()
    total_cost = df_assign["costo_wdist"].sum()

    # Carga por planta
    df_load = df_assign.groupby("planta_asignada", as_index=False)["rsu_t_anio"].sum()
    df_load = df_load.rename(columns={"rsu_t_anio": "rsu_asignado_t_anio"})
    df_load["cap_t_anio"] = cap_t_anio
    df_load["utilizacion_%"] = 100 * df_load["rsu_asignado_t_anio"] / cap_t_anio

    return {
        "status": status,
        "opened": opened,
        "objective": value(prob.objective),
        "total_cost_wdist": total_cost,
        "wavg_dist_km": wavg_dist,
        "max_dist_km": max_dist,
        "z_maxdist_km": value(z),
        "df_assign": df_assign,
        "df_load": df_load
    }

def make_map(coords, df_assign, opened, radius_km: float):
    """
    Mapa con:
    - puntos municipios
    - puntos plantas (opened)
    - líneas municipio->planta
    - círculos radio en plantas
    """
    # Data de puntos
    pts = []
    for m, (lat, lon) in coords.items():
        pts.append({"municipio": m, "lat": lat, "lon": lon, "tipo": "municipio"})

    df_pts = pd.DataFrame(pts)
    df_pts["is_planta"] = df_pts["municipio"].isin(opened)

    # Líneas (paths)
    paths = []
    for _, r in df_assign.iterrows():
        m = r["municipio"]
        p = r["planta_asignada"]
        lat1, lon1 = coords[m]
        lat2, lon2 = coords[p]
        paths.append({
            "name": f"{m} → {p}",
            "path": [[lon1, lat1], [lon2, lat2]],
            "dist_km": float(r["dist_km"]),
            "rsu_t_anio": float(r["rsu_t_anio"])
        })
    df_paths = pd.DataFrame(paths)

    # Círculos de plantas
    df_plants = df_pts[df_pts["is_planta"]].copy()
    df_plants["radius_m"] = float(radius_km) * 1000.0

    # Layers
    layer_munis = pdk.Layer(
        "ScatterplotLayer",
        data=df_pts[df_pts["is_planta"] == False],
        get_position=["lon", "lat"],
        get_radius=300,
        pickable=True,
    )

    layer_plants = pdk.Layer(
        "ScatterplotLayer",
        data=df_plants,
        get_position=["lon", "lat"],
        get_radius=550,
        pickable=True,
    )

    layer_paths = pdk.Layer(
        "PathLayer",
        data=df_paths,
        get_path="path",
        width_scale=15,
        width_min_pixels=2,
        pickable=True,
    )

    layer_radius = pdk.Layer(
        "ScatterplotLayer",
        data=df_plants,
        get_position=["lon", "lat"],
        get_radius="radius_m",
        stroked=True,
        filled=False,
        pickable=False,
    )

    # View
    mean_lat = float(df_pts["lat"].mean())
    mean_lon = float(df_pts["lon"].mean())

    view_state = pdk.ViewState(
        latitude=mean_lat,
        longitude=mean_lon,
        zoom=10,
        pitch=0
    )

    deck = pdk.Deck(
        layers=[layer_radius, layer_paths, layer_plants, layer_munis],
        initial_view_state=view_state,
        tooltip={
            "text": "{municipio}"
        }
    )
    return deck

# =========================
# UI
# =========================
st.title("📍 Optimizador Territorial CDR – Oriente Antioqueño (distancias viales reales)")
st.caption("Calcula matriz vial (km) y ubica 1 o k plantas minimizando costo logístico y max-dist (equidad), con capacidad por planta.")

with st.sidebar:
    st.header("0) Preset RSU")
    if st.button("↩️ Restaurar preset", use_container_width=True):
        st.session_state.df_rsu_state = pd.DataFrame({
            "municipio": DEFAULT_MUNICIPIOS,
            "rsu_t_anio": [float(DEFAULT_RSU_PRESET.get(m, 0.0)) for m in DEFAULT_MUNICIPIOS]
        })
        st.rerun()

    st.header("1) Municipios del clúster")
    municipios = st.multiselect(
        "Selecciona municipios (candidatos y aportantes):",
        options=DEFAULT_MUNICIPIOS,
        default=DEFAULT_MUNICIPIOS
    )

    st.header("2) RSU (t/año) – Preset editable")
    st.caption("Arranca con preset. Puedes editar aquí o subir CSV con columnas: municipio, rsu_t_anio")

    # Estado persistente
    if "df_rsu_state" not in st.session_state:
        st.session_state.df_rsu_state = pd.DataFrame({
            "municipio": DEFAULT_MUNICIPIOS,
            "rsu_t_anio": [float(DEFAULT_RSU_PRESET.get(m, 0.0)) for m in DEFAULT_MUNICIPIOS]
        })

    df_state = st.session_state.df_rsu_state.copy()
    df_state = df_state[df_state["municipio"].isin(municipios)].copy()

    faltantes = [m for m in municipios if m not in set(df_state["municipio"])]
    if faltantes:
        df_add = pd.DataFrame({
            "municipio": faltantes,
            "rsu_t_anio": [float(DEFAULT_RSU_PRESET.get(m, 0.0)) for m in faltantes]
        })
        df_state = pd.concat([df_state, df_add], ignore_index=True)

    df_state = df_state.set_index("municipio").reindex(municipios).reset_index()
    df_state["rsu_t_anio"] = df_state["rsu_t_anio"].fillna(0.0).astype(float)

    uploaded = st.file_uploader("Subir CSV RSU", type=["csv"])
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        df_up.columns = [c.strip().lower() for c in df_up.columns]
        if {"municipio", "rsu_t_anio"}.issubset(df_up.columns):
            df_up = df_up[df_up["municipio"].isin(municipios)].copy()
            df_state = df_state.drop(columns=["rsu_t_anio"]).merge(
                df_up[["municipio", "rsu_t_anio"]],
                on="municipio",
                how="left"
            )
            df_state["rsu_t_anio"] = df_state["rsu_t_anio"].fillna(
                df_state["municipio"].map(DEFAULT_RSU_PRESET).fillna(0.0)
            ).astype(float)
        else:
            st.error("El CSV debe tener columnas: municipio, rsu_t_anio")

    st.write("Editar RSU aquí (t/año):")
    df_state = st.data_editor(df_state, use_container_width=True, num_rows="fixed")
    st.session_state.df_rsu_state = df_state.copy()
    df_rsu = df_state.copy()

    st.header("3) Capacidad por planta")
    cap_tph = st.number_input("Capacidad nominal (t/h)", min_value=0.1, value=12.0, step=0.5)
    horas_anio = st.number_input("Horas operativas por año", min_value=100.0, value=8000.0, step=100.0)
    disp = st.slider("Disponibilidad (%)", 10, 100, 85, 1)
    cap_t_anio = cap_tph * horas_anio * (disp / 100.0)
    st.caption(f"Capacidad por planta ≈ **{cap_t_anio:,.0f} t/año**")

    st.header("4) Optimización")
    k = st.slider("Número de plantas (k)", min_value=1, max_value=max(1, len(municipios)), value=1)

    alpha = st.slider("α (mezcla costo vs equidad)", 0.0, 1.0, 0.75, 0.05)
    st.caption(f"Interpretación: **{alpha*100:.0f}% costo / {(1-alpha)*100:.0f}% equidad**")

    st.header("5) Mapa / radio de acción")
    mode_radius = st.radio("Radio de acción", ["Usar max distancia asignada", "Definir radio fijo"], index=0)
    radio_fijo_km = st.slider("Radio fijo (km)", 5, 80, 25, 1)

    run = st.button("🚀 Calcular distancias + Optimizar", type="primary", use_container_width=True)

# =========================
# Run
# =========================
if run:
    if len(municipios) < 2:
        st.error("Selecciona al menos 2 municipios.")
        st.stop()

    w = df_rsu.set_index("municipio")["rsu_t_anio"].astype(float)

    # Validación rápida de capacidad vs total RSU
    total_rsu = float(w.sum())
    if k * cap_t_anio < total_rsu:
        st.warning(
            f"⚠️ Capacidad insuficiente: k·capacidad = {(k*cap_t_anio):,.0f} t/año < RSU total {total_rsu:,.0f} t/año. "
            "El solver puede fallar o forzar soluciones inviables. Sube k o la capacidad."
        )

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
        st.error(f"No pude geocodificar: {failed}. Prueba renombrar (p.ej. 'El Peñol' -> 'Peñol') o intenta de nuevo.")
        st.stop()

    # Distances
    with st.spinner("Calculando matriz de distancias viales (OSRM, por carretera)..."):
        dfD = build_distance_matrix_osrm(coords)

    st.subheader("1) dfD – Matriz de distancias viales (km) todos vs todos")
    st.dataframe(dfD.style.format("{:.1f}"), use_container_width=True)

    # Ranking single-plant (always useful)
    st.subheader("2) Ranking de candidatos (k=1) por costo logístico ponderado")
    df_rank = candidate_ranking_single_plant(dfD, w)
    st.dataframe(df_rank, use_container_width=True)

    # --- Comparación 1 planta vs k plantas ---
    st.subheader("3) Comparación: 1 planta vs k plantas")
    try:
        sol1 = solve_facility_location(dfD, w, k=1, alpha=alpha, cap_t_anio=cap_t_anio)
        solk = solve_facility_location(dfD, w, k=k, alpha=alpha, cap_t_anio=cap_t_anio)
    except Exception as e:
        st.error(str(e))
        st.stop()

    def kpi_row(sol, k_label):
        return {
            "Caso": k_label,
            "Estado": sol["status"],
            "Plantas": ", ".join(sol["opened"]) if sol["opened"] else "—",
            "Dist. prom ponderada (km)": round(sol["wavg_dist_km"], 2),
            "Dist. máxima (km)": round(sol["max_dist_km"], 2),
            "Costo logístico (t·km/año)": round(sol["total_cost_wdist"], 0),
        }

    df_kpi = pd.DataFrame([
        kpi_row(sol1, "k=1"),
        kpi_row(solk, f"k={k}")
    ])
    st.dataframe(df_kpi, use_container_width=True)

    # KPIs destacados (para el caso k elegido)
    st.subheader("4) Solución optimizada (caso k elegido)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado solver", solk["status"])
    c2.metric("Distancia promedio ponderada (km)", f"{solk['wavg_dist_km']:.1f}")
    c3.metric("Distancia máxima asignada (km)", f"{solk['max_dist_km']:.1f}")
    c4.metric("Costo logístico ponderado (t·km/año)", f"{solk['total_cost_wdist']:.0f}")

    st.write("**Plantas seleccionadas:**", ", ".join(solk["opened"]) if solk["opened"] else "—")

    df_assign = solk["df_assign"].copy()
    df_assign["%RSU"] = 100 * df_assign["rsu_t_anio"] / df_assign["rsu_t_anio"].sum()

    st.markdown("**Asignación municipio → planta**")
    st.dataframe(df_assign.sort_values("rsu_t_anio", ascending=False), use_container_width=True)

    st.markdown("**Carga / utilización por planta**")
    st.dataframe(solk["df_load"].sort_values("utilizacion_%", ascending=False), use_container_width=True)

    # “%Residuos / Distancia_a_Planta” (heurística)
    st.subheader("5) %RSU / Distancia_a_Planta (heurística de priorización)")
    eps = 1e-6
    df_ratio = df_assign.copy()
    df_ratio["ratio_%RSU_por_km"] = df_ratio["%RSU"] / (df_ratio["dist_km"] + eps)
    df_ratio = df_ratio.sort_values("ratio_%RSU_por_km", ascending=False)
    st.dataframe(df_ratio[["municipio", "planta_asignada", "dist_km", "%RSU", "ratio_%RSU_por_km"]], use_container_width=True)

    # Mapa con radio de acción
    st.subheader("6) Visual territorial (radio de acción + asignaciones)")
    if mode_radius == "Usar max distancia asignada":
        radius_km = float(df_assign["dist_km"].max())
    else:
        radius_km = float(radio_fijo_km)

    deck = make_map(coords, df_assign, solk["opened"], radius_km=radius_km)
    st.pydeck_chart(deck, use_container_width=True)

    # (Opcional) Mantengo un gráfico útil de %RSU (puedes quitarlo luego)
    st.subheader("7) %RSU sugerido (proporcional a RSU PGIRS)")
    fig1 = px.bar(
        df_assign.sort_values("%RSU", ascending=False),
        x="municipio", y="%RSU",
        title="%RSU sugerido (proporcional a RSU PGIRS)"
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.info("Tip: Usa la tabla comparativa (k=1 vs k) para decidir el trade-off entre centralización y cobertura territorial.")
else:
    st.warning("Configura municipios + RSU (PGIRS) y presiona **Calcular distancias + Optimizar**.")
