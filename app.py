import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value

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
    # OSRM usa "lon,lat;lon,lat;..."
    coord_str = ";".join([f"{coords_dict[n][1]},{coords_dict[n][0]}" for n in names])
    url = OSRM_TABLE_URL + coord_str
    params = {"annotations": "distance"}  # distancias en metros
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    dist_m = np.array(js["distances"], dtype=float)  # metros
    dist_km = dist_m / 1000.0
    dfD = pd.DataFrame(dist_km, index=names, columns=names)
    return dfD

def solve_facility_location(dfD: pd.DataFrame, w: pd.Series, k: int, alpha: float):
    """
    Optimiza:
      min alpha * sum_i sum_j w_i * d_ij * x_ij + (1-alpha) * z
    s.a. asignación + apertura k plantas + z >= d_ij * x_ij (equidad por max distance)

    Retorna: dict resultados + df asignaciones
    """
    I = list(dfD.index)
    J = list(dfD.columns)

    # Limpieza
    w = w.reindex(I).fillna(0.0).astype(float)
    if (w < 0).any():
        raise ValueError("Hay RSU negativos. Corrige entradas.")
    if w.sum() <= 0:
        raise ValueError("La suma total de RSU debe ser > 0 para optimizar.")

    prob = LpProblem("CDR_Oriente_Antioquia", LpMinimize)

    x = LpVariable.dicts("x", (I, J), lowBound=0, upBound=1, cat=LpBinary)
    y = LpVariable.dicts("y", J, lowBound=0, upBound=1, cat=LpBinary)
    z = LpVariable("z_maxdist", lowBound=0)

    # Objective
    prob += alpha * lpSum(w[i] * dfD.loc[i, j] * x[i][j] for i in I for j in J) + (1 - alpha) * z

    # Constraints
    for i in I:
        prob += lpSum(x[i][j] for j in J) == 1, f"assign_{i}"
    for i in I:
        for j in J:
            prob += x[i][j] <= y[j], f"open_link_{i}_{j}"

    prob += lpSum(y[j] for j in J) == int(k), "k_facilities"

    # Equity: z >= d_ij * x_ij
    for i in I:
        for j in J:
            prob += z >= dfD.loc[i, j] * x[i][j], f"maxdist_{i}_{j}"

    # Solve
    prob.solve()

    status = LpStatus.get(prob.status, str(prob.status))

    # Extract
    opened = [j for j in J if value(y[j]) > 0.5]
    assign = []
    for i in I:
        for j in J:
            if value(x[i][j]) > 0.5:
                assign.append((i, j, dfD.loc[i, j], float(w[i])))
                break

    df_assign = pd.DataFrame(assign, columns=["municipio", "planta_asignada", "dist_km", "rsu_t_anio"])
    df_assign["costo_wdist"] = df_assign["rsu_t_anio"] * df_assign["dist_km"]

    # KPIs
    total_w = df_assign["rsu_t_anio"].sum()
    wavg_dist = (df_assign["rsu_t_anio"] * df_assign["dist_km"]).sum() / total_w
    max_dist = df_assign["dist_km"].max()
    total_cost = df_assign["costo_wdist"].sum()

    return {
        "status": status,
        "opened": opened,
        "objective": value(prob.objective),
        "total_cost_wdist": total_cost,
        "wavg_dist_km": wavg_dist,
        "max_dist_km": max_dist,
        "z_maxdist_km": value(z),
        "df_assign": df_assign
    }

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
# UI
# =========================
st.title("📍 Optimizador Territorial CDR – Oriente Antioqueño (distancias viales reales)")
st.caption("Calcula matriz vial (km) y ubica 1 o k plantas minimizando costo logístico y max-dist (equidad territorial).")

with st.sidebar:
    st.header("1) Municipios del clúster")
    municipios = st.multiselect(
        "Selecciona municipios (candidatos y aportantes):",
        options=DEFAULT_MUNICIPIOS,
        default=DEFAULT_MUNICIPIOS
    )

    st.header("2) RSU (t/año) – Línea base PGIRS")
    st.caption("Puedes editar manualmente o subir un CSV con columnas: municipio, rsu_t_anio")

    uploaded = st.file_uploader("Subir CSV RSU", type=["csv"])
    if uploaded is not None:
        df_rsu_up = pd.read_csv(uploaded)
        df_rsu_up.columns = [c.strip().lower() for c in df_rsu_up.columns]
        if not {"municipio", "rsu_t_anio"}.issubset(set(df_rsu_up.columns)):
            st.error("El CSV debe tener columnas: municipio, rsu_t_anio")
            df_rsu = pd.DataFrame({"municipio": municipios, "rsu_t_anio": [0.0]*len(municipios)})
        else:
            df_rsu = df_rsu_up.copy()
    else:
        df_rsu = pd.DataFrame({"municipio": municipios, "rsu_t_anio": [0.0]*len(municipios)})

    df_rsu = df_rsu[df_rsu["municipio"].isin(municipios)].copy()
    df_rsu = df_rsu.drop_duplicates(subset=["municipio"], keep="last")
    df_rsu = df_rsu.set_index("municipio").reindex(municipios).fillna(0.0).reset_index()

    st.write("Editar RSU aquí:")
    df_rsu = st.data_editor(df_rsu, use_container_width=True, num_rows="fixed")

    st.header("3) Optimización")
    k = st.slider("Número de plantas (k)", min_value=1, max_value=max(1, len(municipios)), value=1)
    alpha = st.slider("Peso costo logístico (α)", 0.0, 1.0, 0.75, 0.05)
    run = st.button("🚀 Calcular distancias + Optimizar", type="primary", use_container_width=True)

# =========================
# Run
# =========================
if run:
    if len(municipios) < 2:
        st.error("Selecciona al menos 2 municipios.")
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
        st.error(f"No pude geocodificar: {failed}. Prueba a renombrar (ej. 'El Peñol' -> 'Peñol') o intenta de nuevo.")
        st.stop()

    # Distances
    with st.spinner("Calculando matriz de distancias viales (OSRM, por carretera)..."):
        dfD = build_distance_matrix_osrm(coords)

    st.subheader("1) dfD – Matriz de distancias viales (km) todos vs todos")
    st.dataframe(dfD.style.format("{:.1f}"), use_container_width=True)

    # RSU series
    w = df_rsu.set_index("municipio")["rsu_t_anio"].astype(float)

    # Ranking single-plant (always useful)
    st.subheader("2) Ranking de candidatos (k=1) por costo logístico ponderado")
    df_rank = candidate_ranking_single_plant(dfD, w)
    st.dataframe(df_rank, use_container_width=True)

    # Solve for chosen k
    st.subheader("3) Solución optimizada")
    try:
        sol = solve_facility_location(dfD, w, k=k, alpha=alpha)
    except Exception as e:
        st.error(str(e))
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estado solver", sol["status"])
    c2.metric("Distancia promedio ponderada (km)", f"{sol['wavg_dist_km']:.1f}")
    c3.metric("Distancia máxima asignada (km)", f"{sol['max_dist_km']:.1f}")
    c4.metric("Costo logístico ponderado (t·km/año)", f"{sol['total_cost_wdist']:.0f}")

    st.write("**Plantas seleccionadas:**", ", ".join(sol["opened"]) if sol["opened"] else "—")

    df_assign = sol["df_assign"].copy()
    df_assign["%RSU"] = 100 * df_assign["rsu_t_anio"] / df_assign["rsu_t_anio"].sum()

    st.dataframe(df_assign.sort_values("rsu_t_anio", ascending=False), use_container_width=True)

    # “%Residuos / Distancia_a_Planta” (heurística pedida)
    st.subheader("4) %RSU / Distancia_a_Planta (heurística de priorización)")
    eps = 1e-6
    df_ratio = df_assign.copy()
    df_ratio["ratio_%RSU_por_km"] = (df_ratio["%RSU"] / (df_ratio["dist_km"] + eps))
    df_ratio = df_ratio.sort_values("ratio_%RSU_por_km", ascending=False)
    st.dataframe(df_ratio[["municipio","planta_asignada","dist_km","%RSU","ratio_%RSU_por_km"]], use_container_width=True)

    # Visual: barras %RSU
    fig1 = px.bar(df_assign.sort_values("%RSU", ascending=False), x="municipio", y="%RSU",
                  title="%RSU sugerido (proporcional a RSU PGIRS)")
    st.plotly_chart(fig1, use_container_width=True)

    # Visual: distancias asignadas
    fig2 = px.bar(df_assign.sort_values("dist_km", ascending=False), x="municipio", y="dist_km",
                  title="Distancia vial (km) a la planta asignada")
    st.plotly_chart(fig2, use_container_width=True)

    st.info("Tip: para comparar '1 planta vs varias plantas', corre k=1 y luego k=2 o k=3 y compara los KPIs.")
else:
    st.warning("Configura municipios + RSU (PGIRS) y presiona **Calcular distancias + Optimizar**.")
