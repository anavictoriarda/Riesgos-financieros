"""
===========================================================================
MÓDULO XII · TAREA INTEGRADORA · LA CARTERA DEL BANCO — maquinaria
Riesgos Financieros (FIN516) · Universidad Panamericana · Gerardo Alcalá
===========================================================================

Cada equipo recibe 20 empresas de un pool de 50 (EUA, no financieras), saca
sus números de Bloomberg y, con los cuatro modelos del módulo, construye la
cartera de un banco que le presta 100 millones de dólares a cada una:

    Paso 1  Bloomberg      14 números por empresa + 3 de mercado (plantilla xlsx)
    Paso 2  Merton         de E y σ_E a V y σ_V; DD y PD; el ranking
    Paso 3  Altman Z''     puntaje, zona, holgura y covenant
    Paso 4  Duffie-Saita-Wang   λ y PD de hoy y de estrés
    Paso 5  Vasicek        reserva, capital, RORAC y spread mínimo; veredicto

Capa fina sobre empresas.py y altman.py: no recalcula nada que ya esté ahí.
    merton_implicito, merton, cadena_dd          (empresas.py)
    zpp_en_cadena, CORTES, ALTMAN_ZPP            (altman.py)
    intensidad_covariables, DSW_COEF             (empresas.py)
    vasicek_cuantil                              (empresas.py)

UNIDADES: montos en millones de USD; volatilidad, retornos y tasas EN POR
CIENTO en la tabla de Bloomberg (35 = 35 %); las funciones convierten.
"""
import math
import os
import numpy as np
import pandas as pd
from scipy.stats import norm

import empresas as E
import altman as A

NAVY, GOLD, STEEL, RED, MUTE = "#0B2545", "#B08D3C", "#5B7DA3", "#9E2A2B", "#6B7280"

# ===========================================================================
# PARÁMETROS FIJOS DE LA TAREA (nadie los cambia; se imprimen al inicio)
# ===========================================================================
PARAMETROS = {
    "T": 1.0,              # horizonte, años
    "mu": 0.05,            # deriva real de los activos
    "EAD": 100.0,          # exposición por empresa, millones USD
    "LGD": 0.45,
    "rho": 0.20,           # factor común
    "q": 0.999,            # confianza del capital (el peor año de mil)
    "fondeo_pb": 50,       # costo de fondeo sobre SOFR
    "gasto_pb": 50,        # gasto operativo sobre EAD
    "spread_pb": 250,      # spread inicial del crédito
    "hurdle": 0.15,        # RORAC objetivo
    "spread_max_pb": 500,  # arriba de esto el banco no presta sin garantía
    "PD_piso": 0.0005,     # piso de la PD (5 pb, el de Basilea III para corporativos)
    "estres": {"dDD": -2.0, "ret": -0.30, "r3m": 0.1, "sp500": -0.15},   # marzo de 2020, estilizado
}

# Números de mercado del día: los saca el profesor UNA vez, en por ciento.
MERCADO_CAMPOS = [
    ("r_1a",   "USGG12M Index", "PX_LAST",     "T-bill a 1 año, % (Merton: r)"),
    ("r3m",    "USGG3M Index",  "PX_LAST",     "T-bill a 3 meses, % (DSW)"),
    ("sp500",  "SPX Index",     "CHG_PCT_1YR", "retorno 12 m del S&P 500, % (DSW)"),
]

# ===========================================================================
# EL POOL DE 50 Y LOS 9 EQUIPOS
# ===========================================================================
POOL = pd.DataFrame([
    ("MSFT", "Microsoft", "tecnología"), ("AAPL", "Apple", "tecnología"), ("CSCO", "Cisco", "tecnología"),
    ("JNJ", "Johnson & Johnson", "salud"), ("KO", "Coca-Cola", "consumo"), ("PG", "Procter & Gamble", "consumo"),
    ("PEP", "PepsiCo", "consumo"), ("WMT", "Walmart", "comercio"), ("HD", "Home Depot", "comercio"),
    ("MCD", "McDonald's", "restaurantes"), ("XOM", "ExxonMobil", "energía"), ("CVX", "Chevron", "energía"),
    ("CAT", "Caterpillar", "industrial"), ("DE", "Deere", "industrial"), ("GE", "GE Aerospace", "industrial"),
    ("MAR", "Marriott", "hoteles"), ("F", "Ford", "autos"), ("GM", "General Motors", "autos"),
    ("TSLA", "Tesla", "autos"), ("T", "AT&T", "telecom"), ("VZ", "Verizon", "telecom"),
    ("CHTR", "Charter", "medios"), ("INTC", "Intel", "tecnología"), ("ORCL", "Oracle", "tecnología"),
    ("IBM", "IBM", "tecnología"), ("KHC", "Kraft Heinz", "consumo"), ("GIS", "General Mills", "consumo"),
    ("NKE", "Nike", "comercio"), ("SBUX", "Starbucks", "restaurantes"), ("UAL", "United Airlines", "aerolíneas"),
    ("DAL", "Delta", "aerolíneas"), ("RCL", "Royal Caribbean", "cruceros"), ("CVS", "CVS Health", "salud"),
    ("PFE", "Pfizer", "salud"), ("HAL", "Halliburton", "energía"), ("OXY", "Occidental", "energía"),
    ("DVN", "Devon", "energía"), ("AA", "Alcoa", "materiales"), ("FCX", "Freeport-McMoRan", "materiales"),
    ("AAL", "American Airlines", "aerolíneas"), ("JBLU", "JetBlue", "aerolíneas"), ("CCL", "Carnival", "cruceros"),
    ("NCLH", "Norwegian Cruise", "cruceros"), ("KSS", "Kohl's", "comercio"), ("M", "Macy's", "comercio"),
    ("LUMN", "Lumen", "telecom"), ("WBD", "Warner Bros. Discovery", "medios"), ("XRX", "Xerox", "industrial"),
    ("BA", "Boeing", "aeroespacial"), ("WYNN", "Wynn Resorts", "juego"),
], columns=["ticker", "empresa", "sector"])

# 9 equipos × 20 empresas: cada empresa cae en 3 a 5 equipos (para cotejar).
EQUIPOS = {
    1: ["AAPL", "PEP", "WMT", "XOM", "CAT", "MAR", "F", "T", "ORCL", "KHC", "NKE", "SBUX", "UAL", "HAL", "AA", "AAL", "JBLU", "NCLH", "LUMN", "WBD"],
    2: ["CSCO", "JNJ", "KO", "PG", "CVX", "DE", "TSLA", "VZ", "CHTR", "IBM", "GIS", "RCL", "CVS", "PFE", "FCX", "CCL", "M", "XRX", "BA", "WYNN"],
    3: ["MSFT", "PG", "HD", "MCD", "DE", "GE", "F", "GM", "INTC", "DAL", "RCL", "HAL", "OXY", "DVN", "AA", "CCL", "KSS", "M", "XRX", "WYNN"],
    4: ["MSFT", "JNJ", "KO", "HD", "MCD", "CVX", "TSLA", "VZ", "CHTR", "IBM", "GIS", "DAL", "CVS", "PFE", "FCX", "AAL", "JBLU", "KSS", "LUMN", "WBD"],
    5: ["AAPL", "PEP", "XOM", "CAT", "GE", "MAR", "GM", "T", "INTC", "ORCL", "KHC", "SBUX", "UAL", "OXY", "DVN", "AAL", "JBLU", "NCLH", "LUMN", "BA"],
    6: ["AAPL", "CSCO", "PEP", "WMT", "XOM", "GE", "GM", "CHTR", "INTC", "GIS", "NKE", "DAL", "OXY", "DVN", "FCX", "NCLH", "M", "XRX", "BA", "WYNN"],
    7: ["CSCO", "PG", "WMT", "CAT", "DE", "MAR", "F", "T", "ORCL", "KHC", "NKE", "SBUX", "UAL", "HAL", "AA", "CCL", "KSS", "WBD", "XRX", "WYNN"],
    8: ["MSFT", "JNJ", "KO", "HD", "MCD", "CVX", "GM", "TSLA", "T", "VZ", "IBM", "RCL", "CVS", "PFE", "DVN", "AAL", "JBLU", "KSS", "LUMN", "WBD"],
    9: ["MSFT", "CSCO", "JNJ", "KO", "HD", "CVX", "F", "ORCL", "KHC", "NKE", "SBUX", "UAL", "RCL", "HAL", "AA", "CCL", "NCLH", "M", "LUMN", "BA"],
}

# ===========================================================================
# LOS 14 NÚMEROS DE BLOOMBERG (columna de la tabla, campo BDP, qué es)
# ===========================================================================
CAMPOS = [
    ("E",                    "CUR_MKT_CAP",                 "capitalización de mercado, millones USD",   "Merton"),
    ("sigma_E_pct",          "VOLATILITY_260D",             "volatilidad realizada 260 días, %",          "Merton"),
    ("deuda_cp",             "BS_ST_BORROW",                "deuda de corto plazo, millones",             "Merton"),
    ("deuda_lp",             "BS_LT_BORROW",                "deuda de largo plazo, millones",             "Merton"),
    ("activo_circulante",    "BS_CUR_ASSET_REPORT",         "activo circulante, millones",                "Altman X1"),
    ("pasivo_circulante",    "BS_CUR_LIAB",                 "pasivo circulante, millones",                "Altman X1"),
    ("activo_total",         "BS_TOT_ASSET",                "activo total, millones",                     "Altman X1-X3"),
    ("utilidades_retenidas", "BS_RETAIN_EARN",              "utilidades retenidas, millones",             "Altman X2"),
    ("ebit",                 "EBIT",                        "EBIT último año fiscal, millones",           "Altman X3"),
    ("capital_contable",     "TOT_COMMON_EQY",              "capital contable, millones",                 "Altman X4"),
    ("pasivo_total",         "BS_TOT_LIAB2",                "pasivo total, millones",                     "Altman X4"),
    ("ret_12m_pct",          "CHG_PCT_1YR",                 "retorno de la acción 12 meses, %",           "DSW"),
    ("pd_drsk_pct",          "BB_1YR_DEFAULT_PROB",         "PD a 1 año de Bloomberg (DRSK), %",          "comparación"),
    ("rating",               "RTG_SP_LT_LC_ISSUER_CREDIT",  "calificación S&P del emisor, largo plazo",   "comparación"),
]
COLUMNAS = ["ticker"] + [c[0] for c in CAMPOS]
NUMERICAS = [c[0] for c in CAMPOS if c[0] != "rating"]
EN_MILLONES = ["E", "deuda_cp", "deuda_lp", "activo_circulante", "pasivo_circulante", "activo_total",
               "utilidades_retenidas", "ebit", "capital_contable", "pasivo_total"]

# DATOS DE EJEMPLO, INVENTADOS, para correr el cuaderno de punta a punta ANTES
# de tener Bloomberg. Seis empresas ficticias con balances estilizados (una con
# capital contable negativo, una en problemas). No son empresas reales.
EJEMPLO = pd.DataFrame([
    ("ALFA",    400_000, 22,  10_000,  60_000, 120_000,  90_000, 350_000,  80_000, 60_000, 110_000, 240_000,  15, 0.05, "AA-"),
    ("BRAVO",     8_000, 55,   4_000,  30_000,  12_000,  20_000,  60_000,  -8_000,  2_500,  -3_000,  63_000, -25, 4.50, "B"),
    ("CHARLIE",  45_000, 35,  50_000, 100_000, 110_000,  95_000, 270_000,  30_000,  8_000,  45_000, 225_000,   5, 0.60, "BBB-"),
    ("DELTA",   120_000, 20,   8_000, 130_000,  40_000,  50_000, 400_000,  20_000, 25_000, 110_000, 290_000,  10, 0.30, "BBB"),
    ("ECO",       1_500, 70,     500,   4_000,   4_000,   3_500,  10_000,  -2_000,   -100,   2_000,   8_000, -45, 9.00, "CCC+"),
    ("FOX",     150_000, 30,   3_000,  25_000,  40_000,  30_000, 200_000,  90_000, 30_000, 120_000,  80_000, -10, 0.10, "A"),
], columns=COLUMNAS)
MERCADO_EJEMPLO = {"r_1a": 4.0, "r3m": 4.2, "sp500": 12.0}


# ===========================================================================
# PASO 1 · BLOOMBERG: la plantilla y la carga
# ===========================================================================
def equipo(n, verbose=True):
    """Las 20 empresas del equipo n, con nombre y sector."""
    if n not in EQUIPOS:
        raise ValueError(f"el equipo debe ser 1..9; recibí {n}")
    t = POOL.set_index("ticker").loc[EQUIPOS[n]].reset_index()
    if verbose:
        print(f"EQUIPO {n} · {len(t)} empresas")
        print("   " + " · ".join(t["ticker"]))
    return t


def plantilla_bloomberg(n, ruta=None, verbose=True):
    """Escribe bloomberg_equipo_N.xlsx con los 20 tickers del equipo y una fórmula
    =BDP(...) en cada celda. Se abre en el Excel de la terminal (complemento
    Bloomberg), se calculan las fórmulas, se GUARDA y se sube a Colab. La hoja
    «mercado» trae los tres números del día. NO se editan los encabezados."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill
    ruta = ruta or f"bloomberg_equipo_{n}.xlsx"
    tk = EQUIPOS[n]
    wb = Workbook()
    ws = wb.active
    ws.title = "datos"
    ws.append(["ticker"] + [c[0] for c in CAMPOS])
    ws.append(["(campo BDP)"] + [c[1] for c in CAMPOS])
    ws.append(["(qué es)"] + [c[2] for c in CAMPOS])
    for i, t in enumerate(tk, start=4):
        ws.cell(row=i, column=1, value=f"{t} US Equity")
        for j, (col, campo, _, _) in enumerate(CAMPOS, start=2):
            ws.cell(row=i, column=j, value=f'=BDP($A{i},"{campo}")')
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="0B2545")
    for row in (ws[2], ws[3]):
        for c in row:
            c.font = Font(italic=True, color="6B7280")
    ws.column_dimensions["A"].width = 18
    wm = wb.create_sheet("mercado")
    wm.append(["clave", "ticker", "campo", "valor", "qué es"])
    for k, tick, campo, que in MERCADO_CAMPOS:
        wm.append([k, tick, campo, f'=BDP("{tick}","{campo}")', que])
    for c in wm[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="0B2545")
    wi = wb.create_sheet("instrucciones")
    for linea in [
        "1. Abran este archivo en el Excel de la terminal de Bloomberg (complemento activo).",
        "2. Esperen a que las fórmulas =BDP se llenen (F9 si hace falta). Un #N/A: busquen el campo con FLDS <GO>.",
        "3. Revisen unidades: todo en millones de USD; volatilidad y retornos en %.",
        "4. GUARDEN el archivo (Ctrl+S) con las fórmulas ya calculadas y súbanlo a Colab.",
        "5. En el cuaderno: datos, mercado = C.cargar_datos('bloomberg_equipo_N.xlsx').",
        "No cambien los encabezados de la fila 1: el cuaderno los usa para leer las columnas.",
    ]:
        wi.append([linea])
    wb.save(ruta)
    if verbose:
        print(f"plantilla escrita: {ruta}  ({len(tk)} empresas, {len(CAMPOS)} campos + hoja «mercado»)")
        print("   ábranla en el Excel de la terminal, dejen que se calcule, guarden y súbanla a Colab")
    return ruta


def _a_numero(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return np.nan
    if isinstance(x, str):
        s = x.strip().replace(",", "").replace("%", "")
        if s in ("", "#N/A", "N/A", "#N/A N/A", "#N/A Field Not Applicable", "#NAME?"):
            return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    return float(x)


def cargar_datos(ruta, hoja="datos", verbose=True):
    """Lee la plantilla ya calculada (xlsx guardado desde el Excel de Bloomberg)
    o un CSV con las mismas columnas. Devuelve (datos, mercado):
    datos = DataFrame con COLUMNAS; mercado = dict r_1a, r3m, sp500 en %.
    Corrige unidades obvias (dólares → millones; σ en fracción → %) y lo dice.
    REVIENTA si faltan columnas o si las fórmulas no se calcularon."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"no encuentro {ruta}: súbanlo a la carpeta de Colab (ícono de carpeta → subir)")
    mercado = {}
    if ruta.lower().endswith(".xlsx"):
        raw = pd.read_excel(ruta, sheet_name=hoja, header=0)
        try:
            m = pd.read_excel(ruta, sheet_name="mercado")
            for _, r in m.iterrows():
                mercado[str(r["clave"]).strip()] = _a_numero(r["valor"])
        except Exception:
            pass
    else:
        raw = pd.read_csv(ruta)
    raw.columns = [str(c).strip() for c in raw.columns]
    faltan = [c for c in COLUMNAS if c not in raw.columns]
    if faltan:
        raise ValueError(f"{ruta} necesita las columnas {COLUMNAS}; faltan {faltan}")
    d = raw[COLUMNAS].copy()
    d["ticker"] = d["ticker"].astype(str).str.strip()
    d = d[~d["ticker"].str.startswith("(")]                       # quita las filas de ayuda de la plantilla
    d = d[d["ticker"].str.len() > 0].copy()
    d["ticker"] = d["ticker"].str.replace(r"\s+(US|UN|UW|UQ)?\s*Equity$", "", regex=True).str.strip()
    for c in NUMERICAS:
        d[c] = d[c].map(_a_numero)
    d["rating"] = d["rating"].astype(str).str.strip().replace({"nan": "", "#N/A": ""})
    d = d.reset_index(drop=True)
    notas = []
    if d["E"].isna().all():
        raise ValueError("todas las E están vacías: las fórmulas =BDP no se calcularon. Abran el archivo en el Excel de la terminal, esperen, guarden y vuelvan a subirlo.")
    # unidades: millones. Si la mediana del activo total pasa de 20 millones de millones, está en dólares.
    if d["activo_total"].median() > 2e7:
        for c in EN_MILLONES:
            d[c] = d[c] / 1e6
        notas.append("los montos venían en dólares: se dividieron entre 1,000,000 (ahora millones)")
    if d["sigma_E_pct"].median() < 1.5:
        d["sigma_E_pct"] = d["sigma_E_pct"] * 100
        notas.append("la volatilidad venía en fracción: se multiplicó por 100 (ahora en %)")
    if verbose:
        print(f"datos cargados de {ruta}: {len(d)} empresas · " + " · ".join(d["ticker"]))
        for n in notas:
            print("   AVISO:", n)
        vacios = {c: int(d[c].isna().sum()) for c in NUMERICAS if d[c].isna().any()}
        if vacios:
            print("   celdas vacías (#N/A) por columna:", vacios, "→ esas empresas saldrán con NaN en el modelo que las use")
        if mercado:
            print("   mercado:", ", ".join(f"{k} = {v:g} %" for k, v in mercado.items() if v == v))
    return d, mercado


def revisar(datos, verbose=True):
    """Chequeos de unidades antes de modelar: razón E / capital contable,
    σ_E en %, deuda contra activo. Imprime lo raro; no corrige."""
    d = datos
    avisos = []
    for _, r in d.iterrows():
        if r["capital_contable"] > 0 and (r["E"] / r["capital_contable"] > 60 or r["E"] / r["capital_contable"] < 0.1):
            avisos.append(f"{r['ticker']}: E / capital contable = {r['E'] / r['capital_contable']:.1f}; revisen unidades (E en millones, balance en millones)")
        if r["capital_contable"] < 0:
            avisos.append(f"{r['ticker']}: capital contable NEGATIVO ({r['capital_contable']:,.0f}); X4 será negativo, el covenant no puede ir sobre X4")
        if r["sigma_E_pct"] > 150 or r["sigma_E_pct"] < 8:
            avisos.append(f"{r['ticker']}: σ_E = {r['sigma_E_pct']:.0f} %; fuera del rango usual (8–150 %)")
        if r["deuda_cp"] + r["deuda_lp"] > r["activo_total"]:
            avisos.append(f"{r['ticker']}: la deuda supera el activo total; revisen BS_ST_BORROW / BS_LT_BORROW")
    if verbose:
        print("REVISIÓN DE UNIDADES")
        if avisos:
            for a in avisos:
                print("   ·", a)
        else:
            print("   nada raro: E, σ_E y deuda en rangos usuales")
    return avisos


# ===========================================================================
# PASO 2 · MERTON
# ===========================================================================
def merton_cartera(datos, mercado, P=PARAMETROS, verbose=True):
    """De E y σ_E a V y σ_V (dos ecuaciones), DD y PD con deriva μ, para cada
    empresa. Devuelve la tabla ordenada por DD (menor = más riesgosa)."""
    r = mercado["r_1a"] / 100
    filas = []
    for _, x in datos.iterrows():
        D = x["deuda_cp"] + 0.5 * x["deuda_lp"]                    # punto de default de KMV
        fila = {"ticker": x["ticker"], "E": x["E"], "sigma_E": x["sigma_E_pct"] / 100, "D": D}
        try:
            if not (D > 0 and x["E"] > 0 and x["sigma_E_pct"] > 0):
                raise ValueError("E, σ_E o D no positivos")
            m = E.merton_implicito(x["E"], x["sigma_E_pct"] / 100, D, T=P["T"], r=r, verbose=False)
            mp = E.merton(m["V"], D, m["sigma_V"], T=P["T"], r=r, mu=P["mu"], verbose=False)
            fila.update({"V": m["V"], "sigma_V": m["sigma_V"], "DD": mp["DD"], "PD_merton": mp["PD_P"]})
        except Exception as e:
            fila.update({"V": np.nan, "sigma_V": np.nan, "DD": np.nan, "PD_merton": np.nan, "nota": str(e)[:50]})
        fila["PD_drsk"] = x["pd_drsk_pct"] / 100 if x["pd_drsk_pct"] == x["pd_drsk_pct"] else np.nan
        fila["rating"] = x["rating"]
        filas.append(fila)
    t = pd.DataFrame(filas).sort_values("DD").reset_index(drop=True)
    t.index = t.index + 1
    t.index.name = "rango"
    if verbose:
        print(f"MERTON · r = {mercado['r_1a']:.2f} % · μ = {P['mu']:.0%} · T = {P['T']:g} · D = corto + ½ largo")
        print(_fmt(t[["ticker", "E", "sigma_E", "D", "V", "sigma_V", "DD", "PD_merton", "PD_drsk", "rating"]],
                   {"E": "{:,.0f}", "D": "{:,.0f}", "V": "{:,.0f}", "sigma_E": "{:.0%}", "sigma_V": "{:.0%}",
                    "DD": "{:.2f}", "PD_merton": "{:.2%}", "PD_drsk": "{:.2%}"}))
        if t["sigma_V"].gt(t["sigma_E"]).any():
            print("   AVISO: alguna σ_V salió mayor que σ_E; revisen unidades de E o σ_E en esa empresa")
    return t


def merton_detalle(datos, ticker, mercado, P=PARAMETROS):
    """La cadena completa de UNA empresa, con los números a la vista."""
    x = datos.set_index("ticker").loc[ticker]
    D = x["deuda_cp"] + 0.5 * x["deuda_lp"]
    r = mercado["r_1a"] / 100
    print(f"{ticker} · punto de default D = {x['deuda_cp']:,.0f} + ½·{x['deuda_lp']:,.0f} = {D:,.0f} millones")
    res = E.merton_implicito(x["E"], x["sigma_E_pct"] / 100, D, T=P["T"], r=r, verbose="simple")
    print()
    return E.cadena_dd(res["V"], D, res["sigma_V"], T=P["T"], mu=P["mu"], etiqueta=f"{ticker} · la cadena de la DD (μ = {P['mu']:.0%})")


# ===========================================================================
# PASO 3 · ALTMAN Z'' Y EL COVENANT
# ===========================================================================
PESOS = {k: A.ALTMAN_ZPP[k] for k in ("X1", "X2", "X3", "X4")}
NOMBRE_RAZON = {"X1": "capital de trabajo / activo total", "X3": "EBIT / activo total", "X4": "capital contable / pasivo total"}


def covenant(z, verbose=False):
    """La regla mecánica del covenant a partir del resultado de zpp_en_cadena.

    1. La razón que SOSTIENE el Z'' (aporte positivo más grande), sin X2
       (utilidades retenidas: es historia acumulada, no se pacta) y sin razones
       negativas (si X4 < 0, el covenant no puede ir sobre X4).
    2. Holgura = cuánto puede caer esa razón, sola, antes de cruzar 5.85:
           holgura_k = (Z'' − 5.85) / peso_k
    3. Nivel del covenant = punto medio entre hoy y el cruce; si la empresa ya
       está bajo 5.85 (holgura negativa), el nivel es el que la regresa a 5.85.
    """
    X, ap, Z = z["X"], z["aportes"], z["Z"]
    candidatas = {k: ap[k] for k in ("X1", "X3", "X4") if ap[k] > 0}
    if not candidatas:
        return {"razon": None, "holgura": np.nan, "nivel": np.nan, "covenant": "sin razón positiva que pactar: sólo con garantía"}
    k = max(candidatas, key=candidatas.get)
    holgura = (Z - A.CORTES["segura"]) / PESOS[k]
    cruce = X[k] - holgura
    nivel = (X[k] + cruce) / 2 if holgura >= 0 else cruce
    tipo = "mantenimiento" if holgura >= 0 else "recuperación"
    if k == "X4":
        texto = f"pasivo total / capital contable ≤ {1 / nivel:.2f}" if nivel > 0 else "sin covenant sobre X4"
    else:
        texto = f"{NOMBRE_RAZON[k]} ≥ {nivel:.3f}"
    out = {"razon": k, "holgura": holgura, "nivel": nivel, "covenant": f"{texto} ({tipo})"}
    if verbose:
        print(f"   la sostiene {k} ({NOMBRE_RAZON[k]}): {X[k]:+.3f}, aporte {ap[k]:+.2f}")
        print(f"   holgura = (Z'' − 5.85) / {PESOS[k]} = ({Z:.2f} − 5.85) / {PESOS[k]} = {holgura:+.3f}  →  cruza 5.85 en {k} = {cruce:+.3f}")
        print(f"   covenant de {tipo}: {texto}")
    return out


def altman_cartera(datos, verbose=True):
    """Z'', zona, aportes, la razón que sostiene y el covenant, para cada empresa."""
    filas = []
    for _, x in datos.iterrows():
        try:
            z = A.zpp_en_cadena(x, verbose=False)
            c = covenant(z)
            filas.append({"ticker": x["ticker"], **z["X"], "Z''": z["Z"], "zona": z["zona"],
                          "la hunde": min(z["aportes"], key=z["aportes"].get),
                          "la sostiene": c["razon"], "holgura": c["holgura"], "nivel": c["nivel"], "covenant": c["covenant"]})
        except Exception as e:
            filas.append({"ticker": x["ticker"], "Z''": np.nan, "zona": "sin datos", "covenant": str(e)[:40]})
    t = pd.DataFrame(filas).sort_values("Z''").reset_index(drop=True)
    t.index = t.index + 1
    t.index.name = "rango"
    if verbose:
        print(f"ALTMAN Z'' · zonas: distress < {A.CORTES['distress']:.2f} · gris · segura > {A.CORTES['segura']:.2f}")
        print(_fmt(t[["ticker", "X1", "X2", "X3", "X4", "Z''", "zona", "la hunde", "la sostiene", "holgura", "covenant"]],
                   {"X1": "{:+.3f}", "X2": "{:+.3f}", "X3": "{:+.3f}", "X4": "{:+.3f}", "Z''": "{:.2f}", "holgura": "{:+.3f}"}))
    return t


def altman_detalle(datos, ticker):
    """El Z'' en cadena de UNA empresa y la regla del covenant explicada."""
    x = datos.set_index("ticker").loc[ticker].copy()
    x["ticker"] = ticker
    z = A.zpp_en_cadena(x, verbose=True)
    print("EL COVENANT")
    return covenant(z, verbose=True)


# ===========================================================================
# PASO 4 · DUFFIE, SAITA Y WANG: PD base y de estrés
# ===========================================================================
def dsw_cartera(tabla_merton, datos, mercado, P=PARAMETROS, verbose=True):
    """λ y PD de hoy (lecturas observadas) y de estrés (marzo de 2020 estilizado)
    para cada empresa. Entrada: la tabla de merton_cartera (por la DD)."""
    est = P["estres"]
    ret = datos.set_index("ticker")["ret_12m_pct"] / 100
    filas = []
    for _, x in tabla_merton.iterrows():
        if x["DD"] != x["DD"]:
            filas.append({"ticker": x["ticker"], "DD": np.nan}); continue
        b = E.intensidad_covariables(x["DD"], ret=ret[x["ticker"]], r3m=mercado["r3m"], sp500=mercado["sp500"] / 100, verbose=False)
        s = E.intensidad_covariables(x["DD"] + est["dDD"], ret=est["ret"], r3m=est["r3m"], sp500=est["sp500"], verbose=False)
        filas.append({"ticker": x["ticker"], "DD": x["DD"], "ret_12m": ret[x["ticker"]],
                      "lambda_base": b["lambda"], "PD_base": b["PD_1a"],
                      "lambda_estres": s["lambda"], "PD_estres": s["PD_1a"], "cociente": s["PD_1a"] / b["PD_1a"]})
    t = pd.DataFrame(filas)
    t.index = t.index + 1
    t.index.name = "rango"
    if verbose:
        print(f"DUFFIE-SAITA-WANG · base: r3m = {mercado['r3m']:.2f} %, S&P 12 m = {mercado['sp500']:+.1f} % · "
              f"estrés: DD {est['dDD']:+.0f}, acción {est['ret']:+.0%}, r3m {est['r3m']} %, S&P {est['sp500']:+.0%}")
        print(_fmt(t, {"DD": "{:.2f}", "ret_12m": "{:+.0%}", "lambda_base": "{:.4f}", "PD_base": "{:.2%}",
                       "lambda_estres": "{:.4f}", "PD_estres": "{:.2%}", "cociente": "×{:.1f}"}))
    return t


def dsw_detalle(tabla_dsw, ticker, datos, mercado, P=PARAMETROS):
    """La λ en cadena de UNA empresa, base y estrés."""
    import reducida as R
    x = tabla_dsw.set_index("ticker").loc[ticker]
    est = P["estres"]
    R.lambda_en_cadena(x["DD"], ret=x["ret_12m"], r3m=mercado["r3m"], sp500=mercado["sp500"] / 100, nombre=f"{ticker} · base")
    print()
    return R.lambda_en_cadena(x["DD"] + est["dDD"], ret=est["ret"], r3m=est["r3m"], sp500=est["sp500"], nombre=f"{ticker} · estrés")


# ===========================================================================
# PASO 5 · VASICEK: reserva, capital, RORAC, spread mínimo y veredicto
# ===========================================================================
def credito(PD, P=PARAMETROS):
    """Un crédito de EAD con PD: reserva (EL), PD del año malo, capital (K),
    margen al spread inicial, RORAC y spread mínimo para el hurdle."""
    PDc = min(max(PD, P["PD_piso"]), 0.99)
    v = E.vasicek_cuantil(PDc, P["rho"], q=P["q"], LGD=P["LGD"], verbose=False)
    EL = PDc * P["LGD"] * P["EAD"]
    K = v["capital"] * P["EAD"]
    margen = (P["spread_pb"] - P["fondeo_pb"] - P["gasto_pb"]) / 1e4 * P["EAD"]
    rorac = (margen - EL) / K if K > 0 else np.nan
    spread_min = ((P["hurdle"] * K + EL) / P["EAD"]) * 1e4 + P["fondeo_pb"] + P["gasto_pb"]
    return {"EL": EL, "PD_malo": v["PD_q"], "K": K, "margen": margen, "RORAC": rorac, "spread_min_pb": spread_min}


def veredicto(rorac, spread_min, P=PARAMETROS):
    if rorac >= P["hurdle"]:
        return f"sí, a {P['spread_pb']:.0f} pb"
    if spread_min <= P["spread_max_pb"]:
        return f"sí, a {spread_min:.0f} pb"
    return "no sin garantía"


def vasicek_cartera(tabla_dsw, P=PARAMETROS, verbose=True):
    """Para cada crédito: EL, K, RORAC al spread inicial, spread mínimo y
    veredicto, en base; EL y K en estrés."""
    filas = []
    for _, x in tabla_dsw.iterrows():
        if x["DD"] != x["DD"]:
            filas.append({"ticker": x["ticker"]}); continue
        b = credito(x["PD_base"], P)
        s = credito(x["PD_estres"], P)
        filas.append({"ticker": x["ticker"], "PD_base": x["PD_base"], "EL": b["EL"], "PD_malo": b["PD_malo"], "K": b["K"],
                      "RORAC": b["RORAC"], "spread_min_pb": b["spread_min_pb"],
                      "veredicto": veredicto(b["RORAC"], b["spread_min_pb"], P),
                      "PD_estres": x["PD_estres"], "EL_estres": s["EL"], "K_estres": s["K"]})
    t = pd.DataFrame(filas)
    t.index = t.index + 1
    t.index.name = "rango"
    if verbose:
        print(f"VASICEK · EAD {P['EAD']:.0f} · LGD {P['LGD']:.0%} · ρ {P['rho']:.0%} · el peor año de {1 / (1 - P['q']):,.0f} · "
              f"spread {P['spread_pb']} pb − fondeo {P['fondeo_pb']} − gasto {P['gasto_pb']} · hurdle {P['hurdle']:.0%} · piso de PD {P['PD_piso']:.2%}")
        print(_fmt(t, {"PD_base": "{:.2%}", "EL": "{:.2f}", "PD_malo": "{:.1%}", "K": "{:.2f}", "RORAC": "{:.1%}",
                       "spread_min_pb": "{:.0f}", "PD_estres": "{:.1%}", "EL_estres": "{:.2f}", "K_estres": "{:.2f}"}))
    return t


def credito_detalle(tabla_vasicek, ticker, P=PARAMETROS):
    """La cadena del crédito de UNA empresa, con los números a la vista."""
    x = tabla_vasicek.set_index("ticker").loc[ticker]
    PD = x["PD_base"]
    a, zq = norm.ppf(PD), norm.ppf(P["q"])
    print(f"{ticker} · PD base = {PD:.2%} · EAD {P['EAD']:.0f} · LGD {P['LGD']:.0%} · ρ {P['rho']:.0%}")
    print(f"   reserva      EL = PD × LGD × EAD = {PD:.4f} × {P['LGD']} × {P['EAD']:.0f} = {x['EL']:.2f}")
    print(f"   PD año malo  Φ((Φ⁻¹(PD) + √ρ·Φ⁻¹({P['q']})) / √(1−ρ)) = Φ(({a:.2f} + {math.sqrt(P['rho']):.3f}·{zq:.2f}) / {math.sqrt(1 - P['rho']):.3f}) = {x['PD_malo']:.1%}")
    print(f"   capital      K = (LGD·PD_malo − LGD·PD) × EAD = ({P['LGD']}·{x['PD_malo']:.4f} − {P['LGD']}·{PD:.4f}) × {P['EAD']:.0f} = {x['K']:.2f}")
    margen = (P["spread_pb"] - P["fondeo_pb"] - P["gasto_pb"]) / 1e4 * P["EAD"]
    print(f"   margen       ({P['spread_pb']} − {P['fondeo_pb']} − {P['gasto_pb']}) pb × {P['EAD']:.0f} = {margen:.2f}")
    print(f"   RORAC        (margen − EL) / K = ({margen:.2f} − {x['EL']:.2f}) / {x['K']:.2f} = {x['RORAC']:.1%}   (hurdle {P['hurdle']:.0%})")
    print(f"   spread mín.  (hurdle·K + EL) / EAD + 100 pb = ({P['hurdle']}·{x['K']:.2f} + {x['EL']:.2f}) / {P['EAD']:.0f} + 100 pb = {x['spread_min_pb']:.0f} pb")
    print(f"   veredicto    {x['veredicto']}")
    print(f"   estrés       PD {x['PD_estres']:.1%} → EL {x['EL_estres']:.2f} (×{x['EL_estres'] / x['EL']:.1f}) · K {x['K_estres']:.2f} (×{x['K_estres'] / x['K']:.1f})")


# ===========================================================================
# CIERRE · la cartera completa, el resumen, las gráficas y la exportación
# ===========================================================================
def cartera_completa(datos, mercado, P=PARAMETROS, verbose=False):
    """Corre los cuatro pasos y junta todo en una tabla por empresa."""
    tm = merton_cartera(datos, mercado, P, verbose=verbose)
    ta = altman_cartera(datos, verbose=verbose)
    td = dsw_cartera(tm, datos, mercado, P, verbose=verbose)
    tv = vasicek_cartera(td, P, verbose=verbose)
    t = (tm[["ticker", "DD", "PD_merton", "PD_drsk", "rating"]]
         .merge(ta[["ticker", "Z''", "zona", "covenant"]], on="ticker", how="left")
         .merge(td[["ticker", "PD_base", "PD_estres"]], on="ticker", how="left")
         .merge(tv[["ticker", "EL", "K", "RORAC", "spread_min_pb", "veredicto", "EL_estres", "K_estres"]], on="ticker", how="left"))
    t.index = t.index + 1
    t.index.name = "rango"
    return t


def resumen(tabla, P=PARAMETROS, verbose=True):
    """Totales de la cartera: reserva y capital base contra estrés; veredictos."""
    n = int(tabla["EL"].notna().sum())
    tot = {"cartera": n * P["EAD"], "EL": tabla["EL"].sum(), "K": tabla["K"].sum(),
           "EL_estres": tabla["EL_estres"].sum(), "K_estres": tabla["K_estres"].sum(),
           "veredictos": tabla["veredicto"].dropna().map(_clase_veredicto).value_counts().to_dict()}
    if verbose:
        print(f"LA CARTERA · {n} créditos de {P['EAD']:.0f} = {tot['cartera']:,.0f} millones")
        print(f"   base     reserva {tot['EL']:7.1f} ({tot['EL'] / tot['cartera']:.2%})   capital {tot['K']:7.1f} ({tot['K'] / tot['cartera']:.2%})")
        print(f"   estrés   reserva {tot['EL_estres']:7.1f} (×{tot['EL_estres'] / tot['EL']:.1f})     capital {tot['K_estres']:7.1f} (×{tot['K_estres'] / tot['K']:.1f})")
        print("   la reserva paga el año promedio; el capital, el año malo. En estrés revienta primero la reserva.")
        print("   veredictos:", ", ".join(f"{k}: {v}" for k, v in tot["veredictos"].items()))
        peor = tabla.dropna(subset=["DD"]).iloc[0]
        mejor = tabla.dropna(subset=["DD"]).iloc[-1]
        print(f"   a quién mirar primero: {peor['ticker']} (DD {peor['DD']:.2f}) · el ancla: {mejor['ticker']} (DD {mejor['DD']:.2f})")
    return tot


def _clase_veredicto(v, P=PARAMETROS):
    if v.startswith(f"sí, a {P['spread_pb']:.0f} pb"):
        return f"sí a {P['spread_pb']:.0f} pb"
    return "sí al spread mínimo" if v.startswith("sí") else "no sin garantía"


FORMATOS = {"E": "{:,.0f}", "D": "{:,.0f}", "V": "{:,.0f}", "sigma_E": "{:.0%}", "sigma_V": "{:.0%}", "DD": "{:.2f}",
            "PD_merton": "{:.2%}", "PD_drsk": "{:.2%}", "X1": "{:+.3f}", "X2": "{:+.3f}", "X3": "{:+.3f}", "X4": "{:+.3f}",
            "Z''": "{:.2f}", "holgura": "{:+.3f}", "nivel": "{:.3f}", "ret_12m": "{:+.0%}", "lambda_base": "{:.4f}",
            "lambda_estres": "{:.4f}", "PD_base": "{:.2%}", "PD_estres": "{:.1%}", "cociente": "×{:.1f}", "EL": "{:.2f}",
            "PD_malo": "{:.1%}", "K": "{:.2f}", "RORAC": "{:.0%}", "spread_min_pb": "{:.0f}", "EL_estres": "{:.2f}", "K_estres": "{:.2f}"}


def mostrar(tabla, columnas=None):
    """La tabla con formato legible (porcentajes, miles), para la celda del cuaderno."""
    t = tabla if columnas is None else tabla[columnas]
    s = t.copy()
    for c, f in FORMATOS.items():
        if c in s.columns:
            s[c] = s[c].map(lambda v: f.format(v) if v == v else "—")
    return s


def graficas(tabla, titulo=""):
    """Dos gráficas: el ranking por DD y la reserva contra el capital, base y estrés."""
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    t = tabla.dropna(subset=["DD"]).sort_values("DD")
    fig, ax = plt.subplots(1, 2, figsize=(11, 0.28 * len(t) + 2.2))
    cols = [RED if d < 2 else (GOLD if d < 4 else NAVY) for d in t["DD"]]
    ax[0].barh(t["ticker"], t["DD"], color=cols)
    ax[0].axvline(2, color=RED, ls="--", lw=1); ax[0].axvline(4, color=GOLD, ls="--", lw=1)
    ax[0].invert_yaxis(); ax[0].set_xlabel("distancia al default, sacudidas")
    ax[0].set_title("a quién mirar primero: el ranking por DD" + (f" · {titulo}" if titulo else ""), fontsize=10, color=NAVY)
    y = np.arange(len(t)); h = 0.2
    ax[1].barh(y - 1.5 * h, t["EL"], h, color=STEEL, label="reserva base")
    ax[1].barh(y - 0.5 * h, t["K"], h, color=NAVY, label="capital base")
    ax[1].barh(y + 0.5 * h, t["EL_estres"], h, color=GOLD, label="reserva estrés")
    ax[1].barh(y + 1.5 * h, t["K_estres"], h, color=RED, label="capital estrés")
    ax[1].set_yticks(y); ax[1].set_yticklabels(t["ticker"]); ax[1].invert_yaxis()
    ax[1].set_xlabel("millones por crédito de 100"); ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("lo que cuesta cada crédito: año promedio y año malo", fontsize=10, color=NAVY)
    plt.tight_layout(); plt.show()


def exportar(tabla, n, datos=None, mercado=None, carpeta=".", verbose=True):
    """cartera_equipo_N.xlsx (hojas: cartera, datos, parametros) y .csv."""
    base = os.path.join(carpeta, f"cartera_equipo_{n}")
    with pd.ExcelWriter(base + ".xlsx") as w:
        tabla.to_excel(w, sheet_name="cartera")
        if datos is not None:
            datos.to_excel(w, sheet_name="datos_bloomberg", index=False)
        par = {k: v for k, v in PARAMETROS.items() if k != "estres"}
        par.update({f"estres_{k}": v for k, v in PARAMETROS["estres"].items()})
        if mercado:
            par.update({f"mercado_{k}": v for k, v in mercado.items()})
        pd.Series(par, name="valor").to_excel(w, sheet_name="parametros")
    tabla.to_csv(base + ".csv")
    if verbose:
        print(f"exportado: {base}.xlsx y {base}.csv  (Colab: panel de archivos → clic derecho → Descargar)")
    return base + ".xlsx"


def _fmt(t, formatos):
    """Imprime una tabla con formatos por columna, sin romper las columnas de texto."""
    s = t.copy()
    for c, f in formatos.items():
        if c in s.columns:
            s[c] = s[c].map(lambda v: f.format(v) if v == v else "—")
    return s.to_string()


def parametros(P=PARAMETROS):
    print("PARÁMETROS FIJOS DE LA TAREA")
    print(f"   Merton    T = {P['T']:g} año · μ = {P['mu']:.0%} · D = deuda CP + ½ deuda LP · σ_E = volatilidad 260 d · r = T-bill 1 año del día")
    print(f"   crédito   EAD = {P['EAD']:.0f} millones USD por empresa · LGD = {P['LGD']:.0%} · ρ = {P['rho']:.0%} · año malo = el peor de {1 / (1 - P['q']):,.0f}")
    print(f"   precio    spread {P['spread_pb']} pb · fondeo SOFR + {P['fondeo_pb']} pb · gasto {P['gasto_pb']} pb · hurdle {P['hurdle']:.0%} · máximo sin garantía {P['spread_max_pb']} pb")
    e = P["estres"]
    print(f"   estrés    DD {e['dDD']:+.0f} · acción {e['ret']:+.0%} · T-bill 3 m {e['r3m']} % · S&P {e['sp500']:+.0%}")


# ===========================================================================
# A MANO Y EN WIDGETS · lo que el equipo llena antes de ver la tabla
# ===========================================================================
def comparar(tabla, ticker, columna, valor, tol=0.02, etiqueta=None):
    """Compara un número que el equipo calculó a mano con el de la tabla.
    Imprime los dos y dice si coinciden (tolerancia relativa `tol`)."""
    fila = tabla.set_index("ticker").loc[ticker]
    ref = float(fila[columna])
    nombre = etiqueta or columna
    ok = abs(valor - ref) <= tol * max(abs(ref), 1e-9) + 1e-9
    fmt = "{:.2%}" if abs(ref) < 1 and columna.startswith("PD") else "{:,.3f}"
    print(f"{ticker} · {nombre}: a mano {fmt.format(valor)} · en la tabla {fmt.format(ref)} → " + ("COINCIDE" if ok else "NO coincide: revisen los números que metieron"))
    return ok


def _widgets():
    import ipywidgets as W
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    return W, plt


def _titulo(W, txt, color=NAVY):
    return W.HTML(f"<div style='background:{color};color:white;padding:6px 10px;font-weight:600;border-radius:4px;margin:4px 0'>{txt}</div>")


def _nota(W, txt):
    return W.HTML(f"<div style='color:{MUTE};font-size:12px;margin:2px 0 8px 2px'>{txt}</div>")


def _slider(W, valor, minimo, maximo, paso, desc, leyenda, fmt=".2f"):
    s = W.FloatSlider(value=valor, min=minimo, max=maximo, step=paso, description=desc, continuous_update=False,
                      readout_format=fmt, style={"description_width": "80px"}, layout=W.Layout(width="440px"))
    return W.HBox([s, W.HTML(f"<span style='color:{MUTE};font-size:12px'>{leyenda}</span>")]), s


def juega_merton(V0, D0, sigma_V0, r, mu=PARAMETROS["mu"], ticker=""):
    """Deslizadores de V, D, σ_V, T y μ a partir de los números que el equipo despejó.
    Imprime la DD sustituida, la PD y el spread implícito; grafica PD contra el plazo
    y la distribución de los activos frente a la barrera."""
    W, plt = _widgets()
    from IPython.display import clear_output
    hV, sV = _slider(W, round(V0, 1), max(0.5, V0 * 0.4), V0 * 2.0, max(0.1, V0 / 200), "V", "valor de los activos, millones (lo que despejaron)", ",.0f")
    hD, sD = _slider(W, round(D0, 1), max(0.1, D0 * 0.2), max(D0 * 3.0, 1.0), max(0.1, D0 / 200), "D", "punto de default: deuda CP + ½ LP", ",.0f")
    hs, ss = _slider(W, round(sigma_V0, 2), 0.03, 1.0, 0.01, "σ_V", "sacudida anual de los activos")
    hT, sT = _slider(W, 1.0, 0.5, 7.0, 0.5, "T", "años al vencimiento", ".1f")
    hm, sm = _slider(W, mu, -0.10, 0.20, 0.01, "μ", "deriva real de los activos; r es la neutral al riesgo")
    out = W.Output()

    def _dibuja(V, D, sigma_V, T, mu):
        m = E.merton(V, D, sigma_V, T=T, r=r, mu=mu, verbose=False)
        with out:
            clear_output(wait=True)
            sT_ = sigma_V * math.sqrt(T)
            print("DD = [ln(V/D) + (μ − σ_V²/2)·T] / (σ_V·√T)")
            print(f"   = [ln({V:,.0f}/{D:,.0f}) + ({mu:.2f} − {sigma_V:.2f}²/2)·{T:g}] / ({sigma_V:.2f}·√{T:g})")
            print(f"   = [{math.log(V / D):.3f} {(mu - 0.5 * sigma_V ** 2) * T:+.3f}] / {sT_:.3f} = {m['DD']:.2f} sacudidas")
            print(f"PD hasta T = Φ(−{m['DD']:.2f}) = {m['PD_P']:.2%}   (neutral al riesgo, con r: {m['PD_Q']:.2%})   ·   spread implícito de la deuda {m['spread'] * 1e4:,.0f} pb")
            Ts = np.arange(0.5, 7.01, 0.5)
            pds = [E.merton(V, D, sigma_V, T=t, r=r, mu=mu, verbose=False)["PD_P"] for t in Ts]
            fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.0))
            ax[0].plot(Ts, [p * 100 for p in pds], color=NAVY, lw=2); ax[0].scatter([T], [m["PD_P"] * 100], color=GOLD, zorder=3, s=50)
            ax[0].set_xlabel("años al vencimiento T"); ax[0].set_ylabel("PD acumulada, %"); ax[0].set_title("PD acumulada contra el plazo", fontsize=10, color=NAVY)
            x = np.linspace(max(0.05, V * 0.2), V * 2.2, 300)
            mu_ln = math.log(V) + (mu - 0.5 * sigma_V ** 2) * T; s_ln = sigma_V * math.sqrt(T)
            dens = (1 / (x * s_ln * math.sqrt(2 * math.pi))) * np.exp(-(np.log(x) - mu_ln) ** 2 / (2 * s_ln ** 2))
            ax[1].plot(x, dens, color=NAVY); ax[1].fill_between(x, dens, where=x < D, color=RED, alpha=0.35, label=f"V_T < D: {m['PD_P']:.1%}")
            ax[1].axvline(D, color=RED, ls="--", lw=1); ax[1].set_yticks([]); ax[1].set_xlabel("valor de los activos en T")
            ax[1].set_title("dónde terminan los activos y la barrera D", fontsize=10, color=NAVY); ax[1].legend(frameon=False, fontsize=9)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo(W, f"JUEGA · Merton con los números de {ticker or 'su empresa'}"),
                 _nota(W, "Muevan UN deslizador a la vez. Primero D (¿cuánta deuda más aguanta?), luego σ_V (¿y si la sacudida se duplica?), luego T."),
                 hV, hD, hs, hT, hm, out])
    W.interactive_output(_dibuja, {"V": sV, "D": sD, "sigma_V": ss, "T": sT, "mu": sm})
    return ui


def juega_altman(balance, ticker=""):
    """Deslizadores sobre el balance de la empresa (pasivo circulante, utilidades
    retenidas, EBIT, pasivo total; el capital contable se ajusta para que activo =
    pasivo + capital). Imprime el Z'' en cadena y la zona; grafica los aportes."""
    W, plt = _widgets()
    from IPython.display import clear_output
    b0 = dict(balance); AT = float(b0["activo_total"])
    def rng(v, lo, hi): return (min(lo, v * 0.5 if v > 0 else v * 1.5), max(hi, v * 1.5 if v > 0 else v * 0.5))
    hpc, spc = _slider(W, b0["pasivo_circulante"], 0, max(AT, b0["pasivo_circulante"] * 2), AT / 200, "vence pronto", f"pasivo circulante (activo circulante fijo en {b0['activo_circulante']:,.0f})", ",.0f")
    lo, hi = rng(b0["utilidades_retenidas"], -AT, AT)
    hur, sur = _slider(W, b0["utilidades_retenidas"], lo, hi, AT / 200, "ut. retenidas", "lo ganado y guardado en los años anteriores (negativo si hubo pérdidas)", ",.0f")
    heb, seb = _slider(W, b0["ebit"], -AT * 0.3, AT * 0.5, AT / 500, "EBIT", "utilidad de operación del año", ",.0f")
    hpt, spt = _slider(W, b0["pasivo_total"], AT * 0.05, AT * 1.5, AT / 200, "pasivo total", "todo lo que se debe; capital contable = activo total − pasivo total", ",.0f")
    out = W.Output()

    def _dibuja(pc, ur, eb, pt):
        b = dict(b0); b.update({"pasivo_circulante": pc, "utilidades_retenidas": ur, "ebit": eb, "pasivo_total": pt, "capital_contable": AT - pt, "ticker": ticker})
        z = A.zpp_en_cadena(b, verbose=False); X = z["X"]; ap = z["aportes"]
        with out:
            clear_output(wait=True)
            print(f"X1 = ({b['activo_circulante']:,.0f} − {pc:,.0f})/{AT:,.0f} = {X['X1']:+.3f} · X2 = {ur:,.0f}/{AT:,.0f} = {X['X2']:+.3f} · X3 = {eb:,.0f}/{AT:,.0f} = {X['X3']:+.3f} · X4 = {AT - pt:,.0f}/{pt:,.0f} = {X['X4']:+.3f}")
            print(f"Z'' = 3.25 + 6.56·({X['X1']:+.2f}) + 3.26·({X['X2']:+.2f}) + 6.72·({X['X3']:+.2f}) + 1.05·({X['X4']:+.2f})")
            print(f"    = 3.25 {ap['X1']:+.2f} {ap['X2']:+.2f} {ap['X3']:+.2f} {ap['X4']:+.2f} = {z['Z']:.2f}  →  zona {z['zona'].upper()}   (cortes {A.CORTES['distress']:.2f} y {A.CORTES['segura']:.2f})")
            fig, ax = plt.subplots(figsize=(8.5, 2.9))
            nombres = ["constante", "X1 liquidez", "X2 historia", "X3 rentabilidad", "X4 apalancamiento"]
            vals = [A.ALTMAN_ZPP["const"], ap["X1"], ap["X2"], ap["X3"], ap["X4"]]
            acum = np.cumsum([0] + vals[:-1]); cols = [MUTE] + [NAVY if v >= 0 else RED for v in vals[1:]]
            ax.bar(nombres, vals, bottom=acum, color=cols)
            ax.axhline(A.CORTES["distress"], color=RED, ls="--", lw=1); ax.text(4.45, A.CORTES["distress"] + 0.05, "distress", color=RED, ha="right", va="bottom", fontsize=9)
            ax.axhline(A.CORTES["segura"], color=GOLD, ls="--", lw=1); ax.text(4.45, A.CORTES["segura"] + 0.05, "segura", color=GOLD, ha="right", va="bottom", fontsize=9)
            ax.axhline(z["Z"], color=NAVY, lw=1.6); ax.set_ylim(min(0, z["Z"] - 0.5), max(8.3, z["Z"] + 0.5))
            ax.set_ylabel("aporte a Z''"); ax.set_title(f"{ticker}: Z'' = {z['Z']:.2f}, de qué está hecho", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo(W, f"JUEGA · el balance de {ticker or 'su empresa'}"),
                 _nota(W, "Empieza en el balance real. Muevan UN rubro a la vez: ¿qué la saca de la zona con menos esfuerzo? Esa razón es la del covenant."),
                 hpc, hur, heb, hpt, out])
    W.interactive_output(_dibuja, {"pc": spc, "ur": sur, "eb": seb, "pt": spt})
    return ui


def juega_dsw(DD0, ret0, r3m0, sp0, ticker=""):
    """Las cuatro lecturas de la empresa en deslizadores, a partir de sus valores de
    hoy. Imprime el exponente término por término, λ y PD; grafica λ contra DD."""
    W, plt = _widgets()
    from IPython.display import clear_output
    c = E.DSW_COEF
    hd, sd = _slider(W, float(DD0), -1.0, 12.0, 0.1, "DD", "distancia al default (de Merton)", ".1f")
    hr, sr = _slider(W, float(ret0), -0.8, 0.8, 0.05, "acción", "retorno de la acción en 12 meses (−0.30 = −30 %)")
    ht, st = _slider(W, float(r3m0), 0.0, 8.0, 0.1, "tasa 3 m", "T-bill a 3 meses, EN POR CIENTO (4.2 = 4.2 %)", ".2f")
    hs, ssp = _slider(W, float(sp0), -0.5, 0.5, 0.05, "S&P 500", "retorno del mercado en 12 meses: el ciclo")
    out = W.Output()

    def _dibuja(DD, ret, r3m, sp):
        res = E.intensidad_covariables(DD, ret=ret, r3m=r3m, sp500=sp, verbose=False)
        base = E.intensidad_covariables(DD0, ret=ret0, r3m=r3m0, sp500=sp0, verbose=False)
        with out:
            clear_output(wait=True)
            print(f"exponente = {c['const']:+.3f} {c['DD']:+.3f}·{DD:.2f} {c['ret']:+.3f}·{ret:.2f} {c['r3m']:+.3f}·{r3m:.2f} {c['sp500']:+.3f}·{sp:.2f}")
            print(f"          = {c['const']:+.3f} {c['DD'] * DD:+.3f} {c['ret'] * ret:+.3f} {c['r3m'] * r3m:+.3f} {c['sp500'] * sp:+.3f} = {res['exponente']:+.3f}")
            print(f"λ = e^({res['exponente']:+.3f}) = {res['lambda'] * 1e4:,.0f} pb por año · PD a un año = 1 − e^(−λ) = {res['PD_1a']:.2%}   (hoy: {base['PD_1a']:.2%}  → ×{res['PD_1a'] / base['PD_1a']:.1f})")
            dds = np.linspace(-1, 12, 200)
            l0 = [E.intensidad_covariables(x, ret=ret0, r3m=r3m0, sp500=sp0, verbose=False)["lambda"] * 1e4 for x in dds]
            l1 = [E.intensidad_covariables(x, ret=ret, r3m=r3m, sp500=sp, verbose=False)["lambda"] * 1e4 for x in dds]
            fig, ax = plt.subplots(figsize=(8.5, 3.0))
            ax.plot(dds, l0, color=STEEL, lw=1.5, label="lecturas de hoy"); ax.plot(dds, l1, color=NAVY, lw=2, label="con sus lecturas")
            ax.scatter([DD], [res["lambda"] * 1e4], color=GOLD, zorder=3, s=50)
            ax.set_yscale("log"); ax.set_xlabel("distancia al default DD"); ax.set_ylabel("λ, pb anuales (escala log)")
            ax.set_title("λ contra DD: cada sacudida menos multiplica, no suma", fontsize=10, color=NAVY); ax.legend(frameon=False, fontsize=9)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo(W, f"JUEGA · el termómetro de {ticker or 'su empresa'}"),
                 _nota(W, "Parte de las lecturas de hoy. Lleven la DD a DD − 2 y luego pongan el ciclo de 2020 (acción −0.30, tasa 0.1, S&P −0.15): ese es el estrés de la tabla."),
                 hd, hr, ht, hs, out])
    W.interactive_output(_dibuja, {"DD": sd, "ret": sr, "r3m": st, "sp": ssp})
    return ui


def juega_credito(PD0, ticker="", P=PARAMETROS):
    """PD, ρ, LGD y spread en deslizadores, a partir de la PD de la empresa. Imprime la
    cadena del crédito (EL, PD del año malo, K, RORAC, spread mínimo) y grafica."""
    W, plt = _widgets()
    from IPython.display import clear_output
    hp, sp = _slider(W, max(PD0, P["PD_piso"]) * 100, 0.05, 25.0, 0.05, "PD (%)", "la PD de hoy, en por ciento (1 = 1 %)", ".2f")
    hr, sr = _slider(W, P["rho"] * 100, 1.0, 60.0, 1.0, "ρ (%)", "cuánto pesa el factor común", ".0f")
    hl, sl = _slider(W, P["LGD"] * 100, 10.0, 90.0, 5.0, "LGD (%)", "lo que se pierde si cae; una garantía la baja", ".0f")
    hs, ss = _slider(W, P["spread_pb"], 50, 1000, 10, "spread (pb)", "lo que cobra el banco sobre SOFR", ".0f")
    out = W.Output()

    def _dibuja(PDp, rho, LGD, spread):
        Q = dict(P); Q.update({"rho": rho / 100, "LGD": LGD / 100, "spread_pb": spread})
        PD = PDp / 100
        c = credito(PD, Q)
        with out:
            clear_output(wait=True)
            a, zq = norm.ppf(PD), norm.ppf(Q["q"])
            print(f"reserva      EL = {PD:.4f} × {Q['LGD']:.2f} × {Q['EAD']:.0f} = {c['EL']:.2f}")
            print(f"PD año malo  Φ(({a:.2f} + {math.sqrt(Q['rho']):.3f}·{zq:.2f}) / {math.sqrt(1 - Q['rho']):.3f}) = {c['PD_malo']:.1%}")
            print(f"capital      K = ({Q['LGD']:.2f}·{c['PD_malo']:.4f} − {Q['LGD']:.2f}·{PD:.4f}) × {Q['EAD']:.0f} = {c['K']:.2f}")
            print(f"margen       ({spread:.0f} − {Q['fondeo_pb']} − {Q['gasto_pb']}) pb × {Q['EAD']:.0f} = {c['margen']:.2f}")
            print(f"RORAC        ({c['margen']:.2f} − {c['EL']:.2f}) / {c['K']:.2f} = {c['RORAC']:.1%}   (hurdle {Q['hurdle']:.0%})   ·   spread mínimo {c['spread_min_pb']:.0f} pb   ·   {veredicto(c['RORAC'], c['spread_min_pb'], Q)}")
            pds = np.linspace(0.0005, 0.25, 200)
            ks = [credito(p, Q)["K"] for p in pds]; els = [p * Q["LGD"] * Q["EAD"] for p in pds]
            fig, ax = plt.subplots(figsize=(8.5, 3.0))
            ax.plot(pds * 100, els, color=STEEL, lw=1.8, label="reserva (EL)"); ax.plot(pds * 100, ks, color=NAVY, lw=2, label="capital (K)")
            ax.axhline(c["margen"], color=GOLD, ls="--", lw=1.2, label=f"margen a {spread:.0f} pb")
            ax.scatter([PD * 100], [c["K"]], color=GOLD, zorder=3, s=50)
            ax.set_xlabel("PD, %"); ax.set_ylabel("millones por crédito de 100"); ax.legend(frameon=False, fontsize=9)
            ax.set_title("lo que cuesta el crédito contra la PD: el capital crece más despacio que la reserva", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo(W, f"JUEGA · el crédito de {ticker or 'su empresa'}"),
                 _nota(W, "Primero suban el spread hasta que el RORAC pase de 15 % (ese es el spread mínimo). Luego bajen la LGD a 25 %: eso hace una garantía. Luego suban ρ a 30 %: eso hace una crisis."),
                 hp, hr, hl, hs, out])
    W.interactive_output(_dibuja, {"PDp": sp, "rho": sr, "LGD": sl, "spread": ss})
    return ui
