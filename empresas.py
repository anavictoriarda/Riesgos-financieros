"""
===========================================================================
MÓDULO XII · RIESGO DE CRÉDITO II — EMPRESAS
Riesgos Financieros (FIN516) · Universidad Panamericana · Gerardo Alcalá
===========================================================================

El Módulo XI leyó la PD en un PRECIO. Este módulo la estima cuando el precio
no basta o no existe: del BALANCE y del EQUITY de la empresa, y después la
agrega a un portafolio. La pregunta de la sesión:

    Cuando el precio no basta o no existe, ¿cómo se estima la probabilidad
    de incumplimiento de una empresa a partir de su balance, y cómo se
    agrega a un portafolio?

---------------------------------------------------------------------------
FORMATO DE LA SESIÓN (claude/convenciones.md, §2.7 y §2.8)
---------------------------------------------------------------------------
    Cuatro LABORATORIOS; en cada uno el alumno es un rol que toma una
    decisión real con el modelo, apuesta antes de jugar, juega con
    deslizadores, compara y resuelve un reto. La bitácora de cada alumno
    alimenta la TAREA DE DECISIÓN (memo) y las preguntas de corroboración
    P1–P3 que el profesor genera desde este archivo.

    L1  Merton            el acreedor de Cemex, agosto de 2009
    L2  Altman Z''        el analista con mandato, con sus empresas de EDGAR
    L3  Forma reducida    el CRO de un banco mexicano, marzo de 2020
    L4  Vasicek           el comité de crédito y la concentración (Pemex)

    Este archivo es la MAQUINARIA: funciones puras, verificadas en
    pruebas_empresas.py. Los deslizadores viven en laboratorios.py y son
    una capa delgada encima; nunca tienen lógica propia.

---------------------------------------------------------------------------
TERMINOLOGÍA (claude/convenciones.md, §1.4)
---------------------------------------------------------------------------
    Lo que el XI definió y aquí se asume: PD · LGD · EAD · RR · EL · UL ·
    intensidad (hazard rate) · supervivencia · Q contra P · calificación ·
    notch · grado de inversión · TTC / PIT · matriz de transición.

    Lo que este módulo DEFINE (glosario, sección XII):
      modelo estructural · modelo de forma reducida (como familias)
      Merton · valor de activos V · volatilidad de activos σ_V · punto de
      default · distancia al default (DD) · PD de Merton
      KMV · EDF · mapeo empírico DD → PD
      Altman Z'' · análisis discriminante · zona de distress / gris / segura
      intensidad con covariables (Duffie-Saita-Wang) · covariable
      factor sistemático · correlación de activos ρ · PD condicional ·
      distribución de Vasicek · ASRF · cartera granular
      pérdida inesperada (UL) · capital económico · IRB · HHI · granularidad

    Notación: la duración es Dur (XI); aquí D es la DEUDA (punto de default)
    y DD la distancia al default. No se usa D para factor de descuento en
    este módulo. Φ es la normal estándar acumulada; Φ⁻¹ su inversa.

---------------------------------------------------------------------------
SECCIONES Y FUNCIONES
---------------------------------------------------------------------------
    L1  Merton          merton · merton_implicito · distancia_default ·
                        kmv_punto_default · mapeo_edf
    L2  Altman          altman_zpp · zona_zpp
    L3  Forma reducida  intensidad_covariables
    L4  Vasicek         vasicek_pd_condicional · vasicek_cuantil · vasicek_dist
                        · irb_rho · irb_capital · hhi · capital_concentracion
    Laboratorio         Bitacora · tarea · preguntas_p1p3 · revisa

---------------------------------------------------------------------------
FUENTES (ver investigacion_xii.md; lo marcado NO VERIFICADO no se proyecta)
---------------------------------------------------------------------------
"""

import os
import json
import math
import datetime as _dt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq, fsolve

__all__ = [
    "merton", "merton_implicito", "merton_pasos", "cadena_dd", "cadena_opcion", "distancia_default", "kmv_punto_default", "mapeo_edf",
    "altman_zpp", "zona_zpp", "intensidad_covariables",
    "vasicek_pd_condicional", "vasicek_cuantil", "vasicek_dist", "irb_rho", "irb_capital",
    "hhi", "capital_concentracion",
    "Bitacora", "tarea", "preguntas_p1p3", "revisa", "bajar_cx",
    "pagos_al_vencimiento", "simular_caminos", "bajar_precios", "dd_diaria", "anticipacion", "CASOS_ALERTA", "cargar_precios_casos",
    "ALTMAN_ZPP", "ZONAS_ZPP", "DSW_COEF", "MAPEO_EDF", "CEMEX_2009", "CEMEX_2009_BASE", "CEMEX_VENCIMIENTOS", "CNBV_COVID_2020", "LIMITES_CUB", "PEMEX_EXPOSICION",
]

# ===========================================================================
# CONSTANTES VERIFICADAS Y CONSTANTES PENDIENTES
# ===========================================================================

# Altman (2000, «Predicting Financial Distress of Companies: Revisiting the
# Z-Score and ZETA Models», NYU Stern), modelo Z'' para no manufactureras y
# mercados emergentes (origen: Altman, Hartzell y Peck 1995, Eurobonos mexicanos):
#   Z'' = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4      [V, textual]
#   X1 capital de trabajo / activo total     X2 utilidades retenidas / activo total
#   X3 EBIT / activo total                   X4 capital contable EN LIBROS / pasivo total
# Constante +3.25 «so as to standardize the scores» (EM Score): D = 0 … AAA = 8.15 [V].
ALTMAN_ZPP = {"const": 3.25, "X1": 6.56, "X2": 3.26, "X3": 6.72, "X4": 1.05,
              "fuente": "Altman (2000), NYU Stern; constante 3.25 y equivalencia AAA 8.15 … D 0 [V]"}

# Zonas del Z'' sin la constante: distress < 1.10, gris 1.10–2.60, segura > 2.60.
# FUENTE SECUNDARIA (Altman et al. 2017 vía Wikipedia): Altman (2000) no da cortes
# para Z''. Con la constante 3.25 los cortes quedan en 4.35 y 5.85.
ZONAS_ZPP = {"distress": 1.10, "segura": 2.60, "fuente": "secundaria: Altman et al. (2017) vía Wikipedia [S]"}

# Duffie, Saita y Wang (2007), JFE 83(3), 635–665, Tabla 2, fila «Default»,
# estimadores MV (errores estándar): constante −2.156 (0.113); DTD −1.129 (0.036);
# retorno de la acción 12 m −0.694 (0.075); tasa Tesoro 3 m −0.105 (0.021);
# S&P 500 12 m +1.203 (0.289). Muestra 1980–2004, 2,770 industriales de EUA,
# 497 incumplimientos. UNIDADES: DTD en desviaciones; retornos en fracción
# (0.10 = 10 %); la tasa a 3 meses EN POR CIENTO (2.0 = 2 %).  [V]
DSW_COEF = {"const": -2.156, "DD": -1.129, "ret": -0.694, "r3m": -0.105, "sp500": 1.203,
            "ee": {"const": 0.113, "DD": 0.036, "ret": 0.075, "r3m": 0.021, "sp500": 0.289},
            "fuente": "Duffie, Saita y Wang (2007), JFE 83(3), Tabla 2 [V]"}

# Mapeo DD → EDF. Crosbie y Bohn (KMV, rev. 2002) publican sólo ilustraciones:
# DD 7 ↔ «about 5 bp» (≈ AA); Philip Morris DD 3.5 ↔ EDF 25 pb; DD 3 ↔ 13 pb
# bajo Black-Scholes; y «around 100 bp» para DD 4 bajo la distribución empírica
# en la discusión normal-contra-empírica. La tabla propietaria no es pública.
# La de abajo es una CONSTRUCCIÓN del curso, suave y decreciente, anclada en
# DD 7 ↔ 5 pb y DD 3.5 ↔ 25 pb. Se proyecta como «estilizada».
MAPEO_EDF = pd.DataFrame({
    "DD":  [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0],
    "EDF": [0.15, 0.08, 0.045, 0.025, 0.012, 0.0050, 0.0025, 0.0016, 0.0009, 0.0006, 0.0005, 0.0004],
})

# Cemex, 13 de agosto de 2009: el día antes de firmar el refinanciamiento.
# Unidades: millones de dólares.  [V] = 6-K / 20-F; [S] = cálculo propio con
# precios de agregadores; [NV] = pendiente de calcular en Colab con yfinance.
CEMEX_2009 = {
    "fecha": "2009-08-13",
    "deuda_total_musd": 19_250,          # [V] deuda total al 30-jun-2009, Exhibit 2 6-K 3T09 (sin perpetuos)
    "perpetuos_musd": 3_020,             # [V] obligaciones perpetuas (US$3.0 mmdd); el covenant de apalancamiento SÍ las incluye
    "deuda_refinanciada_musd": 15_000,   # [V] «approximately US$15 billion», 6-K 14-ago-2009; vence 14-feb-2014 (US$6.8–6.9 mmdd ese día)
    "acreedores": 75,                    # [V] «approximately 75 banks and private placement noteholders»
    "margen_pct": 4.5,                   # [V] margen de las facilidades bancarias
    "cpo_equivalentes_mm": 8_250,        # [S] 8.1–8.4 mil millones de CPO-equivalentes antes de la colocación (20-F FY2008 + emisiones)
    "precio_ads_usd": 13.0,              # [S] ≈ US$13 el ADS en agosto de 2009 (digrin, cierre mensual 13.28); 1 ADS = 10 CPO [V]
    "capitalizacion_musd": 10_700,       # [S] ≈ 8,250 / 10 × 13.0; Cemex no publica capitalización. En Colab se recalcula con el precio del día
    "sigma_E": 0.85,                     # [NV] volatilidad anualizada de CX en los 12 meses previos: NO calculada aquí (red bloqueada);
                                         #      meses de −56 % (oct-08) y +41 % (ago-09) la ponen arriba de 70–80 %. El cuaderno la calcula en Colab
    "capital_colocado_musd": 1_782,      # [V] netos de la oferta global del 28-sep-2009: 1,495 M CPO = 149.5 M ADS a US$12.50
    "capital_exigido_musd": 1_000,       # [V] el acuerdo obligaba a levantar ≥ US$1.0 mmdd de capital antes del 30-jun-2010
    "ebitda_2008_musd": 4_343,           # [V] comunicado 4T08 (reexpresado 4,080 sin Australia)
    "ebitda_2009_musd": 2_657,           # [V] reporte 4T09
    "covenant_apalancamiento": "deuda total (incl. perpetuos)/EBITDA ≤ 7.75x (jun-2010) bajando a 3.50x (dic-2013)",  # [V]
    "calificaciones": "S&P: BBB− (oct-08) → BB+ (22-ene-09) → B− (10-mar-09) → B (8-oct-09, tras la colocación); Fitch: BB+ (31-oct-08) → B (11-mar-09)",  # [S]
    "r": 0.03,
    "fuente": "6-K 14-ago-2009 y 3T09; 20-F FY2008 y 2009; comunicados Cemex 22 y 28-sep-2009 (investigacion_xii.md §1)",
}

# Copia CONGELADA del caso base: el cuaderno puede actualizar CEMEX_2009 con el
# precio y la σ_E bajados en Colab; los RETOS y el deck usan esta copia para que
# «Debe dar» no cambie con el día.
CEMEX_2009_BASE = dict(CEMEX_2009)

# Perfil de vencimientos pro forma tras el refinanciamiento (Exhibit 1 del 6-K de
# septiembre de 2009), millones de dólares [V]. Sirve para el punto de default
# «corto + ½ largo»: corto = 2S09 + 2010; largo = 2011 en adelante.
CEMEX_VENCIMIENTOS = {"2S09": 1_222, "2010": 2_888, "2011": 2_835, "2012": 1_610, "2013": 2_395, "2014+": 8_353}

# CNBV y COVID, para el laboratorio L3 (rol: CRO en marzo de 2020).  [V] salvo nota.
CNBV_COVID_2020 = {
    "oficio": "P285/2020 del 26 de marzo de 2020 (y P293/2020 del 15 de abril)",       # [V]
    "que_permitia": "diferir capital y/o intereses hasta 4 meses, con 2 adicionales; los créditos siguen «vigentes»: no pasan a cartera vencida",  # [V/S]
    "elegibilidad": "cartera vigente al 28 de febrero de 2020 (ampliado al 31 de marzo)",  # [V]
    "ventana": "hasta el 31 de julio de 2020 (prórroga P325/2020)",                         # [V]
    "acogida_abm": "9.3 millones de créditos · $1.1 billones de pesos · 22 % del crédito al sector privado (ABM, oct-2020)",  # [V]
    "reservas_2T20": {"Banorte": 4_868, "Santander México": 3_915},   # [V] mdp, reservas adicionales COVID, reportes 2T20
    "bbva_1T20_estimacion_preventiva": 16_237,                        # [V] mdp; BBVA concentró el esfuerzo en 1T20
    "bbva_cartera_apoyada": "Ps324,306 M, 25 % de la cartera; cartera vencida dentro de la apoyada: 0.01 %",  # [V]
    "fuente": "CNBV, ABM (22-oct-2020), reportes 2T20 de BBVA, Banorte y Santander (investigacion_xii.md §2)",
}

# Límites de concentración de la CUB, régimen de Grandes Exposiciones vigente
# desde el 1 de octubre de 2023 (DOF 17-abr-2023).  [V]
LIMITES_CUB = {
    "gran_exposicion": "≥ 10 % de la parte básica del capital neto a una persona o grupo de riesgo común",
    "limite_general_pct": 25,          # de la parte básica del capital neto, por Gran Exposición
    "limite_sistemicos_pct": 15,       # entre instituciones de importancia sistémica
    "limite_paraestatales_pct": 100,   # entidades y organismos paraestatales federales (Pemex, CFE)
    "exentos": "Gobierno Federal y créditos con su garantía · Banco de México · IPAB",
    "regimen_anterior": "tabla 12–40 % del capital básico según ICAP (hasta sep-2023) — NO VERIFICADO textualmente",
    "fuente": "CUB Título Segundo, Cap. III, Arts. 54–58; catálogo R35 de la CNBV (investigacion_xii.md §6.3)",
}

# Pemex como nombre concentrado: no hay cifra pública de Banxico. Moody's (jun-2025,
# vía El CEO): ≈ 10 % del capital tangible común de los bancos comerciales; 24 % Nafin;
# 8 % Bancomext.  [S]
PEMEX_EXPOSICION = {"bancos_comerciales_pct_tce": 10, "nafin_pct_tce": 24, "bancomext_pct_tce": 8,
                    "fecha": "2025-06", "fuente": "Moody's vía El CEO, 11-ago-2025 [S]; sin cifra de Banxico"}


# ===========================================================================
# DATOS DE MERCADO (sólo con red; en Colab)
# ===========================================================================

def bajar_cx(fecha_corte="2009-08-13", ventana=252, ticker="CX", verbose=True):
    """Baja el ADR de Cemex (CX, NYSE) con yfinance y calcula, a la fecha de
    corte: el precio del ADS y la volatilidad anualizada de los log-retornos
    diarios de la ventana previa (σ = desv. est. × √252). Devuelve un
    diccionario para actualizar CEMEX_2009. REVIENTA si no hay red o si el
    ticker no trae historia: no hay respaldo silencioso (convenciones §5.1)."""
    import yfinance as yf
    px = yf.download(ticker, start="2007-01-01", end="2010-12-31", progress=False, auto_adjust=False)
    if px is None or len(px) == 0:
        raise RuntimeError(f"yfinance no devolvió historia para {ticker}: ¿hay red?")
    serie = px["Close"]
    if hasattr(serie, "columns"):
        serie = serie.iloc[:, 0]
    serie = serie.dropna()
    hasta = serie.loc[:fecha_corte]
    if len(hasta) < ventana + 1:
        raise RuntimeError(f"historia insuficiente antes de {fecha_corte}: {len(hasta)} días")
    precio = float(hasta.iloc[-1])
    lr = np.log(hasta).diff().dropna().iloc[-ventana:]
    sigma = float(lr.std(ddof=1) * math.sqrt(252))
    out = {"fecha": str(hasta.index[-1].date()), "precio_ads_usd": precio, "sigma_E": sigma,
           "capitalizacion_musd": CEMEX_2009["cpo_equivalentes_mm"] / 10 * precio, "n_dias": int(len(lr))}
    if verbose:
        print(f"CX al {out['fecha']}: ADS = US${precio:.2f} · σ_E ({ventana} días previos) = {sigma:.1%} · "
              f"capitalización ≈ {CEMEX_2009['cpo_equivalentes_mm']:,.0f} M CPO / 10 × {precio:.2f} = US${out['capitalizacion_musd']:,.0f} M")
    return out


# ===========================================================================
# L1 · HERRAMIENTAS DE NARRATIVA Y DE ALERTA TEMPRANA
# ===========================================================================

def pagos_al_vencimiento(V_T, D=80.0):
    """La tabla de la primera lámina: con activos V_T al vencimiento y deuda D,
    el banco cobra mín(V_T, D) y el accionista se queda con máx(V_T − D, 0).
    Es el payoff de un call (accionista) y de deuda segura menos una put (banco)."""
    V_T = np.asarray(V_T, dtype=float)
    return pd.DataFrame({"activos": V_T, "banco cobra": np.minimum(V_T, D), "accionista se queda con": np.maximum(V_T - D, 0.0)})


def simular_caminos(V0=100.0, sigma=0.20, T=1.0, n_caminos=40, pasos=252, mu=0.0, semilla=11):
    """Caminos aleatorios del valor de los activos (movimiento browniano
    geométrico, el del Módulo III): cada día un paso con media mu y
    desviación sigma anualizadas. Devuelve (t, matriz de caminos)."""
    rng = np.random.default_rng(semilla)
    dt = T / pasos
    z = rng.standard_normal((n_caminos, pasos))
    incr = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    caminos = V0 * np.exp(np.cumsum(incr, axis=1))
    caminos = np.hstack([np.full((n_caminos, 1), V0), caminos])
    return np.linspace(0, T, pasos + 1), caminos


def bajar_precios(tickers, inicio, fin, verbose=True):
    """Precios de cierre diarios con yfinance para varios tickers (p. ej.
    "AEROMEX.MX"). REVIENTA si no hay red o si un ticker no trae historia:
    sin respaldo silencioso (convenciones §5.1). Devuelve un DataFrame
    fecha × ticker."""
    import logging
    import yfinance as yf
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)      # sin los ERROR rojos: el aviso lo da esta función
    px = yf.download(list(tickers), start=inicio, end=fin, progress=False, auto_adjust=False)
    if px is None or len(px) == 0:
        raise RuntimeError("yfinance no devolvió datos: ¿hay red?")
    cierre = px["Close"] if "Close" in px else px
    if isinstance(cierre, pd.Series):
        cierre = cierre.to_frame(list(tickers)[0])
    faltan = [t for t in tickers if t not in cierre.columns or cierre[t].dropna().empty]
    if faltan:
        raise RuntimeError(f"sin historia para {faltan}")
    if verbose:
        print("precios bajados:", ", ".join(f"{t} ({cierre[t].dropna().index.min().date()} → {cierre[t].dropna().index.max().date()})" for t in tickers))
    return cierre


def dd_diaria(precios, acciones_mm, deuda_mm, ventana=252, r=0.0, mu=0.0, T=1.0, cada=5, verbose=True):
    """LA DISTANCIA AL DEFAULT DÍA POR DÍA · la alerta temprana de KMV.

    Para cada fecha (cada `cada` días hábiles, para que sea rápido): E = precio
    × acciones, σ_E = desviación de los log-retornos de los `ventana` días
    previos × √252, y de (E, σ_E, D) se despejan V y σ_V con las dos
    ecuaciones de Merton; DD y PD a un año con deriva mu.

    SIMPLIFICACIÓN DEL LABORATORIO: la deuda y el número de acciones se
    toman CONSTANTES (los del último reporte antes del evento), porque
    yfinance no trae balances históricos. Se dice en la lámina. Precios y
    deuda deben estar en la misma moneda.

    Devuelve un DataFrame con E, sigma_E, V, sigma_V, DD, PD."""
    p = precios.dropna()
    lr = np.log(p).diff()
    filas = []
    idx = list(range(ventana, len(p), cada))
    for i in idx:
        sigma_E = float(lr.iloc[i - ventana + 1:i + 1].std(ddof=1) * math.sqrt(252))
        E_ = float(p.iloc[i]) * acciones_mm
        try:
            m = merton_implicito(E_, sigma_E, deuda_mm, T=T, r=r, verbose=False)
            mp = merton(m["V"], deuda_mm, m["sigma_V"], T=T, r=r, mu=mu, verbose=False)
            filas.append({"fecha": p.index[i], "E": E_, "sigma_E": sigma_E, "V": m["V"], "sigma_V": m["sigma_V"], "DD": mp["DD"], "PD": mp["PD_P"]})
        except Exception:
            filas.append({"fecha": p.index[i], "E": E_, "sigma_E": sigma_E, "V": np.nan, "sigma_V": np.nan, "DD": np.nan, "PD": np.nan})
    out = pd.DataFrame(filas).set_index("fecha")
    if verbose and len(out):
        print(f"DD diaria calculada en {len(out)} fechas ({out.index.min().date()} → {out.index.max().date()}); deuda {deuda_mm:,.0f} M y {acciones_mm:,.0f} M acciones constantes")
    return out


def anticipacion(dd, fecha_evento, umbral=2.0):
    """Cuántos meses antes del evento la DD cruzó por primera vez (y ya no
    volvió a subir de forma sostenida) el umbral. Devuelve (fecha del
    cruce, meses de anticipación) o (None, None)."""
    fe = pd.Timestamp(fecha_evento)
    serie = dd["DD"].dropna()
    serie = serie[serie.index <= fe]
    bajo = serie[serie < umbral]
    if bajo.empty:
        return None, None
    # primer cruce desde el cual la serie permanece bajo el umbral al menos la mitad del tiempo
    for f in bajo.index:
        despues = serie[serie.index >= f]
        if (despues < umbral).mean() >= 0.5:
            return f, round((fe - f).days / 30.44, 1)
    f = bajo.index[0]
    return f, round((fe - f).days / 30.44, 1)


# Casos para el laboratorio de alerta temprana (investigacion_merton_alerta.md,
# consultado el 25-oct-2026). Deuda y acciones CONSTANTES del último reporte
# antes del evento. Precios: Cemex (CX, en dólares) está en Yahoo; Aeroméxico,
# TV Azteca e ICA fueron canceladas de la BMV y NO están en Yahoo: sus precios
# se leen de precios/<ticker>.csv (columnas fecha, cierre), traídos de
# Bloomberg o de la BMV. [V] fuente primaria · [S] secundaria.
CASOS_ALERTA = {
    "Cemex": {
        "ticker": "CX", "moneda": "USD", "csv": None,
        "deuda_mm": 18_784,            # [V] deuda total sin perpetuos al 31-dic-2008, 20-F FY2008 (US$ M)
        "acciones_mm": 836,            # [V] ≈ 8,363 M CPO-equivalentes / 10 (1 ADS = 10 CPO), portada 20-F FY2008
        "evento": "2009-01-22",        # [S] S&P baja a BB+: pierde el grado de inversión
        "que_paso": "pierde el grado de inversión (S&P BB+); en marzo S&P B− y Fitch B; el 14-ago-2009 refinancia ~US$15,000 M",
        "calificaciones": [("2008-10-15", "S&P BBB−"), ("2008-10-31", "Fitch BB+"), ("2009-01-22", "S&P BB+"), ("2009-03-10", "S&P B−"), ("2009-03-11", "Fitch B")],
        "desde": "2007-01-01", "hasta": "2009-12-31",
        "fuente": "20-F FY2008; CemNet/LatinFinance para las fechas de calificación [S]",
    },
    "Aeroméxico": {
        "ticker": "AEROMEX.MX", "moneda": "MXN", "csv": "precios/AEROMEX.csv",
        "deuda_mm": 34_923,            # [V] deuda financiera al 31-dic-2019 (reporte 4T20); sin los 28,497 M de arrendamientos (IFRS 16)
        "acciones_mm": 682.12,         # [V] 682,119,793 acciones (dic-2020; sin cambios de capital en 2020)
        "evento": "2020-06-30",        # [V] solicitud de Capítulo 11 en Nueva York
        "que_paso": "la pandemia para los vuelos; solicita Capítulo 11 el 30-jun-2020",
        "calificaciones": [("2020-03-18", "S&P BB−→B+"), ("2020-03-19", "Moody's B1→B2"), ("2020-06-15", "Moody's →Caa1 [S]"), ("2020-07-02", "Moody's →Ca")],
        "desde": "2018-01-01", "hasta": "2020-09-30",
        "fuente": "reporte 4T20 y eventos relevantes BMV [V]; deuda 1T20 (47,733 M) sólo en prensa [S]",
    },
    "TV Azteca": {
        "ticker": "AZTECACPO.MX", "moneda": "MXN", "csv": "precios/AZTECACPO.csv",
        "deuda_mm": 14_410,            # [V] deuda total al 30-sep-2020 (reporte 3T20); 13,114 M a dic-2019
        "acciones_mm": 2_984,          # [V] CPOs en circulación
        "evento": "2021-02-09",        # [V] aplaza el cupón (~US$16.5 M) de las Senior Notes 8.25 % 2024 (US$400 M); Fitch RD el 12-mar-2021
        "que_paso": "no paga el cupón de sus bonos en dólares; años de negociación y concurso mercantil en 2026",
        "calificaciones": [("2019-12-24", "HR A+ perspectiva negativa"), ("2020-03-25", "Fitch B+→B [S]"), ("2020-11-11", "Verum A+→A"), ("2021-02-11", "HR →DS")],
        "desde": "2019-01-01", "hasta": "2021-06-30",
        "fuente": "reportes 3T20/4T20 y comunicados de HR Ratings y Verum en la BMV [V]",
    },
    "ICA": {
        "ticker": "ICA.MX", "moneda": "MXN", "csv": "precios/ICA.csv",
        "deuda_mm": 57_929,            # [V] deuda total al 30-sep-2015 (6-K 3T15)
        "acciones_mm": 611.03,         # [V] 611,029,276 acciones (20-F)
        "evento": "2015-11-30",        # [V] no paga US$31 M de cupón de las notas 8.875 % 2024; el consejo decide no pagar el 18-dic; gracia vence 29-dic
        "que_paso": "deja de pagar el cupón de sus bonos en dólares; concurso mercantil en 2017 con quita de 90.9 %",
        "calificaciones": [("2015-11-05", "Moody's B2→B3"), ("2015-11-17", "S&P B→CCC+"), ("2015-12-02", "S&P →CC; Moody's →Caa3"), ("2015-12-21", "S&P →D")],
        "desde": "2013-06-01", "hasta": "2016-03-31",
        "fuente": "6-K 3T15 y 6-K de nov–dic 2015; 20-F [V]",
    },
    "Bimbo": {
        "ticker": "BIMBOA.MX", "moneda": "MXN", "csv": None,
        "deuda_mm": 87_000,            # [S] «total debt was 87 billion pesos» a dic-2019 (prensa especializada citando el reporte)
        "acciones_mm": 4_703,          # [S] 4,703 M acciones serie A
        "evento": None,
        "que_paso": "contraste sin evento: BBB / Baa2 / BBB estables en 2019",
        "calificaciones": [],
        "desde": "2018-01-01", "hasta": "2020-12-31",
        "fuente": "comunicado Bimbo 20-feb-2020 [V]; deuda en pesos y acciones [S]",
    },
}


def cargar_precios_casos(casos=None, carpeta_csv="precios", verbose=True):
    """Para cada caso: lee precios/<ticker>.csv (columnas fecha, cierre) si
    existe; si no, intenta yfinance; si tampoco, lo salta y lo dice. Devuelve
    dict nombre → Serie de cierres. Nunca inventa precios."""
    casos = CASOS_ALERTA if casos is None else casos
    out = {}
    for nombre, c in casos.items():
        serie = None
        ruta = c.get("csv")
        if ruta and os.path.exists(ruta):
            df = pd.read_csv(ruta)
            col_f = [k for k in df.columns if k.lower().startswith("fecha") or k.lower() == "date"][0]
            col_p = [k for k in df.columns if k.lower().startswith("cierre") or k.lower() in ("close", "px_last", "precio")][0]
            serie = pd.Series(df[col_p].values, index=pd.to_datetime(df[col_f])).sort_index()
            origen = ruta
        else:
            try:
                serie = bajar_precios([c["ticker"]], c["desde"], c["hasta"], verbose=False)[c["ticker"]].dropna()
                origen = "yfinance"
            except Exception as e:
                if verbose:
                    print(f"{nombre:12s} sin precios: Yahoo no tiene la historia de {c['ticker']} (salió de bolsa, o no hay red) "
                          f"→ se salta; para incluirlo, poner {ruta or 'un CSV'} (columnas fecha, cierre)")
                continue
        serie = serie.loc[c["desde"]:c["hasta"]]
        out[nombre] = serie
        if verbose:
            print(f"{nombre:12s} {len(serie):5d} precios de {origen} ({serie.index.min().date()} → {serie.index.max().date()})")
    return out


# ===========================================================================
# L1 · MERTON
# ===========================================================================

def merton(V, D, sigma_V, T=1.0, r=0.03, mu=None, verbose=True):
    """EL MODELO DE MERTON · la empresa como opción.

    EL PROBLEMA QUE EXISTÍA. La deuda de una empresa tiene riesgo de
    incumplimiento, pero antes de 1974 no había una teoría que ligara ese
    riesgo con lo que sí se observa todos los días: el precio y la
    volatilidad de la acción.

    CÓMO LO PLANTEÓ MERTON. Los accionistas son dueños de la empresa sólo si
    al vencimiento T los activos V valen más que la deuda D; si no, entregan
    la empresa a los acreedores y se quedan sin nada. Eso es exactamente el
    pago de un call sobre V con precio de ejercicio D:

        E = V·Φ(d1) − D·e^(−rT)·Φ(d2)
        d1 = [ln(V/D) + (r + σ_V²/2)·T] / (σ_V·√T)
        d2 = d1 − σ_V·√T

    La probabilidad de incumplimiento es la probabilidad de que V_T < D:

        PD_Q = Φ(−d2)                        neutral al riesgo (con r)
        DD   = [ln(V/D) + (μ − σ_V²/2)·T] / (σ_V·√T)
        PD_P = Φ(−DD)                        física (con μ, la deriva real)

    DD es la distancia al default: cuántas desviaciones estándar separan a
    los activos de la barrera. Es el número que KMV convierte en EDF.

    LA ADVERTENCIA. Una sola deuda, un solo vencimiento, activos que se
    mueven suavemente (sin saltos), sin liquidez ni refinanciamiento. Cemex
    en 2009 violó las cuatro: tenía muchos vencimientos, el problema era de
    refinanciamiento, y el precio de la acción saltaba. El modelo sirve para
    ordenar y para ver qué parámetro manda; no para la cifra exacta.

    Parámetros: V valor de activos, D deuda (punto de default), sigma_V
    volatilidad de activos, T años, r tasa libre de riesgo, mu deriva física
    (si None, se usa r y PD_P = PD_Q).
    """
    if V <= 0 or D <= 0 or sigma_V <= 0 or T <= 0:
        raise ValueError("V, D, sigma_V y T deben ser positivos")
    mu = r if mu is None else mu
    sT = sigma_V * math.sqrt(T)
    d1 = (math.log(V / D) + (r + 0.5 * sigma_V ** 2) * T) / sT
    d2 = d1 - sT
    E = V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2)
    DD = (math.log(V / D) + (mu - 0.5 * sigma_V ** 2) * T) / sT
    pd_q = norm.cdf(-d2)
    pd_p = norm.cdf(-DD)
    deuda = V - E                                   # valor de mercado de la deuda
    y = -math.log(deuda / D) / T                    # rendimiento de la deuda
    spread = y - r
    sigma_E = norm.cdf(d1) * sigma_V * V / E if E > 0 else float("nan")
    out = {"E": E, "deuda": deuda, "d1": d1, "d2": d2, "DD": DD,
           "PD_Q": pd_q, "PD_P": pd_p, "spread": spread, "sigma_E": sigma_E,
           "apalancamiento": D / V}
    if verbose:
        print("MERTON · la empresa como opción")
        print(f"   activos V = {V:,.1f} · deuda D = {D:,.1f} · σ_V = {sigma_V:.0%} · T = {T:g} años · r = {r:.2%}")
        print(f"   d1 = [ln({V:,.0f}/{D:,.0f}) + ({r:.3f} + {sigma_V:.2f}²/2)·{T:g}] / ({sigma_V:.2f}·√{T:g}) = {d1:.3f}")
        print(f"   d2 = d1 − σ_V·√T = {d1:.3f} − {sT:.3f} = {d2:.3f}")
        print(f"   E  = V·Φ(d1) − D·e^(−rT)·Φ(d2) = {E:,.1f}   (el equity es el call)")
        print(f"   DD = [ln(V/D) + (μ − σ_V²/2)·T] / (σ_V·√T) = {DD:.3f} desviaciones")
        print(f"   PD a {T:g} año(s): neutral al riesgo Φ(−d2) = {pd_q:.2%} · física Φ(−DD) = {pd_p:.2%}")
        print(f"   deuda a mercado = V − E = {deuda:,.1f} → spread implícito {spread*1e4:,.0f} pb")
    return out


def cadena_dd(V, D, sigma_V, T=1.0, mu=0.0, etiqueta=""):
    """Imprime la distancia al default como cadena legible, sin d1 ni d2:
    colchón → ajuste pequeño → sacudida en T → DD → PD. Devuelve (DD, PD)."""
    colchon = math.log(V / D)
    ajuste = (mu - 0.5 * sigma_V ** 2) * T
    sac = sigma_V * math.sqrt(T)
    DD = (colchon + ajuste) / sac
    PD = float(norm.cdf(-DD))
    if etiqueta:
        print(etiqueta)
    print(f"   colchón          ln(V/D)          = ln({V:g}/{D:g}) = {colchon:.3f}  ({colchon:.0%})")
    signo = "−" if ajuste < 0 else "+"
    print(f"   ajuste pequeño   (μ − σ_V²/2)·T   = ({mu:g} − {sigma_V:.2f}²/2)·{T:g} = {signo}{abs(ajuste):.3f}")
    print(f"   sacudida en T    σ_V·√T           = {sigma_V:.2f}·√{T:g} = {sac:.3f}")
    print(f"   DD = ({colchon:.3f} {signo} {abs(ajuste):.3f}) / {sac:.3f} = {DD:.2f} sacudidas")
    print(f"   PD = Φ(−{DD:.2f}) = {PD:.1%}   ← el área bajo la deuda en el abanico")
    return DD, PD


def cadena_opcion(V, D, sigma_V, T=1.0, r=0.0):
    """Imprime la lectura del balance como contrato: el equity es el call, la
    deuda es deuda segura menos la put, y la put es el spread. Devuelve el
    diccionario de merton()."""
    m = merton(V, D, sigma_V, T=T, r=r, mu=r, verbose=False)
    segura = D * math.exp(-r * T)
    put = segura - m["deuda"]
    print(f"   activos {V:g} · deuda {D:g} · sacudida {sigma_V:.0%} · r = {r:.0%}")
    print(f"   el equity es un call sobre los activos:    E = {m['E']:.1f}   (el colchón contable era {V - D:g}: el call vale más porque gana sin límite y pierde sólo hasta cero)")
    print(f"   la deuda es lo demás:                      V − E = {V:g} − {m['E']:.1f} = {m['deuda']:.1f}")
    print(f"   deuda segura menos la put:                 {segura:.1f} − put = {m['deuda']:.1f}  →  put = {put:.1f}")
    print(f"   la put como spread (Módulo XI):            {m['spread']*1e4:.0f} pb por año")
    return m


def merton_implicito(E, sigma_E, D, T=1.0, r=0.03, verbose=True):
    """DE LO OBSERVABLE A LO NO OBSERVABLE · las dos ecuaciones de Merton.

    V y σ_V no se observan; E (capitalización) y σ_E (volatilidad de la
    acción) sí. Dos ecuaciones, dos incógnitas:

        E     = V·Φ(d1) − D·e^(−rT)·Φ(d2)
        σ_E·E = Φ(d1)·σ_V·V            (lema de Itô: la acción es el call)

    Se resuelven numéricamente. El resultado es lo que un analista haría con
    Cemex en agosto de 2009: capitalización y volatilidad de CX en la mano,
    deuda del 20-F, y de ahí V, σ_V, DD y PD.

    Devuelve el diccionario de merton() más V y sigma_V.
    """
    if E <= 0 or sigma_E <= 0 or D <= 0:
        raise ValueError("E, sigma_E y D deben ser positivos")

    def ecuaciones(x):
        V, sV = x
        if V <= 0 or sV <= 0:
            return [1e6, 1e6]
        sT = sV * math.sqrt(T)
        d1 = (math.log(V / D) + (r + 0.5 * sV ** 2) * T) / sT
        d2 = d1 - sT
        e1 = V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2) - E
        e2 = norm.cdf(d1) * sV * V - sigma_E * E
        return [e1, e2]

    x0 = [E + D, sigma_E * E / (E + D)]
    sol, info, ier, msg = fsolve(ecuaciones, x0, full_output=True)
    if ier != 1:
        raise RuntimeError("merton_implicito no convergió: " + msg)
    V, sV = float(sol[0]), float(sol[1])
    out = merton(V, D, sV, T=T, r=r, verbose=False)
    out.update({"V": V, "sigma_V": sV})
    if verbose == "simple":
        print("DE LA ACCIÓN A LOS ACTIVOS")
        print(f"   lo que se ve:     la acción vale E = {E:,.2f} y se mueve σ_E = {sigma_E:.0%} al año; la deuda es D = {D:,.2f}")
        print(f"   lo que se despeja: activos V = {V:,.2f} (≈ E + D) · sacudida σ_V = {sV:.1%} (menos que σ_E: la acción amplifica)")
    elif verbose:
        print("MERTON IMPLÍCITO · de E y σ_E a V y σ_V")
        print(f"   observables: E = {E:,.1f} · σ_E = {sigma_E:.0%} · deuda D = {D:,.1f} · T = {T:g} · r = {r:.2%}")
        print(f"   solución:    V = {V:,.1f} · σ_V = {sV:.1%}   (σ_V < σ_E: el apalancamiento amplifica)")
        print(f"   DD = {out['DD']:.3f} · PD_Q = {out['PD_Q']:.2%} · spread implícito {out['spread']*1e4:,.0f} pb")
    return out


def merton_pasos(E, sigma_E, D, T=1.0, r=0.0, pasos=5, verbose=True):
    """DE LA ACCIÓN A LOS ACTIVOS, PASO POR PASO · la versión que se sigue a mano.

    PRIMER INTENTO: si la deuda valiera siempre D (no se moviera), cada peso que
    ganan o pierden los activos lo gana o pierde la acción:
        V ≈ E + D            σ_V ≈ σ_E · E / V
    Con E = 21.2, σ_E = 84 %, D = 80: V ≈ 101.2 y σ_V ≈ 17.6 %.

    SEGUNDO INTENTO: la deuda sí se mueve un poco cuando la empresa se acerca
    al problema (la put vale algo). Se corrige en pasos: con el V y σ_V que se
    tienen se calcula cuánto valdría la acción (el call); si sale más que la
    real, los activos valen un poco menos; y la fracción del movimiento de los
    activos que absorbe la acción no es 1 sino Φ(d1):
        V_nuevo  = (E + D·e^(−rT)·Φ(d2)) / Φ(d1)
        σ_V nuevo = σ_E · E / (V_nuevo · Φ(d1))
    En cuatro o cinco pasos se queda quieto: V = 100, σ_V = 20 %. Es lo que
    merton_implicito() hace con un algoritmo numérico.
    """
    V = E + D
    sV = sigma_E * E / V
    filas = [{"paso": 0, "V": V, "sigma_V": sV, "E_modelo": float("nan"), "Phi_d1": float("nan")}]
    for k in range(1, pasos + 1):
        d1 = (math.log(V / D) + (r + 0.5 * sV ** 2) * T) / (sV * math.sqrt(T))
        d2 = d1 - sV * math.sqrt(T)
        Em = V * norm.cdf(d1) - D * math.exp(-r * T) * norm.cdf(d2)
        V_new = (E + D * math.exp(-r * T) * norm.cdf(d2)) / norm.cdf(d1)
        sV_new = sigma_E * E / (V_new * norm.cdf(d1))
        filas.append({"paso": k, "V": V_new, "sigma_V": sV_new, "E_modelo": Em, "Phi_d1": norm.cdf(d1)})
        V, sV = V_new, sV_new
    df = pd.DataFrame(filas).set_index("paso")
    if verbose:
        print("DE LA ACCIÓN A LOS ACTIVOS · paso por paso")
        print(f"   primer intento (la deuda no se mueve): V = E + D = {E:.2f} + {D:.2f} = {E + D:.2f} · σ_V = σ_E·E/V = {sigma_E:.0%}·{E:.2f}/{E + D:.2f} = {df.loc[0, 'sigma_V']:.1%}")
        for k in range(1, pasos + 1):
            print(f"   paso {k}: con V = {df.loc[k-1, 'V']:.2f} y σ_V = {df.loc[k-1, 'sigma_V']:.1%} la acción valdría {df.loc[k, 'E_modelo']:.2f} (real {E:.2f}); "
                  f"la acción absorbe Φ(d1) = {df.loc[k, 'Phi_d1']:.3f} del movimiento → V = {df.loc[k, 'V']:.2f}, σ_V = {df.loc[k, 'sigma_V']:.1%}")
        print(f"   se queda quieto en V = {V:.2f}, σ_V = {sV:.1%}")
    return df


def distancia_default(V, D, sigma_V, T=1.0, mu=0.0):
    """DD = [ln(V/D) + (μ − σ_V²/2)·T] / (σ_V·√T). Sin impresión; para widgets."""
    return (math.log(V / D) + (mu - 0.5 * sigma_V ** 2) * T) / (sigma_V * math.sqrt(T))


def kmv_punto_default(deuda_corto, deuda_largo, verbose=True):
    """EL PUNTO DE DEFAULT EMPÍRICO DE KMV.

    Merton usa toda la deuda como barrera. KMV observó que las empresas
    incumplen antes de que los activos caigan bajo la deuda total, y después
    de que caigan bajo la de corto plazo: el punto de default «generally lies
    somewhere between total liabilities and current, or short-term,
    liabilities» (Crosbie y Bohn, 2002). La regla operativa que usa la
    literatura que replica KMV (Duffie, Saita y Wang 2007, Apéndice A;
    Bharath y Shumway 2008):

        punto de default = deuda de corto plazo + ½ · deuda de largo plazo

    Es una convención calibrada a incumplimientos observados, no un teorema.
    """
    dp = deuda_corto + 0.5 * deuda_largo
    if verbose:
        print(f"KMV · punto de default = corto + ½·largo = {deuda_corto:,.1f} + ½·{deuda_largo:,.1f} = {dp:,.1f}")
    return dp


def mapeo_edf(DD, tabla=None, verbose=True):
    """DE LA DISTANCIA AL DEFAULT A LA EDF · el paso Q → P de KMV.

    Merton da PD = Φ(−DD) bajo normalidad. KMV no se la cree: las colas
    reales son más pesadas y la normal subestima la PD cuando DD es grande.
    En vez de Φ, usan la frecuencia OBSERVADA de incumplimiento de empresas
    con esa DD en su base histórica: eso es la EDF (expected default
    frequency). Es el mismo movimiento del XI (de la PD de un precio a la
    PD física), ahora con una tabla empírica en lugar de una fórmula.

    La tabla propietaria no es pública; la del curso (MAPEO_EDF) es una
    construcción estilizada anclada en dos ilustraciones publicadas por
    Crosbie y Bohn (2002): DD 7 ↔ unos 5 pb (≈ AA) y DD 3.5 ↔ 25 pb (Philip
    Morris). Interpola log-linealmente. Se proyecta como «estilizada».
    """
    t = MAPEO_EDF if tabla is None else tabla
    DDc = float(np.clip(DD, t["DD"].min(), t["DD"].max()))
    edf = float(np.exp(np.interp(DDc, t["DD"], np.log(t["EDF"]))))
    pd_normal = float(norm.cdf(-DD))
    if verbose:
        print(f"MAPEO DD → EDF · DD = {DD:.2f}")
        print(f"   con la normal: Φ(−{DD:.2f}) = {pd_normal:.3%}")
        print(f"   con la tabla empírica (estilizada): EDF = {edf:.3%}"
              + ("   ← la cola real es más pesada" if edf > pd_normal else ""))
    return {"DD": DD, "EDF": edf, "PD_normal": pd_normal}


# ===========================================================================
# L2 · ALTMAN Z''
# ===========================================================================

def altman_zpp(X1=None, X2=None, X3=None, X4=None, *, capital_trabajo=None,
               utilidades_retenidas=None, ebit=None, capital_contable=None,
               activo_total=None, pasivo_total=None, coef=None, verbose=True):
    """ALTMAN Z'' · una señal contable de distress.

    EL PROBLEMA QUE EXISTÍA (1968). Predecir la quiebra con razones
    financieras, cuando no había precios de deuda ni modelos de opciones.

    CÓMO LO PLANTEÓ. Análisis discriminante: la combinación lineal de
    razones que mejor separa, en una muestra histórica, a las empresas que
    quebraron de las que no. La versión Z'' (1995) quita ventas/activos y
    usa capital contable en libros, para que sirva en empresas no
    manufactureras y en mercados emergentes:

        Z'' = 3.25 + 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4
        X1 = capital de trabajo / activo total      (liquidez)
        X2 = utilidades retenidas / activo total    (historia de rentabilidad)
        X3 = EBIT / activo total                    (rentabilidad operativa)
        X4 = capital contable / pasivo total        (apalancamiento)

    SU LUGAR HOY. Señal, no PD: no sale de una probabilidad y no se calibra
    a tasas de incumplimiento. Se usa como filtro de universo y como alarma
    temprana con rezago contable. Las zonas van en zona_zpp().

    Acepta las razones X1..X4 directamente, o los rubros del balance.
    """
    c = ALTMAN_ZPP if coef is None else coef
    if X1 is None:
        if None in (capital_trabajo, utilidades_retenidas, ebit, capital_contable, activo_total, pasivo_total):
            raise ValueError("Den las cuatro razones X1..X4 o los seis rubros del balance")
        X1 = capital_trabajo / activo_total
        X2 = utilidades_retenidas / activo_total
        X3 = ebit / activo_total
        X4 = capital_contable / pasivo_total
    z = c["const"] + c["X1"] * X1 + c["X2"] * X2 + c["X3"] * X3 + c["X4"] * X4
    aportes = {"X1": c["X1"] * X1, "X2": c["X2"] * X2, "X3": c["X3"] * X3, "X4": c["X4"] * X4}
    zona = zona_zpp(z, verbose=False)
    if verbose:
        print("ALTMAN Z''")
        print(f"   Z'' = {c['const']} + {c['X1']}·{X1:.3f} + {c['X2']}·{X2:.3f} + {c['X3']}·{X3:.3f} + {c['X4']}·{X4:.3f}")
        print(f"       = {c['const']} + {aportes['X1']:.2f} + {aportes['X2']:.2f} + {aportes['X3']:.2f} + {aportes['X4']:.2f} = {z:.2f}  → zona {zona}")
        peso = max(aportes, key=lambda k: abs(aportes[k]))
        print(f"   la razón que más pesa: {peso} (aporta {aportes[peso]:+.2f})")
    return {"Z": z, "zona": zona, "X": {"X1": X1, "X2": X2, "X3": X3, "X4": X4}, "aportes": aportes}


def zona_zpp(z, zonas=None, verbose=True):
    """Zonas del Z'' con la constante 3.25: distress < 4.35, gris 4.35–5.85,
    segura > 5.85 (los cortes 1.10 y 2.60 de Altman desplazados por la
    constante). PENDIENTE DE VERIFICAR contra Altman (2005)."""
    zz = ZONAS_ZPP if zonas is None else zonas
    k = ALTMAN_ZPP["const"]
    if z < zz["distress"] + k:
        zona = "distress"
    elif z < zz["segura"] + k:
        zona = "gris"
    else:
        zona = "segura"
    if verbose:
        print(f"Z'' = {z:.2f} → zona {zona}  (cortes: {zz['distress']+k:.2f} y {zz['segura']+k:.2f})")
    return zona


# ===========================================================================
# L3 · FORMA REDUCIDA CON COVARIABLES
# ===========================================================================

def intensidad_covariables(DD, ret=0.0, r3m=0.0, sp500=0.0, coef=None, verbose=True):
    """LA INTENSIDAD COMO FUNCIÓN DE COVARIABLES · Duffie, Saita y Wang (2007).

    EL HILO DE LA INTENSIDAD, segunda aparición. En el XI, λ salía del
    precio. Aquí λ es una función de variables observables del emisor y del
    ciclo:

        λ(x) = exp(β0 + β_DD·DD + β_ret·ret + β_r·r3m + β_sp·SP500)

        DD      distancia al default del emisor (de Merton / KMV)
        ret     rendimiento de la acción del emisor en el último año
        r3m     tasa del Tesoro a 3 meses, EN POR CIENTO (2.0 = 2 %)
        SP500   rendimiento del S&P 500 en el último año (fracción)

    EL RESULTADO. La distancia al default es, por mucho, la covariable más
    fuerte: el modelo estructural (Merton) y el de forma reducida (una
    intensidad) convergen porque la mejor covariable de la intensidad es la
    salida del modelo estructural. Y el ciclo entra por las macro: cuando
    todas las DD caen a la vez, todas las λ suben a la vez, que es
    exactamente lo que un CRO ve en marzo de 2020.

    LA ADVERTENCIA. El exponencial hace que la respuesta a la DD sea
    multiplicativa: bajar la DD de 4 a 2 no suma, multiplica (e^(1.129·2) ≈
    9.6 veces). Los coeficientes vienen de 2,770 industriales de EEUU
    1980–2004 (497 incumplimientos), con la DD calculada como KMV (volatilidad
    histórica de activos, horizonte de un año); para un banco mexicano son el
    orden de magnitud y los signos, no la calibración.

    Coeficientes: DSW_COEF, Tabla 2 de Duffie, Saita y Wang (2007), fila
    «Default». Devuelve λ anual y la PD a un año 1 − e^(−λ).
    """
    c = DSW_COEF if coef is None else coef
    expo = c["const"] + c["DD"] * DD + c["ret"] * ret + c["r3m"] * r3m + c["sp500"] * sp500
    lam = math.exp(expo)
    pd1 = 1 - math.exp(-lam)
    if verbose:
        print("INTENSIDAD CON COVARIABLES · λ(x) = exp(β·x)")
        print(f"   exponente = {c['const']:+.3f} {c['DD']:+.3f}·{DD:.2f} {c['ret']:+.3f}·{ret:.2f} {c['r3m']:+.3f}·{r3m:.2f} {c['sp500']:+.3f}·{sp500:.2f} = {expo:+.3f}")
        print(f"   λ = e^({expo:+.3f}) = {lam:.4f} por año = {lam*1e4:,.0f} pb · PD a un año = 1 − e^(−λ) = {pd1:.2%}")
        print(f"   coeficientes: {c.get('fuente', 'sin fuente')}")
    return {"lambda": lam, "PD_1a": pd1, "exponente": expo}


# ===========================================================================
# L4 · VASICEK, IRB Y CONCENTRACIÓN
# ===========================================================================

def vasicek_pd_condicional(PD, rho, Z):
    """PD condicional al factor sistemático Z (Z < 0 es un año malo):
    PD(Z) = Φ( (Φ⁻¹(PD) − √ρ·Z) / √(1−ρ) ). Sin impresión; para widgets."""
    return float(norm.cdf((norm.ppf(PD) - math.sqrt(rho) * Z) / math.sqrt(1 - rho)))


def vasicek_cuantil(PD, rho, q=0.999, LGD=0.45, verbose=True):
    """VASICEK · la distribución de pérdida de un portafolio con un factor.

    EL PROBLEMA QUE EXISTÍA. Sumar las PD de cien acreditados da la pérdida
    ESPERADA del portafolio, no su distribución; y la distribución es lo que
    decide el capital. Si los incumplimientos fueran independientes, cien
    nombres bastarían para que la pérdida fuera casi segura. No lo son: el
    ciclo los mueve juntos.

    CÓMO LO PLANTEÓ VASICEK (1987, 2002). Cada acreditado incumple cuando su
    variable latente (sus «activos», como en Merton) cae bajo un umbral, y
    esa variable tiene un componente común a todos:

        X_i = √ρ·Z + √(1−ρ)·ε_i          Z el factor del ciclo, ε_i lo propio

    Condicional a Z, los incumplimientos son independientes, y en una
    cartera grande (granular) la pérdida converge a su media condicional:

        PD(Z) = Φ( (Φ⁻¹(PD) − √ρ·Z) / √(1−ρ) )

    El cuantil q de la pérdida es la PD condicional al cuantil del factor:

        L_q = LGD · Φ( (Φ⁻¹(PD) + √ρ·Φ⁻¹(q)) / √(1−ρ) )
        capital = L_q − EL = L_q − PD·LGD        (EL se provisiona, UL se capitaliza)

    LA ADVERTENCIA. Un solo factor, cartera infinitamente granular, ρ
    constante. Un nombre que pesa 10 % del portafolio rompe la granularidad;
    una crisis sube ρ. Por eso existen los límites de concentración y el
    escenario de estrés sobre ρ.
    """
    if not (0 < PD < 1) or not (0 <= rho < 1) or not (0 < q < 1):
        raise ValueError("PD y q en (0,1); rho en [0,1)")
    a = norm.ppf(PD)
    zq = norm.ppf(q)
    pd_q = float(norm.cdf((a + math.sqrt(rho) * zq) / math.sqrt(1 - rho)))
    L_q = LGD * pd_q
    EL = PD * LGD
    cap = L_q - EL
    if verbose:
        print("VASICEK · cuantil de la pérdida")
        print(f"   PD = {PD:.2%} · ρ = {rho:.2f} · q = {q:.1%} · LGD = {LGD:.0%}")
        print(f"   PD condicional al cuantil = Φ((Φ⁻¹({PD:.4f}) + √{rho:.2f}·Φ⁻¹({q})) / √(1−{rho:.2f}))")
        print(f"                             = Φ(({a:.3f} + {math.sqrt(rho):.3f}·{zq:.3f}) / {math.sqrt(1-rho):.3f}) = {pd_q:.2%}")
        print(f"   L_q = LGD·PD_q = {LGD:.2f}·{pd_q:.4f} = {L_q:.2%} del portafolio")
        print(f"   EL = PD·LGD = {EL:.2%} → capital = L_q − EL = {cap:.2%}  ({cap/EL:,.1f} veces la EL)")
    return {"PD_q": pd_q, "L_q": L_q, "EL": EL, "capital": cap}


def vasicek_dist(PD, rho, LGD=0.45, puntos=400):
    """Densidad de la pérdida de Vasicek sobre una malla (para graficar).
    f(L) con L = LGD·p, p la fracción que incumple:
    f(p) = √((1−ρ)/ρ) · exp( ½[Φ⁻¹(p)² − ((√(1−ρ)·Φ⁻¹(p) − Φ⁻¹(PD))/√ρ)²] )."""
    if rho <= 0:
        raise ValueError("vasicek_dist requiere rho > 0")
    p = np.linspace(1e-4, 0.999, puntos)
    a = norm.ppf(PD)
    x = norm.ppf(p)
    f = math.sqrt((1 - rho) / rho) * np.exp(0.5 * (x ** 2 - ((math.sqrt(1 - rho) * x - a) / math.sqrt(rho)) ** 2))
    return pd.DataFrame({"perdida": LGD * p, "densidad": f / LGD})


def irb_rho(PD):
    """Correlación de activos de Basilea para exposiciones corporativas
    (CRE31): ρ = 0.12·w + 0.24·(1−w), con w = (1 − e^(−50·PD)) / (1 − e^(−50)).
    Va de 0.24 (PD → 0) a 0.12 (PD alta): los buenos créditos son más
    sistemáticos; los malos, más propios."""
    w = (1 - math.exp(-50 * PD)) / (1 - math.exp(-50))
    return 0.12 * w + 0.24 * (1 - w)


def irb_capital(PD, LGD=0.45, M=2.5, verbose=True):
    """EL CAPITAL IRB DE BASILEA · Vasicek con los parámetros del regulador.

        ρ(PD)  = 0.12·w + 0.24·(1−w),  w = (1 − e^(−50·PD)) / (1 − e^(−50))
        b(PD)  = (0.11852 − 0.05478·ln PD)²                       ajuste por plazo
        K      = [ LGD·Φ((Φ⁻¹(PD) + √ρ·Φ⁻¹(0.999)) / √(1−ρ)) − PD·LGD ]
                 · (1 + (M − 2.5)·b) / (1 − 1.5·b)

    K es capital por unidad de exposición; los activos ponderados son
    12.5·K·EAD. Es exactamente vasicek_cuantil() al 99.9 % con ρ fijada por
    la fórmula y un ajuste por vencimiento M. En México la CUB lo recoge
    para el método basado en calificaciones internas; la mayoría de los
    bancos usa el método estándar (investigación XII, bloque 6).
    """
    rho = irb_rho(PD)
    b = (0.11852 - 0.05478 * math.log(PD)) ** 2
    base = vasicek_cuantil(PD, rho, q=0.999, LGD=LGD, verbose=False)
    K = base["capital"] * (1 + (M - 2.5) * b) / (1 - 1.5 * b)
    if verbose:
        print("CAPITAL IRB (corporativos)")
        print(f"   PD = {PD:.2%} → ρ(PD) = {rho:.3f} · b(PD) = {b:.4f} · M = {M:g} · LGD = {LGD:.0%}")
        print(f"   Vasicek al 99.9 %: L_q = {base['L_q']:.2%} · EL = {base['EL']:.2%} · UL = {base['capital']:.2%}")
        print(f"   K = UL · (1 + (M − 2.5)·b) / (1 − 1.5·b) = {K:.2%} de la exposición · RWA = 12.5·K = {12.5*K:.0%}")
    return {"rho": rho, "b": b, "K": K, "RWA": 12.5 * K, **base}


def hhi(pesos, verbose=True):
    """ÍNDICE DE HERFINDAHL · cuántos nombres «efectivos» tiene el portafolio.

        HHI = Σ w_i²           n_efectivo = 1 / HHI

    Veinte nombres iguales: HHI = 0.05, n_efectivo = 20. Veinte nombres con
    uno al 30 %: HHI ≈ 0.115, n_efectivo ≈ 8.7. La granularidad que Vasicek
    supone se mide aquí.
    """
    w = np.asarray(pesos, dtype=float)
    w = w / w.sum()
    h = float((w ** 2).sum())
    n_ef = 1 / h
    if verbose:
        print(f"HHI = Σ w² = {h:.4f} → nombres efectivos = 1/HHI = {n_ef:.1f} (nominales: {len(w)})")
    return {"HHI": h, "n_efectivo": n_ef, "n": len(w)}


def capital_concentracion(PD, rho, pesos, LGD=0.45, q=0.999, n_sim=100_000, semilla=7, verbose=True):
    """CAPITAL CON UN NOMBRE GRANDE · lo que Vasicek no ve.

    Vasicek supone cartera granular. Con un nombre que pesa mucho, la
    pérdida ya no converge a su media condicional: hay un salto si ESE
    nombre incumple. Se simula el modelo de un factor con los pesos reales
    y se compara el cuantil simulado con el de Vasicek. La diferencia es el
    costo de la concentración, y es lo que un límite por nombre acota. En
    México (CUB, desde octubre de 2023): 25 % de la parte básica del capital
    neto por Gran Exposición, 15 % entre bancos sistémicos y hasta 100 % para
    paraestatales federales como Pemex (LIMITES_CUB).

    Mismo PD y ρ para todos los nombres (simplificación del laboratorio);
    el peso del mayor es lo que el alumno mueve.
    """
    rng = np.random.default_rng(semilla)
    w = np.asarray(pesos, dtype=float)
    w = w / w.sum()
    n = len(w)
    umbral = norm.ppf(PD)
    Z = rng.standard_normal((n_sim, 1))
    eps = rng.standard_normal((n_sim, n))
    X = math.sqrt(rho) * Z + math.sqrt(1 - rho) * eps
    perdidas = LGD * ((X < umbral) * w).sum(axis=1)
    L_sim = float(np.quantile(perdidas, q))
    EL = PD * LGD
    vas = vasicek_cuantil(PD, rho, q=q, LGD=LGD, verbose=False)
    out = {"L_q_simulado": L_sim, "capital_simulado": L_sim - EL,
           "L_q_vasicek": vas["L_q"], "capital_vasicek": vas["capital"],
           "costo_concentracion": (L_sim - EL) - vas["capital"], "peso_mayor": float(w.max()),
           **hhi(w, verbose=False)}
    if verbose:
        print("CAPITAL CON CONCENTRACIÓN (simulación de un factor)")
        print(f"   {n} nombres · peso del mayor {w.max():.0%} · nombres efectivos {out['n_efectivo']:.1f} · PD = {PD:.2%} · ρ = {rho:.2f}")
        print(f"   capital Vasicek (granular) = {vas['capital']:.2%} · capital simulado = {out['capital_simulado']:.2%}")
        print(f"   costo de la concentración = {out['costo_concentracion']:+.2%} del portafolio")
    return out


# ===========================================================================
# CAPA DE LABORATORIO · bitácora, tarea, preguntas
# ===========================================================================

class Bitacora:
    """La bitácora del alumno: apuestas, parámetros, resultados, retos y
    respuestas, con nombre y hora. Es el insumo del memo (tarea) y de las
    preguntas P1–P3 (convenciones §2.7 y §2.8).

    Se GUARDA SOLA en un archivo JSON cada vez que registra algo, y al crearla
    con el mismo nombre y carpeta vuelve a cargar lo que ya había: si Colab se
    desconecta o se vuelve a correr la celda del nombre, no se pierde nada.
    Con carpeta en Google Drive sobrevive también al cierre de la sesión."""

    def __init__(self, nombre="(sin nombre)", carpeta="."):
        self.nombre = nombre
        self.carpeta = carpeta
        self.eventos = []
        os.makedirs(carpeta, exist_ok=True)
        self.ruta = os.path.join(carpeta, f"bitacora_{self._slug()}.json")
        if os.path.exists(self.ruta):
            with open(self.ruta, encoding="utf-8") as f:
                self.eventos = json.load(f).get("eventos", [])

    def _slug(self):
        return "".join(ch if ch.isalnum() else "_" for ch in self.nombre).strip("_") or "sin_nombre"

    def registrar(self, lab, tipo, **datos):
        """tipo ∈ {'apuesta', 'parametros', 'resultado', 'reto', 'respuesta', 'nota'}."""
        ev = {"hora": _dt.datetime.now().isoformat(timespec="seconds"), "alumno": self.nombre,
              "lab": lab, "tipo": tipo, **datos}
        self.eventos.append(ev)
        self.exportar(self.ruta)
        return ev

    def tabla(self, lab=None):
        df = pd.DataFrame(self.eventos)
        return df if lab is None or df.empty else df[df["lab"] == lab].reset_index(drop=True)

    def ultimo(self, lab, tipo):
        for ev in reversed(self.eventos):
            if ev["lab"] == lab and ev["tipo"] == tipo:
                return ev
        return None

    def resumen(self):
        n = {}
        for ev in self.eventos:
            n.setdefault(ev["lab"], set()).add(ev["tipo"])
        partes = [f"{lab}: {', '.join(sorted(t))}" for lab, t in sorted(n.items())]
        return f"bitácora de {self.nombre} · {len(self.eventos)} registros · {' | '.join(partes) or 'vacía'} · archivo: {self.ruta}"

    def exportar(self, ruta=None):
        ruta = ruta or self.ruta
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"alumno": self.nombre, "eventos": self.eventos}, f, ensure_ascii=False, indent=1)
        return ruta

    @classmethod
    def cargar(cls, ruta):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        b = cls.__new__(cls)
        b.nombre = d["alumno"]; b.carpeta = os.path.dirname(ruta) or "."; b.ruta = ruta
        b.eventos = d["eventos"]
        return b


LABS = {
    "L1": {"modelo": "Merton", "rol": "acreedor de Cemex en el sindicato de bancos", "fecha": "agosto de 2009",
           "decision": "extender el plazo de ~15 mmdd, exigir capital, o las dos",
           "parametro_clave": "sigma_V", "resultado_clave": "PD_P"},
    "L2": {"modelo": "Altman Z''", "rol": "gestor con mandato que prohíbe emisores en distress", "fecha": "hoy, con sus empresas de EDGAR",
           "decision": "qué empresas salen del universo y por qué variable",
           "parametro_clave": "X3", "resultado_clave": "Z"},
    "L3": {"modelo": "forma reducida (intensidad con covariables)", "rol": "director de riesgos de un banco mexicano", "fecha": "marzo de 2020",
           "decision": "subir las provisiones ya, con el modelo, o esperar los atrasos",
           "parametro_clave": "DD", "resultado_clave": "lambda"},
    "L4": {"modelo": "Vasicek", "rol": "comité de crédito que fija el límite por nombre", "fecha": "2026, con los números de Pemex del XI",
           "decision": "cuánto capital cuesta subir la exposición de 5 % a 10 %, y qué pasa si ρ sube en crisis",
           "parametro_clave": "rho", "resultado_clave": "capital"},
}


def tarea(bitacora, lab, ruta=None):
    """GENERA EL MEMO DE DECISIÓN (convenciones §2.8): encabezado y sección 1
    llenos desde la bitácora, las respuestas escritas en clase (celdas RESPONDE)
    pegadas en la sección 3, y las demás secciones con dos líneas de instrucción.
    Escribe el memo en .md y en .docx (para terminarlo en Word y exportar PDF) y
    el anexo JSON de la bitácora. Imprime dónde quedó cada archivo."""
    if lab not in LABS:
        raise ValueError("lab debe ser L1, L2, L3 o L4")
    L = LABS[lab]
    ap = bitacora.ultimo(lab, "apuesta")
    res = bitacora.ultimo(lab, "resultado")
    if res is None:
        raise ValueError(f"La bitácora no tiene resultados de {lab}: juega el laboratorio antes de generar la tarea.")
    if ap is None:
        print(f"AVISO: no registraste tu apuesta en {lab}; el memo lo dirá y la sección 1 vale menos.")
        ap = {"valor": "(no registrada)", "hora": "—"}
    apuesta = ap.get("valor", "(sin registrar)")
    salio = res.get(L["resultado_clave"], res.get("valor", "(sin registrar)"))
    if isinstance(salio, float):
        salio = f"{salio:.2%}" if salio < 1 else f"{salio:.2f}"
    comp = bitacora.ultimo(lab, "nota") or {}
    resp = bitacora.ultimo(lab, "respuesta") or {}
    reto = bitacora.ultimo(lab, "reto") or {}
    reto_txt = ", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in reto.items() if k not in ("hora", "alumno", "lab", "tipo")) or "(sin registrar)"
    md = f"""# Memo de decisión · Módulo XII · {lab} {L['modelo']}

**Alumno:** {bitacora.nombre}  ·  **Rol:** {L['rol']}  ·  **Fecha del caso:** {L['fecha']}
**Decisión:** {L['decision']}

## 1 · Mi predicción y lo que salió

- Aposté: **{apuesta}**  (registrado {ap['hora']})
- Salió: **{salio}**  (registrado {res['hora']}){'  ·  ' + comp.get('compara', '') if comp.get('compara') else ''}
- Mi reto: {reto_txt}
- Por cuánto fallé y qué supuesto mío estaba mal: _(escribe aquí; una o dos oraciones)_

## 2 · La decisión

_Qué le recomiendas al rol y con qué número del modelo la sostienes._
_Tabla 2 × 2: tu decisión con los parámetros base, y con el parámetro que más la cambia._

| | parámetros base | con el parámetro que más la cambia |
|---|---|---|
| decisión | | |
| número que la sostiene | | |

## 3 · Lo que el modelo no te dijo

_El hueco del laboratorio, explicado con el caso real: qué hizo el actor real que el modelo no habría recomendado._

Lo que escribiste en clase: {resp.get('texto', '_(no registraste respuesta en la celda RESPONDE)_')}

## 4 · La variable que compraría

_Si sólo pudieras conocer con certeza un parámetro, ¿cuál, y cuánto vale saberlo? Sensibilidad en palabras._

## 5 · Anexo

Bitácora de {lab} adjunta (archivo JSON generado por el cuaderno). Sin anexo no hay reporte.
"""
    ruta = ruta or os.path.join(bitacora.carpeta, f"memo_{lab}_{bitacora._slug()}.md")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write(md)
    anexo = bitacora.exportar(ruta.replace(".md", "_bitacora.json"))
    docx_ruta = _memo_docx(md, ruta.replace(".md", ".docx"))
    print("MEMO GENERADO")
    print(f"   Word:     {docx_ruta}   ← ábranlo, completen las secciones 1 a 4 y exporten a PDF")
    print(f"   Markdown: {ruta}")
    print(f"   Anexo:    {anexo}   ← se entrega junto con el PDF")
    print("   En Colab: panel de archivos (ícono de carpeta a la izquierda) → clic derecho sobre cada archivo → Descargar.")
    print("   Si guardaron la bitácora en Drive, los tres archivos ya están en esa carpeta.")
    return ruta


def _memo_docx(md, ruta):
    """Convierte el memo (Markdown sencillo) a .docx con python-docx; si no está
    instalado, devuelve None y el memo queda sólo en Markdown."""
    try:
        import docx
    except ImportError:
        return None
    d = docx.Document()
    tabla = []
    for linea in md.splitlines():
        if linea.startswith("| ") and "|" in linea[2:]:
            celdas = [c.strip() for c in linea.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in celdas):
                continue
            tabla.append(celdas)
            continue
        if tabla:
            t = d.add_table(rows=len(tabla), cols=len(tabla[0])); t.style = "Table Grid"
            for i, fila in enumerate(tabla):
                for j, c in enumerate(fila):
                    t.cell(i, j).text = c
            tabla = []
        if linea.startswith("# "):
            d.add_heading(linea[2:], level=1)
        elif linea.startswith("## "):
            d.add_heading(linea[3:], level=2)
        elif linea.startswith("- "):
            d.add_paragraph(linea[2:].replace("**", ""), style="List Bullet")
        elif linea.strip():
            d.add_paragraph(linea.replace("**", "").replace("_", ""))
    d.save(ruta)
    return ruta


def preguntas_p1p3(bitacora, lab):
    """GENERA P1–P3 PARA EL PROFESOR desde la bitácora de un alumno
    (convenciones §2.8). P1 usa el número base del alumno; P2 su parámetro
    clave; P3 es la del hueco. Devuelve texto imprimible."""
    L = LABS[lab]
    res = bitacora.ultimo(lab, "resultado") or {}
    par = bitacora.ultimo(lab, "parametros") or {}
    base = res.get(L["resultado_clave"], None)
    pk = L["parametro_clave"]
    pv = par.get(pk, None)
    def fmt(v, estilo):
        if v is None:
            return "(sin registro)"
        return {"pct": f"{v:.2%}", "pb": f"{v*1e4:,.0f} pb", "num1": f"{v:.1f}", "num2": f"{v:.2f}"}[estilo]
    p1 = {
        "L1": f"Tu PD física base fue {fmt(base, 'pct')} con σ_V = {fmt(pv, 'pct')}. Sin computadora: si σ_V sube 50 %, ¿la PD queda cerca de la mitad, del doble o de cinco veces? ¿Por qué ese orden?",
        "L2": f"Tu Z'' base fue {fmt(base, 'num2')}. Sin computadora: si el EBIT/activos (X3) cae a la mitad, ¿Z'' baja menos de 1 punto, entre 1 y 3, o más de 3? ¿Por qué?",
        "L3": f"Tu λ base fue {fmt(base, 'pb')} con DD = {fmt(pv, 'num1')}. Sin computadora: si la DD cae a la mitad, ¿λ sube menos del doble, entre 2 y 5 veces, o más de 5? ¿Qué parte de la fórmula lo decide?",
        "L4": f"Tu capital base fue {fmt(base, 'pct')} del portafolio con ρ = {fmt(pv, 'num2')}. Sin computadora: si ρ se duplica, ¿el capital sube menos de 50 %, entre 50 % y 150 %, o más? ¿Por qué?",
    }[lab]
    p2 = f"Recomendaste una decisión en tu memo. ¿Con qué cambio de UN solo parámetro habrías recomendado lo contrario? Da el parámetro y la dirección."
    p3 = f"Nombra una cosa que pasó en el caso real ({L['fecha']}) que {L['modelo']} no podía ver, y qué dato habrías pedido para verla."
    txt = f"{bitacora.nombre} · {lab} {L['modelo']}\n  P1  {p1}\n  P2  {p2}\n  P3  {p3}\n"
    return txt


def revisa(**kw):
    """Para el cuaderno del alumno: lanza ValueError nombrando la variable que
    quedó en None. Ningún hueco falla en silencio (convenciones §5.3)."""
    faltan = [k for k, v in kw.items() if v is None]
    if faltan:
        raise ValueError("Falta completar: " + ", ".join(faltan)
                         + " — busquen la línea marcada COMPLETEN ESTA LÍNEA.")
    return True
