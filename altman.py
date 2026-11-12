"""
===========================================================================
MÓDULO XII · BLOQUE L2 · ALTMAN Z'' — maquinaria del laboratorio
Riesgos Financieros (FIN516) · Universidad Panamericana · Gerardo Alcalá
===========================================================================

El modelo de cuando NO hay acción en bolsa: sólo el balance. Cuatro razones
que cualquiera lee en un estado financiero, una mezcla de pesos que Altman
encontró separando empresas que quebraron de empresas parecidas que no, y
un número que ordena: Z''. Es una SEÑAL ORDINAL, no una probabilidad.

Importa de empresas.py los coeficientes y las dos funciones base
(altman_zpp, zona_zpp) y agrega lo que este bloque necesita:

    DOS_EMPRESAS              las dos empresas de juguete (activo total 100)
    razones(balance)          las cuatro razones X1..X4 con su lectura
    zpp_en_cadena(balance)    razón → qué mide → valor → peso → aporte → suma
    x3_para_salir(...)        cuánto EBIT/activos hace falta para salir de distress
    MUESTRA                   tabla estilizada con la trampa (la de menor EBIT es
                              segura; la de EBIT alto está en distress)
    empresas_edgar(tickers)   razones desde el API público de la SEC (Colab)
    cargar_empresas(...)      mis_empresas.csv → EDGAR → MUESTRA, en ese orden

Fuentes: Altman (2000) para coeficientes y constante [V]; Altman, Hartzell
y Peck (1995) para el origen (Eurobonos mexicanos) [V]; cortes de zona 1.10
y 2.60 de fuente secundaria (Altman et al. 2017 vía Wikipedia) [S]. Ver
investigacion_xii.md §5.
"""
import os
import numpy as np
import pandas as pd

from empresas import ALTMAN_ZPP, ZONAS_ZPP, altman_zpp, zona_zpp, revisa, Bitacora, tarea, preguntas_p1p3  # noqa: F401

__all__ = ["ALTMAN_ZPP", "ZONAS_ZPP", "altman_zpp", "zona_zpp", "revisa", "Bitacora", "tarea",
           "DOS_EMPRESAS", "MUESTRA", "COLUMNAS", "razones", "zpp_en_cadena", "x3_para_salir",
           "empresas_edgar", "cargar_empresas", "tabla_zpp", "CORTES"]

COLUMNAS = ["ticker", "activo_total", "pasivo_total", "activo_circulante", "pasivo_circulante",
            "utilidades_retenidas", "ebit", "capital_contable"]

# Cortes de zona con la constante 3.25 ya sumada (1.10 + 3.25 y 2.60 + 3.25) [S]
CORTES = {"distress": ZONAS_ZPP["distress"] + ALTMAN_ZPP["const"], "segura": ZONAS_ZPP["segura"] + ALTMAN_ZPP["const"]}

# Dos empresas DE JUGUETE con activo total 100. Son construcción del curso
# (estilizadas): sirven para ver en qué se nota, en el balance, la que quiebra.
# Las dos ganan casi lo mismo en operación (EBIT 12 contra 11); la diferencia
# está en el capital de trabajo, en la historia (utilidades retenidas) y en la deuda.
DOS_EMPRESAS = pd.DataFrame({
    "ticker":               ["Sobrevivió", "Quebró"],
    "activo_total":         [100.0, 100.0],
    "pasivo_total":         [40.0, 85.0],
    "activo_circulante":    [35.0, 25.0],
    "pasivo_circulante":    [20.0, 35.0],
    "utilidades_retenidas": [30.0, -25.0],
    "ebit":                 [12.0, 11.0],
    "capital_contable":     [60.0, 15.0],
})

# Tabla MUESTRA estilizada para correr el laboratorio sin red. Diseñada para
# que la empresa de MENOR EBIT (DELTA) sea segura y una de EBIT alto (BRAVO)
# esté en distress: la trampa del PREDICE.
MUESTRA = pd.DataFrame({
    "ticker":               ["ALFA", "BRAVO", "CHARLIE", "DELTA", "ECO"],
    "activo_total":         [100.0, 100.0, 100.0, 100.0, 100.0],
    "pasivo_total":         [40.0, 85.0, 60.0, 45.0, 70.0],
    "activo_circulante":    [35.0, 25.0, 30.0, 30.0, 20.0],
    "pasivo_circulante":    [20.0, 35.0, 25.0, 15.0, 22.0],
    "utilidades_retenidas": [30.0, -25.0, 10.0, 20.0, 5.0],
    "ebit":                 [12.0, 11.0, 4.0, 1.0, 7.0],
    "capital_contable":     [60.0, 15.0, 40.0, 55.0, 30.0],
})

LECTURA = {
    "X1": ("capital de trabajo / activo total", "liquidez: lo que queda del activo circulante después de pagar lo que vence pronto"),
    "X2": ("utilidades retenidas / activo total", "historia: lo que la empresa ha ganado y guardado a lo largo de los años; negativa si lleva años perdiendo"),
    "X3": ("EBIT / activo total", "rentabilidad operativa: cuánto gana el negocio por cada peso de activos, antes de intereses e impuestos"),
    "X4": ("capital contable en libros / pasivo total", "apalancamiento: cuánto capital de los dueños respalda cada peso que se debe"),
}


def _fila(balance):
    """Acepta una fila (Series) o un dict con los rubros; devuelve un dict."""
    if isinstance(balance, pd.Series):
        return balance.to_dict()
    return dict(balance)


def razones(balance, verbose=True):
    """LAS CUATRO RAZONES, una por una, con su lectura en palabras.

        X1 = (activo circulante − pasivo circulante) / activo total
        X2 = utilidades retenidas / activo total
        X3 = EBIT / activo total
        X4 = capital contable (en libros) / pasivo total
    """
    b = _fila(balance)
    X = {
        "X1": (b["activo_circulante"] - b["pasivo_circulante"]) / b["activo_total"],
        "X2": b["utilidades_retenidas"] / b["activo_total"],
        "X3": b["ebit"] / b["activo_total"],
        "X4": b["capital_contable"] / b["pasivo_total"],
    }
    if verbose:
        nombre = b.get("ticker", "la empresa")
        print(f"RAZONES DE {nombre}")
        print(f"   X1 = (activo circulante − pasivo circulante) / activo total = ({b['activo_circulante']:.0f} − {b['pasivo_circulante']:.0f}) / {b['activo_total']:.0f} = {X['X1']:+.2f}   ← {LECTURA['X1'][1]}")
        print(f"   X2 = utilidades retenidas / activo total = {b['utilidades_retenidas']:.0f} / {b['activo_total']:.0f} = {X['X2']:+.2f}   ← {LECTURA['X2'][1]}")
        print(f"   X3 = EBIT / activo total = {b['ebit']:.0f} / {b['activo_total']:.0f} = {X['X3']:+.2f}   ← {LECTURA['X3'][1]}")
        print(f"   X4 = capital contable / pasivo total = {b['capital_contable']:.0f} / {b['pasivo_total']:.0f} = {X['X4']:+.2f}   ← {LECTURA['X4'][1]}")
    return X


def zpp_en_cadena(balance, verbose=True):
    """EL Z'' EN CADENA · razón → valor → peso → aporte, y la suma.

    Z'' = 3.25 + 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4
    zonas: distress < 4.35 · gris 4.35–5.85 · segura > 5.85
    """
    b = _fila(balance)
    X = razones(b, verbose=False)
    c = ALTMAN_ZPP
    aportes = {k: c[k] * X[k] for k in ("X1", "X2", "X3", "X4")}
    Z = c["const"] + sum(aportes.values())
    zona = zona_zpp(Z, verbose=False)
    tabla = pd.DataFrame({
        "razón": ["constante", "X1 liquidez", "X2 historia", "X3 rentabilidad", "X4 apalancamiento"],
        "valor": [np.nan, X["X1"], X["X2"], X["X3"], X["X4"]],
        "peso": [np.nan, c["X1"], c["X2"], c["X3"], c["X4"]],
        "aporte": [c["const"], aportes["X1"], aportes["X2"], aportes["X3"], aportes["X4"]],
    })
    if verbose:
        nombre = b.get("ticker", "la empresa")
        print(f"Z'' DE {nombre}, EN CADENA")
        print(f"   constante                                   {c['const']:+.2f}")
        for k, nom in (("X1", "liquidez"), ("X2", "historia"), ("X3", "rentabilidad"), ("X4", "apalancamiento")):
            print(f"   {k} {nom:15s} {X[k]:+.2f} × {c[k]:.2f} = {aportes[k]:+.2f}")
        print(f"   suma: Z'' = {c['const']:.2f} {aportes['X1']:+.2f} {aportes['X2']:+.2f} {aportes['X3']:+.2f} {aportes['X4']:+.2f} = {Z:.2f}  →  zona {zona.upper()}"
              f"   (cortes: {CORTES['distress']:.2f} y {CORTES['segura']:.2f})")
        peor = min(aportes, key=lambda k: aportes[k])
        mejor = max(aportes, key=lambda k: aportes[k])
        print(f"   lo que más la hunde: {peor} ({aportes[peor]:+.2f}) · lo que más la sostiene: {mejor} ({aportes[mejor]:+.2f})")
    return {"Z": Z, "zona": zona, "X": X, "aportes": aportes, "tabla": tabla}


def x3_para_salir(balance, objetivo=None, verbose=True):
    """¿CUÁNTO EBIT/ACTIVOS HACE FALTA para que la empresa llegue al corte?

    Con las otras tres razones fijas, se despeja X3 de Z'' = objetivo:
        X3 = (objetivo − 3.25 − 6.56·X1 − 3.26·X2 − 1.05·X4) / 6.72
    Por defecto el objetivo es el corte de distress (4.35). Devuelve el X3
    necesario y cuántos puntos de EBIT/activos hay que subir.
    """
    objetivo = CORTES["distress"] if objetivo is None else objetivo
    b = _fila(balance)
    X = razones(b, verbose=False)
    c = ALTMAN_ZPP
    x3 = (objetivo - c["const"] - c["X1"] * X["X1"] - c["X2"] * X["X2"] - c["X4"] * X["X4"]) / c["X3"]
    delta = x3 - X["X3"]
    if verbose:
        nombre = b.get("ticker", "la empresa")
        print(f"EBIT/ACTIVOS QUE NECESITA {nombre} PARA LLEGAR A Z'' = {objetivo:.2f}")
        print(f"   X3 = ({objetivo:.2f} − {c['const']} − {c['X1']}·{X['X1']:+.2f} − {c['X2']}·{X['X2']:+.2f} − {c['X4']}·{X['X4']:+.2f}) / {c['X3']}")
        print(f"      = {x3:+.3f}   (hoy {X['X3']:+.3f}: tendría que subir {delta:+.1%} del activo, es decir pasar de un EBIT de {b['ebit']:.0f} a {x3 * b['activo_total']:.0f} sobre activos de {b['activo_total']:.0f})")
    return {"X3_necesario": x3, "X3_hoy": X["X3"], "delta": delta}


def tabla_zpp(datos, verbose=False):
    """Z'', zona y la razón que más pesa para cada empresa de una tabla con COLUMNAS."""
    filas = []
    for _, r in datos.iterrows():
        z = zpp_en_cadena(r, verbose=False)
        filas.append({"ticker": r["ticker"], "Z''": z["Z"], "zona": z["zona"],
                      "la hunde": min(z["aportes"], key=lambda k: z["aportes"][k]),
                      "la sostiene": max(z["aportes"], key=lambda k: z["aportes"][k]),
                      "EBIT/activos": z["X"]["X3"]})
    t = pd.DataFrame(filas).sort_values("Z''").reset_index(drop=True)
    if verbose:
        print(t.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return t


# EDGAR no pide API key: pide identificarse. La SEC exige un User-Agent con
# nombre y correo de contacto ("Nombre Apellido correo@dominio"); sin correo
# responde 403 Forbidden. Se fija una vez en el cuaderno:  A.AGENTE = "..."
AGENTE = None


def _leer(resp):
    """Lee la respuesta; la SEC manda gzip cuando se le pide."""
    import gzip
    datos = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip":
        datos = gzip.decompress(datos)
    return datos.decode("utf-8")


def empresas_edgar(tickers, agente=None):
    """Razones de Altman desde data.sec.gov/api/xbrl/companyfacts (último
    10-K, US GAAP). Requiere red; sólo empresas listadas en EUA que reportan
    en US GAAP (una emisora mexicana con ADR reporta en IFRS y no sale).
    REVIENTA si falla: no hay respaldo silencioso."""
    import json
    import urllib.request
    agente = agente or AGENTE
    if not agente or "@" not in agente:
        raise RuntimeError('falta A.AGENTE = "Nombre Apellido correo@dominio" (la SEC lo exige; no es una API key)')
    cab = {"User-Agent": agente, "Accept-Encoding": "gzip, deflate", "Host": "www.sec.gov"}
    req = urllib.request.Request("https://www.sec.gov/files/company_tickers.json", headers=cab)
    mapa = {v["ticker"]: int(v["cik_str"]) for v in json.loads(_leer(urllib.request.urlopen(req, timeout=30))).values()}
    # Cada razón tiene un tag principal y respaldos: no todas las empresas
    # reportan los mismos conceptos en XBRL (muchas no reportan "Liabilities").
    tags = {"activo_total": ["Assets"],
            "pasivo_total": ["Liabilities"],
            "activo_circulante": ["AssetsCurrent"],
            "pasivo_circulante": ["LiabilitiesCurrent"],
            "utilidades_retenidas": ["RetainedEarningsAccumulatedDeficit"],
            "ebit": ["OperatingIncomeLoss",
                     "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
                     "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments"],
            "capital_contable": ["StockholdersEquity",
                                 "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"]}

    def _ultimo(facts, candidatos):
        """El último valor anual (10-K, FY) del primer tag que exista; (valor, fecha, tag) o None."""
        for tag in candidatos:
            if tag not in facts or "USD" not in facts[tag]["units"]:
                continue
            vals = [v for v in facts[tag]["units"]["USD"] if v.get("form") == "10-K" and v.get("fp") == "FY"]
            if vals:
                vals.sort(key=lambda v: v["end"])
                return vals[-1]["val"], vals[-1]["end"], tag
        return None

    filas, omitidas = [], []
    for t in tickers:
        if t not in mapa:
            omitidas.append(f"{t}: no está en la lista de la SEC (¿ticker de EUA?)"); continue
        cab["Host"] = "data.sec.gov"
        req = urllib.request.Request(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{mapa[t]:010d}.json", headers=cab)
        facts = json.loads(_leer(urllib.request.urlopen(req, timeout=30)))["facts"].get("us-gaap", {})
        fila, faltan = {"ticker": t}, []
        for col, candidatos in tags.items():
            r = _ultimo(facts, candidatos)
            if r is None and col == "pasivo_total":                      # pasivo = activo + pasivo − capital
                tot = _ultimo(facts, ["LiabilitiesAndStockholdersEquity"]); cap = _ultimo(facts, tags["capital_contable"])
                r = (tot[0] - cap[0], tot[1], "Assets − Equity") if tot and cap else None
            if r is None:
                faltan.append(col); continue
            fila[col] = r[0]
            if col == "activo_total":
                fila["fecha"] = r[1]
        if faltan:
            omitidas.append(f"{t}: sin {', '.join(faltan)} en su XBRL (balance no clasificado o IFRS); se omite")
        else:
            filas.append(fila)
    for o in omitidas:
        print("   ·", o)
    if not filas:
        raise RuntimeError("ninguna empresa con las siete partidas: " + "; ".join(omitidas)[:120])
    return pd.DataFrame(filas)


def cargar_empresas(tickers=None, csv="mis_empresas.csv", verbose=True):
    """En este orden: mis_empresas.csv (opción B) → EDGAR por ticker (opción A,
    Colab) → MUESTRA estilizada (opción C). Dice cuál usó."""
    if os.path.exists(csv):
        datos = pd.read_csv(csv)
        faltan = [c for c in COLUMNAS if c not in datos.columns]
        if faltan:
            raise ValueError(f"{csv} necesita las columnas {COLUMNAS}; faltan {faltan}")
        origen = f"{csv} (opción B)"
    else:
        try:
            datos = empresas_edgar(tickers or [])
            if datos.empty:
                raise RuntimeError("sin tickers")
            origen = "API de la SEC (opción A)"
        except Exception as e:
            datos = MUESTRA.copy()
            origen = f"tabla MUESTRA estilizada (opción C) — EDGAR no respondió: {str(e)[:90]}"
    if verbose:
        print("usando", origen)
        print("empresas:", ", ".join(datos["ticker"]))
    return datos
