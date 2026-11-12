"""
LABORATORIO L2 · ALTMAN Z'' · capa de widgets sobre altman.py
Riesgos Financieros (FIN516) · Universidad Panamericana

Cinco partes (convenciones §2.7): PREGUNTA (markdown), PREDICE, JUEGA, COMPARA
(aquí, widgets) y RETO (celda con hueco). Todo se registra en la Bitacora del
alumno. Esta capa no calcula nada por su cuenta: llama a altman.py.
Diseñada para Google Colab con ipywidgets 8.
"""
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output

import altman as A

NAVY, GOLD, STEEL, RED, MUTE = "#0B2545", "#B08D3C", "#5B7DA3", "#9E2A2B", "#6B7280"
matplotlib.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _titulo(txt, color=NAVY):
    return W.HTML(f"<div style='background:{color};color:white;padding:6px 10px;font-weight:600;"
                  f"border-radius:4px;margin:4px 0'>{txt}</div>")


def _nota(txt):
    return W.HTML(f"<div style='color:{MUTE};font-size:12px;margin:2px 0 8px 2px'>{txt}</div>")


def _slider(valor, minimo, maximo, paso, desc, leyenda, fmt=".2f"):
    s = W.FloatSlider(value=valor, min=minimo, max=maximo, step=paso, description=desc,
                      continuous_update=False, readout_format=fmt,
                      style={"description_width": "70px"}, layout=W.Layout(width="420px"))
    return W.HBox([s, W.HTML(f"<span style='color:{MUTE};font-size:12px'>{leyenda}</span>")]), s


def _predice(bit, lab, pregunta, opciones, nota=""):
    radio = W.RadioButtons(options=opciones, layout=W.Layout(width="auto"))
    boton = W.Button(description="Registrar mi apuesta", button_style="warning")
    salida = W.Output()

    def _click(_):
        ev = bit.registrar(lab, "apuesta", valor=radio.value)
        with salida:
            clear_output()
            print(f"Apuesta registrada: «{radio.value}» ({ev['hora'][11:]}). Ahora sí: a jugar.")
        radio.disabled = True
        boton.disabled = True

    boton.on_click(_click)
    piezas = [_titulo(f"PREDICE · {lab}", GOLD), W.HTML(f"<b>{pregunta}</b>"), radio, boton, salida]
    if nota:
        piezas.insert(2, _nota(nota))
    return W.VBox(piezas)


# ---------------------------------------------------------------------------
# el balance de juguete: mueve un rubro y mira cómo cambian las razones y el Z''
# ---------------------------------------------------------------------------

def lab2_balance(bit, base=None):
    """JUEGA: empieza en la empresa que quebró; deslizadores de pasivo circulante,
    utilidades retenidas, EBIT y pasivo total (el capital se ajusta para que
    activo = pasivo + capital). Muestra las razones, el Z'' en cadena y la zona."""
    b0 = (A.DOS_EMPRESAS.iloc[1] if base is None else base).to_dict()
    hpc, spc = _slider(b0["pasivo_circulante"], 0, 60, 1, "vence pronto", "pasivo circulante: lo que hay que pagar en el año (activo circulante fijo en %.0f)" % b0["activo_circulante"], ".0f")
    hur, sur = _slider(b0["utilidades_retenidas"], -40, 40, 1, "ut. retenidas", "lo ganado y guardado en los años anteriores (negativo si hubo pérdidas)", ".0f")
    heb, seb = _slider(b0["ebit"], -5, 25, 1, "EBIT", "utilidad de operación del año, antes de intereses e impuestos", ".0f")
    hpt, spt = _slider(b0["pasivo_total"], 10, 95, 1, "pasivo total", "todo lo que se debe; el capital contable es 100 − pasivo total", ".0f")
    out = W.Output()

    def _dibuja(pc, ur, eb, pt):
        b = dict(b0); b.update({"pasivo_circulante": pc, "utilidades_retenidas": ur, "ebit": eb, "pasivo_total": pt, "capital_contable": 100.0 - pt, "ticker": "tu empresa"})
        z = A.zpp_en_cadena(b, verbose=False)
        bit.registrar("L2", "parametros", **{k: b[k] for k in ("pasivo_circulante", "utilidades_retenidas", "ebit", "pasivo_total")}, **z["X"])
        bit.registrar("L2", "resultado", Z=z["Z"], zona=z["zona"], ticker="tu empresa")
        c = A.ALTMAN_ZPP
        with out:
            clear_output(wait=True)
            X = z["X"]; ap = z["aportes"]
            print(f"X1 = ({b['activo_circulante']:.0f} − {pc:.0f})/100 = {X['X1']:+.2f} · X2 = {ur:.0f}/100 = {X['X2']:+.2f} · X3 = {eb:.0f}/100 = {X['X3']:+.2f} · X4 = {100-pt:.0f}/{pt:.0f} = {X['X4']:+.2f}")
            print(f"Z'' = 3.25 + 6.56·({X['X1']:+.2f}) + 3.26·({X['X2']:+.2f}) + 6.72·({X['X3']:+.2f}) + 1.05·({X['X4']:+.2f})")
            print(f"    = 3.25 {ap['X1']:+.2f} {ap['X2']:+.2f} {ap['X3']:+.2f} {ap['X4']:+.2f} = {z['Z']:.2f}  →  zona {z['zona'].upper()}  (distress < {A.CORTES['distress']:.2f} · segura > {A.CORTES['segura']:.2f})")
            fig, ax = plt.subplots(figsize=(8.5, 3.0))
            nombres = ["constante", "X1 liquidez", "X2 historia", "X3 rentabilidad", "X4 apalancamiento"]
            vals = [c["const"], ap["X1"], ap["X2"], ap["X3"], ap["X4"]]
            acum = np.cumsum([0] + vals[:-1]); cols = [MUTE] + [NAVY if v >= 0 else RED for v in vals[1:]]
            ax.bar(nombres, vals, bottom=acum, color=cols)
            ax.axhline(A.CORTES["distress"], color=RED, ls="--", lw=1); ax.text(4.45, A.CORTES["distress"] + 0.05, "distress", color=RED, ha="right", va="bottom", fontsize=9)
            ax.axhline(A.CORTES["segura"], color=GOLD, ls="--", lw=1); ax.text(4.45, A.CORTES["segura"] + 0.05, "segura", color=GOLD, ha="right", va="bottom", fontsize=9)
            ax.axhline(z["Z"], color=NAVY, lw=1.6); ax.set_ylim(min(0, z["Z"] - 0.5), max(8.3, z["Z"] + 0.5))
            ax.set_ylabel("aporte a Z''"); ax.set_title(f"Z'' = {z['Z']:.2f}: de qué está hecho", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L2 · el balance de juguete"),
                 _nota("Empieza en la empresa que quebró. Mueve UN rubro a la vez y mira qué razón cambia y cuánto se mueve el Z''. ¿Qué la saca de la zona?"),
                 hpc, hur, heb, hpt, out])
    W.interactive_output(_dibuja, {"pc": spc, "ur": sur, "eb": seb, "pt": spt})
    return ui


# ---------------------------------------------------------------------------
# sus empresas
# ---------------------------------------------------------------------------

def lab2_predice(bit, tickers):
    return _predice(
        bit, "L2",
        "De sus empresas, ¿cuál tiene el Z'' MÁS BAJO (la más cerca del distress)?",
        list(tickers),
        "Sólo tienen los nombres: apuesten con lo que sepan de cada empresa, antes de ver sus razones.")


def lab2_juega(bit, datos):
    """JUEGA: menú de empresas → razones, Z'' en cadena y aportes; deslizador ΔX3
    («¿cuánto tendría que mejorar la rentabilidad para cambiar de zona?»)."""
    d = datos.set_index("ticker") if "ticker" in datos.columns else datos
    dd = W.Dropdown(options=list(d.index), description="empresa", style={"description_width": "70px"})
    hx, sx = _slider(0.0, -0.20, 0.40, 0.01, "ΔX3", "puntos que se suman a EBIT/activos: ¿cuánto tendría que mejorar la rentabilidad para cambiar de zona?")
    out = W.Output()

    def _dibuja(ticker, dX3):
        r = d.loc[ticker].to_dict(); r["ticker"] = ticker
        X = A.razones(r, verbose=False); X["X3"] += dX3
        z = A.altman_zpp(X["X1"], X["X2"], X["X3"], X["X4"], verbose=False)
        bit.registrar("L2", "parametros", ticker=ticker, dX3=dX3, **X)
        bit.registrar("L2", "resultado", Z=z["Z"], zona=z["zona"], ticker=ticker)
        c = A.ALTMAN_ZPP; ap = z["aportes"]
        with out:
            clear_output(wait=True)
            print(f"{ticker}:  X1 liquidez {X['X1']:+.3f} · X2 historia {X['X2']:+.3f} · X3 rentabilidad {X['X3']:+.3f}{' (con ΔX3)' if dX3 else ''} · X4 apalancamiento {X['X4']:+.3f}")
            print(f"Z'' = 3.25 + 6.56·({X['X1']:+.2f}) + 3.26·({X['X2']:+.2f}) + 6.72·({X['X3']:+.2f}) + 1.05·({X['X4']:+.2f})")
            print(f"    = 3.25 {ap['X1']:+.2f} {ap['X2']:+.2f} {ap['X3']:+.2f} {ap['X4']:+.2f} = {z['Z']:.2f}  →  zona {z['zona'].upper()}")
            fig, ax = plt.subplots(figsize=(8.5, 3.0))
            nombres = ["constante", "X1 liquidez", "X2 historia", "X3 rentabilidad", "X4 apalancamiento"]
            vals = [c["const"], ap["X1"], ap["X2"], ap["X3"], ap["X4"]]
            acum = np.cumsum([0] + vals[:-1]); cols = [MUTE] + [NAVY if v >= 0 else RED for v in vals[1:]]
            ax.bar(nombres, vals, bottom=acum, color=cols)
            ax.axhline(A.CORTES["distress"], color=RED, ls="--", lw=1); ax.text(4.45, A.CORTES["distress"] + 0.05, "distress", color=RED, ha="right", va="bottom", fontsize=9)
            ax.axhline(A.CORTES["segura"], color=GOLD, ls="--", lw=1); ax.text(4.45, A.CORTES["segura"] + 0.05, "segura", color=GOLD, ha="right", va="bottom", fontsize=9)
            ax.axhline(z["Z"], color=NAVY, lw=1.6); ax.set_ylim(min(0, z["Z"] - 0.5), max(8.3, z["Z"] + 0.5))
            ax.set_ylabel("aporte a Z''"); ax.set_title(f"{ticker}: Z'' = {z['Z']:.2f}, de qué está hecho", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L2 · sus empresas"), _nota("Elijan empresa; luego muevan ΔX3 para ver cuánta rentabilidad extra haría falta para cambiar de zona."), dd, hx, out])
    W.interactive_output(_dibuja, {"ticker": dd, "dX3": sx})
    return ui


def lab2_compara(bit, datos):
    boton = W.Button(description="Comparar mi apuesta con el ranking", button_style="info")
    out = W.Output()

    def _click(_):
        ap = bit.ultimo("L2", "apuesta")
        with out:
            clear_output()
            if ap is None:
                print("Registren su apuesta primero."); return
            tab = A.tabla_zpp(datos)
            peor = tab.iloc[0]["ticker"]
            menor_ebit = tab.sort_values("EBIT/activos").iloc[0]["ticker"]
            bit.registrar("L2", "nota", compara=f"apuesta {ap['valor']} · peor {peor} · menor EBIT {menor_ebit}")
            print(f"Su apuesta: {ap['valor']}. La más baja: {peor}." + ("" if peor == ap["valor"] else " Fallaron."))
            print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            if menor_ebit != peor:
                print(f"\nLa de menor EBIT/activos es {menor_ebit} y no es la peor: lo que hunde a {peor} es el balance (historia y liquidez), no la utilidad del año.")
            print("Lo que el modelo no les dijo: Z'' usa el último balance; el precio y la calificación ya reaccionaron. ¿A cuál le creen cuando se contradicen?")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L2", STEEL), boton, out])
