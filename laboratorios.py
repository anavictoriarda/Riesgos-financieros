"""
LABORATORIOS DEL MÓDULO XII · capa de widgets sobre empresas.py
Riesgos Financieros (FIN516) · Universidad Panamericana

Cada laboratorio tiene cinco partes (convenciones §2.7): PREGUNTA (en el
markdown del cuaderno), PREDICE, JUEGA, COMPARA (aquí, como widgets) y
RETO (una celda con hueco en el cuaderno). Todo lo que el alumno hace se
registra en su Bitacora; esta capa NO calcula nada por su cuenta: llama a
empresas.py.

Probado con ipywidgets 8 en Jupyter; diseñado para Google Colab (los
widgets se recrean al ejecutar la celda; no sobreviven al guardar).
"""
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output, HTML

import empresas as E

NAVY, GOLD, STEEL, RED, MUTE = "#0B2545", "#B08D3C", "#5B7DA3", "#9E2A2B", "#6B7280"
matplotlib.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------------------
# utilería
# ---------------------------------------------------------------------------

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
    """Bloque PREDICE: apuesta registrada con nombre antes de jugar."""
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
# L1 · MERTON · el acreedor de Cemex, agosto de 2009
# ---------------------------------------------------------------------------

def lab1_predice(bit):
    return _predice(
        bit, "L1",
        "Cemex pide extender sus vencimientos de 1 a 5 años. Si el banco acepta y nada más cambia, "
        "¿la probabilidad de incumplimiento ACUMULADA hasta el vencimiento sube, baja o queda igual?",
        ["sube", "baja", "queda igual"],
        "Decide con la fórmula de DD enfrente, antes de mover T.")


def lab1_juega(bit, caso=None):
    """JUEGA L1: V, D, σ_V, T → PD de Merton, con la fórmula sustituida en vivo."""
    c = E.CEMEX_2009 if caso is None else caso
    base = E.merton_implicito(c["capitalizacion_musd"] / 1000, c["sigma_E"], c["deuda_total_musd"] / 1000,
                              T=1.0, r=c["r"], verbose=False)
    V0, sV0, D0 = base["V"], base["sigma_V"], c["deuda_total_musd"] / 1000

    hV, sV = _slider(round(V0, 1), 10, 60, 0.5, "V", "valor de los activos de la empresa, miles de millones de dólares (de E y σ_E por las dos ecuaciones)", ".1f")
    hD, sD = _slider(round(D0, 1), 5, 40, 0.5, "D", "deuda que vence: el punto de default, mmdd", ".1f")
    hs, ss = _slider(round(sV0, 2), 0.05, 1.0, 0.01, "σ_V", "volatilidad anual de los activos (qué tanto se mueve V en un año)")
    hT, sT = _slider(1.0, 0.5, 7.0, 0.5, "T", "años al vencimiento de la deuda", ".1f")
    hm, sm = _slider(0.05, -0.10, 0.20, 0.01, "μ", "deriva real de los activos (crecimiento esperado); r = 3 % es la neutral al riesgo")
    out = W.Output()

    def _dibuja(V, D, sigma_V, T, mu):
        m = E.merton(V, D, sigma_V, T=T, r=c["r"], mu=mu, verbose=False)
        bit.registrar("L1", "parametros", V=V, D=D, sigma_V=sigma_V, T=T, mu=mu)
        bit.registrar("L1", "resultado", PD_P=m["PD_P"], PD_Q=m["PD_Q"], DD=m["DD"], E=m["E"])
        with out:
            clear_output(wait=True)
            sT_ = sigma_V * math.sqrt(T)
            print(f"DD = [ln(V/D) + (μ − σ_V²/2)·T] / (σ_V·√T)")
            print(f"   = [ln({V:.1f}/{D:.1f}) + ({mu:.2f} − {sigma_V:.2f}²/2)·{T:g}] / ({sigma_V:.2f}·√{T:g})")
            print(f"   = [{math.log(V/D):.3f} {(mu - 0.5*sigma_V**2)*T:+.3f}] / {sT_:.3f} = {m['DD']:.2f} desviaciones")
            print(f"PD física hasta T = Φ(−DD) = Φ(−{m['DD']:.2f}) = {m['PD_P']:.2%}      "
                  f"(neutral al riesgo, con r: {m['PD_Q']:.2%})")
            print(f"equity = call sobre V = {m['E']:.1f} mmdd · deuda a mercado = {m['deuda']:.1f} · spread implícito {m['spread']*1e4:,.0f} pb")
            Ts = np.arange(0.5, 7.01, 0.5)
            pds = [E.merton(V, D, sigma_V, T=t, r=c["r"], mu=mu, verbose=False)["PD_P"] for t in Ts]
            fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
            ax[0].plot(Ts, [p * 100 for p in pds], color=NAVY, lw=2)
            ax[0].scatter([T], [m["PD_P"] * 100], color=GOLD, zorder=3, s=50)
            ax[0].set_xlabel("años al vencimiento T"); ax[0].set_ylabel("PD acumulada, %")
            ax[0].set_title("PD acumulada contra el plazo", fontsize=10, color=NAVY)
            x = np.linspace(max(0.05, V * 0.2), V * 2.2, 300)
            mu_ln = math.log(V) + (mu - 0.5 * sigma_V ** 2) * T
            s_ln = sigma_V * math.sqrt(T)
            dens = (1 / (x * s_ln * math.sqrt(2 * math.pi))) * np.exp(-(np.log(x) - mu_ln) ** 2 / (2 * s_ln ** 2))
            ax[1].plot(x, dens, color=NAVY)
            ax[1].fill_between(x, dens, where=x < D, color=RED, alpha=0.35, label=f"V_T < D: {m['PD_P']:.1%}")
            ax[1].axvline(D, color=RED, ls="--", lw=1)
            ax[1].set_xlabel("valor de los activos en T, mmdd"); ax[1].set_yticks([])
            ax[1].set_title("distribución de V en T y la barrera D", fontsize=10, color=NAVY)
            ax[1].legend(frameon=False, fontsize=9)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L1 · Merton con los números de Cemex 2009"),
                 _nota("Cemex al 13 de agosto de 2009: deuda 19.25 mmdd (sin 3.0 de perpetuos), capitalización ≈ 10.7 mmdd, σ_E de CX calculada en la celda anterior. Mueve un deslizador a la vez."),
                 hV, hD, hs, hT, hm, out])
    W.interactive_output(_dibuja, {"V": sV, "D": sD, "sigma_V": ss, "T": sT, "mu": sm})
    return ui


def lab1_compara(bit):
    """COMPARA L1: la apuesta contra lo que el modelo dice de 1 a 5 años."""
    boton = W.Button(description="Comparar mi apuesta con el modelo", button_style="info")
    out = W.Output()

    def _click(_):
        ap = bit.ultimo("L1", "apuesta")
        par = bit.ultimo("L1", "parametros")
        with out:
            clear_output()
            if ap is None or par is None:
                print("Registra tu apuesta y mueve los deslizadores primero."); return
            p1 = E.merton(par["V"], par["D"], par["sigma_V"], T=1, r=E.CEMEX_2009["r"], mu=par["mu"], verbose=False)["PD_P"]
            p5 = E.merton(par["V"], par["D"], par["sigma_V"], T=5, r=E.CEMEX_2009["r"], mu=par["mu"], verbose=False)["PD_P"]
            real = "sube" if p5 > p1 else ("baja" if p5 < p1 else "queda igual")
            bit.registrar("L1", "nota", compara=f"apuesta {ap['valor']} · modelo {real}", PD_1=p1, PD_5=p5)
            print(f"Tu apuesta: «{ap['valor']}». Con tus parámetros (V={par['V']:.1f}, D={par['D']:.1f}, σ_V={par['sigma_V']:.2f}, μ={par['mu']:.2f}):")
            print(f"   PD acumulada a 1 año = {p1:.2%}   ·   a 5 años = {p5:.2%}   →  {real.upper()}")
            if real == "sube":
                print("Más tiempo es más oportunidad de cruzar la barrera: la PD ACUMULADA sube con T (salvo μ muy alta).")
                print("Lo que baja al extender no es la PD de Merton: es la presión de LIQUIDEZ, que el modelo no tiene.")
                print("Lo que pasó: el 14 de agosto de 2009, ~75 acreedores extendieron ~15 mmdd hasta febrero de 2014 a un margen de 4.5 %,")
                print("y el acuerdo OBLIGÓ a Cemex a levantar al menos 1.0 mmdd de capital; el 28 de septiembre colocó 1.78 mmdd netos.")
                print("El sindicato hizo las dos cosas: extendió (liquidez) y bajó D (lo que Merton sí ve).")
            else:
                print("Con tu μ, los activos crecen tan rápido que el tiempo juega a favor. Prueba con μ = 3 % y vuelve a comparar.")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L1", STEEL), boton, out])


# ---------------------------------------------------------------------------
# L2 · ALTMAN Z'' · el analista con mandato, con sus empresas
# ---------------------------------------------------------------------------

def lab2_predice(bit, tickers):
    return _predice(
        bit, "L2",
        "De tus empresas, ¿cuál crees que tiene el Z'' MÁS BAJO (la más cerca del distress)?",
        list(tickers),
        "Sólo tienes los nombres: apuesta con lo que sepas de la empresa.")


def lab2_juega(bit, datos):
    """JUEGA L2: menú de empresas → Z'' por aportes; deslizador de mejora en la razón que más pesa.

    `datos` es un DataFrame con columnas: ticker, activo_total, pasivo_total,
    activo_circulante, pasivo_circulante, utilidades_retenidas, ebit,
    capital_contable (mismas unidades). Ver empresas_edgar() en el cuaderno.
    """
    datos = datos.set_index("ticker") if "ticker" in datos.columns else datos
    dd = W.Dropdown(options=list(datos.index), description="empresa", style={"description_width": "70px"})
    hx, sx = _slider(0.0, -0.20, 0.20, 0.01, "ΔX3", "cambio en EBIT/activos (puntos): ¿cuánto tendría que mejorar la rentabilidad para cambiar de zona?")
    out = W.Output()

    def _dibuja(ticker, dX3):
        r = datos.loc[ticker]
        X1 = (r["activo_circulante"] - r["pasivo_circulante"]) / r["activo_total"]
        X2 = r["utilidades_retenidas"] / r["activo_total"]
        X3 = r["ebit"] / r["activo_total"] + dX3
        X4 = r["capital_contable"] / r["pasivo_total"]
        z = E.altman_zpp(X1, X2, X3, X4, verbose=False)
        bit.registrar("L2", "parametros", ticker=ticker, X1=X1, X2=X2, X3=X3, X4=X4, dX3=dX3)
        bit.registrar("L2", "resultado", Z=z["Z"], zona=z["zona"], ticker=ticker)
        c = E.ALTMAN_ZPP
        with out:
            clear_output(wait=True)
            print(f"{ticker}:  X1 capital de trabajo/activos = {X1:+.3f} · X2 utilidades retenidas/activos = {X2:+.3f}")
            print(f"       X3 EBIT/activos = {X3:+.3f}{' (ajustado)' if dX3 else ''} · X4 capital contable/pasivos = {X4:+.3f}")
            print(f"Z'' = {c['const']} + {c['X1']}·X1 + {c['X2']}·X2 + {c['X3']}·X3 + {c['X4']}·X4")
            print(f"    = {c['const']} {z['aportes']['X1']:+.2f} {z['aportes']['X2']:+.2f} {z['aportes']['X3']:+.2f} {z['aportes']['X4']:+.2f} = {z['Z']:.2f}  →  zona {z['zona'].upper()}")
            k = c["const"]
            fig, ax = plt.subplots(figsize=(8.5, 3.0))
            nombres = ["constante", "X1 liquidez", "X2 ut. retenidas", "X3 EBIT", "X4 apalancamiento"]
            vals = [k, z["aportes"]["X1"], z["aportes"]["X2"], z["aportes"]["X3"], z["aportes"]["X4"]]
            acum = np.cumsum([0] + vals[:-1])
            cols = [MUTE] + [NAVY if v >= 0 else RED for v in vals[1:]]
            ax.bar(nombres, vals, bottom=acum, color=cols)
            ax.axhline(E.ZONAS_ZPP["distress"] + k, color=RED, ls="--", lw=1); ax.text(4.4, E.ZONAS_ZPP["distress"] + k, " distress", color=RED, va="center", fontsize=9)
            ax.axhline(E.ZONAS_ZPP["segura"] + k, color=GOLD, ls="--", lw=1); ax.text(4.4, E.ZONAS_ZPP["segura"] + k, " segura", color=GOLD, va="center", fontsize=9)
            ax.axhline(z["Z"], color=NAVY, lw=1.5); ax.text(-0.4, z["Z"], f" Z'' = {z['Z']:.2f}", color=NAVY, va="bottom", fontsize=9, fontweight="bold")
            ax.set_ylabel("aporte a Z''"); ax.set_title(f"{ticker}: de qué está hecho su Z''", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L2 · Altman Z'' sobre tus empresas"), dd, hx, out])
    W.interactive_output(_dibuja, {"ticker": dd, "dX3": sx})
    return ui


def lab2_compara(bit, datos):
    boton = W.Button(description="Comparar mi apuesta con el ranking", button_style="info")
    out = W.Output()
    d = datos.set_index("ticker") if "ticker" in datos.columns else datos

    def _click(_):
        ap = bit.ultimo("L2", "apuesta")
        with out:
            clear_output()
            if ap is None:
                print("Registra tu apuesta primero."); return
            filas = []
            for t, r in d.iterrows():
                z = E.altman_zpp((r["activo_circulante"] - r["pasivo_circulante"]) / r["activo_total"],
                                 r["utilidades_retenidas"] / r["activo_total"], r["ebit"] / r["activo_total"],
                                 r["capital_contable"] / r["pasivo_total"], verbose=False)
                filas.append((t, z["Z"], z["zona"], max(z["aportes"], key=lambda k: z["aportes"][k])))
            tab = pd.DataFrame(filas, columns=["ticker", "Z''", "zona", "razón que más pesa"]).sort_values("Z''")
            peor = tab.iloc[0]["ticker"]
            bit.registrar("L2", "nota", compara=f"apuesta {ap['valor']} · peor {peor}")
            print(f"Tu apuesta: {ap['valor']}. La más baja: {peor}.")
            print(tab.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
            print("\nLo que el modelo no te dijo: Z'' usa el último balance; el precio y la calificación ya reaccionaron. ¿A cuál le crees?")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L2", STEEL), boton, out])


# ---------------------------------------------------------------------------
# L3 · FORMA REDUCIDA · el CRO en marzo de 2020
# ---------------------------------------------------------------------------

def lab3_predice(bit):
    return _predice(
        bit, "L3",
        "Marzo de 2020: la distancia al default promedio de tu cartera cae de 4 a 2 desviaciones. "
        "Según el modelo de intensidad con covariables, ¿λ sube menos del doble, entre 2 y 5 veces, o más de 5 veces?",
        ["menos del doble", "entre 2 y 5 veces", "más de 5 veces"],
        "La fórmula es λ = exp(β·x) y el coeficiente de DD es −1.129. Decide antes de mover nada.")


def lab3_juega(bit, cartera_mmdp=100.0, LGD=0.45):
    hd, sd = _slider(4.0, 0.5, 7.0, 0.1, "DD", "distancia al default promedio de los acreditados (de Merton / KMV)", ".1f")
    hr, sr = _slider(0.0, -0.6, 0.6, 0.05, "ret", "rendimiento de la acción del emisor en el último año")
    hs, ssp = _slider(0.0, -0.4, 0.4, 0.05, "SP500", "rendimiento del mercado en el último año: el ciclo")
    ht, st = _slider(1.5, 0.0, 8.0, 0.25, "r3m", "tasa del Tesoro a 3 meses, en por ciento (1.5 = 1.5 %)", ".2f")
    out = W.Output()

    def _dibuja(DD, ret, sp, r3m):
        res = E.intensidad_covariables(DD, ret=ret, r3m=r3m, sp500=sp, verbose=False)
        base = E.intensidad_covariables(4.0, ret=0.0, r3m=1.5, sp500=0.0, verbose=False)
        prov = res["PD_1a"] * LGD * cartera_mmdp
        prov0 = base["PD_1a"] * LGD * cartera_mmdp
        bit.registrar("L3", "parametros", DD=DD, ret=ret, sp500=sp, r3m=r3m)
        bit.registrar("L3", "resultado", **{"lambda": res["lambda"]}, PD_1a=res["PD_1a"], provision_mmdp=prov)
        c = E.DSW_COEF
        with out:
            clear_output(wait=True)
            print(f"λ = exp({c['const']:+.3f} {c['DD']:+.3f}·DD {c['ret']:+.3f}·ret {c['r3m']:+.3f}·r3m {c['sp500']:+.3f}·SP500)")
            print(f"  = exp({c['const']:+.3f} {c['DD']*DD:+.3f} {c['ret']*ret:+.3f} {c['r3m']*r3m:+.3f} {c['sp500']*sp:+.3f}) = exp({res['exponente']:+.3f}) = {res['lambda']*1e4:,.0f} pb por año")
            print(f"PD a un año = 1 − e^(−λ) = {res['PD_1a']:.2%} · contra la base (DD 4, ret 0, r3m 1.5 %, SP500 0): {base['PD_1a']:.2%}  → ×{res['PD_1a']/base['PD_1a']:.1f}")
            print(f"provisión esperada a un año = PD × LGD × cartera = {res['PD_1a']:.4f} × {LGD} × {cartera_mmdp:,.0f} mmdp = {prov:,.2f} mmdp   (base: {prov0:,.2f})")
            print("coeficientes: Duffie, Saita y Wang (2007), Tabla 2; industriales de EUA 1980–2004")
            dds = np.linspace(0.5, 7, 200)
            lams = [E.intensidad_covariables(x, ret=ret, r3m=r3m, sp500=sp, verbose=False)["lambda"] * 1e4 for x in dds]
            lams0 = [E.intensidad_covariables(x, ret=0.0, r3m=1.5, sp500=0.0, verbose=False)["lambda"] * 1e4 for x in dds]
            fig, ax = plt.subplots(figsize=(8.5, 3.2))
            ax.plot(dds, lams0, color=STEEL, lw=1.5, label="ciclo neutro")
            ax.plot(dds, lams, color=NAVY, lw=2, label="con tus covariables")
            ax.scatter([DD], [res["lambda"] * 1e4], color=GOLD, zorder=3, s=50)
            ax.set_yscale("log"); ax.set_xlabel("distancia al default DD"); ax.set_ylabel("λ, pb anuales (escala log)")
            ax.set_title("la intensidad como función de la DD: cada unidad de DD multiplica, no suma", fontsize=10, color=NAVY)
            ax.legend(frameon=False, fontsize=9); plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L3 · la intensidad con covariables"),
                 _nota(f"Cartera estilizada de {cartera_mmdp:,.0f} mmdp con LGD {LGD:.0%}. Primero mueve sólo DD; después el ciclo (SP500)."),
                 hd, hr, hs, ht, out])
    W.interactive_output(_dibuja, {"DD": sd, "ret": sr, "sp": ssp, "r3m": st})
    return ui


def lab3_compara(bit):
    boton = W.Button(description="Comparar mi apuesta con el modelo", button_style="info")
    out = W.Output()

    def _click(_):
        ap = bit.ultimo("L3", "apuesta")
        with out:
            clear_output()
            if ap is None:
                print("Registra tu apuesta primero."); return
            l4 = E.intensidad_covariables(4.0, ret=0.0, r3m=1.5, sp500=0.0, verbose=False)["lambda"]
            l2 = E.intensidad_covariables(2.0, ret=0.0, r3m=1.5, sp500=0.0, verbose=False)["lambda"]
            factor = l2 / l4
            real = "menos del doble" if factor < 2 else ("entre 2 y 5 veces" if factor <= 5 else "más de 5 veces")
            bit.registrar("L3", "nota", compara=f"apuesta {ap['valor']} · modelo {real}", factor=factor)
            print(f"Tu apuesta: «{ap['valor']}». El modelo: λ(DD=2)/λ(DD=4) = e^(−β_DD·(2−4)) = e^({-E.DSW_COEF['DD']*2:.2f}) = ×{factor:.1f}  →  {real.upper()}")
            print("Con el ciclo encima (SP500 negativo) el factor es mayor todavía: todas las λ suben a la vez.")
            print("Lo que el modelo no te dijo: cuánto va a durar. La provisión de hoy es la PD de hoy; el overlay (XIV) existe por eso.")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L3", STEEL), boton, out])


# ---------------------------------------------------------------------------
# L4 · VASICEK · el comité de crédito y la concentración
# ---------------------------------------------------------------------------

def lab4_predice(bit):
    return _predice(
        bit, "L4",
        "Un portafolio de 20 nombres iguales con PD 2 % y ρ = 0.2 pasa a 100 nombres iguales. "
        "¿El capital al 99.9 % baja a la quinta parte, a la mitad, o casi no baja?",
        ["a la quinta parte", "a la mitad", "casi no baja"],
        "Sólo cambia el número de nombres: PD, ρ y LGD quedan iguales. Decide antes de mover n.")


def lab4_juega(bit, LGD=0.45):
    hp, sp = _slider(0.02, 0.002, 0.15, 0.002, "PD", "probabilidad de incumplimiento de cada nombre a un año", ".3f")
    hr, sr = _slider(0.20, 0.0, 0.6, 0.02, "ρ", "correlación de activos: cuánto del movimiento de cada nombre es el ciclo")
    hn, sn = _slider(20, 5, 200, 5, "n", "número de nombres en el portafolio", ".0f")
    hw, sw = _slider(0.05, 0.005, 0.5, 0.005, "w_max", "peso del nombre más grande (el resto se reparte igual)", ".3f")
    hq, sq = _slider(0.999, 0.90, 0.9999, 0.0005, "q", "nivel de confianza del capital", ".4f")
    out = W.Output()

    def _dibuja(PD, rho, n, wmax, q):
        n = int(n)
        wmax = min(wmax, 1.0)
        pesos = [wmax] + [(1 - wmax) / (n - 1)] * (n - 1)
        vas = E.vasicek_cuantil(PD, max(rho, 1e-6), q=q, LGD=LGD, verbose=False)
        con = E.capital_concentracion(PD, max(rho, 1e-6), pesos, LGD=LGD, q=q, n_sim=30_000, verbose=False)
        bit.registrar("L4", "parametros", PD=PD, rho=rho, n=n, w_max=wmax, q=q)
        bit.registrar("L4", "resultado", capital=vas["capital"], capital_simulado=con["capital_simulado"],
                      costo_concentracion=con["costo_concentracion"], n_efectivo=con["n_efectivo"])
        with out:
            clear_output(wait=True)
            a, zq = E.norm.ppf(PD), E.norm.ppf(q)
            print(f"PD condicional al cuantil = Φ((Φ⁻¹(PD) + √ρ·Φ⁻¹(q)) / √(1−ρ)) = Φ(({a:.3f} + {math.sqrt(rho):.3f}·{zq:.3f}) / {math.sqrt(1-rho):.3f}) = {vas['PD_q']:.2%}")
            print(f"capital Vasicek (granular) = LGD·PD_q − PD·LGD = {LGD}·{vas['PD_q']:.4f} − {PD:.4f}·{LGD} = {vas['capital']:.2%} del portafolio")
            print(f"con {n} nombres y el mayor al {wmax:.1%} (nombres efectivos: {con['n_efectivo']:.1f}): capital simulado = {con['capital_simulado']:.2%}"
                  f"  →  costo de la concentración {con['costo_concentracion']:+.2%}")
            fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
            if rho > 0:
                xs = np.linspace(vas["EL"] * 0.2, max(1.6 * vas["L_q"], 0.02), 300)
                cola = 1 - E.norm.cdf((math.sqrt(1 - rho) * E.norm.ppf(np.clip(xs / LGD, 1e-9, 1 - 1e-9)) - a) / math.sqrt(rho))
                ax[0].plot(xs * 100, cola, color=NAVY, lw=2)
                ax[0].set_yscale("log")
                ax[0].axhline(1 - q, color=RED, ls="--", lw=1, label=f"1 − q = {1-q:.2%}: una vez en {1/(1-q):,.0f}")
                ax[0].axvline(vas["EL"] * 100, color=STEEL, ls="--", lw=1, label=f"EL {vas['EL']:.2%}")
                ax[0].axvline(vas["L_q"] * 100, color=GOLD, ls="--", lw=1, label=f"L_q {vas['L_q']:.2%}")
                ax[0].set_xlabel("pérdida del portafolio x, %"); ax[0].set_ylabel("P(pérdida > x), escala log")
                ax[0].legend(frameon=False, fontsize=8)
            ax[0].set_title("la cola de la pérdida de Vasicek", fontsize=10, color=NAVY)
            rhos = np.linspace(0.02, 0.6, 30)
            caps = [E.vasicek_cuantil(PD, r_, q=q, LGD=LGD, verbose=False)["capital"] * 100 for r_ in rhos]
            ax[1].plot(rhos, caps, color=NAVY, lw=2)
            ax[1].scatter([rho], [vas["capital"] * 100], color=GOLD, zorder=3, s=50)
            ax[1].set_xlabel("ρ"); ax[1].set_ylabel("capital, % del portafolio")
            ax[1].set_title("el capital contra ρ: el ciclo no se diversifica", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L4 · Vasicek y la concentración"),
                 _nota("Primero mueve n con w_max pequeño; luego ρ; al final sube w_max: es el nombre grande (Pemex) entrando al portafolio."),
                 hp, hr, hn, hw, hq, out])
    W.interactive_output(_dibuja, {"PD": sp, "rho": sr, "n": sn, "wmax": sw, "q": sq})
    return ui


def lab4_compara(bit, LGD=0.45):
    boton = W.Button(description="Comparar mi apuesta con el modelo", button_style="info")
    out = W.Output()

    def _click(_):
        ap = bit.ultimo("L4", "apuesta")
        with out:
            clear_output()
            if ap is None:
                print("Registra tu apuesta primero."); return
            c20 = E.capital_concentracion(0.02, 0.2, [1] * 20, LGD=LGD, n_sim=60_000, verbose=False)["capital_simulado"]
            c100 = E.capital_concentracion(0.02, 0.2, [1] * 100, LGD=LGD, n_sim=60_000, verbose=False)["capital_simulado"]
            cvas = E.vasicek_cuantil(0.02, 0.2, LGD=LGD, verbose=False)["capital"]
            razon = c100 / c20
            real = "a la quinta parte" if razon < 0.35 else ("a la mitad" if razon < 0.75 else "casi no baja")
            bit.registrar("L4", "nota", compara=f"apuesta {ap['valor']} · modelo {real}", c20=c20, c100=c100)
            print(f"Tu apuesta: «{ap['valor']}». Simulado: capital con 20 nombres = {c20:.2%}; con 100 = {c100:.2%}; el límite granular (Vasicek) = {cvas:.2%}  →  {real.upper()}")
            print("Diversificar quita el riesgo propio; ρ deja el piso. Con ρ = 0.2 el piso ya está casi alcanzado a 100 nombres.")
            print("Lo que el modelo no te dijo: ρ no es constante. En crisis sube, y el capital que calculaste con 0.2 era el de los años buenos.")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L4", STEEL), boton, out])


# ---------------------------------------------------------------------------
# bienvenida: un deslizador que mueve una recta
# ---------------------------------------------------------------------------

def bienvenida():
    hm, sm = _slider(1.0, -3, 3, 0.1, "pendiente", "mueve este deslizador: así funcionan todos los de hoy", ".1f")
    out = W.Output()

    def _dibuja(m):
        with out:
            clear_output(wait=True)
            print(f"y = {m:.1f} · x      (la fórmula cambia con el deslizador; en los laboratorios verás la sustitución completa)")
            x = np.linspace(-2, 2, 50)
            fig, ax = plt.subplots(figsize=(4.5, 2.4)); ax.plot(x, m * x, color=NAVY); ax.axhline(0, color=MUTE, lw=0.5); ax.axvline(0, color=MUTE, lw=0.5)
            plt.tight_layout(); plt.show()

    W.interactive_output(_dibuja, {"m": sm})
    return W.VBox([_titulo("BIENVENIDA · el gesto"), hm, out])


# ---------------------------------------------------------------------------
# L1 · versión narrativa: el abanico, la opción y la alerta temprana
# ---------------------------------------------------------------------------

def lab1_abanico(bit, V0=100.0):
    """JUEGA: la figura clásica. Deslizadores de deuda y de sacudidas; 40 caminos y la
    distribución de lado; la zona roja es la PD."""
    hD, sD = _slider(80.0, 50.0, 98.0, 1.0, "deuda", "lo que hay que pagar en un año (los activos hoy valen 100)", ".0f")
    hs, ss = _slider(0.20, 0.05, 0.50, 0.01, "sacudida", "cuánto se mueven los activos en un año (σ_V)")
    out = W.Output()

    def _dibuja(D, sigma):
        t, cam = E.simular_caminos(V0, sigma, n_caminos=40)
        m = E.merton(V0, D, sigma, T=1.0, r=0.0, mu=0.0, verbose=False)
        colchon = math.log(V0 / D)
        bit.registrar("L1", "parametros", V=V0, D=D, sigma_V=sigma, T=1.0, mu=0.0)
        bit.registrar("L1", "resultado", PD_P=m["PD_P"], DD=m["DD"], colchon=colchon)
        with out:
            clear_output(wait=True)
            print(f"colchón = ln(100/{D:.0f}) = {colchon:.0%} · sacudida = {sigma:.0%} · DD = colchón ÷ sacudida (con un ajuste pequeño) = {m['DD']:.2f}")
            print(f"PD = área roja = {m['PD_P']:.1%}   ·   de los 40 caminos, {(cam[:, -1] < D).sum()} terminan bajo la deuda")
            fig = plt.figure(figsize=(9.5, 3.6)); gs = fig.add_gridspec(1, 2, width_ratios=[3.2, 1], wspace=0.02)
            ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1], sharey=ax)
            for k in range(cam.shape[0]):
                bajo = cam[k, -1] < D
                ax.plot(t, cam[k], color=RED if bajo else STEEL, lw=1.1 if bajo else 0.7, alpha=0.9 if bajo else 0.55)
            ax.axhline(D, color=RED, ls="--", lw=1.3); ax.text(0.02, D - 4, f"deuda {D:.0f}", color=RED, fontsize=9)
            ax.set_xlabel("años"); ax.set_ylabel("valor de los activos"); ax.set_ylim(max(20, V0 * 0.35), V0 * 1.75)
            y = np.linspace(max(20, V0 * 0.35), V0 * 1.75, 300); mu_ln = math.log(V0) - 0.5 * sigma ** 2
            dens = (1 / (y * sigma * math.sqrt(2 * math.pi))) * np.exp(-(np.log(y) - mu_ln) ** 2 / (2 * sigma ** 2))
            ax2.plot(dens, y, color=NAVY, lw=2); ax2.fill_betweenx(y, 0, dens, where=y < D, color=RED, alpha=0.45)
            ax2.axhline(D, color=RED, ls="--", lw=1.3); ax2.set_xticks([]); ax2.tick_params(labelleft=False)
            for sp in ax2.spines.values(): sp.set_visible(False)
            ax2.set_title("dónde terminan", fontsize=10, color=NAVY)
            plt.show()

    ui = W.VBox([_titulo("JUEGA · L1 · el abanico: deuda y sacudidas"),
                 _nota("Primero mueve la deuda con la sacudida fija; luego la sacudida con la deuda fija. Mira la zona roja."), hD, hs, out])
    W.interactive_output(_dibuja, {"D": sD, "sigma": ss})
    return ui


def lab1_alerta_predice(bit, nombre_caso, fecha_evento):
    return _predice(
        bit, "L1",
        f"{nombre_caso}: ¿cuántos meses ANTES del evento ({fecha_evento}) cruzó la distancia al default el umbral de 2 "
        "(«aguanta dos sacudidas») y ya no volvió?",
        ["menos de 3 meses", "entre 3 y 12 meses", "más de 12 meses", "nunca lo cruzó antes del evento"],
        "Decide antes de ver la gráfica; piensa qué tan rápido cae el precio de una acción cuando una empresa se mete en problemas.")


def lab1_alerta(bit, series, umbral0=2.0):
    """JUEGA: la DD día por día de varios emisores con la fecha de su evento y las
    acciones de calificación marcadas. `series` es un dict nombre → dict con
    'dd' (DataFrame de dd_diaria), 'evento' (fecha o None), 'calificaciones'
    (lista de (fecha, texto)) y 'nota' (deuda y acciones usadas)."""
    hu, su = _slider(umbral0, 0.5, 4.0, 0.25, "umbral", "la DD por debajo de la cual el gestor manda el nombre a revisión")
    out = W.Output()

    def _dibuja(umbral):
        with out:
            clear_output(wait=True)
            fig, axs = plt.subplots(1, len(series), figsize=(4.6 * len(series), 3.4), sharey=True)
            axs = np.atleast_1d(axs)
            res = {}
            for ax, (nombre, d) in zip(axs, series.items()):
                dd = d["dd"]
                ax.plot(dd.index, dd["DD"], color=NAVY, lw=1.6)
                ax.axhline(umbral, color=GOLD, ls="--", lw=1.2)
                if d.get("evento"):
                    ax.axvline(pd.Timestamp(d["evento"]), color=RED, lw=1.5); ax.text(pd.Timestamp(d["evento"]), ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 1, " evento", color=RED, fontsize=9, va="top")
                    f, meses = E.anticipacion(dd, d["evento"], umbral)
                    res[nombre] = meses
                    if f is not None:
                        ax.scatter([f], [dd.loc[f, "DD"]], color=GOLD, zorder=5, s=45)
                for (fc, txt) in d.get("calificaciones", []):
                    ax.axvline(pd.Timestamp(fc), color=STEEL, ls=":", lw=1)
                    ax.text(pd.Timestamp(fc), ax.get_ylim()[0] + 0.1, " " + txt, color=STEEL, fontsize=7, rotation=90, va="bottom")
                ax.set_title(nombre, fontsize=10, color=NAVY); ax.set_xlabel(""); ax.tick_params(axis="x", labelsize=8)
            axs[0].set_ylabel("distancia al default (sacudidas)")
            plt.tight_layout(); plt.show()
            for nombre, meses in res.items():
                print(f"{nombre}: " + (f"la DD cruzó {umbral:.2f} {meses:.0f} meses antes del evento" if meses is not None else f"la DD no cruzó {umbral:.2f} antes del evento"))
            bit.registrar("L1", "resultado", umbral=umbral, **{f"meses_{k}": v for k, v in res.items() if v is not None})
            bit.registrar("L1", "parametros", umbral=umbral)
            print("líneas punteadas: acciones de las calificadoras. Línea roja: el evento de crédito.")

    ui = W.VBox([_titulo("JUEGA · L1 · la distancia al default día por día"), hu, out])
    W.interactive_output(_dibuja, {"umbral": su})
    return ui


def lab1_alerta_compara(bit, nombre_caso):
    boton = W.Button(description="Comparar mi apuesta con la gráfica", button_style="info")
    out = W.Output()

    def _click(_):
        ap = bit.ultimo("L1", "apuesta"); res = bit.ultimo("L1", "resultado") or {}
        with out:
            clear_output()
            if ap is None:
                print("Registra tu apuesta primero."); return
            meses = res.get(f"meses_{nombre_caso}")
            if meses is None:
                real = "nunca lo cruzó antes del evento"
            elif meses < 3:
                real = "menos de 3 meses"
            elif meses <= 12:
                real = "entre 3 y 12 meses"
            else:
                real = "más de 12 meses"
            bit.registrar("L1", "nota", compara=f"apuesta {ap['valor']} · gráfica {real}", meses=meses)
            print(f"Tu apuesta: «{ap['valor']}». La gráfica, con el umbral que dejaste: {real.upper()}" + (f" ({meses:.0f} meses)." if meses is not None else "."))
            print("Lo que no te dijo el modelo: la DD avisa que el colchón se está acabando; no dice si habrá quien refinancie, ni cuándo vence la deuda.")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L1", STEEL), boton, out])
