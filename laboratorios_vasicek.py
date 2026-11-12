"""
LABORATORIO L4 · VASICEK, IRB Y CONCENTRACIÓN · capa de widgets sobre vasicek.py
Riesgos Financieros (FIN516) · Universidad Panamericana

Cinco partes (convenciones §2.7): PREGUNTA (markdown), PREDICE, JUEGA, COMPARA
(aquí, widgets) y RETO (celda con hueco). Todo se registra en la Bitacora del
alumno. Esta capa no calcula nada por su cuenta: llama a vasicek.py, que a su
vez llama a empresas.py. Diseñada para Google Colab con ipywidgets 8.
"""
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output

import empresas as E
import vasicek as V

NAVY, GOLD, STEEL, RED, MUTE = "#0B2545", "#B08D3C", "#5B7DA3", "#9E2A2B", "#6B7280"
matplotlib.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = V.CARTERA


def _titulo(txt, color=NAVY):
    return W.HTML(f"<div style='background:{color};color:white;padding:6px 10px;font-weight:600;"
                  f"border-radius:4px;margin:4px 0'>{txt}</div>")


def _nota(txt):
    return W.HTML(f"<div style='color:{MUTE};font-size:12px;margin:2px 0 8px 2px'>{txt}</div>")


def _slider(valor, minimo, maximo, paso, desc, leyenda, fmt=".2f"):
    s = W.FloatSlider(value=valor, min=minimo, max=maximo, step=paso, description=desc,
                      continuous_update=False, readout_format=fmt,
                      style={"description_width": "80px"}, layout=W.Layout(width="420px"))
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
# PREDICE · el peor año de cien
# ---------------------------------------------------------------------------

def lab4_predice(bit):
    return _predice(
        bit, "L4",
        "Cartera de 1,000 créditos de 1 mdp, PD 1 %, viento con ρ = 20 %. "
        "En el peor año de cien, ¿cuántos caen: unos 10, unos 30 o unos 75?",
        ["unos 10", "unos 30", "unos 75"],
        "Decidan antes de mover nada. Si cada uno cayera por su cuenta, el peor año de cien serían 18.")


# ---------------------------------------------------------------------------
# JUEGA · el viento: PD, ρ, LGD y la rareza del año malo
# ---------------------------------------------------------------------------

def lab4_juega(bit, n=C["n"], monto_mdp=C["monto_mdp"]):
    """La cartera de juguete. Mueva la PD, el viento (ρ), la LGD y la rareza del
    año malo; lea la cadena del capital y la distribución «de lado»."""
    hp, sp = _slider(C["PD"] * 100, 0.1, 10.0, 0.1, "PD (%)", "la PD de cada crédito, por año, en por ciento (1 = 1 %)", ".1f")
    hr, sr = _slider(C["rho"] * 100, 0.0, 60.0, 1.0, "ρ (%)", "cuánto pesa el viento: 0 = cada uno por su cuenta", ".0f")
    hl, sl = _slider(C["LGD"] * 100, 10.0, 90.0, 5.0, "LGD (%)", "lo que se pierde si cae, en por ciento", ".0f")
    hu, su = _slider(100, 10, 1000, 10, "uno en", "la rareza del año malo: 100 = el peor de cien; 1,000 = el peor de mil (Basilea)", ".0f")
    out = W.Output()
    base = V.cadena_capital(n=n, monto_mdp=monto_mdp, verbose=False)

    def _dibuja(PD, rho, LGD, uno_en):
        PD, rho, LGD, uno_en = PD / 100, rho / 100, LGD / 100, int(uno_en)
        res = V.cadena_capital(n=n, monto_mdp=monto_mdp, PD=PD, LGD=LGD, rho=rho, uno_en=uno_en, verbose=False) if rho > 0 else None
        bit.registrar("L4", "parametros", PD=PD, rho=rho, LGD=LGD, uno_en=uno_en)
        with out:
            clear_output(wait=True)
            if rho <= 0:
                ind = V.cada_una_por_su_cuenta(n=n, PD=PD, LGD=LGD, monto_mdp=monto_mdp, verbose=False)
                q = int(np.ceil(__import__("scipy.stats").stats.binom.ppf(1 - 1 / uno_en, n, PD)))
                cap = (q - ind["media"]) * LGD * monto_mdp
                bit.registrar("L4", "resultado", capital=cap / (n * monto_mdp), EL=ind["EL_mdp"] / (n * monto_mdp), caen=q)
                print(f"sin viento (ρ = 0): cada uno cae por su cuenta. Caen {ind['media']:.0f} ± {ind['sd']:.1f}; el peor año de {uno_en:,}: {q}")
                print(f"provisión = {ind['EL_mdp']:.1f} mdp · capital = ({q} − {ind['media']:.0f}) × {LGD:.2f} × {monto_mdp:g} = {cap:.1f} mdp")
                return
            bit.registrar("L4", "resultado", capital=res["capital"], EL=res["EL"], PD_malo=res["PD_malo"], caen=res["caen"])
            V.cadena_capital(n=n, monto_mdp=monto_mdp, PD=PD, LGD=LGD, rho=rho, uno_en=uno_en, verbose=True)
            print(f"contra el caso base (PD 1 %, ρ 20 %, LGD 45 %, el peor de cien → capital {base['capital_mdp']:.1f} mdp): ×{res['capital_mdp']/base['capital_mdp']:.2f}")
            df = E.vasicek_dist(PD, rho, LGD=LGD, puntos=1500)
            x = df["perdida"].values * res["cartera_mdp"]; y = df["densidad"].values
            fig, ax = plt.subplots(figsize=(8.5, 3.0))
            xmax = max(80, res["L_malo_mdp"] * 1.3)
            m = x <= xmax
            ax.plot(x[m], y[m], color=NAVY, lw=2)
            ax.fill_between(x, 0, y, where=(x <= res["EL_mdp"]), color=STEEL, alpha=0.6)
            ax.fill_between(x, 0, y, where=(x > res["EL_mdp"]) & (x <= res["L_malo_mdp"]), color=GOLD, alpha=0.5)
            ax.axvline(res["EL_mdp"], color=STEEL, ls="--"); ax.axvline(res["L_malo_mdp"], color=RED, ls="--")
            ymax = y[m].max()
            ax.text(res["EL_mdp"] + 0.5, ymax * 0.9, f"provisión {res['EL_mdp']:.1f}", color=STEEL, fontsize=9, va="top")
            ax.text(res["L_malo_mdp"] + 0.5, ymax * 0.6, f"año malo {res['L_malo_mdp']:.1f}\ncapital {res['capital_mdp']:.1f} mdp", color=RED, fontsize=9, va="top")
            ax.set_xlim(0, xmax); ax.set_ylim(0, ymax * 1.05); ax.set_yticks([])
            ax.set_xlabel("pérdida del año, mdp"); ax.set_title("la pérdida del año: provisión (azul) y capital (dorado)", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L4 · el viento: PD, ρ, LGD y la rareza del año malo"),
                 _nota(f"Cartera de {n:,} créditos de {monto_mdp:g} mdp. Primero muevan sólo ρ (de 20 a 0 y de 20 a 40); luego la PD; luego «uno en» a 1,000 (lo que pide Basilea)."),
                 hp, hr, hl, hu, out])
    W.interactive_output(_dibuja, {"PD": sp, "rho": sr, "LGD": sl, "uno_en": su})
    return ui


# ---------------------------------------------------------------------------
# JUEGA · la simulación: cuántos caen por año, con y sin viento
# ---------------------------------------------------------------------------

def lab4_simula(bit, n=C["n"], n_anios=5_000):
    """Simula n_anios años con el viento que elijan y muestra 100 años y la distribución de lado."""
    hr, sr = _slider(C["rho"] * 100, 0.0, 60.0, 5.0, "ρ (%)", "cuánto pesa el viento", ".0f")
    hp, sp = _slider(C["PD"] * 100, 0.5, 5.0, 0.5, "PD (%)", "la PD de cada crédito, en por ciento", ".1f")
    out = W.Output()

    def _dibuja(rho, PD):
        rho, PD = rho / 100, PD / 100
        c0 = V.simular_anios(n, PD, 0.0, n_anios, semilla=7)
        c1 = V.simular_anios(n, PD, rho, n_anios, semilla=7) if rho > 0 else c0
        q0, q1 = np.quantile(c0, 0.99), np.quantile(c1, 0.99)
        bit.registrar("L4", "nota", simula_rho=rho, simula_PD=PD, peor100_sin=float(q0), peor100_con=float(q1))
        with out:
            clear_output(wait=True)
            print(f"{n_anios:,} años simulados · PD {PD:.1%} · ρ {rho:.0%}")
            print(f"   sin viento: promedio {c0.mean():.1f} · ± {c0.std():.1f} · el peor de cien {q0:.0f}")
            print(f"   con viento: promedio {c1.mean():.1f} · ± {c1.std():.1f} · el peor de cien {q1:.0f}   (la fórmula: {n*V.pd_anio_malo(PD, rho, 100, verbose=False)['PD_malo']:.0f})" if rho > 0 else "   (ρ = 0: no hay viento)")
            fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.0), sharey=True, gridspec_kw={"width_ratios": [2.2, 1]})
            a = np.arange(1, 101)
            ax.bar(a - 0.2, c0[:100], width=0.4, color=STEEL, label="sin viento")
            ax.bar(a + 0.2, c1[:100], width=0.4, color=RED, label=f"ρ = {rho:.0%}")
            ax.axhline(n * PD, color=NAVY, ls=":"); ax.legend(frameon=False, fontsize=9); ax.set_xlabel("año"); ax.set_ylabel(f"caen de {n:,}")
            top = max(40, np.quantile(c1, 0.999) * 1.1)
            bins = np.linspace(0, top, 50)
            ax2.hist(c0, bins=bins, orientation="horizontal", color=STEEL, density=True)
            ax2.hist(c1, bins=bins, orientation="horizontal", color=RED, alpha=0.6, density=True)
            ax2.axhline(q1, color=NAVY, ls="--"); ax2.text(ax2.get_xlim()[1] * 0.98, q1 + top * 0.02, f"peor de cien: {q1:.0f}", ha="right", color=NAVY, fontsize=9)
            ax2.set_xticks([]); ax2.set_ylim(0, top); ax2.set_xlabel("de lado")
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L4 · la simulación: cuántos caen por año"), _nota("Muevan ρ. La media no cambia; miren los años malos y la cola de lado."), hr, hp, out])
    W.interactive_output(_dibuja, {"rho": sr, "PD": sp})
    return ui


# ---------------------------------------------------------------------------
# JUEGA · el crédito grande: concentración
# ---------------------------------------------------------------------------

def lab4_concentra(bit, n_sim=20_000):
    """900 créditos de 1 mdp más uno grande; mueva el tamaño del grande y lea el costo."""
    hg, sg = _slider(C["grande_mdp"] if "grande_mdp" in C else 100.0, 1.0, 250.0, 1.0, "grande (mdp)", "el tamaño del crédito grande; los otros 900 son de 1 mdp", ".0f")
    out = W.Output()

    def _dibuja(grande):
        r = V.concentracion(grande_mdp=grande, n_sim=n_sim, verbose=False)
        bit.registrar("L4", "nota", grande_mdp=grande, costo_mdp=r["costo_mdp"], n_efectivo=r["n_efectivo"])
        with out:
            clear_output(wait=True)
            print(f"900 de 1 mdp + 1 de {grande:,.0f} mdp = {r['cartera_mdp']:,.0f} mdp · el grande pesa {r['peso_grande']:.1%} · nombres efectivos {r['n_efectivo']:.0f}")
            print(f"capital si todos fueran iguales (Vasicek): {r['capital_vasicek_mdp']:.1f} mdp · con el grande dentro (simulado): {r['capital_simulado_mdp']:.1f} mdp   ×{r['factor']:.2f}")
            print(f"costo de la concentración: {r['costo_mdp']:+.1f} mdp")
            gs = [1, 10, 25, 50, 100, 150, 200, 250]
            costos = [V.concentracion(grande_mdp=g, n_sim=8_000, verbose=False)["costo_mdp"] for g in gs]
            fig, ax = plt.subplots(figsize=(7, 2.6))
            ax.plot(gs, costos, "-o", color=NAVY, ms=4); ax.scatter([grande], [r["costo_mdp"]], color=GOLD, s=70, zorder=4)
            ax.set_xlabel("tamaño del crédito grande, mdp"); ax.set_ylabel("costo de la concentración, mdp")
            ax.set_title("lo que la fórmula de Vasicek no ve: crece más que proporcional", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L4 · el crédito grande"), _nota("Muevan el tamaño del grande: 1 mdp es «todos iguales»; 100 mdp es «pesa 10 %»."), hg, out])
    W.interactive_output(_dibuja, {"grande": sg})
    return ui


# ---------------------------------------------------------------------------
# COMPARA
# ---------------------------------------------------------------------------

def lab4_compara(bit):
    boton = W.Button(description="Comparar mi apuesta con el modelo", button_style="info")
    out = W.Output()

    def _click(_):
        ap = bit.ultimo("L4", "apuesta")
        with out:
            clear_output()
            if ap is None:
                print("Registren su apuesta primero."); return
            pm = V.pd_anio_malo(verbose=False)
            caen = C["n"] * pm["PD_malo"]
            real = "unos 10" if caen < 20 else ("unos 30" if caen < 50 else "unos 75")
            bit.registrar("L4", "nota", compara=f"apuesta {ap['valor']} · modelo {real}", caen=caen)
            ind = V.cada_una_por_su_cuenta(verbose=False)
            print(f"Su apuesta: «{ap['valor']}». El modelo: PD del peor año de cien = Φ((−2.33 + 0.447 × 2.33) / 0.894) = {pm['PD_malo']:.1%} → caen {caen:.0f} de 1,000  →  {real.upper()}")
            print(f"   sin viento serían {ind['peor100']}; con viento de 20 % son {caen:.0f}: ×{caen/ind['peor100']:.1f}. La media sigue siendo 10 en los dos casos.")
            cc = V.cadena_capital(verbose=False)
            print(f"   en pesos: provisión {cc['EL_mdp']:.1f} mdp · capital {cc['capital_mdp']:.1f} mdp ({cc['capital_mdp']/cc['EL_mdp']:.1f} veces la provisión)")
            print("La trampa era pensar en línea recta: el viento no suma unos cuantos; mueve a todas a la vez, y la cola se estira.")
            print("Lo que el modelo no dijo: cuánto pesa el viento de verdad. Nadie observa ρ; Basilea lo fija por fórmula, y en crisis sube.")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L4", STEEL), boton, out])
