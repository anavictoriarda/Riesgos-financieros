"""
LABORATORIO L3 · FORMA REDUCIDA · capa de widgets sobre reducida.py
Riesgos Financieros (FIN516) · Universidad Panamericana

Cinco partes (convenciones §2.7): PREGUNTA (markdown), PREDICE, JUEGA, COMPARA
(aquí, widgets) y RETO (celda con hueco). Todo se registra en la Bitacora del
alumno. Esta capa no calcula nada por su cuenta: llama a reducida.py, que a su
vez llama a empresas.intensidad_covariables. Diseñada para Google Colab con
ipywidgets 8.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output

import empresas as E
import reducida as R

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
# el termómetro: mueve UNA lectura a la vez y mira por cuánto se multiplica λ
# ---------------------------------------------------------------------------

def lab3_termometro(bit):
    """Las cuatro lecturas de UNA empresa. Parte de la empresa tranquila; cada
    movimiento imprime el multiplicador de esa lectura y la λ en cadena."""
    t = R.DOS_EMPRESAS_R.loc["tranquila"]
    hd, sd = _slider(float(t["DD"]), 0.5, 7.0, 0.5, "DD", "distancia al default: sacudidas que aguanta el colchón", ".1f")
    hr, sr = _slider(float(t["ret"]), -0.6, 0.6, 0.05, "acción", "retorno de la acción en 12 meses (−0.30 = −30 %)")
    ht, st = _slider(float(t["r3m"]), 0.0, 8.0, 0.25, "tasa 3 m", "tasa del Tesoro a 3 meses, EN POR CIENTO (1.5 = 1.5 %)", ".2f")
    hs, ssp = _slider(float(t["sp500"]), -0.4, 0.4, 0.05, "S&P 500", "retorno del mercado en 12 meses: el ciclo")
    out = W.Output()
    base = R.lambda_en_cadena(verbose=False, **t.to_dict())

    def _dibuja(DD, ret, r3m, sp):
        res = R.lambda_en_cadena(DD, ret=ret, r3m=r3m, sp500=sp, verbose=False)
        bit.registrar("L3", "termometro", DD=DD, ret=ret, r3m=r3m, sp500=sp, **{"lambda": res["lambda"]})
        df = R.multiplicadores(a={"DD": DD, "ret": ret, "r3m": r3m, "sp500": sp, "nombre": "ahora"}, de="tranquila", verbose=False)
        with out:
            clear_output(wait=True)
            print(f"λ en cadena: exponente = {E.DSW_COEF['const']:+.3f} "
                  f"{E.DSW_COEF['DD']:+.3f}·{DD:.2f} {E.DSW_COEF['ret']:+.3f}·{ret:.2f} "
                  f"{E.DSW_COEF['r3m']:+.3f}·{r3m:.2f} {E.DSW_COEF['sp500']:+.3f}·{sp:.2f} = {res['exponente']:+.3f}")
            print(f"λ = e^({res['exponente']:+.3f}) = {res['lambda']*1e4:,.0f} pb por año · PD a un año = 1 − e^(−λ) = {res['PD_1a']:.2%}")
            print(f"desde la tranquila ({base['lambda']*1e4:,.0f} pb), cada lectura multiplica:")
            for k, r in df.iterrows():
                marca = "   ←" if abs(r["multiplica"] - 1) > 0.005 else ""
                print(f"   {k:<28s} {r['de']:>6.2f} → {r['a']:>6.2f}   ×{r['multiplica']:.2f}{marca}")
            print(f"   producto de los cuatro: ×{df['acumulado'].iloc[-1]:.2f}   →   {base['lambda']*1e4:,.0f} pb × {df['acumulado'].iloc[-1]:.2f} = {res['lambda']*1e4:,.0f} pb")
            fig, ax = plt.subplots(figsize=(8.5, 2.6))
            etq = ["constante", "DD", "acción", "tasa", "S&P 500"]
            partes = [E.DSW_COEF["const"]] + [res["aportes"][k] for k in ["DD", "ret", "r3m", "sp500"]]
            acum = 0.0
            for i, p in enumerate(partes):
                ax.bar(i, p, bottom=acum, color=MUTE if i == 0 else (RED if p > 0 else NAVY), width=0.6)
                ax.text(i, acum + p + (0.15 if p > 0 else -0.15), f"{p:+.2f}", ha="center", va="bottom" if p > 0 else "top", fontsize=9)
                acum += p
            ax.axhline(acum, color=NAVY, ls=":", lw=1)
            ax.set_xticks(range(5)); ax.set_xticklabels(etq); ax.set_ylim(min(-8.5, acum - 1), 1.0)
            ax.set_title(f"el exponente, término por término: {acum:+.2f}  →  λ = {res['lambda']*1e4:,.0f} pb", fontsize=10, color=NAVY)
            plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L3 · el termómetro: una lectura a la vez"),
                 _nota("Parte de la empresa tranquila. Mueva UNA lectura, lea el multiplicador; luego otra. Los aportes azules bajan λ; los rojos la suben."),
                 hd, hr, ht, hs, out])
    W.interactive_output(_dibuja, {"DD": sd, "ret": sr, "r3m": st, "sp": ssp})
    return ui


# ---------------------------------------------------------------------------
# PREDICE · JUEGA · COMPARA del CRO: la DD promedio de la cartera cae de 4 a 2
# ---------------------------------------------------------------------------

def lab3_predice(bit):
    return _predice(
        bit, "L3",
        "Marzo de 2020: la distancia al default promedio de su cartera cae de 4 a 2 sacudidas. "
        "Según el termómetro, ¿λ sube menos del doble, entre 2 y 5 veces, o más de 5 veces?",
        ["menos del doble", "entre 2 y 5 veces", "más de 5 veces"],
        "Decidan antes de mover nada. La DD se parte a la mitad; ¿la λ se duplica?")


def lab3_juega(bit, cartera_mmdp=100.0, LGD=0.45):
    """La cartera del CRO: DD promedio y el ciclo. Imprime la cadena término por
    término, los multiplicadores contra la base y la provisión esperada."""
    hd, sd = _slider(4.0, 0.5, 7.0, 0.1, "DD", "distancia al default promedio de los acreditados (de Merton / KMV)", ".1f")
    hr, sr = _slider(0.0, -0.6, 0.6, 0.05, "acción", "retorno de la acción del emisor en 12 meses (fracción)")
    hs, ssp = _slider(0.0, -0.4, 0.4, 0.05, "S&P 500", "retorno del mercado en 12 meses: el ciclo")
    ht, st = _slider(1.5, 0.0, 8.0, 0.25, "tasa 3 m", "tasa del Tesoro a 3 meses, EN POR CIENTO (1.5 = 1.5 %)", ".2f")
    out = W.Output()
    base = R.provision_esperada(cartera_mmdp=cartera_mmdp, LGD=LGD, verbose=False, **R.BASE)

    def _dibuja(DD, ret, sp, r3m):
        res = R.provision_esperada(DD, ret=ret, r3m=r3m, sp500=sp, cartera_mmdp=cartera_mmdp, LGD=LGD, verbose=False)
        df = R.multiplicadores(a={"DD": DD, "ret": ret, "r3m": r3m, "sp500": sp, "nombre": "ahora"}, de=R.BASE, verbose=False)
        bit.registrar("L3", "parametros", DD=DD, ret=ret, sp500=sp, r3m=r3m)
        bit.registrar("L3", "resultado", **{"lambda": res["lambda"]}, PD_1a=res["PD_1a"], provision_mmdp=res["provision_mmdp"])
        c = E.DSW_COEF
        with out:
            clear_output(wait=True)
            print("en símbolos:  λ = exp(β0 + β_DD·DD + β_ret·acción + β_r·tasa + β_sp·S&P500)")
            print(f"con números:  exponente = {c['const']:+.3f} {c['DD']:+.3f}·{DD:.2f} {c['ret']:+.3f}·{ret:.2f} {c['r3m']:+.3f}·{r3m:.2f} {c['sp500']:+.3f}·{sp:.2f}")
            print(f"            = {c['const']:+.3f} {c['DD']*DD:+.3f} {c['ret']*ret:+.3f} {c['r3m']*r3m:+.3f} {c['sp500']*sp:+.3f} = {res['exponente']:+.3f}")
            print(f"λ = e^({res['exponente']:+.3f}) = {res['lambda']*1e4:,.0f} pb por año · PD a un año = 1 − e^(−λ) = {res['PD_1a']:.2%}")
            print(f"contra la base (DD 4, acción 0, tasa 1.5 %, S&P 0 → {base['lambda']*1e4:,.0f} pb), cada lectura multiplica:")
            for k, r in df.iterrows():
                print(f"   {k:<28s} {r['de']:>6.2f} → {r['a']:>6.2f}   ×{r['multiplica']:.2f}")
            print(f"   producto: ×{df['acumulado'].iloc[-1]:.1f} en λ   (en PD: ×{res['PD_1a']/base['PD_1a']:.1f})")
            print(f"provisión esperada a un año = PD × LGD × cartera = {res['PD_1a']:.4f} × {LGD} × {cartera_mmdp:,.0f} mmdp = {res['provision_mmdp']:.3f} mmdp   (base: {base['provision_mmdp']:.3f})")
            print("pesos: Duffie, Saita y Wang (2007), Tabla 2; industriales de EUA 1980–2004 · cartera estilizada")
            dds, lams0 = R.curva_lambda_dd()
            _, lams = R.curva_lambda_dd(ret=ret, r3m=r3m, sp500=sp)
            fig, ax = plt.subplots(figsize=(8.5, 3.2))
            ax.plot(dds, lams0, color=STEEL, lw=1.5, label="ciclo neutro")
            ax.plot(dds, lams, color=NAVY, lw=2, label="con sus lecturas")
            ax.scatter([DD], [res["lambda"] * 1e4], color=GOLD, zorder=3, s=50)
            ax.set_yscale("log"); ax.set_xlabel("distancia al default DD"); ax.set_ylabel("λ, pb anuales (escala log)")
            ax.set_title("λ contra DD: cada sacudida menos multiplica, no suma", fontsize=10, color=NAVY)
            ax.legend(frameon=False, fontsize=9); plt.tight_layout(); plt.show()

    ui = W.VBox([_titulo("JUEGA · L3 · la cartera del CRO: DD promedio y ciclo"),
                 _nota(f"Cartera estilizada de {cartera_mmdp:,.0f} mmdp con LGD {LGD:.0%}. Primero muevan sólo la DD de 4 a 2; después el ciclo (acción y S&P 500)."),
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
                print("Registren su apuesta primero."); return
            r = R.cuanto_se_multiplica(verbose=False)
            factor = r["factor_lambda"]
            real = "menos del doble" if factor < 2 else ("entre 2 y 5 veces" if factor <= 5 else "más de 5 veces")
            bit.registrar("L3", "nota", compara=f"apuesta {ap['valor']} · modelo {real}", factor=factor)
            print(f"Su apuesta: «{ap['valor']}». El modelo: λ(DD 2) / λ(DD 4) = e^(−β_DD·(2 − 4)) = e^({-E.DSW_COEF['DD']*2:.3f}) = ×{factor:.1f}  →  {real.upper()}")
            print(f"   λ: {r['lambda_antes']*1e4:,.0f} pb → {r['lambda_despues']*1e4:,.0f} pb · provisión esperada: {r['prov_antes']:.3f} → {r['prov_despues']:.3f} mmdp (×{r['factor_prov']:.1f})")
            r2 = R.cuanto_se_multiplica(ciclo=R.CICLO_2020, verbose=False)
            print(f"   con el ciclo de marzo de 2020 encima (acción −30 %, tasa 0.1 %, S&P −15 %): ×{r2['factor_lambda']:.1f} en λ, provisión {r2['prov_despues']:.3f} mmdp")
            print("La trampa era pensar en línea recta: la mitad de colchón no es el doble de λ, porque λ es un exponencial de la DD.")
            print("Lo que el modelo no dijo: cuánto va a durar. La provisión de hoy es la PD de hoy; el overlay (XIV) existe por eso.")

    boton.on_click(_click)
    return W.VBox([_titulo("COMPARA · L3", STEEL), boton, out])
