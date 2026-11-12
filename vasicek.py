"""
VASICEK, IRB Y CONCENTRACIÓN · de una empresa a mil (la cartera)
Riesgos Financieros (FIN516) · Universidad Panamericana

Capa fina sobre empresas.py para el bloque L4, construido de abajo hacia arriba:
de una empresa a mil → pero caen juntas (el viento) → la PD de un año malo →
dos carteras, misma PD, distinta cola → esperada vs inesperada (provisión vs
capital) → de dónde sale ρ y la fórmula IRB → lo que Vasicek supone y Pemex
rompe (concentración) → para qué sirve (el CRO) → lo que no ve → la fórmula.

Metáfora del bloque: el VIENTO. Cada empresa tiene su colchón (Merton, L1) y
cada año sopla un viento que baja TODOS los colchones a la vez; lo que le pasa
a cada una es parte viento y parte lo suyo. ρ (léase «rho», la correlación de
activos) dice cuánto pesa el viento.

UNIDADES: PD, LGD y ρ en fracción (0.01 = 1 %); montos en mdp (millones de
pesos); «uno en» es la rareza del año malo (100 = el peor año de cien).
"""
import math
import numpy as np
import pandas as pd
from scipy.stats import norm, binom

import empresas as E
from empresas import (vasicek_pd_condicional, vasicek_cuantil, vasicek_dist, irb_rho, irb_capital,
                      hhi, capital_concentracion, LIMITES_CUB, PEMEX_EXPOSICION, CNBV_COVID_2020,
                      revisa, Bitacora, tarea, preguntas_p1p3)

# El rol del laboratorio L4 (se registra en la maquinaria de la tarea, sin tocar empresas.py).
E.LABS["L4"].update({
    "modelo": "Vasicek (un factor) / IRB",
    "rol": "director de riesgos (CRO) de un banco mexicano",
    "fecha": "marzo de 2020, con la exposición a Pemex de 2025",
    "decision": "cuánto capital pide la cartera en un año malo, y qué límite por nombre se defiende ante el consejo",
    "parametro_clave": "rho", "resultado_clave": "capital",
})

# ---------------------------------------------------------------------------
# la cartera de juguete (construcción del curso: ESTILIZADA)
# ---------------------------------------------------------------------------
CARTERA = {"n": 1_000, "monto_mdp": 1.0, "PD": 0.01, "LGD": 0.45, "rho": 0.20, "uno_en": 100}
# 1,000 créditos de 1 mdp cada uno (1,000 mdp en total), PD 1 % por año, LGD 45 %,
# ρ 20 % y «año malo» = el peor año de cien.

# Marzo de 2020, estilizado: la PD se duplica y el viento pesa más (ρ sube).
ESCENARIO_2020 = {"PD": 0.02, "rho": 0.30}

# Rareza de los años, en palabras.
ANIOS = [("año típico (viento 0)", None), ("promedio de todos los años", "promedio"),
         ("año malo: el peor de cien", 100), ("año muy malo: el peor de mil", 1_000)]

# El crédito grande del juguete de concentración: 900 créditos de 1 mdp + 1 de 100 mdp.
CONCENTRACION = {"n_chicos": 900, "grande_mdp": 100.0}

# Lo que hicieron las autoridades y los bancos con el capital en 2020 (para la lámina del CRO).
CAPITAL_2020 = {
    "cnbv_dividendos": "31 de marzo de 2020: la CNBV recomendó a los bancos no pagar dividendos ni recomprar acciones, para conservar capital",  # [V]
    "icap_sistema": "el índice de capitalización del sistema se mantuvo arriba de 17 % durante 2020, contra un mínimo regulatorio de 10.5 %",  # [S]
    "fuente": "CNBV, comunicado 31-mar-2020 [V]; boletines de la CNBV sobre ICAP 2020 [S]",
}


def _z_malo(uno_en):
    """El viento del año malo: el cuantil 1/uno_en de la normal (negativo).
    uno_en = 100 → −2.33; uno_en = 1,000 → −3.09."""
    return float(norm.ppf(1.0 / uno_en))


# ---------------------------------------------------------------------------
# 1 · de una empresa a mil: si cada una cae por su cuenta
# ---------------------------------------------------------------------------
def cada_una_por_su_cuenta(n=CARTERA["n"], PD=CARTERA["PD"], LGD=CARTERA["LGD"], monto_mdp=CARTERA["monto_mdp"], verbose=True):
    """Mil créditos que caen cada uno por su cuenta (independientes): cuántos caen
    en promedio, cuánto se mueve ese número, y el peor año de cien y de mil."""
    media = n * PD
    sd = math.sqrt(n * PD * (1 - PD))
    peor100 = int(binom.ppf(0.99, n, PD))
    peor1000 = int(binom.ppf(0.999, n, PD))
    p_mas20 = float(1 - binom.cdf(20, n, PD))
    EL = media * LGD * monto_mdp
    if verbose:
        print("CADA UNA POR SU CUENTA (independientes)")
        print(f"   {n:,} créditos de {monto_mdp:g} mdp con PD {PD:.0%}: en promedio caen {media:.0f} por año")
        print(f"   ese número se mueve ± {sd:.1f} (la desviación de una binomial: √(n·PD·(1−PD)))")
        print(f"   el peor año de cien: caen {peor100} · el peor de mil: {peor1000} · más de 20 caen en {p_mas20:.2%} de los años")
        print(f"   pérdida esperada = caen × LGD × monto = {media:.0f} × {LGD:.0%} × {monto_mdp:g} mdp = {EL:.1f} mdp")
    return {"media": media, "sd": sd, "peor100": peor100, "peor1000": peor1000, "p_mas20": p_mas20, "EL_mdp": EL}


# ---------------------------------------------------------------------------
# 2 y 3 · el viento y la PD de un año malo
# ---------------------------------------------------------------------------
def pd_anio_malo(PD=CARTERA["PD"], rho=CARTERA["rho"], uno_en=CARTERA["uno_en"], verbose=True):
    """La PD de un año malo, en cadena: PD → umbral → viento → PD condicional.

    Cada empresa cae cuando su variable X (su colchón en sacudidas, centrado)
    queda por debajo de un umbral; el umbral es el que hace que caiga con
    probabilidad PD. En un año con viento Z, lo que le queda de propio tiene
    que caer más (o menos) para cruzar el umbral.
    """
    umbral = float(norm.ppf(PD))
    Z = _z_malo(uno_en)
    pd_malo = vasicek_pd_condicional(PD, rho, Z)
    if verbose:
        print(f"LA PD DE UN AÑO MALO (el peor de {uno_en:,})")
        print("   en palabras:  la empresa cae si su variable queda bajo el umbral; el viento del año la empuja hacia abajo,")
        print("                 y lo suyo tiene que cubrir lo que falte. En un año malo falta menos: la PD sube.")
        print("   en símbolos:  PD(Z) = Φ( (Φ⁻¹(PD) − √ρ·Z) / √(1 − ρ) )      Φ: la cola de la normal; Z: el viento (negativo = malo)")
        print(f"   con números:  umbral = Φ⁻¹({PD:.2f}) = {umbral:.2f}  ·  viento del peor año de {uno_en:,} = {Z:.2f}  ·  √ρ = √{rho:.2f} = {math.sqrt(rho):.3f}  ·  √(1−ρ) = {math.sqrt(1-rho):.3f}")
        print(f"                 PD(año malo) = Φ( ({umbral:.2f} − {math.sqrt(rho):.3f}·({Z:.2f})) / {math.sqrt(1-rho):.3f} ) = Φ({(umbral - math.sqrt(rho)*Z)/math.sqrt(1-rho):.2f}) = {pd_malo:.1%}")
        print(f"   de {PD:.1%} a {pd_malo:.1%}: ×{pd_malo/PD:.1f}")
    return {"umbral": umbral, "Z": Z, "PD_malo": pd_malo, "factor": pd_malo / PD}


def tabla_anios(PD=CARTERA["PD"], rho=CARTERA["rho"], n=CARTERA["n"]):
    """Un año típico, el promedio, el peor de cien y el peor de mil: la PD de cada
    empresa y cuántas de n caen."""
    filas = []
    for nombre, k in ANIOS:
        if k is None:
            Z, p = 0.0, vasicek_pd_condicional(PD, rho, 0.0)
        elif k == "promedio":
            Z, p = float("nan"), PD
        else:
            Z = _z_malo(k); p = vasicek_pd_condicional(PD, rho, Z)
        filas.append({"año": nombre, "viento Z": Z, "PD de cada empresa": p, f"caen de {n:,}": n * p})
    return pd.DataFrame(filas).set_index("año")


# ---------------------------------------------------------------------------
# 4 · dos carteras, misma PD, distinta cola: simulación de muchos años
# ---------------------------------------------------------------------------
def simular_anios(n=CARTERA["n"], PD=CARTERA["PD"], rho=CARTERA["rho"], n_anios=10_000, semilla=7):
    """Cuántos de n caen en cada uno de n_anios años simulados con el modelo de un
    factor: X = √ρ·viento + √(1−ρ)·lo suyo; cae si X < Φ⁻¹(PD). Devuelve un array."""
    rng = np.random.default_rng(semilla)
    umbral = norm.ppf(PD)
    Z = rng.standard_normal((n_anios, 1))
    eps = rng.standard_normal((n_anios, n))
    X = math.sqrt(rho) * Z + math.sqrt(1 - rho) * eps
    return (X < umbral).sum(axis=1)


def dos_carteras(n=CARTERA["n"], PD=CARTERA["PD"], rho=CARTERA["rho"], n_anios=10_000, semilla=7, verbose=True):
    """Misma PD, dos ρ (0 y rho): media, desviación y el peor año de cien y de mil."""
    out = {}
    for r in [0.0, rho]:
        c = simular_anios(n, PD, r, n_anios, semilla)
        out[r] = {"media": float(c.mean()), "sd": float(c.std()), "peor100": float(np.quantile(c, 0.99)),
                  "peor1000": float(np.quantile(c, 0.999)), "max": int(c.max()), "p_mas20": float((c > 20).mean()), "caidas": c}
    if verbose:
        print(f"DOS CARTERAS, MISMA PD ({PD:.0%}), {n_anios:,} años simulados")
        for r, d in out.items():
            et = "cada una por su cuenta (ρ = 0)" if r == 0 else f"con viento (ρ = {r:.0%})"
            print(f"   {et:<34s} promedio {d['media']:5.1f} · se mueve ± {d['sd']:4.1f} · peor de cien {d['peor100']:5.0f} · peor de mil {d['peor1000']:5.0f} · más de 20 en {d['p_mas20']:.1%} de los años")
        print("   la media es la misma; la cola no. Lo que cambia no es cuántas caen por año en promedio, sino cuántas caen JUNTAS en el año malo.")
    return out


# ---------------------------------------------------------------------------
# 5 · esperada vs inesperada = provisión vs capital
# ---------------------------------------------------------------------------
def cadena_capital(n=CARTERA["n"], monto_mdp=CARTERA["monto_mdp"], PD=CARTERA["PD"], LGD=CARTERA["LGD"],
                   rho=CARTERA["rho"], uno_en=CARTERA["uno_en"], verbose=True, etiqueta=""):
    """La cadena del capital: pérdida esperada (provisión) → PD del año malo →
    pérdida del año malo → capital = lo inesperado.

    Devuelve montos en mdp y fracciones de la cartera. `capital` (fracción) es
    el número que se registra en la bitácora como resultado clave de L4.
    """
    cartera = n * monto_mdp
    EL = PD * LGD * cartera
    pm = pd_anio_malo(PD, rho, uno_en, verbose=False)
    L_malo = pm["PD_malo"] * LGD * cartera
    capital = L_malo - EL
    if verbose:
        et = f" · {etiqueta}" if etiqueta else ""
        print(f"LA CADENA DEL CAPITAL{et}")
        print(f"   cartera: {n:,} créditos de {monto_mdp:g} mdp = {cartera:,.0f} mdp · PD {PD:.1%} · LGD {LGD:.0%} · ρ {rho:.0%} · año malo = el peor de {uno_en:,}")
        print("   en palabras:  la provisión cubre lo que se pierde en un año promedio; el capital cubre lo que el año malo pierde DE MÁS")
        print("   en símbolos:  EL = PD · LGD · cartera        capital = (PD(año malo) − PD) · LGD · cartera")
        print(f"   con números:  EL = {PD:.3f} × {LGD:.2f} × {cartera:,.0f} = {EL:.1f} mdp  (provisión)")
        print(f"                 PD(año malo) = {pm['PD_malo']:.1%}  →  caen {n*pm['PD_malo']:.0f} de {n:,}  →  pérdida del año malo = {pm['PD_malo']:.3f} × {LGD:.2f} × {cartera:,.0f} = {L_malo:.1f} mdp")
        print(f"                 capital = {L_malo:.1f} − {EL:.1f} = {capital:.1f} mdp = {capital/cartera:.1%} de la cartera  ({capital/EL:.1f} veces la provisión)")
    return {"cartera_mdp": cartera, "EL_mdp": EL, "PD_malo": pm["PD_malo"], "caen": n * pm["PD_malo"],
            "L_malo_mdp": L_malo, "capital_mdp": capital, "capital": capital / cartera, "EL": EL / cartera}


def curva_capital_rho(PD=CARTERA["PD"], LGD=CARTERA["LGD"], uno_en=CARTERA["uno_en"], cartera_mdp=1_000.0, rhos=None):
    """Capital (mdp) como función de ρ, con lo demás fijo. ρ = 0 usa la binomial."""
    rhos = np.linspace(0, 0.5, 101) if rhos is None else np.asarray(rhos, dtype=float)
    cap = []
    for r in rhos:
        if r <= 0:
            k = binom.ppf(1 - 1 / uno_en, int(cartera_mdp), PD)
            cap.append((k - cartera_mdp * PD) * LGD)
        else:
            cap.append(vasicek_cuantil(PD, r, q=1 - 1 / uno_en, LGD=LGD, verbose=False)["capital"] * cartera_mdp)
    return rhos, np.array(cap)


def duplicar_rho(PD=CARTERA["PD"], LGD=CARTERA["LGD"], uno_en=CARTERA["uno_en"], cartera_mdp=1_000.0, verbose=True):
    """La trampa: de ρ 0 a 12 % y de 12 % a 24 %, el capital no sube por lo mismo."""
    rhos, cap = curva_capital_rho(PD, LGD, uno_en, cartera_mdp, rhos=[0.0, 0.12, 0.24])
    out = dict(zip([0.0, 0.12, 0.24], cap))
    if verbose:
        print(f"DUPLICAR ρ (PD {PD:.0%}, LGD {LGD:.0%}, el peor año de {uno_en:,}, cartera {cartera_mdp:,.0f} mdp)")
        print(f"   ρ = 0 %:  capital {out[0.0]:.1f} mdp (la binomial: casi nada de cola)")
        print(f"   ρ = 12 %: capital {out[0.12]:.1f} mdp   ×{out[0.12]/out[0.0]:.1f} contra ρ = 0")
        print(f"   ρ = 24 %: capital {out[0.24]:.1f} mdp   ×{out[0.24]/out[0.12]:.1f} contra ρ = 12 %")
        print("   el salto grande es de «nada de viento» a «algo de viento»; después, duplicar ρ casi duplica el capital.")
    return out


# ---------------------------------------------------------------------------
# 6 · de dónde sale ρ: Basilea (CRE31)
# ---------------------------------------------------------------------------
def tabla_rho_irb(PDs=(0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20), LGD=CARTERA["LGD"], M=2.5):
    """ρ(PD) de Basilea y el capital K por unidad de exposición, para varias PD."""
    filas = []
    for p in PDs:
        r = irb_capital(p, LGD=LGD, M=M, verbose=False)
        filas.append({"PD": p, "ρ de Basilea": r["rho"], "K (capital por peso prestado)": r["K"], "ponderador de riesgo": r["RWA"]})
    return pd.DataFrame(filas).set_index("PD")


def capital_irb(PD=CARTERA["PD"], LGD=CARTERA["LGD"], M=2.5, cartera_mdp=1_000.0, verbose=True):
    """El capital IRB de Basilea, en palabras: es la cadena del capital con tres
    cambios del regulador: el año malo es el peor de MIL (99.9 %), ρ sale de la
    PD con una fórmula, y hay un ajuste por plazo (M, en años)."""
    r = irb_capital(PD, LGD=LGD, M=M, verbose=False)
    if verbose:
        print("EL CAPITAL DE BASILEA (IRB, corporativos)")
        print("   en palabras:  la misma cadena, pero el año malo es el peor de mil, la ρ la da una fórmula que baja con la PD,")
        print("                 y un crédito más largo pide más capital (ajuste por plazo).")
        print(f"   con números:  PD {PD:.1%} → ρ de Basilea = {r['rho']:.1%} · plazo M = {M:g} años → ajuste b = {r['b']:.4f}")
        print(f"                 PD del peor año de mil = {r['PD_q']:.1%} · pérdida = {r['L_q']:.2%} · EL = {r['EL']:.2%} · inesperada = {r['capital']:.2%}")
        print(f"                 K = inesperada × (1 + (M − 2.5)·b) / (1 − 1.5·b) = {r['K']:.2%} de lo prestado")
        print(f"                 ponderador de riesgo = 12.5 × K = {r['RWA']:.2%} · para {cartera_mdp:,.0f} mdp: capital {r['K']*cartera_mdp:.1f} mdp")
    return {**r, "capital_mdp": r["K"] * cartera_mdp}


# ---------------------------------------------------------------------------
# 7 · concentración: lo que Vasicek supone y Pemex rompe
# ---------------------------------------------------------------------------
def pesos_con_grande(n_chicos=CONCENTRACION["n_chicos"], grande_mdp=CONCENTRACION["grande_mdp"], monto_mdp=CARTERA["monto_mdp"]):
    """Una cartera de n_chicos créditos de monto_mdp más uno de grande_mdp."""
    return [monto_mdp] * n_chicos + [grande_mdp]


def concentracion(PD=CARTERA["PD"], rho=CARTERA["rho"], LGD=CARTERA["LGD"], uno_en=CARTERA["uno_en"],
                  n_chicos=CONCENTRACION["n_chicos"], grande_mdp=CONCENTRACION["grande_mdp"], monto_mdp=CARTERA["monto_mdp"],
                  n_sim=100_000, semilla=7, verbose=True):
    """Dos carteras del mismo tamaño total: mil créditos iguales contra n_chicos
    iguales más uno grande. Nombres efectivos (1/HHI), capital de Vasicek (que
    supone todos iguales) y capital simulado con los pesos reales. La
    diferencia es el costo de la concentración."""
    w = pesos_con_grande(n_chicos, grande_mdp, monto_mdp)
    cartera = float(sum(w))
    q = 1 - 1 / uno_en
    r = capital_concentracion(PD, rho, w, LGD=LGD, q=q, n_sim=n_sim, semilla=semilla, verbose=False)
    out = {"cartera_mdp": cartera, "peso_grande": grande_mdp / cartera, "n_efectivo": r["n_efectivo"], "HHI": r["HHI"],
           "capital_vasicek_mdp": r["capital_vasicek"] * cartera, "capital_simulado_mdp": r["capital_simulado"] * cartera,
           "costo_mdp": r["costo_concentracion"] * cartera, "factor": r["capital_simulado"] / r["capital_vasicek"]}
    if verbose:
        print(f"CONCENTRACIÓN · {n_chicos:,} créditos de {monto_mdp:g} mdp + 1 de {grande_mdp:,.0f} mdp = {cartera:,.0f} mdp")
        print(f"   el grande pesa {out['peso_grande']:.0%} de la cartera · nombres efectivos = 1/HHI = 1/{r['HHI']:.4f} = {r['n_efectivo']:.0f} (nominales: {n_chicos+1:,})")
        print(f"   capital si fueran mil iguales (Vasicek): {out['capital_vasicek_mdp']:.1f} mdp")
        print(f"   capital con el grande dentro (simulado):  {out['capital_simulado_mdp']:.1f} mdp   ×{out['factor']:.2f}")
        print(f"   costo de la concentración = {out['costo_mdp']:+.1f} mdp: lo que un límite por nombre acota")
    return out


def limites_cub(capital_basico_mdp=100.0, verbose=True):
    """Los límites de la CUB (desde octubre de 2023) aplicados a un capital básico dado."""
    L = LIMITES_CUB
    filas = [("una persona o grupo (límite general)", L["limite_general_pct"]),
             ("entre bancos de importancia sistémica", L["limite_sistemicos_pct"]),
             ("paraestatales federales (Pemex, CFE)", L["limite_paraestatales_pct"])]
    df = pd.DataFrame([{"a quién": a, "% del capital básico": p, f"con capital básico de {capital_basico_mdp:,.0f} mdp": p / 100 * capital_basico_mdp} for a, p in filas]).set_index("a quién")
    if verbose:
        print("LÍMITES POR NOMBRE (CUB, Grandes Exposiciones, desde el 1 de octubre de 2023)")
        print(f"   Gran Exposición: {L['gran_exposicion']}")
        print(df.to_string())
        print(f"   exentos: {L['exentos']}")
        print(f"   Pemex en los bancos comerciales: ≈ {PEMEX_EXPOSICION['bancos_comerciales_pct_tce']} % del capital tangible común ({PEMEX_EXPOSICION['fuente']})")
    return df


# ---------------------------------------------------------------------------
# 8 · para qué sirve: el CRO en marzo de 2020
# ---------------------------------------------------------------------------
def escenario_cro(PD0=CARTERA["PD"], rho0=CARTERA["rho"], PD1=ESCENARIO_2020["PD"], rho1=ESCENARIO_2020["rho"],
                  n=CARTERA["n"], monto_mdp=CARTERA["monto_mdp"], LGD=CARTERA["LGD"], uno_en=CARTERA["uno_en"], verbose=True):
    """El escenario del CRO: la PD sube y el viento pesa más. Capital base, sólo
    con la PD, sólo con ρ, y con los dos."""
    kw = dict(n=n, monto_mdp=monto_mdp, LGD=LGD, uno_en=uno_en, verbose=False)
    base = cadena_capital(PD=PD0, rho=rho0, **kw)
    solo_pd = cadena_capital(PD=PD1, rho=rho0, **kw)
    solo_rho = cadena_capital(PD=PD0, rho=rho1, **kw)
    ambos = cadena_capital(PD=PD1, rho=rho1, **kw)
    out = {"base": base, "solo_PD": solo_pd, "solo_rho": solo_rho, "ambos": ambos}
    if verbose:
        print(f"EL ESCENARIO DEL CRO (cartera {base['cartera_mdp']:,.0f} mdp, LGD {LGD:.0%}, el peor año de {uno_en:,})")
        print(f"   base:            PD {PD0:.0%}, ρ {rho0:.0%}  →  provisión {base['EL_mdp']:.1f} mdp · capital {base['capital_mdp']:.1f} mdp")
        print(f"   sólo sube la PD: PD {PD1:.0%}, ρ {rho0:.0%}  →  provisión {solo_pd['EL_mdp']:.1f} mdp · capital {solo_pd['capital_mdp']:.1f} mdp  (×{solo_pd['capital_mdp']/base['capital_mdp']:.1f})")
        print(f"   sólo sube ρ:     PD {PD0:.0%}, ρ {rho1:.0%}  →  provisión {solo_rho['EL_mdp']:.1f} mdp · capital {solo_rho['capital_mdp']:.1f} mdp  (×{solo_rho['capital_mdp']/base['capital_mdp']:.1f})")
        print(f"   los dos:         PD {PD1:.0%}, ρ {rho1:.0%}  →  provisión {ambos['EL_mdp']:.1f} mdp · capital {ambos['capital_mdp']:.1f} mdp  (×{ambos['capital_mdp']/base['capital_mdp']:.1f})")
        print("   la provisión sólo ve la PD; el capital ve la PD y el viento. En una crisis suben los dos.")
    return out


if __name__ == "__main__":
    cada_una_por_su_cuenta(); print()
    pd_anio_malo(); print()
    print(tabla_anios().round(4)); print()
    dos_carteras(); print()
    cadena_capital(); print()
    duplicar_rho(); print()
    capital_irb(); print()
    print(tabla_rho_irb().round(4)); print()
    concentracion(); print()
    limites_cub(); print()
    escenario_cro()
