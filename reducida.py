"""
FORMA REDUCIDA · la intensidad con covariables (Duffie, Saita y Wang 2007)
Riesgos Financieros (FIN516) · Universidad Panamericana

Capa fina sobre empresas.py para el bloque L3, construido de abajo hacia arriba:
recordar λ → una empresa, cuatro lecturas → cómo se combinan (multiplican) →
de dónde salen los pesos → las dos empresas en cadena → para qué sirve → lo que
no ve → la fórmula. No recalcula nada que ya esté en empresas.py: toda λ sale de
empresas.intensidad_covariables con los coeficientes DSW_COEF.

Metáfora del bloque: un termómetro que toma CUATRO lecturas de una empresa y las
combina en un solo número, λ, la tasa a la que incumple. Cada lectura mala no
suma: MULTIPLICA.

UNIDADES (las de DSW_COEF): DD en desviaciones («sacudidas»); retorno de la
acción y del S&P 500 en fracción (0.10 = 10 %); tasa del Tesoro a 3 meses EN
POR CIENTO (1.5 = 1.5 %).
"""
import math
import numpy as np
import pandas as pd

from empresas import (DSW_COEF, CNBV_COVID_2020, intensidad_covariables,
                      revisa, Bitacora, tarea, preguntas_p1p3)

# ---------------------------------------------------------------------------
# las dos empresas de juguete (construcción del curso: ESTILIZADAS)
# ---------------------------------------------------------------------------
# Misma tasa a 3 meses (1.5 %) porque la tasa es del país, no de la empresa.
DOS_EMPRESAS_R = pd.DataFrame({
    "DD":    [4.0, 2.0],       # sacudidas de colchón (de Merton / KMV)
    "ret":   [0.10, -0.30],    # retorno de la acción en 12 meses (fracción)
    "r3m":   [1.5, 1.5],       # tasa del Tesoro a 3 meses, en por ciento
    "sp500": [0.10, -0.15],    # retorno del S&P 500 en 12 meses (fracción)
}, index=["tranquila", "en problemas"])

# Cómo se lee cada lectura, en palabras (para tablas y cuadernos).
LECTURAS = {
    "DD":    ("distancia al default", "cuántas sacudidas aguanta el colchón de activos sobre la deuda (de Merton / KMV)"),
    "ret":   ("retorno de la acción, 12 m", "lo que el mercado piensa de ESTA empresa en el último año"),
    "r3m":   ("tasa del Tesoro a 3 m, en %", "el costo del dinero en el país; es igual para todas las empresas"),
    "sp500": ("retorno del S&P 500, 12 m", "el ciclo: lo que le pasó a TODO el mercado en el último año"),
}

# Caso base del bloque y de los laboratorios: ciclo neutro (ret 0, S&P 0), tasa 1.5 %.
BASE = {"DD": 4.0, "ret": 0.0, "r3m": 1.5, "sp500": 0.0}

# El ciclo de marzo de 2020, estilizado (los mismos valores que usa mknb12_alumno):
CICLO_2020 = {"ret": -0.30, "r3m": 0.1, "sp500": -0.15}

# La muestra de Duffie, Saita y Wang (2007), Tabla 1 y §4.  [V]
MUESTRA_DSW = {
    "empresas": 2_770, "periodo": "1980–2004", "empresa_meses": 392_404, "incumplimientos": 497,
    "universo": "industriales de Estados Unidos (Compustat y CRSP), con la DD calculada como KMV",
    "fuente": "Duffie, Saita y Wang (2007), JFE 83(3), 635–665, Tabla 1 y Tabla 2 [V]",
}

# NOTA HONESTA sobre el signo del S&P 500 (+1.203): en la Tabla 2 del artículo el
# coeficiente es positivo. No significa «un mercado que sube hace quebrar»: la DD
# y el retorno de la acción ya absorben la parte buena del ciclo, y el S&P recoge
# lo que queda (en un año bueno, la DD de una empresa que NO subió con el mercado
# es una peor señal). Se dice en clase tal cual, sin esconderlo.
SIGNO_SP500 = ("En el artículo el coeficiente del S&P 500 es positivo (+1.203). No es que un mercado que sube "
               "haga quebrar: la DD y el retorno de la propia acción ya absorben casi todo el ciclo, y el "
               "S&P recoge lo que queda. Con los cuatro juntos, un año malo del mercado sube la λ por la DD "
               "y la acción, y el término del S&P lo modera un poco.")


# ---------------------------------------------------------------------------
# λ en una línea: 100 empresas con λ de 2 % → unas 2 incumplen en el año
# ---------------------------------------------------------------------------
def lambda_en_una_linea(lam=0.02, n=100):
    """Recordatorio del XI: λ es una tasa por año. PD a un año = 1 − e^(−λ) ≈ λ."""
    pd1 = 1 - math.exp(-lam)
    print("RECORDAR λ")
    print(f"   {n} empresas, cada una con λ = {lam:.0%} por año → en un año incumplen ≈ {n*lam:.0f} de ellas")
    print(f"   PD a un año = 1 − e^(−λ) = 1 − e^(−{lam:.2f}) = {pd1:.4f} ≈ {lam:.2f}  (para λ chica, PD ≈ λ)")
    return pd1


# ---------------------------------------------------------------------------
# los multiplicadores: cada lectura mala MULTIPLICA la λ
# ---------------------------------------------------------------------------
def multiplicador(lectura, de, a, coef=None):
    """Por cuánto se multiplica λ cuando la lectura pasa de `de` a `a`.

    λ = exp(β0 + β·x): mover x de `de` a `a` multiplica λ por e^(β·(a − de)).
    Todo lo demás se queda igual. Unidades: las de DSW_COEF.
    """
    c = DSW_COEF if coef is None else coef
    return math.exp(c[lectura] * (a - de))


def multiplicadores(fila_a="en problemas", fila_de="tranquila", coef=None, verbose=True):
    """Los cuatro multiplicadores que llevan la λ de una empresa a la de otra.

    Devuelve un DataFrame con la lectura `de`, la lectura `a`, el multiplicador
    e^(β·(a − de)) y el producto acumulado. El producto de los cuatro ES el
    cociente λ(a)/λ(de): es la propiedad del exponencial.
    """
    c = DSW_COEF if coef is None else coef
    de = DOS_EMPRESAS_R.loc[fila_de]
    a = DOS_EMPRESAS_R.loc[fila_a]
    filas, acum = [], 1.0
    for k in ["DD", "ret", "r3m", "sp500"]:
        m = multiplicador(k, de[k], a[k], c)
        acum *= m
        filas.append({"lectura": LECTURAS[k][0], "de": de[k], "a": a[k], "beta": c[k], "multiplica": m, "acumulado": acum})
    df = pd.DataFrame(filas).set_index("lectura")
    if verbose:
        l_de = intensidad_covariables(verbose=False, **de.to_dict())["lambda"]
        l_a = intensidad_covariables(verbose=False, **a.to_dict())["lambda"]
        print(f"DE «{fila_de}» A «{fila_a}»: cada lectura multiplica la λ")
        print(f"   λ de partida = {l_de*1e4:,.0f} pb")
        for k, r in df.iterrows():
            print(f"   {k:<28s} {r['de']:>6.2f} → {r['a']:>6.2f}   × e^({r['beta']:+.3f}·({r['a']:.2f} − {r['de']:.2f})) = ×{r['multiplica']:.2f}   acumulado ×{r['acumulado']:.2f}")
        print(f"   λ de llegada = {l_de*1e4:,.0f} pb × {acum:.2f} = {l_a*1e4:,.0f} pb   (comprobación directa: {l_a*1e4:,.0f} pb)")
    return df


# ---------------------------------------------------------------------------
# la cadena: fórmula → sustitución → resultado, término por término
# ---------------------------------------------------------------------------
def lambda_en_cadena(DD, ret=0.0, r3m=1.5, sp500=0.0, nombre="", coef=None, verbose=True):
    """λ de UNA empresa, en cadena: en palabras → en símbolos → con números.

    Imprime el exponente término por término (lectura × peso = aporte), la λ
    en puntos base y la PD a un año. Devuelve lo mismo que
    empresas.intensidad_covariables más los aportes.
    """
    c = DSW_COEF if coef is None else coef
    x = {"DD": DD, "ret": ret, "r3m": r3m, "sp500": sp500}
    aportes = {k: c[k] * x[k] for k in x}
    res = intensidad_covariables(DD, ret=ret, r3m=r3m, sp500=sp500, coef=c, verbose=False)
    if verbose:
        et = f" · {nombre}" if nombre else ""
        print(f"λ EN CADENA{et}")
        print("   en palabras:  λ = e elevado a (constante + peso·DD + peso·acción + peso·tasa + peso·mercado)")
        print("   en símbolos:  λ = exp(β0 + β_DD·DD + β_ret·ret + β_r·r3m + β_sp·SP500)")
        print(f"   con números:  exponente = {c['const']:+.3f}", end="")
        for k in ["DD", "ret", "r3m", "sp500"]:
            print(f" {c[k]:+.3f}·{x[k]:.2f}", end="")
        print(f" = {res['exponente']:+.3f}")
        print("   término por término:")
        print(f"      constante                       {c['const']:+.3f}")
        for k in ["DD", "ret", "r3m", "sp500"]:
            print(f"      {LECTURAS[k][0]:<28s}  {c[k]:+.3f} × {x[k]:>6.2f} = {aportes[k]:+.3f}")
        print(f"   λ = e^({res['exponente']:+.3f}) = {res['lambda']:.4f} por año = {res['lambda']*1e4:,.0f} pb")
        print(f"   PD a un año = 1 − e^(−λ) = 1 − e^(−{res['lambda']:.4f}) = {res['PD_1a']:.2%}")
        print(f"   pesos: {c.get('fuente', 'sin fuente')}")
    return {**res, "aportes": aportes}


def dos_empresas_en_cadena(coef=None, verbose=True):
    """Las dos empresas de juguete, una después de la otra, y el cociente de sus λ."""
    out = {}
    for nombre, fila in DOS_EMPRESAS_R.iterrows():
        out[nombre] = lambda_en_cadena(nombre=nombre, coef=coef, verbose=verbose, **fila.to_dict())
        if verbose:
            print()
    cociente = out["en problemas"]["lambda"] / out["tranquila"]["lambda"]
    if verbose:
        print(f"λ(en problemas) / λ(tranquila) = {out['en problemas']['lambda']*1e4:,.0f} / {out['tranquila']['lambda']*1e4:,.0f} = ×{cociente:.1f}")
    out["cociente"] = cociente
    return out


def tabla_dos_empresas(coef=None):
    """Tabla resumen: las cuatro lecturas, el exponente, λ en pb y PD a un año."""
    c = DSW_COEF if coef is None else coef
    filas = []
    for nombre, f in DOS_EMPRESAS_R.iterrows():
        r = intensidad_covariables(coef=c, verbose=False, **f.to_dict())
        filas.append({"empresa": nombre, "DD": f["DD"], "acción 12 m": f["ret"], "tasa 3 m (%)": f["r3m"],
                      "S&P 500 12 m": f["sp500"], "exponente": r["exponente"], "λ (pb)": r["lambda"] * 1e4,
                      "PD a un año": r["PD_1a"]})
    return pd.DataFrame(filas).set_index("empresa")


# ---------------------------------------------------------------------------
# para qué sirve: la provisión esperada de una cartera estilizada
# ---------------------------------------------------------------------------
def provision_esperada(DD, ret=0.0, r3m=1.5, sp500=0.0, cartera_mmdp=100.0, LGD=0.45, coef=None, verbose=True):
    """Provisión esperada a un año de una cartera ESTILIZADA: PD × LGD × cartera.

    La cartera se trata como si todos los acreditados tuvieran la misma DD (la
    promedio) y las mismas lecturas de ciclo; por eso es estilizada. Devuelve
    mmdp (miles de millones de pesos).
    """
    r = intensidad_covariables(DD, ret=ret, r3m=r3m, sp500=sp500, coef=coef, verbose=False)
    prov = r["PD_1a"] * LGD * cartera_mmdp
    if verbose:
        print(f"provisión esperada = PD × LGD × cartera = {r['PD_1a']:.4f} × {LGD:.2f} × {cartera_mmdp:,.0f} mmdp = {prov:.3f} mmdp"
              f"   (λ = {r['lambda']*1e4:,.0f} pb)")
    return {**r, "provision_mmdp": prov}


def cuanto_se_multiplica(DD_antes=4.0, DD_despues=2.0, ciclo=None, cartera_mmdp=100.0, LGD=0.45, coef=None, verbose=True):
    """El reto del CRO: la DD promedio cae de DD_antes a DD_despues. ¿Por cuánto se
    multiplican λ y la provisión? Con ciclo=None el ciclo se queda neutro (sólo
    cambia la DD); con ciclo=CICLO_2020 cae todo a la vez."""
    cic = dict(ret=0.0, r3m=1.5, sp500=0.0)
    antes = provision_esperada(DD_antes, cartera_mmdp=cartera_mmdp, LGD=LGD, coef=coef, verbose=False, **cic)
    if ciclo:
        cic = dict(ciclo)
    despues = provision_esperada(DD_despues, cartera_mmdp=cartera_mmdp, LGD=LGD, coef=coef, verbose=False, **cic)
    f_lam = despues["lambda"] / antes["lambda"]
    f_prov = despues["provision_mmdp"] / antes["provision_mmdp"]
    if verbose:
        et = "con el ciclo encima" if ciclo else "ciclo neutro: sólo cambia la DD"
        print(f"DD promedio {DD_antes:.0f} → {DD_despues:.0f} ({et})")
        print(f"   λ: {antes['lambda']*1e4:,.0f} pb → {despues['lambda']*1e4:,.0f} pb   ×{f_lam:.1f}")
        print(f"   provisión esperada ({cartera_mmdp:,.0f} mmdp, LGD {LGD:.0%}): {antes['provision_mmdp']:.3f} → {despues['provision_mmdp']:.3f} mmdp   ×{f_prov:.1f}")
        if abs(f_lam - f_prov) > 0.05:
            print("   (λ y PD no se multiplican por lo mismo: PD = 1 − e^(−λ) crece un poco menos que λ)")
    return {"lambda_antes": antes["lambda"], "lambda_despues": despues["lambda"], "factor_lambda": f_lam,
            "prov_antes": antes["provision_mmdp"], "prov_despues": despues["provision_mmdp"], "factor_prov": f_prov}


# ---------------------------------------------------------------------------
# la recta: λ contra DD en escala log es una recta; el ciclo la corre entera
# ---------------------------------------------------------------------------
def curva_lambda_dd(ret=0.0, r3m=1.5, sp500=0.0, dd_min=0.5, dd_max=7.0, n=200, coef=None):
    """λ (pb) como función de la DD con las otras tres lecturas fijas."""
    dds = np.linspace(dd_min, dd_max, n)
    lams = np.array([intensidad_covariables(d, ret=ret, r3m=r3m, sp500=sp500, coef=coef, verbose=False)["lambda"] * 1e4 for d in dds])
    return dds, lams


if __name__ == "__main__":
    lambda_en_una_linea()
    print()
    print(tabla_dos_empresas().round(4))
    print()
    multiplicadores()
    print()
    dos_empresas_en_cadena()
    print()
    cuanto_se_multiplica()
    cuanto_se_multiplica(ciclo=CICLO_2020)
