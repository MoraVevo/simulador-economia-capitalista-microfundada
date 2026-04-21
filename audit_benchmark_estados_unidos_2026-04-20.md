# Auditoria Benchmark y Estados Unidos

Fecha de revision: 2026-04-20

## Hallazgo previo importante

Los dos archivos no parecen provenir de la misma generacion de export:

- `estado-social-intensivo-benchmark.xlsx` sigue con la hoja `auditoria_familias` en corte final solamente.
- `estados-unidos-mixto.xlsx` ya trae `auditoria_familias` por periodo y columnas nuevas en firmas como `Nomina total periodo`, `Ganancias totales periodo`, `Trabajadores renunciaron` y `Trabajadores despedidos`.

Conclusion:

- Compararlos como si fueran exactamente la misma corrida/export puede inducir errores.
- Aun asi, los patrones economicos que salen son suficientemente fuertes como para sacar varias conclusiones utiles.

## Resumen ejecutivo

### Benchmark

- La macro luce muy fuerte al final, pero la auditoria de firmas muestra señales claras de sobreexpansion y de metas laborales poco disciplinadas.
- Hay muchas observaciones donde la firma amplía plantilla deseada aun vendiendo por debajo de lo esperado.
- En varios casos la plantilla deseada explota a niveles poco creibles justo cuando la firma ya viene perdiendo dinero o desorganizada.
- El archivo parece capturar justamente el comportamiento que te estaba molestando.

### Estados Unidos

- La macro luce bastante mas debil: desempleo alto y muchas familias cerca o por debajo de la canasta.
- El comportamiento de firmas es menos explosivo que en Benchmark, pero todavia hay focos de sobreexpansion persistente en ocio, manufactura y algunos casos de vivienda.
- Aqui el problema no parece ser tanto una explosion generalizada de contratacion, sino bolsillos concretos de firmas que no ajustan bien expectativas y plantilla.

## Benchmark: hallazgos

### 1. Sobreexpansion recurrente

En la muestra auditada:

- `71.9%` de las observaciones de firma tienen perdida operativa aproximada.
- `27.9%` de las observaciones expanden trabajadores deseados aun cuando `Ventas reales < Ventas esperadas`.
- `24.8%` expanden incluso con una desviacion mas fuerte: `Ventas reales < 90% de Ventas esperadas`.
- `18.4%` expanden mientras la firma ya esta en perdida.

Esto es demasiado alto para ser solo ruido de adaptacion.

### 2. Casos concretos muy anormales

#### Firma 8, Alimentos basicos

Ultimos periodos observados:

- Periodo 235: `159` trabajadores iniciales, `683` ventas reales, `25` trabajadores deseados siguiente periodo.
- Periodo 236: con solo `25` trabajadores iniciales salta a `219` deseados.
- Periodo 239: con ventas reales `0`, inventario `845`, pasa a `3519` trabajadores deseados.
- Periodo 240: con `0` trabajadores iniciales y ventas reales `845`, pasa a `6784` trabajadores deseados.

Esto parece directamente una ruptura de la regla de headcount deseado, no una respuesta economica razonable.

#### Firma 6, Alimentos basicos

- Promedio de ganancia aproximada: `-400`.
- Gap medio de trabajadores deseados: `+70.7`.
- `151` periodos expandiendo con ventas por debajo de lo esperado.

No es una reaccion puntual. Es una firma que insiste mucho en crecer aun con señales flojas de ventas.

#### Varias firmas manufactureras

Las firmas `62`, `79`, `136`, `189`, `193` muestran el mismo patron base:

- metas laborales persistentemente altas
- rentabilidad media muy negativa
- muchas salidas laborales
- reempleo desigual de quienes salen

### 3. Mezcla rara entre macro fuerte y micro muy deteriorado

El Benchmark termina con:

- salario promedio anual alto
- razon ingreso/canasta familiar alta
- desempleo moderado

Pero la muestra de firmas enseña muchas perdidas y mucho desorden de plantilla. Eso sugiere una de estas dos cosas:

1. la muestra auditada agarró firmas especialmente disfuncionales
2. la macro se sostiene por unos pocos ganadores o por mecanismos agregados, mientras una parte importante del tejido empresarial esta muy mal coordinado

## Estados Unidos: hallazgos

### 1. El problema principal aqui no es explosividad generalizada, sino debilidad social y focos de mala adaptacion

Ultimos 12 meses promedio:

- desempleo: `29.1%`
- salario promedio: `28.2`
- razon ingreso/canasta al final: `1.04`

La economia queda muy cerca del umbral de reproduccion de hogares.

### 2. Auditoria de hogares

La hoja por periodo muestra una realidad bastante apretada:

- `3279` de `6562` observaciones familiares estan por debajo de canasta.
- `1217` observaciones familiares tienen empleo cero.
- En varios tramos tempranos la muestra llega a `90%` o mas de familias bajo canasta.
- Mejora bastante al final, pero incluso en el periodo `240` todavia `29.2%` de la muestra esta bajo canasta.

Casos de familias estresadas:

- Familia `2153`: `240` periodos observados, empleo medio `0.38`, razon ingreso/canasta media `0.35`.
- Familia `4907`: `240` periodos, empleo medio `0.52`, razon media `0.48`.
- Familia `1919`: `240` periodos, empleo medio `0.42`, razon media `0.52`.

Esto parece mas pobreza laboral persistente que simple desempleo transitorio.

### 3. Firmas problematicas, pero mas localizadas

#### Firma 89, Ocio

- `70` periodos expandiendo con ventas por debajo de lo esperado.
- `52` periodos expandiendo aun con perdida.
- `916` salidas laborales acumuladas.
- reempleo casi nulo de quienes salen.

Se ve como una firma que gira mucha mano de obra y aprende poco.

#### Firma 65, Manufactura

- `64` periodos expandiendo con ventas flojas.
- `54` periodos expandiendo con perdida.
- `1341` salidas acumuladas.
- reempleo de salida casi inexistente.

En los ultimos periodos se queda clavada en `60` trabajadores deseados pese a ingresos insuficientes.

#### Firma 170, Vivienda

Caso muy claro de salto desordenado:

- Periodo 237: con `5` trabajadores iniciales, ventas reales `0`, inventario `85.74`, pasa a `433` trabajadores deseados.
- Periodo 238: con `0` trabajadores iniciales sigue en `416` deseados.
- Periodo 239: con `4` trabajadores iniciales todavia pide `142`.
- Periodo 240: arranca con `142` y luego recorta a `97`, con perdida fuerte.

Este es un ejemplo muy limpio del problema que describiste.

## Comparacion util

### Benchmark

- parece peor en disciplina interna de firmas
- tiene signos de headcount objetivo desanclado
- el problema es mas sistémico en la muestra auditada

### Estados Unidos

- parece peor en bienestar laboral y hogares
- el problema de firmas existe, pero esta mas concentrado en ciertos sectores y firmas
- manufactura, ocio y algunos casos de vivienda son los focos mas claros

## Que creo que esta pasando

1. Algunas firmas siguen usando una regla de expansion que reacciona demasiado al faltante revelado o a shocks de ventas, y eso termina inflando `Trabajadores deseados proximo periodo`.
2. El ancla de capacidad real no siempre esta dominando la decision.
3. Cuando una firma entra en descoordinacion, puede alternar entre despidos bruscos y nuevas metas de contratacion igual de bruscas.
4. En Estados Unidos la demanda agregada y el empleo del hogar estan mas fragiles, asi que una firma mal calibrada pega mucho mas en bienestar familiar.

## Siguiente paso recomendado

1. Regenerar `estado-social-intensivo-benchmark.xlsx` con el export nuevo, porque hoy no esta en formato comparable con `Estados Unidos`.
2. Para la siguiente auditoria automatica, marcar alertas por firma cuando ocurra cualquiera de estas reglas:
   - `Ventas reales < Ventas esperadas` y `Trabajadores deseados proximo periodo > Trabajadores inicio`
   - `Ganancias totales periodo < 0` y aun asi `Trabajadores deseados proximo periodo > Trabajadores inicio`
   - `Inventario > 25% de ventas reales` y aun asi expansion laboral
   - `Trabajadores despedidos / Trabajadores inicio > 20%`
3. Agregar una hoja de alertas resumidas por firma y periodo para no tener que revisarlo a mano cada vez.
