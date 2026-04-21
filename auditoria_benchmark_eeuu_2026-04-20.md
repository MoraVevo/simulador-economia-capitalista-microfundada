# Auditoria Benchmark y Estados Unidos

Archivos auditados:

- `estado-social-intensivo-benchmark.xlsx`
- `estados-unidos-mixto.xlsx`

Fecha de revision: `2026-04-20`

## Alcance y limite

Estos dos `xlsx` de la raiz fueron generados con la version anterior del export:

- `auditoria_firmas` todavia no trae `Trabajadores renunciaron`, `Trabajadores despedidos`, `Ganancias totales periodo` ni `Nomina total periodo`.
- `auditoria_familias` solo trae un corte final de `45` familias en el periodo `240`, no una serie por periodo.

La auditoria de firmas si es bastante informativa porque hay entre `4,461` y `4,651` filas por pais, equivalentes a `20` firmas observadas a lo largo del horizonte.

## Hallazgos principales

### 1. Benchmark muestra un sesgo fuerte a expansion agresiva de empleo en firmas aun cuando su demanda no lo justifica

Senales principales:

- Participacion de periodos con `Ventas reales > Ventas esperadas * 1.05`: `18.7%`.
- Participacion de periodos con `Ventas reales < Ventas esperadas * 0.95`: `28.7%`.
- Aun asi, el sector `Alimentos basicos` tiene `avg_desired_delta = 87.3` trabajadores y `avg_sales_ratio = 0.619`.
- `Manufactura` tiene `avg_desired_delta = 39.5` con inventarios medios de `39.4` unidades y salida media de `4.4` trabajadores por periodo.

Casos especialmente raros:

- Firma `8` de `Alimentos basicos`:
  `avg_sales_ratio = 0.725`, `underperform_share = 60.0%`, `avg_desired_delta = 169.1`, `max_desired_delta = 7090`, margen medio `-1050.9`.
- Firma `205` de `Manufactura`:
  `avg_sales_ratio = 0.885`, `avg_desired_delta = 59.1`, inventario medio `74.6`, margen medio `-53,238.4`.
- Firma `84` de `Ocio`:
  `avg_sales_ratio = 0.296`, `avg_desired_delta = 50.0`, margen medio `-924.4`.

Esto es consistente con la sospecha de que las firmas estaban reaccionando demasiado rapido al crecimiento o a su expectativa de ventas, sin validar suficiente restriccion real de capacidad ni persistencia del faltante.

### 2. En Benchmark hay periodos donde la firma intenta contratar mucho mas aun con ventas muy por debajo de lo esperado y mucho inventario acumulado

Ejemplos de instancias concretas:

- Periodo `208`, firma `205`, `Manufactura`:
  inicio `242`, desea `601`, delta `+359`, ratio ventas/esperadas `0.624`, inventario `10007.7`, margen `-238595.4`.
- Periodo `207`, firma `193`, `Manufactura`:
  inicio `440`, desea `544`, delta `+104`, ratio `0.130`, inventario `10453.9`, margen `-406564.1`.
- Periodo `148`, firma `8`, `Alimentos`:
  inicio `107`, desea `195`, delta `+88`, ratio `0.558`, inventario `3152.3`, margen `-32341.9`.

Interpretacion:

- Aqui no parece un problema de falta de mano de obra.
- Parece mas bien una regla de ajuste de empleo que sigue empujando expansion incluso cuando la firma ya esta sobreinventariada y perdiendo dinero.

### 3. Benchmark macroeconomicamente luce fuerte en cobertura, pero con una trayectoria nominal muy inflada y fiscalmente extrema

Senales macro:

- Desempleo promedio: `11.9%`.
- Desempleo final: `9.5%`.
- Cobertura final de esenciales: `99.9%`.
- Matricula escolar final: `99.7%`.
- Deficit fiscal promedio sobre PIB: `30.1%`.
- PIB nominal multiplica aproximadamente `45.4x` entre el inicio y el final.
- Salario promedio pasa de `9.8` a `855.7`.

Lectura:

- El benchmark cumple bastante bien por el lado social y de cobertura.
- Pero nominalmente se expande de forma muy agresiva.
- Da la impresion de una economia que sostiene muy bien el bienestar, pero a costa de una trayectoria fiscal y monetaria demasiado grande para considerarla estable sin mas validacion.

### 4. Estados Unidos luce mucho mas fragil: alto desempleo, cobertura incompleta y recesion mas persistente

Senales macro:

- Desempleo promedio: `24.6%`.
- Desempleo final: `16.4%`.
- Intensidad media de recesion: `0.723`.
- Cobertura final de esenciales: `88.3%`.
- Crecimiento nominal total mucho menor que Benchmark: aproximadamente `2.9x`.

Lectura:

- Aqui el sistema no colapsa, pero queda atrapado con mucho slack.
- La economia parece mas disciplinada nominalmente que Benchmark, pero mucho peor en resultados laborales y de cobertura.

### 5. En Estados Unidos tambien aparecen firmas con contratacion demasiado agresiva, pero el problema esta mas concentrado que en Benchmark

Firmas mas raras:

- Firma `43`, `Ropa e higiene`:
  `avg_sales_ratio = 0.618`, `underperform_share = 57.9%`, `avg_desired_delta = 15.0`, margen medio `-238.7`.
- Firma `49`, `Ropa e higiene`:
  `avg_sales_ratio = 0.767`, `avg_desired_delta = 13.7`, margen medio `-68.1`.
- Firma `58`, `Ropa e higiene`:
  `avg_sales_ratio = 0.658`, `avg_desired_delta = 12.7`, margen medio `-984.4`.

Instancias concretas:

- Periodo `98`, firma `43`, `Ropa`:
  inicio `0`, desea `57`, ratio ventas/esperadas `0.114`, inventario `22.5`, margen `-901.0`.
- Periodo `18`, firma `49`, `Ropa`:
  inicio `0`, desea `53`, ratio `0.178`, inventario `62.4`, margen `-632.4`.
- Periodo `64`, firma `58`, `Ropa`:
  inicio `12`, desea `60`, ratio `0.122`, inventario `68.4`, margen `-6723.9`.

Lectura:

- El sesgo de sobrecontratacion esta presente, pero en Estados Unidos se ve mas focalizado en `Ropa` y parte de `Manufactura`.
- Benchmark lo muestra de manera mas amplia y mas extrema.

### 6. La muestra de familias sugiere que Benchmark protege mucho mejor a los hogares que Estados Unidos

Benchmark, muestra final de `45` familias:

- Hogares bajo canasta: `6.7%`.
- Hogares con empleo familiar completo: `77.8%`.
- Hogares sin empleo: `8.9%`.
- Ratio promedio ingreso/canasta: `5.15`.
- Costo escolar privado promedio: `0.0`.

Estados Unidos, muestra final de `45` familias:

- Hogares bajo canasta: `20.0%`.
- Hogares con empleo familiar completo: `80.0%`.
- Hogares sin empleo: `0.0%`.
- Ratio promedio ingreso/canasta: `1.49`.
- Costo escolar privado promedio: `0.0`.

Interpretacion:

- Aunque muchas familias en Estados Unidos tienen tasa de empleo alta, sus ingresos quedan mucho mas pegados a la canasta.
- Benchmark produce mucha mas holgura de ingresos en la muestra final.

### 7. Hay una rareza educativa fuerte en Estados Unidos y otra distinta en Benchmark

Benchmark:

- `school_enrollment_share` final casi universal.
- `university_enrollment_share` final `0.0`.

Estados Unidos:

- `school_enrollment_share` final `0.537`.
- `university_enrollment_share` final `1.0`.

Esto no se ve economica o demograficamente natural. Puede indicar:

- un problema de definicion del denominador de matricula universitaria;
- una reasignacion rara entre escuela y universidad;
- o una metrica que ya no esta capturando exactamente la poblacion objetivo correcta.

## Conclusiones operativas

### Benchmark

- Socialmente funciona mejor que Estados Unidos en cobertura e ingreso familiar.
- Pero varias firmas expanden empleo con reglas demasiado agresivas, incluso con subventas, inventarios altos y perdidas severas.
- El nivel macro parece apoyarse en una expansion fiscal y nominal demasiado fuerte.

### Estados Unidos

- Tiene peor desempeno social y laboral.
- El sesgo de sobrecontratacion existe, pero esta mas localizado.
- La economia queda mas reprimida y con mayor recesion estructural.

## Lo que conviene revisar despues

1. Regenerar estos dos paises con el export nuevo para tener:
   `renuncias`, `despidos`, `ganancias`, `nomina` y `familias por periodo`.
2. Revisar por que `desired_workers` puede saltar tan alto desde bases pequenas o incluso desde `0`, especialmente en `Ropa`, `Alimentos` y `Manufactura`.
3. Auditar la metrica de matricula universitaria en Estados Unidos y la de universidad `0` en Benchmark.
4. Confirmar si el enorme crecimiento nominal de Benchmark es una propiedad deseada del modelo o una deriva excesiva por politica fiscal.
