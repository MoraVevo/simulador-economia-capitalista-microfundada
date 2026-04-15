# Simulador de Economia Capitalista Microfundada

Este proyecto es un simulador economico basado en agentes donde la macroeconomia emerge desde la interaccion micro de miles de agentes heterogeneos. No resuelve la economia imponiendo un equilibrio walrasiano sobre agregados representativos. Al contrario: hogares, firmas, empresarios, bancos, banco central y gobierno actuan con informacion local, restricciones presupuestarias, inventarios, credito, impuestos, desempleo, educacion, demografia y quiebras. La macro aparece despues, como resultado agregado de esas decisiones descentralizadas.

La idea central del proyecto es simple pero exigente: si queremos entender desempleo, inflacion, pobreza, desigualdad, movilidad social, restriccion crediticia, fragilidad bancaria o efectos de politica publica, no basta con mover unas pocas curvas agregadas. Necesitamos modelar a los agentes que realmente producen esos resultados.

## Tesis del modelo

La economia de este simulador funciona de abajo hacia arriba:

- Los hogares trabajan, consumen, ahorran, se endeudan, forman familias, estudian, envejecen y enfrentan privaciones materiales.
- Las firmas fijan precios, salarios ofrecidos, metas de inventario, expectativas de ventas, inversion, tecnologia y demanda de trabajo.
- Los empresarios concentran riqueza, crean nuevas firmas, absorben perdidas, reciben dividendos y deciden entrada o permanencia.
- Los bancos comerciales intermedian depositos, otorgan credito, absorben mora, enfrentan insolvencia y ajustan su posicion de capital.
- El banco central regula liquidez, tasa de politica, reservas y emision segun reglas monetarias parametrizables.
- El gobierno recauda impuestos, paga transferencias, compra bienes esenciales, financia educacion, administra empleo publico e invierte en infraestructura.

No hay un subastador walrasiano limpiando mercados en un punto de equilibrio instantaneo. Puede haber:

- desempleo persistente
- demanda insatisfecha
- exceso de inventarios
- firmas con perdidas
- quiebras y reemplazos
- racionamiento por ingreso o por oferta
- desigualdad patrimonial creciente
- ciclos con intervencion fiscal y monetaria
- restricciones crediticias que amplifican desaceleraciones

Ese es precisamente el punto del modelo.

## Que hace diferente a este simulador

Muchos modelos agregados parten de una economia resumida en un hogar representativo, una firma representativa o un conjunto pequeño de ecuaciones de equilibrio. Este proyecto toma otro camino:

- representa explicitamente hogares individuales
- representa firmas individuales por sector
- representa empresarios como propietarios separados de los hogares trabajadores
- separa banca comercial, banco central y gobierno
- integra educacion, demografia y estructura familiar dentro del motor economico
- permite que la macroeconomia sea una propiedad emergente y no una condicion impuesta

En otras palabras: aqui la macro no manda a la micro; la micro construye la macro.

## Enfoque economico

El motor esta mas cerca de una economia descentralizada con ajuste adaptativo que de un sistema de equilibrio general estatico. Algunas caracteristicas metodologicas clave:

- Precios adaptativos en lugar de precios de equilibrio instantaneo.
- Salarios ofrecidos que reaccionan a vacantes, rechazo laboral y condiciones del mercado.
- Produccion limitada por trabajo, capital, productividad, inventarios y financiamiento.
- Consumo de hogares sujeto a ingreso, ahorro, credito, necesidades esenciales y preferencias.
- Inversion dependiente de liquidez, rentabilidad, estabilidad macro y costo del credito.
- Fallas reales posibles: hambre, privacion habitacional, fragilidad de salud, mora, defaults, insolvencia y bancarrota.
- Politica publica no neutral: puede modificar trayectorias de empleo, cobertura esencial, educacion, movilidad social y estabilidad financiera.

## Agentes y logica micro

### 1. Hogares

Cada hogar tiene estado propio y no es un simple punto en una distribucion abstracta. Entre sus atributos estan:

- ahorro
- salario de reserva
- propension a ahorrar
- sensibilidad al precio
- afinidad educativa
- impaciencia de consumo
- escala de necesidades
- edad
- situacion laboral
- historial de ingresos
- condicion familiar
- endeudamiento y mora
- historial educativo
- origen social del hogar

Los hogares:

- ofrecen trabajo y aceptan o rechazan ofertas
- consumen primero bienes esenciales y luego gasto discrecional
- pueden quedar racionados por ingreso o por escasez de oferta
- acumulan ahorros o deuda
- forman parejas, tienen hijos y transfieren condiciones a la siguiente generacion
- invierten tiempo y recursos en escolaridad y universidad
- sufren deterioro cuando no logran cubrir necesidades basicas

Eso permite estudiar no solo promedio de consumo, sino tambien cobertura efectiva de canasta esencial, hambre aguda, fragilidad de salud y movilidad intergeneracional.

### 2. Empresarios

Los empresarios estan separados de los hogares trabajadores y concentran la propiedad de firmas. Tienen riqueza, apetito de entrada, optimismo y relacion con el credito. Esto permite distinguir claramente entre:

- ingreso laboral
- beneficios empresariales
- dividendos
- acumulacion patrimonial

La desigualdad no aparece como una estadistica externa, sino como consecuencia de la estructura de propiedad y de la dinamica del beneficio.

### 3. Firmas

Cada firma es una unidad economica individual con:

- sector
- propietario
- caja
- inventario
- capital
- precio
- salario ofrecido
- productividad
- tecnologia
- costos por insumos, transporte y estructura
- tolerancia de markup
- cautela de pronostico
- ambicion de cuota de mercado
- apetito de credito e inversion
- historial de ventas, produccion, costos y ganancias

Las firmas:

- forman expectativas de ventas
- ajustan produccion y empleo
- pagan salarios
- venden desde inventario
- actualizan precios y salarios segun condiciones del periodo
- invierten en tecnologia
- toman credito si las condiciones lo permiten
- acumulan perdidas o se expanden
- quiebran y pueden ser reemplazadas por nueva entrada empresarial

Esto genera una economia genuinamente heterogenea donde coexistente firmas exitosas, firmas fragiles y firmas fallidas.

### 4. Banca comercial

Los bancos comerciales tienen:

- depositos
- reservas
- prestamos a hogares
- prestamos a firmas
- tenencia de bonos
- tasas activas y pasivas
- beneficios bancarios
- restricciones de capital y reservas

El sistema bancario no es decorativo. Puede:

- expandir o restringir credito
- absorber moras y defaults
- deteriorar su capital
- depender de liquidez del banco central
- ser recapitalizado o resuelto si entra en insolvencia

Eso vuelve endogena la fragilidad financiera.

### 5. Banco central

El banco central maneja:

- oferta monetaria
- objetivo monetario
- tasa de politica
- emision del periodo
- emision acumulada

El motor incluye reglas monetarias parametrizables, entre ellas:

- emision por crecimiento de bienes
- regla tipo Fisher
- dividendo de productividad

Ademas puede ajustar encajes y operar sobre la brecha monetaria. El dinero no es solo un numerario neutro; condiciona credito, liquidez y velocidad de ajuste del sistema.

### 6. Gobierno

El gobierno recauda y gasta de forma explicita. Entre sus componentes estan:

- impuestos laborales, corporativos, a dividendos y a la riqueza
- transferencias
- seguro de desempleo
- apoyo infantil
- apoyo basico a hogares con brecha de canasta
- compras publicas de bienes esenciales
- gasto educativo
- empleo de administracion publica
- inversion en infraestructura
- deuda publica y emision de bonos

Esto permite experimentar con estados mas liberales, mixtos o de bienestar y ver como cambian los resultados emergentes.

## Sectores productivos

El simulador arranca con sectores diferenciados, cada uno con parametros de precio base, salario base, productividad, necesidad esencial, peso discrecional, inventarios objetivo y markup. Los sectores incluidos son:

- alimentacion basica
- vivienda y servicios esenciales
- vestuario e higiene
- manufactura
- ocio y entretenimiento
- escolaridad
- universidad
- administracion publica

La separacion sectorial es importante porque no toda demanda es igual:

- algunos bienes son esenciales
- otros son discrecionales
- algunos requieren trabajo mas calificado
- educacion y administracion publica tienen una logica institucional distinta

## Ciclo de simulacion por periodo

Cada periodo sigue una secuencia economica concreta. A alto nivel, el motor:

1. actualiza caches, edades, contratos y estado crediticio
2. recalcula politicas de firmas a partir del desempleo previo
3. alinea y empareja mercado laboral
4. aplica politica monetaria y fondeo bancario
5. decide credito y condiciones financieras
6. organiza empleo publico
7. produce, paga salarios y actualiza costos
8. aplica apoyo estatal a hogares
9. cobra servicio de deuda
10. ejecuta consumo de hogares
11. realiza compras estatales e inversion publica
12. liquida resultados de firmas
13. cierra cuentas fiscales
14. resuelve quiebras, entrada y reemplazo de firmas
15. aplica demografia
16. resuelve insolvencia bancaria
17. construye el snapshot macro del periodo

Ese orden importa. La macro final de cada mes surge de una cadena causal concreta, no de una identidad impuesta ex ante.

## Mercados: como se ajustan

### Mercado laboral

Las firmas no contratan una cantidad perfecta en equilibrio. Definen demanda de trabajo, ofrecen salarios, enfrentan rechazos y llenan vacantes con fricciones. Los hogares comparan ofertas con su salario de reserva y su situacion. El resultado puede ser:

- desempleo
- vacantes sin cubrir
- presion salarial
- desalineacion entre oferta y demanda de trabajo cualificado

### Mercado de bienes

La demanda se forma desde hogares y gobierno. Las ventas dependen de:

- ingreso disponible
- necesidades esenciales
- preferencias
- precios
- inventario disponible

Esto permite observar brechas entre demanda potencial, demanda monetariamente efectiva y demanda finalmente satisfecha.

### Mercado de credito

El credito tampoco limpia automaticamente. Los bancos imponen criterios de deuda, cobertura y riesgo. Las firmas pueden ver frenada su expansion por costo financiero y los hogares pueden quedar excluidos por mora o fragilidad.

## Educacion y movilidad social

Una parte especialmente fuerte del modelo es que trata la educacion como mecanismo economico, no como adorno estadistico. El simulador incorpora:

- escolaridad publica y privada
- universidad publica y privada
- cupos y costos
- persistencia del apoyo publico
- priorizacion de hogares con bajos recursos
- prima salarial por escolaridad y universidad
- trazabilidad del origen social
- medicion de movilidad ascendente

Asi se puede analizar si una estructura educativa reduce pobreza, aumenta productividad o cambia la composicion del trabajo cualificado.

## Demografia y familia

El motor no congela la poblacion. Los hogares envejecen, forman parejas, tienen hijos, pierden guardianes, generan nuevas cohortes y finalmente mueren. Esto hace posible observar:

- transicion etaria
- dependencia demografica
- cambios de fuerza laboral
- presion sobre gasto social
- reproduccion o ruptura intergeneracional de la pobreza

## Politica economica y perfiles de pais

El proyecto ya incluye perfiles de politica para distintos arreglos institucionales, incluyendo:

- Guatemala (mas liberal)
- Estados Unidos (mixto)
- Noruega (economia del bienestar)
- Estado social intensivo (benchmark)

Estos perfiles modifican parametros como:

- desempleo objetivo
- carga tributaria
- gasto educativo
- amplitud de la red de proteccion
- intensidad del empleo publico
- reglas monetarias y financieras
- prioridad en cobertura escolar y universitaria

El interes del simulador no es solo correr una economia "promedio", sino comparar arquitecturas institucionales completas.

## Que mide

El simulador produce un conjunto amplio de indicadores mensuales y anuales. Entre ellos:

- PIB nominal y aproximaciones reales
- inflacion, deflactor y crecimiento
- empleo, desempleo, vacantes y salarios
- demanda esencial, produccion esencial y cobertura efectiva
- hambre, cobertura alimentaria y fragilidad de salud
- ahorro de hogares y desigualdad tipo Gini
- riqueza de propietarios y concentracion de activos
- credito de hogares y firmas
- capitalizacion, reservas y mora bancaria
- recaudacion, gasto, deficit y deuda publica
- matricula escolar y universitaria
- prima educativa de ingresos
- movilidad ascendente desde origen pobre
- brecha entre PIB por produccion y por gasto

La salida del modelo no es un numero final unico; es una historia dinamica del sistema.

## Estructura del proyecto

```text
app.py                         # interfaz Streamlit para explorar escenarios y dashboards
economy_simulator/
  domain.py                    # dataclasses, configuracion y snapshots del modelo
  engine.py                    # motor principal de simulacion por periodos
  policies.py                  # perfiles de pais y presets institucionales
  reporting.py                 # dataframes, agregaciones y metricas derivadas
  batch_reports.py             # reportes masivos y salidas comparativas
  cli.py                       # ejecucion por linea de comandos
tests/
  test_money_flow.py           # pruebas de consistencia y reporting
pyproject.toml                 # dependencias y configuracion del paquete
```

## Instalacion

Requisitos:

- Python 3.10 o superior
- `pip`

Instalacion recomendada:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Si prefieres instalar dependencias sin modo editable:

```bash
pip install .
```

## Como ejecutar

### Interfaz visual con Streamlit

```bash
streamlit run app.py
```

La app permite:

- seleccionar horizonte temporal
- ajustar parametros institucionales
- comparar perfiles de pais
- inspeccionar series mensuales y anuales
- descargar CSVs y diagnosticos de firmas

### Ejecucion por CLI

```bash
economy-sim --periods 120 --households 10000 --firms-per-sector 40
```

Ejemplo guardando historial en JSON:

```bash
economy-sim --periods 240 --households 5000 --output output\\historial.json
```

## Pruebas

Para correr las pruebas:

```bash
python -m unittest
```

## Como leer el modelo correctamente

Este repositorio no debe leerse como si fuera una version discreta de un modelo de equilibrio general. La intuicion correcta es otra:

- primero existen agentes concretos con reglas y restricciones
- luego esos agentes interactuan mediante mercados incompletos y fricciones
- despues aparecen agregados macroeconomicos observables

Si el simulador muestra desempleo, pobreza, quiebras o tensiones bancarias, no es porque se "movio un parametro agregado" en el vacio. Es porque una secuencia de decisiones micro produjo ese resultado.

## Alcance y filosofia

Este proyecto busca servir como laboratorio para pensar preguntas como:

- Que pasa con la pobreza si sube la cobertura educativa pero el credito sigue restringido.
- Como cambia la estabilidad cuando el banco central endurece liquidez.
- Que tan sensible es el empleo a un estado mas pequeno o mas grande.
- Si la movilidad social mejora cuando la universidad publica prioriza hogares con menos recursos.
- Como se transmite una recesion cuando firmas, hogares y bancos reaccionan con restricciones reales.

No pretende ser una "maquina de pronostico exacto" ni una calibracion cerrada del mundo real. Es una plataforma de simulacion causal, institucional y microfundada.

## Idea fuerza del repositorio

Si hubiera que resumir el proyecto en una sola frase, seria esta:

> La macroeconomia no esta impuesta por un equilibrio sobre agregados; emerge de la accion micro, heterogenea y restringida de todos los agentes que componen la economia.

Ese es el corazon del simulador.
