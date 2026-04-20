# Dataset
Para este proyecto elegimos el dataset COMPAS 
(Correctional Offender Management Profiling for Alternative Sanctions), 
un sistema de evaluación de riesgo que se utiliza en tribunales de Estados Unidos 
para estimar si un acusado volverá a delinquir. Nos pareció un caso especialmente 
interesante porque en 2016 ProPublica demostró que el sistema tenía un sesgo racial 
importante: las tasas de falsos positivos eran mucho más altas para personas
afroamericanas que para blancas. 

Partimos de ese análisis para replicarlo y ampliarlo 
con técnicas actuales de aprendizaje automático.

# Objetivo
Nuestro objetivo principal es estudiar cómo cambia la equidad del modelo según el 
criterio de optimización que se elija. Para ello, entrenaremos un modelo base y lo 
compararemos bajo cuatro enfoques distintos: maximizar la accuracy global sin distinguir 
por grupo demográfico, igualar la accuracy entre grupos raciales, igualar las tasas de 
falsos positivos, e igualar las tasas de falsos negativos. Buscaremos herramientas del 
estado del arte que nos permitan medir y corregir el sesgo, y analizaremos qué compromiso 
implica cada criterio cuando hablamos de decisiones judiciales.

También exploraremos técnicas de explicabilidad para entender qué variables pesan más en 
las predicciones y en el sesgo detectado, así como métodos para evaluar la robustez del 
modelo frente a posibles manipulaciones en los datos de entrada.

Por último, discutiremos los resultados teniendo en cuenta la legislación europea vigente, 
reflexionando sobre los riesgos de usar modelos automáticos para tomar decisiones en el ámbito 
judicial.