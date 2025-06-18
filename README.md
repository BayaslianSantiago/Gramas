# Análisis de N-gramas en Opiniones sobre Educación Superior (Colombia 2025)

Este repositorio contiene un script en Python para realizar un análisis de texto sobre un corpus de opiniones de estudiantes colombianos sobre la educación superior en el año 2025. El objetivo es identificar patrones y temas clave mediante la extracción y comparación de 2-gramas y 3-gramas, aplicando técnicas de preprocesamiento de lenguaje natural (PLN).

## 📊 Características del Análisis

* **Preprocesamiento de Texto:** Limpieza, tokenización, lematización y eliminación de *stop words*.
* **Generación de N-gramas:** Extracción de 2-gramas (pares de palabras) y 3-gramas (tríos de palabras).
* **Filtrado por Frecuencia:** Los n-gramas se filtran para incluir solo aquellos con una frecuencia mínima de aparición (`min_df = 2`).
* **Visualización:** Comparación gráfica de las frecuencias de los n-gramas más relevantes mediante gráficos de barras.

## 📚 Corpus Utilizado

El análisis se realiza sobre el archivo `CorpusEducacion.txt`, que contiene un resumen de opiniones reales de alumnos de Colombia expresando sus deseos con respecto a la educación superior en el año 2025.

## 🚀 Cómo Ejecutar el Proyecto

Sigue estos pasos para configurar y ejecutar el script en tu entorno local.

### 1. Requisitos

Asegúrate de tener Python 3.x instalado. Las siguientes librerías de Python son necesarias:

* `nltk`
* `matplotlib`
* `pandas`

Puedes instalarlas usando `pip`:

pip install nltk matplotlib pandas


2. Descargar Recursos de NLTK
NLTK requiere la descarga de algunos recursos para la tokenización, lematización y listas de stop words. Ejecuta el siguiente código una vez en un entorno Python (por ejemplo, en un script Python separado o en un intérprete interactivo):

Python

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4') # Open Multilingual Wordnet (necesario para lematización en español)


3. Preparar el Corpus
Crea un archivo llamado CorpusEducacion.txt en el mismo directorio donde guardarás el script analisis_ngrams.py. Pega dentro de este archivo el texto del corpus de opiniones.

Importante sobre la codificación del archivo:
Si experimentas un UnicodeDecodeError al cargar el archivo, es probable que la codificación del archivo CorpusEducacion.txt no sea UTF-8. El script intenta cargar el archivo con varias codificaciones comunes (utf-8, latin-1, cp1252). Si el error persiste, verifica la codificación de tu archivo de texto y asegúrate de que sea compatible con las opciones probadas.

4. Ejecutar el Script
Guarda el código principal del análisis (el que contiene todas las funciones y la llamada a main) en un archivo llamado analisis_ngrams.py (o el nombre que prefieras).

Luego, ejecuta el script desde tu terminal:

python analisis_ngrams.py
El script imprimirá el progreso en la consola y, al finalizar, mostrará un gráfico de barras interactivo con la comparación de los n-gramas más frecuentes.

📂 Estructura del Proyecto

├── CorpusEducacion.txt    # Archivo de texto con el corpus de opiniones

└── analisis_ngrams.py     # Script Python para el análisis de n-gramas

├── README.md              # Este archivo


📝 Consideraciones sobre la Lematización
La lematización en español es un desafío. Para este proyecto, la función lematizar_tokens implementa una lematización básica basada en heurísticas simples para el idioma español (por ejemplo, removiendo 's' para plurales o transformando 'a' final a 'o').

Para un análisis más robusto y profesional, se recomienda utilizar librerías de PLN más avanzadas como SpaCy, que ofrecen modelos pre-entrenados para español con una lematización y etiquetado de partes de la oración mucho más precisos.

Si deseas mejorar la lematización con SpaCy, deberías:

Instalar SpaCy: pip install spacy
Descargar el modelo en español: python -m spacy download es_core_news_sm
Reemplazar la implementación de lematizar_tokens en el script por una que utilice SpaCy.
