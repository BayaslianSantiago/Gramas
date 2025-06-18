import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # Aunque usaremos mejor un lematizador específico para español si es posible
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import re

# --- 0. Descargar recursos NLTK (solo la primera vez) ---
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('omw-1.4') # Open Multilingual Wordnet para lematización en español

# --- Funciones de Preprocesamiento ---

def cargar_texto(ruta_archivo):
    """Carga el texto desde un archivo, intentando diferentes codificaciones."""
    codificaciones_a_probar = ['utf-8', 'latin-1', 'cp1252'] # Orden de preferencia
    for encoding in codificaciones_a_probar:
        try:
            with open(ruta_archivo, 'r', encoding=encoding) as f:
                texto = f.read()
            print(f"Archivo cargado exitosamente con codificación: {encoding}")
            return texto
        except UnicodeDecodeError:
            print(f"Fallo al cargar con '{encoding}'. Intentando la siguiente...")
            continue # Intenta con la siguiente codificación en la lista
        except FileNotFoundError:
            print(f"Error: El archivo '{ruta_archivo}' no fue encontrado.")
            return None
        except Exception as e:
            print(f"Ocurrió un error inesperado al cargar el archivo con '{encoding}': {e}")
            return None
    print(f"No se pudo cargar el archivo '{ruta_archivo}' con ninguna de las codificaciones probadas.")
    return None

def limpiar_texto(texto):
    """
    Limpia el texto: convierte a minúsculas, elimina puntuación y números.
    """
    texto = texto.lower()
    # Eliminar URLs
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    # Eliminar menciones (@usuarios) y hashtags (#palabra)
    texto = re.sub(r'@\w+|#\w+', '', texto)
    # Eliminar todo lo que no sea una letra, un espacio o un guion (para palabras compuestas si se mantienen)
    texto = re.sub(r'[^a-záéíóúüñ\s]', '', texto) # Incluye caracteres del español
    texto = re.sub(r'\s+', ' ', texto).strip() # Eliminar espacios extra
    return texto

def tokenizar_texto(texto_limpio):
    """Tokeniza el texto limpio en palabras."""
    return word_tokenize(texto_limpio, language='spanish')

def lematizar_tokens(tokens):
    """
    Lematiza una lista de tokens.
    NLTK's WordNetLemmatizer no es ideal para español.
    Si fuera para producción, usaría SpaCy o un lematizador más específico.
    Para este ejemplo, simularé un lematizador básico o usaré WordNetLemmatizer
    con las limitaciones para español.
    """
    # NLTK WordNetLemmatizer funciona mejor con tags de POS.
    # Para simplicidad, si no tenemos POS tagging, puede no ser muy efectivo en español.
    # nltk_lemmatizer = WordNetLemmatizer()
    # lemas = [nltk_lemmatizer.lemmatize(token) for token in tokens]

    # --- Alternativa simple para simular lematización en español (MEJOR EN ESTE CONTEXTO) ---
    # Esto es una simplificación muy grande. Un lematizador real necesita reglas o un modelo.
    # Para las palabras de tu corpus, NLTK WordNetLemmatizer NO funcionaría bien para español.
    # Sin SpaCy, podríamos hacer esto:
    lemas = []
    for token in tokens:
        if token.endswith('s') and len(token) > 2: # Remover 's' plural si es el caso
            lemas.append(token[:-1])
        elif token.endswith('n') and token not in ['con', 'sin', 'tan', 'bien']: # Para verbos como 'cobran' -> 'cobra'
            lemas.append(token[:-1])
        elif token.endswith('a') and token not in ['la', 'una']: # Para sustantivos como 'empleada' -> 'empleado' (si asumimos la forma 'o' como base)
            lemas.append(token[:-1] + 'o') # Simplificación: empleada -> empleado
        else:
            lemas.append(token)
    return lemas


def eliminar_stopwords(tokens_lematizados):
    """Elimina las stop words de una lista de tokens lematizados."""
    stop_words_spanish = set(stopwords.words('spanish'))
    # Puedes añadir stop words adicionales si encuentras que NLTK no cubre todas
    stop_words_adicionales = {'ser', 'estar', 'hacer', 'tener', 'poder', 'querer', 'deber', 'ir', 'venir', 'decir', 'ver', 'dar', 'saber', 'parecer', 'hay', 'es', 'son', 'del', 'al', 'se', 'lo', 'que', 'más', 'un', 'una', 'unos', 'unas', 'mi', 'mis', 'tu', 'tus', 'su', 'sus'}
    all_stop_words = stop_words_spanish.union(stop_words_adicionales)
    return [word for word in tokens_lematizados if word not in all_stop_words and len(word) > 1] # Evitar tokens de una letra

# --- Funciones para N-gramas ---

def generar_ngrams(tokens, n):
    """Genera n-gramas a partir de una lista de tokens."""
    return list(nltk.ngrams(tokens, n))

def contar_y_filtrar_ngrams(ngrams, min_df=2):
    """
    Cuenta la frecuencia de los n-gramas y filtra los que aparecen menos de min_df veces.
    """
    frecuencias = Counter(ngrams)
    filtrados = {ngram: count for ngram, count in frecuencias.items() if count >= min_df}
    return filtrados

# --- Función de Visualización ---

def plot_ngram_comparison(two_grams_freq, three_grams_freq, top_n=10):
    """
    Crea un gráfico de barras comparando los 2-gramas y 3-gramas más frecuentes.
    """
    # Convertir las claves de tupla a cadenas para una mejor visualización en el gráfico
    two_grams_labels = {str(k): v for k, v in two_grams_freq.items()}
    three_grams_labels = {str(k): v for k, v in three_grams_freq.items()}

    # Ordenar y seleccionar los top_n
    top_2_grams = dict(sorted(two_grams_labels.items(), key=lambda item: item[1], reverse=True)[:top_n])
    top_3_grams = dict(sorted(three_grams_labels.items(), key=lambda item: item[1], reverse=True)[:top_n])

    # Crear el DataFrame para Matplotlib
    df_2 = pd.DataFrame(list(top_2_grams.items()), columns=['N-grama', 'Frecuencia'])
    df_3 = pd.DataFrame(list(top_3_grams.items()), columns=['N-grama', 'Frecuencia'])

    fig, axes = plt.subplots(1, 2, figsize=(18, 7)) # 1 fila, 2 columnas de gráficos

    # Gráfico para 2-gramas
    axes[0].barh(df_2['N-grama'], df_2['Frecuencia'], color='skyblue')
    axes[0].set_xlabel('Frecuencia')
    axes[0].set_ylabel('2-grama')
    axes[0].set_title(f'Top {top_n} 2-gramas más frecuentes (min_df=2)')
    axes[0].invert_yaxis() # Para que el más frecuente esté arriba

    # Gráfico para 3-gramas
    axes[1].barh(df_3['N-grama'], df_3['Frecuencia'], color='lightcoral')
    axes[1].set_xlabel('Frecuencia')
    axes[1].set_ylabel('3-grama')
    axes[1].set_title(f'Top {top_n} 3-gramas más frecuentes (min_df=2)')
    axes[1].invert_yaxis() # Para que el más frecuente esté arriba

    plt.tight_layout() # Ajustar el diseño para evitar superposiciones
    plt.show()

# --- Función Principal (Main) ---

def main(ruta_corpus, min_df=2):
    """
    Función principal que orquesta el proceso de análisis de n-gramas.
    """
    print(f"Cargando texto de: {ruta_corpus}")
    texto_bruto = cargar_texto(ruta_corpus)
    if texto_bruto is None:
        return

    print("Limpiando texto...")
    texto_limpio = limpiar_texto(texto_bruto)

    print("Tokenizando texto...")
    tokens = tokenizar_texto(texto_limpio)

    print("Lematizando tokens...")
    # Atención: Este lematizador es MUY básico para español.
    # En un entorno real, usaría SpaCy: nlp = spacy.load("es_core_news_sm"), [token.lemma_ for token in nlp(texto_limpio)]
    tokens_lematizados = lematizar_tokens(tokens)


    print("Eliminando Stop Words...")
    tokens_finales = eliminar_stopwords(tokens_lematizados)

    print(f"Generando 2-gramas y 3-gramas (min_df={min_df})...")
    two_grams = generar_ngrams(tokens_finales, 2)
    three_grams = generar_ngrams(tokens_finales, 3)

    two_grams_frecuencias = contar_y_filtrar_ngrams(two_grams, min_df)
    three_grams_frecuencias = contar_y_filtrar_ngrams(three_grams, min_df)

    print("\n--- Frecuencias de 2-gramas (filtrado por min_df) ---")
    # Mostrar algunos ejemplos de los más frecuentes
    print(dict(sorted(two_grams_frecuencias.items(), key=lambda item: item[1], reverse=True)[:15]))

    print("\n--- Frecuencias de 3-gramas (filtrado por min_df) ---")
    print(dict(sorted(three_grams_frecuencias.items(), key=lambda item: item[1], reverse=True)[:15]))

    print("\nGenerando gráfico de barras...")
    plot_ngram_comparison(two_grams_frecuencias, three_grams_frecuencias, top_n=15) # Mostrar top 15

    print("\nProceso completado.")

# --- Ejecutar el programa ---
if __name__ == "__main__":
    # Asegúrate de que CorpusEducacion.txt esté en el mismo directorio que tu script,
    # o proporciona la ruta completa al archivo.
    main('CorpusEducacion.txt', min_df=2)