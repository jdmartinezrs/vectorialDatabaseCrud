import streamlit as st
import pandas as pd
from pyvis.network import Network
import streamlit.components.v1 as components
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración de la página
st.set_page_config(layout="wide")
st.title("Visualización de Partidos de Fútbol - FIFA 2022")

# 1. Conectar a ChromaDB y obtener la colección
client = chromadb.PersistentClient(path="./chroma_db")  # Usar persistencia local
collection = client.get_or_create_collection(name="partidos_fifa_2022")

# 2. Obtener todos los datos de la colección
partidos_chroma = collection.get(include=["documents", "metadatas"])

# 3. Convertir los datos a un DataFrame de Pandas
partidos = []
for i in range(len(partidos_chroma["ids"])):
    partido = {
        "id": partidos_chroma["ids"][i],
        "texto": partidos_chroma["documents"][i],
        **partidos_chroma["metadatas"][i]
    }
    partidos.append(partido)

df_partidos = pd.DataFrame(partidos)

# Función para normalizar texto
def normalize_text(text):
    # Convertir a minúsculas
    text = text.lower()
    # Eliminar caracteres especiales y números
    text = re.sub(r'\W+', ' ', text)
    # Eliminar stopwords
    stop_words = set(stopwords.words('spanish'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lematización
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Normalizar los textos de los partidos
df_partidos['texto_normalizado'] = df_partidos['texto'].apply(normalize_text)

# Cargar el modelo de embeddings (Sentence Transformer)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Obtener los embeddings de los textos de los partidos
embeddings = model.encode(df_partidos['texto_normalizado'].tolist())

# Diccionario de banderas (URLs de imágenes de banderas)
banderas = {
    "Argentina": "https://flagcdn.com/ar.svg",
    "Francia": "https://flagcdn.com/fr.svg",
    "Brasil": "https://flagcdn.com/br.svg",
    "Alemania": "https://flagcdn.com/de.svg",
    "España": "https://flagcdn.com/es.svg",
    "Portugal": "https://flagcdn.com/pt.svg",
    "México": "https://flagcdn.com/mx.svg",
    "Qatar": "https://flagcdn.com/qa.svg",
    "Ecuador": "https://flagcdn.com/ec.svg",
    "Senegal": "https://flagcdn.com/sn.svg",
    "Países Bajos": "https://flagcdn.com/nl.svg",
    "Inglaterra": "https://flagcdn.com/gb.svg",
    "Estados Unidos": "https://flagcdn.com/us.svg",
    "Gales": "https://flagcdn.com/gb-wls.svg",
    "Irán": "https://flagcdn.com/ir.svg",
    "Arabia Saudita": "https://flagcdn.com/sa.svg",
    "Polonia": "https://flagcdn.com/pl.svg",
    "Australia": "https://flagcdn.com/au.svg",
    "Dinamarca": "https://flagcdn.com/dk.svg",
    "Túnez": "https://flagcdn.com/tn.svg",
    "Costa Rica": "https://flagcdn.com/cr.svg",
    "Japón": "https://flagcdn.com/jp.svg",
    "Bélgica": "https://flagcdn.com/be.svg",
    "Canadá": "https://flagcdn.com/ca.svg",
    "Marruecos": "https://flagcdn.com/ma.svg",
    "Croacia": "https://flagcdn.com/hr.svg",
    "Suiza": "https://flagcdn.com/ch.svg",
    "Camerún": "https://flagcdn.com/cm.svg",
    "Serbia": "https://flagcdn.com/rs.svg",
    "Uruguay": "https://flagcdn.com/uy.svg",
    "Corea del Sur": "https://flagcdn.com/kr.svg",
    "Ghana": "https://flagcdn.com/gh.svg",
}

# 5. Entrada para buscar partidos semánticamente
st.subheader("Búsqueda semántica de partidos")

# Entrada de búsqueda
texto_buscar = st.text_input("Escribe una descripción, un equipo, un estadio, una etapa o cualquier información relacionada:")

# Calcular y mostrar partidos según la búsqueda semántica
if texto_buscar:
    # Normalizar la consulta del usuario
    texto_buscar_normalizado = normalize_text(texto_buscar)
    
    # Generar embedding para la consulta del usuario
    query_embedding = model.encode([texto_buscar_normalizado])

    # Calcular similitud del coseno entre la consulta y los embeddings de los partidos
    similitudes = cosine_similarity(query_embedding, embeddings)

    # Obtener los índices de los partidos más similares
    indices_similares = np.argsort(similitudes[0])[::-1]  # Ordenar de mayor a menor similitud

    # Crear un DataFrame con los partidos más relevantes y sus puntajes de similitud
    partidos_relevantes = df_partidos.iloc[indices_similares[:5]].copy()
    partidos_relevantes["score"] = similitudes[0][indices_similares[:5]]  # Añadir columna de score

    # Reordenar columnas para mostrar el score primero
    partidos_relevantes = partidos_relevantes[["score", "id", "texto", "equipo_local", "equipo_visitante", "estadio", "resultado", "etapa", "fecha"]]

    # Mostrar los partidos más relevantes
    st.write(f"Partidos más relevantes para '{texto_buscar}':")
    st.dataframe(partidos_relevantes, hide_index=True)

    # Actualizar el gráfico para mostrar los partidos encontrados
    nodes = []  # Lista de nodos (equipos)
    edges = []  # Lista de aristas (partidos)

    # Crear nodos únicos para los equipos
    equipos_unicos = list(set(df_partidos["equipo_local"].tolist() + df_partidos["equipo_visitante"].tolist()))
    for i, equipo in enumerate(equipos_unicos):
        # Obtener la URL de la bandera del equipo
        bandera_url = banderas.get(equipo, "https://flagcdn.com/xx.svg")  # Bandera por defecto si no se encuentra
        nodes.append({
            "id": i,
            "label": equipo,
            "title": equipo,
            "value": 1,  # Tamaño del nodo
            "shape": "circularImage",  # Usar imagen dentro de un círculo
            "image": bandera_url,  # URL de la bandera
            "size": 25,  # Tamaño de la imagen
            "borderWidth": 0,  # Eliminar el borde del círculo
            "color": {
                "border": "transparent",  # Hacer el borde transparente
                "background": "transparent"  # Hacer el fondo transparente
            }
        })

    # Crear aristas para los partidos encontrados
    for _, partido in partidos_relevantes.iterrows():
        src = equipos_unicos.index(partido["equipo_local"])
        dst = equipos_unicos.index(partido["equipo_visitante"])
        edges.append({
            "source": src,
            "target": dst,
            "label": partido["resultado"],  # Resultado como etiqueta de la arista
            "title": f"Estadio: {partido['estadio']}, Etapa: {partido['etapa']}",  # Información adicional
            "width": 2,  # Grosor de la línea
            "color": "#000000"  # Color de la línea
        })

    # Crear el gráfico con Pyvis
    net = Network(height="500px", width="100%", notebook=True)
    for node in nodes:
        net.add_node(node["id"], label=node["label"], title=node["title"], shape=node["shape"], image=node["image"], size=node["size"], borderWidth=node["borderWidth"], color=node["color"])
    for edge in edges:
        net.add_edge(edge["source"], edge["target"], label=edge["label"], title=edge["title"], width=edge["width"], color=edge["color"])

    # Configuración adicional del gráfico
    net.toggle_physics(True)  # Activar física para mejor distribución
    net.set_edge_smooth('dynamic')  # Suavizar las aristas

    # Guardar el gráfico en un archivo HTML
    net.save_graph("grafo_actualizado_busqueda.html")

    # Mostrar el gráfico en Streamlit
    with open("grafo_actualizado_busqueda.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=600)