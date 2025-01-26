# **Visualización Interactiva de Partidos de Fútbol de la FIFA 2022**

Este proyecto es una aplicación interactiva que permite visualizar partidos de fútbol de la FIFA 2022 utilizando **Streamlit**, **Pyvis** y **ChromaDB**. La aplicación incluye filtros para explorar partidos por estadio, resultado, etapa y fecha, y también permite realizar búsquedas semánticas utilizando embeddings generados con **Sentence Transformers**.

## **Requisitos**

Antes de ejecutar el proyecto, asegúrate de tener instalado lo siguiente:

- Python 3.8 o superior.
- Gestor de paquetes `pip`.

1. Clona el repositorio:

   ```python
   git clone https://github.com/jdmartinezrs/vectorialDatabaseCrud.git
   cd tu-repositorio
   ```

2. Crea un entorno virtual (opcional pero recomendado):

   ```python
   python -m venv venv
   source venv/bin/activate  # En Linux/Mac
   venv\Scripts\activate     # En Windows
   ```

3. Instala las dependencias:

   ```python
   pip install -r requirements.txt
   ```



### **Flujo del Código**

#### **a. Configuración Inicial**

- **Streamlit**: Se configura la página con un diseño amplio (`layout="wide"`) y un título descriptivo.
- **ChromaDB**: Se conecta a una base de datos local (`chroma_db`) y se obtiene la colección `partidos_fifa_2022`, que contiene los datos de los partidos.

#### **b. Carga de Datos**

- Los datos de los partidos se obtienen de ChromaDB y se convierten en un DataFrame de Pandas para facilitar su manipulación.
- Cada partido incluye:
  - Un ID único.
  - Un texto descriptivo (por ejemplo, "Partido entre Qatar y Ecuador en el Al Bayt Stadium el 20 de noviembre de 2022").
  - Metadatos como equipos, fecha, estadio, etapa, grupo y resultado.

#### **d. Filtros Interactivos**

- La aplicación permite filtrar los partidos por:
  - **Estadio**: Selecciona partidos jugados en un estadio específico.
  - **Resultado**: Filtra partidos con un resultado específico (por ejemplo, "2-1").
  - **Etapa**: Muestra partidos de una etapa específica (por ejemplo, "Fase de grupos").
  - **Fecha**: Filtra partidos por fecha.

#### **e. Gráfico Interactivo**

- Se crea un gráfico de red interactivo utilizando **Pyvis**:
  - **Nodos**: Representan los equipos, con imágenes de banderas como íconos.
  - **Aristas**: Representan los partidos, con etiquetas que muestran el resultado y tooltips con información adicional (estadio y etapa).


## Muestra de Funcionamiento: Video
[![Miniatura del video](https://img.youtube.com/vi/78yA4XL2HKU/hqdefault.jpg)](https://www.youtube.com/watch?v=78yA4XL2HKU)


## **¿Qué son las Consultas Semánticas?**

Las consultas semánticas se basan en el concepto de **similitud semántica**, que mide cuán similares son dos textos en términos de su significado. A diferencia de las búsquedas tradicionales (que buscan coincidencias exactas de palabras), las consultas semánticas utilizan modelos de lenguaje para capturar el contexto y el significado detrás de las palabras.

### **Ejemplo**:

- **Consulta tradicional**: Si buscas "partidos de Brasil", solo obtendrás resultados que contengan exactamente las palabras "partidos" y "Brasil".
- **Consulta semántica**: Si buscas "partidos emocionantes de Brasil", obtendrás resultados que capturan el significado de "emocionantes", incluso si esa palabra no aparece en el texto.


### **1. Comparación con el Código Anterior**

#### **Mejoras Principales**

1. **Normalización de Texto**:
   - Se ha añadido un proceso de normalización de texto que incluye:
     - Conversión a minúsculas.
     - Eliminación de caracteres especiales y números.
     - Eliminación de stopwords (palabras comunes que no aportan significado).
     - Lematización (reducción de palabras a su forma base).
   - Esto mejora la calidad de los embeddings y, por tanto, la precisión de las búsquedas semánticas.
2. **Puntuación de Similitud**:
   - Ahora se muestra un **score** de similitud para cada partido encontrado, lo que permite al usuario entender qué tan relevante es cada resultado.
3. **Optimización del Gráfico**:
   - El gráfico se actualiza dinámicamente para mostrar solo los partidos relevantes encontrados en la búsqueda semántica.
4. **Integración de NLTK**:
   - Se utiliza la biblioteca **NLTK** para realizar la normalización de texto, lo que añade robustez al procesamiento de lenguaje natural (NLP).

------

### **2. Proceso de Búsqueda Semántica**

#### **a. Normalización del Texto**

- **Objetivo**: Preparar el texto para generar embeddings más precisos.
- **Pasos**:
  1. **Conversión a minúsculas**: Para evitar diferencias entre mayúsculas y minúsculas.
  2. **Eliminación de caracteres especiales**: Se eliminan signos de puntuación y números que no aportan significado.
  3. **Eliminación de stopwords**: Se eliminan palabras comunes como "y", "de", "en", etc., que no aportan significado semántico.
  4. **Lematización**: Se reduce cada palabra a su forma base (por ejemplo, "corriendo" → "correr").

#### **b. Generación de Embeddings**

- **Objetivo**: Convertir el texto normalizado en vectores numéricos que capturen su significado semántico.
- **Proceso**:
  1. Se utiliza el modelo `paraphrase-MiniLM-L6-v2` de **Sentence Transformers** para generar embeddings.
  2. Cada texto normalizado se convierte en un vector de 384 dimensiones.

#### **c. Cálculo de Similitud**

- **Objetivo**: Encontrar los partidos más relevantes basados en la consulta del usuario.
- **Proceso**:
  1. Se genera un embedding para la consulta del usuario.
  2. Se calcula la **similitud del coseno** entre el embedding de la consulta y los embeddings de los textos de los partidos.
  3. Se ordenan los partidos por su puntaje de similitud (de mayor a menor).

#### **d. Visualización de Resultados**

- **Objetivo**: Mostrar los partidos más relevantes y actualizar el gráfico interactivo.
- **Proceso**:
  1. Se muestran los 5 partidos más relevantes en una tabla, junto con su puntaje de similitud.
  2. Se actualiza el gráfico de red para mostrar solo los partidos encontrados.

#### **a. Normalización de Texto**

```
def normalize_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'\W+', ' ', text)  # Eliminar caracteres especiales
    stop_words = set(stopwords.words('spanish'))  # Eliminar stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    lemmatizer = WordNetLemmatizer()  # Lematización
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text
```

#### **b. Búsqueda Semántica**

```python
if texto_buscar:
    texto_buscar_normalizado = normalize_text(texto_buscar)  # Normalizar consulta
    query_embedding = model.encode([texto_buscar_normalizado])  # Generar embedding
    similitudes = cosine_similarity(query_embedding, embeddings)  # Calcular similitud
    indices_similares = np.argsort(similitudes[0])[::-1]  # Ordenar por similitud
    partidos_relevantes = df_partidos.iloc[indices_similares[:5]].copy()  # Obtener los 5 más relevantes
    partidos_relevantes["score"] = similitudes[0][indices_similares[:5]]  # Añadir columna de score
```

#### **c. Actualización del Gráfico**

```python
net = Network(height="500px", width="100%", notebook=True)
for node in nodes:
    net.add_node(node["id"], label=node["label"], title=node["title"], shape=node["shape"], image=node["image"], size=node["size"], borderWidth=node["borderWidth"], color=node["color"])
for edge in edges:
    net.add_edge(edge["source"], edge["target"], label=edge["label"], title=edge["title"], width=edge["width"], color=edge["color"])
net.save_graph("grafo_actualizado_busqueda.html")
```

### **5. Tecnologías Utilizadas**

- **Streamlit**: Para la interfaz de usuario.
- **Pyvis**: Para gráficos de red interactivos.
- **ChromaDB**: Para almacenar los datos de los partidos.
- **Sentence Transformers**: Para generar embeddings.
- **NLTK**: Para normalización de texto.
- **Scikit-learn**: Para calcular la similitud del coseno.

## **Muestra de Funcionamiento: Búsquedas Semánticas**

Aquí puedes ver una demostración en video de cómo funcionan las búsquedas semánticas en la aplicación:

[![Muestra de Funcionamiento: Búsquedas Semánticas](https://img.youtube.com/vi/qL_kZXpux9w/0.jpg)]










---------------------------------------------------------------------------------------------------------------------------

# CRUD en Base Vectorial con ChromaDB

```
# Guía de Instalación y Configuración

Esta guía te ayudará a crear un entorno virtual, instalar dependencias necesarias, y configurar Jupyter Notebook para usarlo con el entorno activado.

### 1. Crear un Entorno Virtual

Para comenzar, crea un entorno virtual en tu proyecto:

```bash
python -m venv venv
```

### 2. Activar el Entorno Virtual

Dependiendo de tu sistema operativo, activa el entorno virtual:

- En Windows:

```
bash


venv\Scripts\activate
```

- En macOS/Linux:

```
bash

source venv/bin/activate
```

### 3. Instalar las Dependencias en el Entorno Virtual

Con el entorno virtual activado, instala las dependencias necesarias, como `sentence-transformers`, usando el siguiente comando:

```
bash

pip install sentence-transformers
```

### 4. Usar el Kernel Correcto en Jupyter Notebook

Si estás utilizando Jupyter Notebook, asegúrate de que el kernel esté vinculado a tu entorno virtual:

- Instala el paquete `ipykernel`:

```
bash

pip install ipykernel
```

- Añade el kernel al Jupyter Notebook:

```
bash

python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

En el Notebook, selecciona el kernel Python (venv) desde el menú desplegable en la parte superior derecha.

### 5. Alternativa: Instalar Dependencias desde el Sistema

Si encuentras problemas con Jupyter Notebook, puedes instalar las dependencias directamente desde la terminal:

```
bashCopiar códigopip install sentence-transformers
pip install chromadb
pip install chroma-migrate
pip install nbconvert
```

Después de instalar las dependencias correctamente, reinicia el servidor Jupyter Notebook para aplicar los cambios.

Ahora podrás trabajar con Jupyter Notebook utilizando el entorno virtual con las dependencias necesarias instaladas.

Para instalar dependencias listadas en un archivo `requirements.txt`, puedes usar el siguiente comando:

```
bash

pip install -r requirements.txt
```

Este comando instalará todas las bibliotecas y versiones especificadas en el archivo


### Embedding en Procesamiento del Lenguaje Natural (PLN)

En PLN, un embedding es una representación matemática de palabras o frases en un espacio vectorial continuo, diseñado para capturar relaciones semánticas y sintácticas. Este concepto permite que las palabras que comparten un significado similar estén representadas por vectores cercanos en dicho espacio. Los embeddings se han convertido en un pilar fundamental del aprendizaje profundo en PLN debido a su capacidad para manejar la dimensionalidad alta y las relaciones contextuales del lenguaje.

Los embeddings son representaciones vectoriales de palabras o frases, donde cada elemento (como una palabra) se mapea en un espacio multidimensional. Esto permite que las similitudes entre textos o palabras sean evaluadas mediante operaciones matemáticas. Un ejemplo común es transformar textos en vectores numéricos para luego compararlos, como se muestra en tu código.

~~~python
markdown

### Similitud Coseno

La similitud coseno es una métrica que mide el ángulo entre dos vectores en un espacio n-dimensional. Se utiliza para determinar qué tan similares son dos textos basándose en su representación vectorial. Un valor de similitud cercano a 1 indica que los textos son muy similares, mientras que valores cercanos a 0 indican que son poco similares.

### Ejemplo de Similitudes entre Textos

**Textos:**
- **Texto 1:** Partido entre Barcelona y Real Madrid en el Camp Nou
- **Texto 2:** Manchester United juega contra Liverpool en Old Trafford
- **Texto 3:** El clásico español en el Camp Nou entre Barça y Madrid
- **Texto 4:** Partido de Champions League en Old Trafford

**Matriz de similitud coseno**  
*(Valores más cercanos a 1 indican mayor similitud)*

```python
Similitud entre texto 1 y texto 1: 1.000  
Similitud entre texto 1 y texto 2: 0.367  
Similitud entre texto 1 y texto 3: 0.789  
Similitud entre texto 1 y texto 4: 0.606  

Similitud entre texto 2 y texto 1: 0.367  
Similitud entre texto 2 y texto 2: 1.000  
Similitud entre texto 2 y texto 3: 0.225  
Similitud entre texto 2 y texto 4: 0.725  

Similitud entre texto 3 y texto 1: 0.789  
Similitud entre texto 3 y texto 2: 0.225  
Similitud entre texto 3 y texto 3: 1.000  
Similitud entre texto 3 y texto 4: 0.397  

Similitud entre texto 4 y texto 1: 0.606  
Similitud entre texto 4 y texto 2: 0.725  
Similitud entre texto 4 y texto 3: 0.397  
Similitud entre texto 4 y texto 4: 1.000  
~~~

**Similitud entre texto 1 y texto 2:** 0.725
**Texto 1:** Manchester United juega contra Liverpool en Old Trafford
**Texto 2:** Partido de Champions League en Old Trafford



### Detalles del Embedding

El modelo 'all-MiniLM-L6-v2' genera embeddings de 384 dimensiones. Este es un tamaño estándar para este modelo específico, y todos los embeddings que generes con él tendrán esta misma dimensión, independientemente de la longitud del texto de entrada.

En tu código original, tanto los embeddings que guardaste en ChromaDB como los que usaste para las consultas tienen estas mismas 384 dimensiones, lo que permite hacer las comparaciones de similitud coseno entre ellos.



```python
**Dimensión del embedding:** (1, 384)  
**Tamaño del vector:** 384 dimensiones

**Primeros 5 valores del vector de embedding:**
[ 0.03745004  0.04752655 -0.02148234 -0.00557694  0.03101636 ]
```



# Add Data



```python
modelo_embeddings = SentenceTransformer('all-MiniLM-L6-v2')

# Datos de partidos de fútbol
partidos = [
    {"id": "1", "texto": "Partido entre Barcelona y Real Madrid en el Camp Nou el 25 de diciembre de 2025.", "metadata": {"equipo_local": "Barcelona", "equipo_visitante": "Real Madrid", "fecha": "2025-12-25", "estadio": "Camp Nou"}},
    {"id": "2", "texto": "Partido entre Manchester United y Liverpool en Old Trafford el 10 de enero de 2025.", "metadata": {"equipo_local": "Manchester United", "equipo_visitante": "Liverpool", "fecha": "2025-01-10", "estadio": "Old Trafford"}},
    {"id": "3", "texto": "Partido entre Juventus y Inter de Milan en el Allianz Stadium el 5 de marzo de 2025.", "metadata": {"equipo_local": "Juventus", "equipo_visitante": "Inter de Milan", "fecha": "2025-03-05", "estadio": "Allianz Stadium"}},
    {"id": "4", "texto": "Partido entre Bayern Munich y Borussia Dortmund en el Allianz Arena el 15 de abril de 2025.", "metadata": {"equipo_local": "Bayern Munich", "equipo_visitante": "Borussia Dortmund", "fecha": "2025-04-15", "estadio": "Allianz Arena"}},
    {"id": "5", "texto": "Partido entre PSG y Olympique de Marsella en el Parc des Princes el 30 de mayo de 2025.", "metadata": {"equipo_local": "PSG", "equipo_visitante": "Olympique de Marsella", "fecha": "2025-05-30", "estadio": "Parc des Princes"}},
    {"id": "6", "texto": "Partido entre Boca Juniors y River Plate en La Bombonera el 20 de junio de 2025.", "metadata": {"equipo_local": "Boca Juniors", "equipo_visitante": "River Plate", "fecha": "2025-06-20", "estadio": "La Bombonera"}},
    {"id": "7", "texto": "Partido entre Flamengo y Palmeiras en el Maracaná el 12 de julio de 2025.", "metadata": {"equipo_local": "Flamengo", "equipo_visitante": "Palmeiras", "fecha": "2025-07-12", "estadio": "Maracaná"}},
    {"id": "8", "texto": "Partido entre Atlético de Madrid y Sevilla en el Wanda Metropolitano el 18 de agosto de 2025.", "metadata": {"equipo_local": "Atlético de Madrid", "equipo_visitante": "Sevilla", "fecha": "2025-08-18", "estadio": "Wanda Metropolitano"}},
    {"id": "9", "texto": "Partido entre Chelsea y Arsenal en Stamford Bridge el 28 de septiembre de 2025.", "metadata": {"equipo_local": "Chelsea", "equipo_visitante": "Arsenal", "fecha": "2025-09-28", "estadio": "Stamford Bridge"}},
    {"id": "10", "texto": "Partido entre AC Milan y Napoli en San Siro el 10 de octubre de 2025.", "metadata": {"equipo_local": "AC Milan", "equipo_visitante": "Napoli", "fecha": "2025-10-10", "estadio": "San Siro"}},
]


# Generar embeddings para los textos
textos = [partido["texto"] for partido in partidos]
ids = [partido["id"] for partido in partidos]
metadatas = [partido["metadata"] for partido in partidos]
embeddings = modelo_embeddings.encode(textos).tolist()

# Insertar partidos con embeddings en la colección
collection.add(ids=ids, documents=textos, metadatas=metadatas, embeddings=embeddings)
print("Partidos con embeddings insertados.")
```



## Query and Get



### Generar embedding para la consulta

consulta = "Partido entre Barcelona y Real Madrid en el Camp Nou"
embedding_consulta = modelo_embeddings.encode([consulta]).tolist()

### Buscar los documentos más similares

resultados_similares = collection.query(query_embeddings=embedding_consulta, n_results=2)


~~~python
# Generar embedding para la consulta
consulta = "Partido entre Barcelona y Real Madrid en el Camp Nou"
embedding_consulta = modelo_embeddings.encode([consulta]).tolist()

# Buscar los documentos más similares
resultados_similares = collection.query(query_embeddings=embedding_consulta, n_results=2)
print("Partidos similares:", resultados_similares)
~~~

##### Resultado de la Consulta

```python
Partidos similares: {'ids': [['2', '3']], 'embeddings': None, 'documents': [['Partido entre Manchester United y Liverpool en Old Trafford el 10 de enero de 2025.', 'Partido entre Juventus y Inter de Milan en el Allianz Stadium el 5 de marzo de 2025.']], 'uris': None, 'data': None, 'metadatas': [[{'equipo_local': 'Bucaramanga', 'equipo_visitante': 'Liverpool', 'estadio': 'Old Trafford', 'estado_partido': 'Finalizado', 'fecha': '2025-01-10', 'resultado': '2-1'}, {'equipo_local': 'Juventus', 'equipo_visitante': 'Inter de Milan', 'estadio': 'Allianz Stadium', 'fecha': '2025-03-05'}]], 'distances': [[0.844056806444902, 0.8645657386614436]], 'included': [<IncludeEnum.distances: 'distances'>, <IncludeEnum.documents: 'documents'>, <IncludeEnum.metadatas: 'metadatas'>]}
```



### Búsqueda y Filtro por Texto Exacto

Este código realiza una búsqueda semántica en la colección utilizando una consulta de texto y aplica un filtro para obtener coincidencias exactas.

1. **Consulta Inicial**:
   - Se busca `"Camp Nou"` en la colección, obteniendo documentos, metadatos y distancias.
   - El número de resultados se limita a `n_results=2`.

2. **Filtro de Coincidencias Exactas**:
   - Se filtran resultados cuyo texto contiene `"Camp Nou"` o cuyo metadato `estadio` coincide exactamente con `"Camp Nou"`.

3. **Impresión de Resultados**:
   - Los documentos filtrados se muestran con su texto, metadatos y distancia.


```python
query_texto = "Camp Nou"
resultados = collection.query(
    query_texts=[query_texto],
    n_results=2,
    include=["documents", "metadatas", "distances"]
)
# Filtro de coincidencia exacta
resultados_filtrados = [
    {
        "texto": doc,
        "metadata": meta,
        "distancia": dist
    }
    for doc, meta, dist in zip(
        resultados["documents"][0],
        resultados["metadatas"][0],
        resultados["distances"][0]
    )
    if "Camp Nou" in doc or meta.get("estadio") == "Camp Nou"
]

for i, res in enumerate(resultados_filtrados, start=1):
    print(f"\nDocumento {i}:")
    print(f"Texto: {res['texto']}")
    print(f"Metadata: {res['metadata']}")
    print(f"Distancia: {res['distancia']:.4f}")
```

##### Resultado de la Consulta

```python
Documento 1:
Texto: Partido entre Barcelona y Real Madrid en el Camp Nou el 25 de diciembre de 2025, con 70,000 espectadores.
Metadata: {'equipo_local': 'Barcelona', 'equipo_visitante': 'Real Madrid', 'estadio': 'Camp Nou', 'fecha': '2025-12-25'}
Distancia: 1.5588
```



### Búsqueda Filtrada por Metadatos equipo local 

Este código realiza una búsqueda semántica con un filtro adicional aplicado sobre los metadatos.

- **Consulta**: Busca documentos relacionados con `"Barcelona"`.
- **Filtro**: Incluye solo resultados donde `equipo_local` es `"Barcelona"`.
- **Resultados**: Imprime los documentos, metadatos, y distancias relevantes.

```python
query_filtrado = collection.query(
    query_texts=["Barcelona"],
    n_results=2,
    where={"equipo_local": "Barcelona"},  
    include=["documents", "metadatas", "distances"]
)

print("Resultados con filtro 'equipo_local = Barcelona':")
for i in range(len(query_filtrado['documents'][0])):
    print(f"\nDocumento {i+1}:")
    print(f"Texto: {query_filtrado['documents'][0][i]}")
    print(f"Metadata: {query_filtrado['metadatas'][0][i]}")
    print(f"Distancia: {query_filtrado['distances'][0][i]:.4f}")
```

##### Resultado de la Consulta

```python
Resultados con filtro 'equipo_local = Barcelona':
```

### Búsqueda por Estadio

Este código realiza una búsqueda en la colección filtrando por un estadio específico, en este caso, "Camp Nou".

```python
resultados_estadio = collection.query(
    query_texts=[""],
    n_results=5,
    where={"estadio": "Camp Nou"},
    include=["documents", "metadatas"]
)
for i, (doc, meta) in enumerate(zip(resultados_estadio['documents'][0], resultados_estadio['metadatas'][0])):
    print(f"\nPartido {i+1}:")
    print(f"Texto: {doc}")
    print(f"Metadata: {meta}")
```

##### Resultado de la Consulta

```python
Partido 1:
Texto: Partido entre Barcelona y Real Madrid en el Camp Nou el 25 de diciembre de 2025, con 70,000 espectadores.
Metadata: {'equipo_local': 'Barcelona', 'equipo_visitante': 'Real Madrid', 'estadio': 'Camp Nou', 'fecha': '2025-12-25'}
```



#### Búsqueda por Fecha Específica

Este código realiza una búsqueda en la colección filtrando por una fecha específica.

```python
resultados_fecha = collection.query(
    query_texts=[""],
    n_results=5,
    where={"fecha": "2025-12-25"},
    include=["documents", "metadatas"]
)

for i, (doc, meta) in enumerate(zip(resultados_fecha['documents'][0], resultados_fecha['metadatas'][0])):
    print(f"\nPartido {i+1}:")
    print(f"Texto: {doc}")
    print(f"Metadata: {meta}")
```



##### Resultado de la Consulta

```python
Texto: Partido entre Barcelona y Real Madrid en el Camp Nou el 25 de diciembre de 2025, con 70,000 espectadores.
Metadata: {'equipo_local': 'Barcelona', 'equipo_visitante': 'Real Madrid', 'estadio': 'Camp Nou', 'fecha': '2025-12-25'}
```



# Update Data

#### Este código actualiza los metadatos de un partido específico en la colección.

```python
collection.update(
    ids=["2"],
    metadatas=[{
        "equipo_local": "Bucaramanga",
        "equipo_visitante": "Liverpool",
        "fecha": "2025-01-10",
        "estadio": "Old Trafford",
        "estado_partido": "Finalizado",
        "resultado": "2-1"
    }]
)

```

#### Actualización de Partido y su Embedding

Este código actualiza la información de un partido en la colección, incluyendo su texto, metadatos, y embedding.

```python
nuevo_texto = "Partido entre Barcelona y Real Madrid en el Camp Nou el 25 de diciembre de 2025, con 70,000 espectadores."
nuevo_embedding = modelo_embeddings.encode([nuevo_texto]).tolist()
collection.delete(ids=["1"])  # Eliminar partido antiguo
collection.add(ids=["1"], documents=[nuevo_texto], metadatas=[{"equipo_local": "Barcelona", "equipo_visitante": "Real Madrid", "fecha": "2025-12-25", "estadio": "Camp Nou"}], embeddings=nuevo_embedding)
print("Partido actualizado con nuevo embedding.")
```

##### Resultado de la Consulta

```python
Delete of nonexisting embedding ID: 1
Partido actualizado con nuevo embedding.
```



# Delete Data

#### Eliminación de Documento por ID

Este código elimina un documento específico de la colección utilizando su ID.

```python
collection.delete(ids=["9"])
print("Documento con ID 9 eliminado.")

```

##### Resultado de la Consulta

```python
Delete of nonexisting embedding ID: 9
Documento con ID 9 eliminado.
```

