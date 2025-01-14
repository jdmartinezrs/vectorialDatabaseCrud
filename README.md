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


