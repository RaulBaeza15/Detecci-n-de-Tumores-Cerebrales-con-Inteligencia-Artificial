🧠 Proyecto: Detección de Tumores Cerebrales con Inteligencia Artificial

Este proyecto utiliza imágenes de resonancias magnéticas para entrenar un sistema que detecta automáticamente si hay presencia de tumores cerebrales. Se compone de varias fases que combinan procesamiento de imágenes, aprendizaje profundo y visualización interpretativa.
________________________________________
1️⃣ Descarga y preparación de datos

•	Se descarga un conjunto de imágenes desde Kaggle usando kagglehub.

o	Utilicé el dataset público Br35H :: Brain Tumor Detection 2020, disponible en Kaggle:

📎 https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

•	Las imágenes están clasificadas en dos carpetas: yes (con tumor) y no (sin tumor).

•	Se organizan en carpetas para entrenamiento, validación y test (70%-15%-15%).

🔧 Además, se aplican transformaciones a las imágenes para mejorar el aprendizaje del modelo:

•	Rotaciones

•	Zoom

•	Volteo horizontal

Esto se llama data augmentation y ayuda a que el modelo generalice mejor.

________________________________________
2️⃣ Autoencoder: comprensión de imágenes

Se entrena un autoencoder, una red neuronal que:

•	Comprime la imagen a una representación más pequeña (llamada espacio latente).

•	Reconstruye la imagen original desde esa representación.

📌 ¿Por qué es útil? Porque obliga al modelo a entender los patrones importantes de las imágenes, como formas y estructuras internas del cerebro.

Se guardan ejemplos visuales de reconstrucciones para comprobar que el modelo ha aprendido correctamente.

________________________________________
3️⃣ Clasificador: detección de tumores

Una vez entrenado el autoencoder, se extrae su parte de compresión (el encoder) y se usa como entrada para un clasificador.

Este clasificador:
•	Toma la representación comprimida de la imagen.

•	Decide si hay tumor o no.

✅ Se entrena con los datos comprimidos y se evalúa con datos nuevos (test), generando:

•	Una matriz de confusión (aciertos y errores)

•	Un informe de clasificación con métricas como precisión y sensibilidad

________________________________________
4️⃣ Interpretabilidad: Grad-CAM

Para entender por qué el modelo toma sus decisiones, se usa una técnica llamada Grad-CAM que genera mapas de calor sobre las imágenes.

📸 ¿Qué muestra?

•	Las zonas de la imagen que han influido más en la decisión del modelo.

•	Se superpone el mapa de calor sobre la imagen original.

Se generan ejemplos tanto de imágenes con tumor como sin tumor, y se guardan en carpetas separadas.


