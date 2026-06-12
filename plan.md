Plan de trabajo extensivo para la integración de captura con muestra de referencia y anotación en el proyecto vision‑inteligente
Este plan amplía y detalla el trabajo a realizar para que el sistema de inspección de moldes industriales se ajuste a la especificación de trabajar exclusivamente con fotografías, utilizando una captura guiada mediante superposición de muestra de referencia (golden sample) y permitiendo la anotación de piezas directamente desde la aplicación. También se incluyen recomendaciones de mejoras en interfaz y experiencia de usuario, un plan de pruebas y sugerencias de arquitectura. Se espera que este plan sirva como guía para el agente Codex a la hora de implementar los cambios en el repositorio F‑Krauss/vision-inteligente-project.
Objetivos generales
Integrar la captura guiada con superposición de referencia para cada zona del molde, asegurando consistencia en ángulos y encuadre durante la toma de fotografías.
Permitir la anotación de elementos (piezas) desde la app y almacenarla en el mismo sistema, de forma que no se requieran herramientas externas para etiquetar imágenes.
Mantener el ciclo de entrenamiento y validación dentro de la plataforma, de modo que operadores no técnicos puedan dar de alta moldes y zonas siguiendo una guía visual.
Mejorar la interfaz de usuario y la experiencia de uso para reducir errores y facilitar la adopción por parte de operarios de planta.
Definir un plan de pruebas que cubra la funcionalidad, integración y rendimiento del sistema.
Orientar la arquitectura hacia una solución escalable y económica que combine Cloud Run y servicios gestionados de Google Cloud.
1. Captura guiada con golden sample
1.1 Definición de referencias
Para cada family + zone_id se debe definir al menos una imagen de referencia (golden sample). Esta imagen representa la foto correcta, con el ángulo, distancia y encuadre deseados.
La referencia se almacenará en Cloud Storage y su URI se guardará en Firestore en la colección de zones.
Las zonas que requieran múltiples referencias (por ejemplo, variantes de ángulo leve) deberán tener un reference_id distintivo para cada variante.
Se debe proporcionar una interfaz en la app para que un administrador pueda subir o actualizar las referencias de cada zona.
1.2 Ajustes en la API
Endpoint GET /v1/zones/{zone_id}/reference: devolverá el URI de la referencia y opcionalmente de la máscara (mask.png) de la zona.
Endpoint POST /v1/zones/{zone_id}/reference: permitirá subir una nueva referencia para la zona. Deberá validar que el usuario tenga permisos de administrador y asociará la imagen con un nuevo reference_id.
Endpoint POST /v1/uploads/align-quality (nuevo) o modificación del endpoint de inspección para validar alineación: antes de procesar la foto, el backend revisará metadatos de la imagen (por ejemplo, orientación y tamaño) y calculaciones básicas de similitud con la referencia para determinar si la alineación es suficiente. En caso negativo, retornará retake_photo.
1.3 Cambios en la interfaz (React)
Pantalla de selección: el usuario selecciona la familia, molde y zona a capturar. Si es la primera vez que se captura una zona, se pedirá a un usuario con rol de administrador que cargue una referencia.
Descarga y superposición: al cargar la pantalla de captura, la app consultará el endpoint GET /v1/zones/{zone_id}/reference y recuperará la imagen de referencia (golden sample).
Visor de cámara con overlay:
Mostrar el streaming de la cámara en horizontal, ya que los moldes se fotografían acostados.
Dibujar la imagen de referencia semitransparente encima, con opacidad ajustable (por ejemplo, 40–60 %).
Colocar puntos de guía o contornos (por ejemplo, esquinas marcadas) para facilitar el alineado manual.
Implementar una lógica de detección de estabilidad: usar la API de cámara para observar si la imagen deja de moverse y si coincide en escala/rotación con la referencia. Tras un tiempo de estabilidad (1–2 segundos) se autocapturará la foto; alternativamente, permitir un botón “Capturar” que solo se activa cuando el encuadre coincide.
Mostrar mensajes claros de “Alinea tu cámara con la referencia” y “Foto aceptada”.
Validación local: antes de enviar la foto al backend, realizar una validación simple en el cliente: comprobar que el brillo no está fuera de rango, que no está borrosa (usando varianza del laplaciano) y que se detectan los contornos principales de la zona.
Subida de imagen: la foto aceptada se enviará al backend mediante los endpoints de subida existentes (POST /v1/uploads/presign, PUT /v1/uploads/{upload_id}), indicando family, zone_id y reference_id usados.
Pantalla de confirmación: una vez subida la foto, se mostrará un resumen y se permitirá al usuario capturar otra sección o finalizar.
1.4 Definición de tolerancias de alineación
Para cada zona habrá parámetros de tolerancia en traslación (desplazamiento), escala (distancia) y rotación. Estos valores se almacenarán en la configuración de la zona (en config/inspection.json o en Firestore) y se usarán en el backend para validar la alineación.
Para simplificar, se puede usar un descriptor global (por ejemplo, ORB + BFMatcher) y fijar un umbral de coincidencia de keypoints entre la referencia y la imagen nueva.
Si la similitud es inferior al umbral, se retorna retake_photo y se sugiere al usuario ajustar.
2. Anotación integrada en la app
2.1 Interfaz de anotación
Componente reutilizable: crear un componente ImageAnnotator en React que permita:
Visualizar la imagen en alta resolución (posiblemente con zoom y desplazamiento).
Dibujar rectángulos de selección (bounding boxes).
Seleccionar la etiqueta de la pieza (de una lista de expected del config/inspection.json para esa zona).
Editar o eliminar anotaciones existentes.
Guardar automáticamente las anotaciones en estado local para evitar pérdida de datos.
Lista de elementos esperados: al abrir la pantalla de anotación, la app consultará el backend para obtener la lista de elementos esperados en la zona (config/inspection.json define expected). Esta lista se presentará como menú desplegable.
Persistencia de anotaciones: al pulsar “Guardar”, las anotaciones se enviarán al endpoint POST /v1/annotations. El backend asignará las coordenadas normalizadas ([x1, y1, x2, y2] en valores 0–1) al image_id y almacenará la anotación en Firestore.
Anotación en sesiones de entrenamiento: al registrar nuevos moldes, se mostrará la imagen con la referencia superpuesta para anotar solo las piezas visibles en la zona.
Navegación de dataset: crear una página donde se muestren todas las imágenes de entrenamiento por family y zone_id. Al seleccionar una imagen, se abrirá el ImageAnnotator con las anotaciones existentes para revisarlas, editarlas o añadir nuevas.
2.2 Integración con entrenamiento
El backend debe incluir un proceso POST /v1/segmenter-datasets/from-annotations que:
Reciba un family y zone_id.
Recupere todas las imágenes de entrenamiento y sus anotaciones asociadas.
Genere el dataset en formato YOLO: crea directorios train/ y val/ con imágenes y archivos .txt con las bounding boxes y etiquetas.
Genere un data.yaml con la configuración de clases.
Una vez creado el dataset, la app permitirá al usuario iniciar un trabajo de entrenamiento (inspector training job) a través del endpoint POST /v1/inspector-training-jobs.
Después del entrenamiento, se mostrará el resultado (loss, validation_recall, false_pass_rate) y se habilitará un botón para promover el modelo (POST /v1/model-candidates/{id}/promote).
3. Mejoras de UI/UX
Jerarquía y simplicidad: organizar la interfaz en pasos claros:
Selección de molde y zona: lista desplegable con filtros y búsqueda.
Captura guiada: visor de cámara con overlay y mensajes en tiempo real.
Revisión y anotación: presentar la foto capturada y permitir anotar inmediatamente.
Historial y dataset: vista de tarjetas o tabla donde se muestran las capturas previas, anotaciones y su estado (train, val, test).
Entrenamiento y modelos: panel con los datasets disponibles, estados de entrenamiento, métricas y modelos promovidos.
Accesibilidad:
Optimizar la aplicación para uso con guantes (botones grandes y fáciles de pulsar).
Garantizar contrastes correctos para uso en planta (iluminación variable).
Incluir retroalimentación audible/vibración opcional cuando la foto se capture correctamente.
Mensajes de error y guía:
Mostrar advertencias en caso de foto borrosa, mal alineada o con iluminación insuficiente.
Sugerir acciones correctivas (acercar cámara, limpiar lente, ajustar iluminación).
Proporcionar un ícono de ayuda con instrucciones rápidas de uso.
Optimización de rendimiento:
Cargar las referencias y máscaras de forma asíncrona y en caché para reducir la latencia.
Minimizar la resolución de previsualización, pero mantener la foto original sin compresión para análisis.
Prevenir recargas completas de la página; usar estado global (por ejemplo, Zustand o Redux) para manejar sesiones de captura y anotación.
Roles de usuario:
Definir roles (operador, supervisor, administrador).
Los operadores pueden capturar fotos, ver sus inspecciones y anotar.
Los supervisores pueden revisar y corregir anotaciones de operadores.
Los administradores pueden crear familias, zonas, referencias y lanzar entrenamientos.
Modo offline (mejora futura):
Permitir capturar y anotar fotos sin conexión y sincronizar cuando haya conectividad. Esta funcionalidad se puede basar en IndexedDB y Service Workers.
4. Plan de pruebas
4.1 Pruebas unitarias
Backend (FastAPI):
Verificar que los endpoints de referencia (GET/POST /v1/zones/{zone_id}/reference) devuelven y almacenan correctamente las imágenes y URIs.
Validar que POST /v1/annotations guarda anotaciones en la base de datos y que GET /v1/annotations recupera las anotaciones de forma consistente.
Testear la generación de datasets a partir de anotaciones y la conversión a formato YOLO.
Comprobar que train-anomaly, train-model-suite y train se ejecutan con los parámetros correctos y crean los artefactos esperados.
Frontend:
Probar que el componente de captura no permite enviar la foto hasta que se cumpla la alineación mínima.
Probar que el componente ImageAnnotator crea, edita y elimina anotaciones adecuadamente.
Probar que la navegación entre imágenes anotadas conserva el estado y evita pérdidas de datos.
4.2 Pruebas de integración
Flujo completo de captura:
Seleccionar un molde y zona; cargar referencia; mostrar overlay; capturar foto; validar imagen; subirla al backend; verificar que se registra con retake_photo cuando la foto no está alineada.
Anotación:
Abrir una imagen capturada; anotar piezas; guardar; recuperar; verificar persistencia.
Exportar dataset y revisar que los archivos .txt generados contienen las etiquetas correctas.
Entrenamiento:
Ejecutar un trabajo de entrenamiento desde la UI; confirmar que el backend crea el job y que se registra en Firestore.
Simular un job exitoso (puede hacerse con un stub o corriendo un entrenamiento corto) y verificar que el resultado se visualice correctamente en la UI.
Roles y permisos:
Probar que usuarios operadores no pueden acceder a la subida de referencias ni a la edición de configuraciones.
Probar que administradores y supervisores ven las opciones adicionales.
4.3 Pruebas de aceptación en planta
Condiciones reales: realizar capturas en planta con diferentes moldes y condiciones de iluminación.
Consistencia: medir el porcentaje de veces que la foto es aceptada a la primera; ajustar las tolerancias si se rechazan capturas correctas.
Operador no técnico: seleccionar varios operadores no técnicos para probar la app y obtener retroalimentación sobre la claridad de la interfaz y las instrucciones.
Pruebas de rendimiento: medir el tiempo desde que el usuario toma la foto hasta que recibe el resultado de inspección (correct, review, retake_photo). Ajustar límites de compresión y tamaño de imagen según sea necesario para cumplir un objetivo de latencia (p. ej., <5 segundos).
Integridad del dataset: asegurarse de que todas las fotos capturadas con golden sample se registran en la misma orientación y tamaño. Revisar manualmente un subconjunto de capturas para verificar la calidad.
5. Arquitectura recomendada
5.1 Componentes principales
Frontend (React/Vite):
Implementa las pantallas de captura, anotación, revisión de datasets, lanzamiento de entrenamientos y visualización de resultados.
Se comunica con la API a través de llamadas HTTP o WebSocket para notificaciones en tiempo real (p. ej., progreso de entrenamiento).
Usa almacenamiento local (IndexedDB) para capturas temporales si se implementa modo offline.
Backend (FastAPI):
Gestiona las zonas, referencias, familias, moldes, sesiones y anotaciones.
Valida y almacena las imágenes y anotaciones.
Orquesta los trabajos de entrenamiento mediante comandos de CLI o llamadas a Vertex AI.
Provee endpoints para obtener modelos entrenados y realizar inferencia (puede llamar al detector/segmentador entrenado).
Servicios de entrenamiento:
Vertex AI Custom Jobs para ejecutar entrenamientos pesados (anomaly detection, inspector).
Cloud Run con GPU para inferencia en línea con mayor latencia.
Opcional: contenedores para entrenamientos en Raspberry Pi (si se usa target=raspberry-pi).
Bases de datos:
Firestore: para almacenar metadatos de imágenes, zonas, moldes, familias y anotaciones.
Cloud Storage: para almacenar imágenes originales, referencias, datasets exportados y modelos.
Artifact Registry: para almacenar los pesos de los modelos entrenados.
Mensajería/colas (opcional pero recomendado):
Pub/Sub para notificar a la UI sobre finalización de entrenamientos y resultados.
Cloud Tasks o Cloud Functions para procesar eventos asíncronos (por ejemplo, generar dataset a partir de anotaciones, limpiar imágenes obsoletas, etc.).
5.2 Mejoras arquitectónicas sugeridas
Desacoplar servicios: separar el servicio de captura/anotación del servicio de entrenamiento e inferencia. Esto permite escalar cada componente de manera independiente y reducir costos.
Caching y CDN: usar Cloud CDN para cachear las referencias y reducir la latencia al cargarlas en la app.
Observabilidad: instrumentar logs estructurados, trazas (Cloud Trace) y métricas para detectar cuellos de botella en la captura y en la inferencia.
Seguridad: implementar autenticación y autorización en la API (por ejemplo, con Firebase Auth o Identity Aware Proxy) para proteger la subida de referencias y anotaciones.
Estrategia de costos:
Ejecutar entrenamientos en Vertex AI con spot VMs o GPU preemptible para reducir costos.
Configurar Cloud Run con escalamiento a cero para ambientes de prueba y entornos de desarrollo.
Utilizar target=raspberry-pi para exportar modelos ligeros para inferencia en dispositivos edge, reduciendo la dependencia de la nube.
Compatibilidad futura: aunque se descarta el uso de 3D y vídeo en esta etapa, la arquitectura debe mantener una capa de abstracción que permita agregar nuevos tipos de sensores o entradas (por ejemplo, cámaras 3D) sin reescribir toda la app.
6. Integración con GitHub y workflow para Codex
6.1 Estructura de ramas
Crear una rama de trabajo a partir de la rama principal (main o master) llamada feature/golden-overlay-annotations.
Subir el archivo de plan a la ruta docs/plan_codex_extended.md.
Crear un pull request (PR) con un título descriptivo (“Add extended codex work plan for golden overlay and annotations”) y en el cuerpo del PR incluir un bloque goal (en idioma inglés) que describa los pasos que debe realizar Codex. Por ejemplo:
goal:
  - Integrate golden sample overlay into capture component.
  - Implement annotation component with bounding boxes and class selection.
  - Add endpoints for reference management and annotations.
  - Update training pipeline to consume annotated data and reference-aligned images.
  - Enhance UI/UX with clear guidance and validation steps.
  - Write unit and integration tests covering API and UI flows.
Etiquetar el PR con codex para indicar que el agente de Codex debe procesarlo y comenzar a iterar en modo extra-high.
Una vez que Codex complete el PR, revisar los cambios, solicitar ajustes si es necesario y, finalmente, aprobar/mergear.
6.2 Checklist para revisión de PR
 Archivo plan_codex_extended.md agregado correctamente.
 Endpoints para referencias y anotaciones implementados y documentados.
 Componente de captura actualizado con overlay y validaciones.
 Componente de anotación funcionando y persistiendo datos.
 Actualizaciones de UI/UX implementadas siguiendo las recomendaciones.
 Plan de pruebas incluido en la carpeta tests/ con ejemplos de unit e integración.
 Documentación actualizada (README.md y docs/).
7. Conclusiones y próximos pasos
Este plan provee una ruta detallada para implementar la captura guiada con referencia y anotación integrada en el prototipo de inspección de moldes.
Se optimiza la captura para asegurar consistencia mediante la superposición de una referencia, lo cual mejorará significativamente la precisión del modelo de visión.
Se facilita la anotación dentro de la app, reduciendo la dependencia de herramientas externas y acelerando la generación de datasets de entrenamiento.
Se plantean mejoras en UI/UX y arquitectura que permitirán escalar a más de 600 moldes sin comprometer costos ni experiencia de uso.
Se define un plan de pruebas robusto que cubrirá desde unidad hasta aceptación en planta.
Codex deberá seguir este plan, generando los cambios de código necesarios y presentando PRs iterativos hasta alcanzar la funcionalidad descrita. Una vez implementado, se recomienda ejecutar pilotos en diferentes líneas de producción para validar el sistema y ajustar parámetros (tolerancias, tiempos de captura, flujos de usuario) antes de un despliegue masivo.