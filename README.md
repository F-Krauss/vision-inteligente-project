# Inspeccion Visual De Moldes Industriales

Proyecto base en Python para inspeccionar moldes por zonas usando fotos de celular.
El flujo esta disenado para detectar presencia, ausencia y casos dudosos de elementos
criticos, priorizando falso rechazo sobre falso aprobado.

## Flujo recomendado

### Modo cloud: API + web app en GCP

Este es el flujo recomendado para produccion cuando la validacion debe correr en
la nube con mayor precision y latencia media cercana a 5 segundos por zona.
La Raspberry queda fuera de este modo.

Componentes:

- Cloud Run `mold-vision-api` con GPU NVIDIA L4.
- FastAPI para cargas, inspecciones, metadata y trabajos de entrenamiento.
- Web app React/Vite para captura guiada desde telefono.
- Cloud Storage para fotos, datasets y artefactos.
- Firestore para metadata, resultados e historial.
- Vertex AI Custom Training para entrenar por `family + zone_id`.

Instalacion local:

```bash
python3 -m pip install -e ".[cloud,vision,dev]"
npm --prefix web install
```

Ejecutar API local:

```bash
mold-inspection-api
```

Ejecutar web local:

```bash
npm --prefix web run dev
```

Endpoints principales:

```text
POST /v1/uploads/presign
PUT  /v1/uploads/{upload_id}
POST /v1/inspections
GET  /v1/inspections/{id}
POST /v1/training-jobs
GET  /v1/training-jobs/{id}
POST /v1/recipes
GET  /v1/recipes
POST /v1/segmenter-datasets/from-annotations
POST /v1/segmenter-training-jobs
POST /v1/inspector-training-jobs
POST /v1/model-candidates/{id}/promote
POST /v1/public-datasets/import
GET/POST /v1/families
GET/POST /v1/molds
GET/POST /v1/zones
GET/POST /v1/datasets
GET/POST /v1/model_versions
```

### Flujo tipo SolVision para moldes

La pantalla `AI Recipes` organiza el flujo por molde/zona: datos de entrenamiento,
estado de dataset, entrenamiento del inspector, candidatos con `loss`,
`confidence`, `validation_recall` y `false_pass_rate`, y modelo promovido. La
promocion se hace automaticamente por menor falso aprobado y mayor recall, con
opcion de promover otro candidato desde la UI.

La captura movil abre pantalla completa, asume toma horizontal, dibuja el
poligono del molde detectado y autocaptura despues de frames estables. La
respuesta de inspeccion incluye `identified_mold`, `identified_zone`,
`confidence`, `mold_polygon`, `missing_regions` y `overlay_image_uri`.

Los adapters de benchmark registran MVTec AD, VisA, KolektorSDD y ABO como
datasets publicos de prueba. MVTec AD y KolektorSDD son no comerciales; se
mantienen como `benchmark_only` salvo permiso explicito.

Bootstrap de GCP:

```bash
gcloud auth login
gcloud config set account <cuenta-con-acceso-a-mia-prod>
PROJECT_ID=mia-prod REGION=us-central1 ./scripts/gcloud_bootstrap.sh
PROJECT_ID=mia-prod REGION=us-central1 ./scripts/deploy_cloud_run.sh
```

El despliegue usa Cloud Run GPU con `--gpu=1`, `--gpu-type=nvidia-l4`,
`--cpu=4`, `--memory=16Gi`, `--no-cpu-throttling` y `--min-instances=1`.
Para costos menores en staging se puede ejecutar con `MIN_INSTANCES=0`.

La API es fail-closed: si no existe modelo productivo para una zona devuelve
`review`; si la foto esta borrosa, sobreexpuesta o no se puede leer devuelve
`retake_photo`.

### Modo recomendado: suite de modelos para Raspberry Pi

Este flujo entrena varios candidatos ligeros por `family + zone_id`, evalua con
fotos `ok` y `fault`, selecciona el mejor con prioridad en evitar falsos aprobados
y exporta un artefacto que corre en Raspberry Pi sin PyTorch en inferencia.

Manifest supervisado minimo:

```csv
image_path,family,zone_id,label,mold_id,session_id,split
/ruta/ok_1.HEIC,molde_a,frontal_zona_01,ok,molde_001,sesion_1,train
/ruta/falla_1.HEIC,molde_a,frontal_zona_01,fault,molde_002,sesion_2,val
```

Entrenar suite:

```bash
python3 -m mold_inspection.cli train-model-suite \
  --family molde_a \
  --zone-id frontal_zona_01 \
  --manifest data/supervised.csv \
  --mask /ruta/mask.png \
  --target raspberry-pi
```

Exportar el mejor modelo:

```bash
python3 -m mold_inspection.cli export-best \
  --family molde_a \
  --zone-id frontal_zona_01 \
  --target raspberry-pi
```

Inspeccionar usando solo el modelo ganador:

```bash
python3 -m mold_inspection.cli inspect-best \
  --family molde_a \
  --zone-id frontal_zona_01 \
  --images /ruta/foto_nueva.HEIC \
  --out reports/best_inspection.json
```

Artefacto exportado:

```text
data/model_registry/<family>/<zone_id>/best_model/
  model.npz
  profile.json
  thresholds.json
  benchmark.json
  mask.png
```

### Modo base: anomaly detection por zona

Este modo no requiere enlistar todos los agujeros o elementos. Cada zona aprende
su apariencia normal usando fotos correctas, aplica una mascara para ignorar fondo
y genera un heatmap con las regiones diferentes.

1. Crear estructura inicial:

```bash
python3 -m mold_inspection.cli init
```

2. Registrar una mascara de zona para ignorar fondo, manos, botas y piso:

```bash
python3 -m mold_inspection.cli set-zone-mask \
  --family familia_a \
  --zone-id frontal_zona_01 \
  --mask /ruta/mask.png
```

3. Entrenar/calibrar la zona con fotos correctas:

```bash
python3 -m mold_inspection.cli train-anomaly \
  --family familia_a \
  --zone-id frontal_zona_01 \
  --images /ruta/golden_1.HEIC /ruta/golden_2.HEIC /ruta/golden_3.HEIC \
  --mask /ruta/mask.png
```

4. Inspeccionar una foto nueva:

```bash
python3 -m mold_inspection.cli inspect-anomaly \
  --family familia_a \
  --zone-id frontal_zona_01 \
  --images /ruta/foto_nueva_zona_01.HEIC \
  --out reports/anomaly_inspection.json
```

Estados posibles:

- `correct`: la zona coincide con el perfil normal.
- `review`: hay regiones anomalas dentro de la mascara.
- `retake_photo`: la foto no sirve para validar; tomar otra foto.

Evidencia visual:

- Imagen alineada: `reports/anomaly_evidence/<family>/<zone>/<image>/aligned.jpg`
- Heatmap: `reports/anomaly_evidence/<family>/<zone>/<image>/heatmap.jpg`
- Overlay con regiones: `reports/anomaly_evidence/<family>/<zone>/<image>/overlay.jpg`

Para los laterales, cree zonas separadas. Por ejemplo:

```bash
python3 -m mold_inspection.cli train-anomaly \
  --family familia_a \
  --zone-id lateral_pistones_01 \
  --images /ruta/lateral_1.HEIC /ruta/lateral_2.HEIC \
  --mask /ruta/mask_lateral_pistones.png
```

### Baseline: referencia maestra 1:1

Este modo queda disponible, pero es mas sensible a perspectiva y escala porque
compara pixeles/SSIM despues de alinear.

Guardar referencias:

```bash
python3 -m mold_inspection.cli set-references \
  --images /ruta/IMG_1755.HEIC /ruta/IMG_1756.HEIC /ruta/IMG_1757.HEIC \
  --family familia_a \
  --zone-id frontal_zona_01 \
  --reference-prefix frontal
```

Inspeccionar:

```bash
python3 -m mold_inspection.cli inspect-golden \
  --family familia_a \
  --zone-id frontal_zona_01 \
  --images /ruta/foto_nueva_zona_01.jpg
```

### Modo opcional: detector YOLO por elementos

Este modo queda disponible para piezas especificas, pero no es el flujo principal
si se necesita comparar el molde completo 1:1.

1. Definir familias, zonas y elementos esperados en `config/inspection.json`.

2. Cargar fotos por zona:

```bash
python3 -m mold_inspection.cli add-images \
  --source /ruta/a/fotos \
  --family familia_a \
  --mold-id molde_001 \
  --session-id sesion_2026_05_13 \
  --zone-id zona_01 \
  --state correct
```

3. Generar plantillas de anotacion:

```bash
python3 -m mold_inspection.cli label-template
```

4. Separar train/val/test sin contaminar por molde/sesion:

```bash
python3 -m mold_inspection.cli split
python3 -m mold_inspection.cli audit-split
```

5. Exportar dataset YOLO y entrenar:

```bash
python3 -m mold_inspection.cli export-yolo
python3 -m mold_inspection.cli train --data-yaml data/yolo/data.yaml --weights yolo11n.pt
```

6. Inspeccionar nuevas fotos:

```bash
python3 -m mold_inspection.cli inspect \
  --weights runs/detect/train/weights/best.pt \
  --family familia_a \
  --zone-id zona_01 \
  --images /ruta/nueva/foto.jpg \
  --out reports/inspection.json
```

## Conceptos

- `family`: familia o layout del molde.
- `mold_id`: identificador del molde fisico.
- `session_id`: corrida o sesion de captura.
- `zone_id`: zona fotografiada siguiendo la guia.
- `state`: `correct`, `incorrect` o `simulated_fault`.
- `expected`: elementos que deben existir en una zona.
- `roi`: region normalizada `[x1, y1, x2, y2]` donde se espera el elemento.
- `reference`: foto maestra correcta para una zona completa.
- `reference_id`: variante aceptada de captura para la misma zona, por ejemplo `frontal_nivelada`, `frontal_angulo` o `lateral`.
- `mask`: imagen binaria donde blanco es area valida de inspeccion y negro se ignora.
- `anomaly_score`: distancia de la foto nueva contra el banco de parches normales.

## Instalacion

Basico:

```bash
python3 -m pip install -e .
```

Con comparacion 1:1, entrenamiento o inferencia visual:

```bash
python3 -m pip install -e ".[vision]"
```

Con pruebas:

```bash
python3 -m pip install -e ".[dev]"
pytest
```
