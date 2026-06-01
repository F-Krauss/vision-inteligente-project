# Dataset

Capture cada molde por zonas con una secuencia fija.

Para la suite supervisada use un CSV con etiquetas por zona:

```csv
image_path,family,zone_id,label,mold_id,session_id,split
/ruta/ok_1.HEIC,molde_a,frontal_zona_01,ok,molde_001,sesion_1,train
/ruta/fault_1.HEIC,molde_a,frontal_zona_01,fault,molde_002,sesion_2,val
```

Etiquetas validas:

- `ok`: zona correcta
- `fault`: zona con faltante, sobrante o diferencia critica

No mezcle el mismo `mold_id/session_id` entre train, val y test.

Para entrenamiento cloud en Vertex AI, suba el manifest, las imagenes y la
mascara a Cloud Storage. El endpoint `/v1/training-jobs` espera:

```json
{
  "family": "molde_a",
  "zone_id": "frontal_zona_01",
  "dataset_uri": "gs://bucket/datasets/molde_a/frontal_zona_01/",
  "manifest_uri": "gs://bucket/datasets/molde_a/frontal_zona_01/manifest.csv",
  "mask_uri": "gs://bucket/datasets/molde_a/frontal_zona_01/mask.png",
  "output_uri": "gs://bucket/models/molde_a/frontal_zona_01/"
}
```

El trainer descarga el manifest, materializa imagenes `gs://` localmente,
entrena candidatos `cloud-gpu` por zona y sube `best_model.tar.gz` junto con
`training_report.json`.

Para anomaly detection use una carpeta por familia y zona:

```text
data/anomaly/<family>/<zone_id>/
  anchor.jpg
  mask.png
  memory_bank.npz
  profile.json
```

Cada molde debe tener zonas frontales y laterales separadas. Ejemplos:

```text
frontal_zona_01/mask.png
lateral_pistones_01/mask.png
```

La mascara debe marcar en blanco solo el area del molde que se inspecciona. Fondo,
manos, botas, herramientas, cables y piso deben quedar en negro.

El sistema rechazara la foto con `retake_photo` si detecta:

- desenfoque
- iluminacion demasiado distinta
- pocos puntos de alineacion
- perspectiva fuera de tolerancia
- zona incorrecta
- poca cobertura de la mascara

Para el baseline 1:1 use fotos maestras por zona en `data/references/<family>/<zone_id>/<reference_id>.jpg`.
Para el modo YOLO opcional, no mezcle sesiones del mismo molde entre train, val y test.
