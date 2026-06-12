# Clasificacion fixture ultimo molde

Fuente local: `try-photos-last-mold-with-without pieces/`

Regla de fixture:
- `present` = muestra correcta/golden para esa seccion.
- `missing` = misma seccion con una o mas piezas pequenas removidas.
- No mover fotos. No commitear fotos.

## Secciones

| Seccion app | Zona | Fotos `present` | Fotos `missing` |
| --- | --- | --- | --- |
| Lado derecho superior cercano | `right_upper_side_close` | `IMG_2563.HEIC`, `IMG_2565.HEIC`, `IMG_2567.HEIC`, `IMG_2569.HEIC` | `IMG_2564.HEIC`, `IMG_2566.HEIC`, `IMG_2568.HEIC` |
| Frente centro medio | `front_center_medium` | `IMG_2570.HEIC`, `IMG_2572.HEIC`, `IMG_2574.HEIC`, `IMG_2576.HEIC`, `IMG_2578.HEIC`, `IMG_2580.HEIC` | `IMG_2571.HEIC`, `IMG_2573.HEIC`, `IMG_2575.HEIC`, `IMG_2577.HEIC`, `IMG_2579.HEIC`, `IMG_2581.HEIC` |
| Frente completo / pieza derecha inferior | `front_full_right_lower` | `IMG_2607.HEIC`, `IMG_2609.HEIC`, `IMG_2611.HEIC`, `IMG_2613.HEIC`, `IMG_2615.HEIC`, `IMG_2618.HEIC`, `IMG_2620.HEIC`, `IMG_2622.HEIC` | `IMG_2608.HEIC`, `IMG_2610.HEIC`, `IMG_2612.HEIC`, `IMG_2614.HEIC`, `IMG_2619.HEIC`, `IMG_2621.HEIC`, `IMG_2623.HEIC` |
| Detalle bajo izquierdo | `left_lower_detail` | `IMG_2616.HEIC` | `IMG_2617.HEIC` |
| Trasera completa | `rear_full` | `IMG_2624.HEIC`, `IMG_2626.HEIC`, `IMG_2628.HEIC` | `IMG_2625.HEIC`, `IMG_2627.HEIC` |
| Oblicua lateral izquierda | `left_oblique_full` | `IMG_2629.HEIC`, `IMG_2631.HEIC`, `IMG_2633.HEIC`, `IMG_2642.HEIC` | `IMG_2630.HEIC`, `IMG_2632.HEIC`, `IMG_2643.HEIC` |
| Detalle superior central | `center_top_detail` | `IMG_2634.HEIC` | `IMG_2635.HEIC` |
| Frente derecho amplio | `front_right_wide` | `IMG_2636.HEIC`, `IMG_2638.HEIC`, `IMG_2640.HEIC` | `IMG_2637.HEIC`, `IMG_2639.HEIC`, `IMG_2641.HEIC` |

## Pares usados para QA del programa

| Zona | Golden | Captura prueba | Esperado |
| --- | --- | --- | --- |
| `front_full_right_lower` | `IMG_2607.HEIC` | `IMG_2607.HEIC` | `correct` |
| `front_full_right_lower` | `IMG_2607.HEIC` | `IMG_2608.HEIC` | `review` con region pequena lado derecho inferior |
| `right_upper_side_close` | `IMG_2563.HEIC` | `IMG_2564.HEIC` | `review` localizado |
| `front_center_medium` | `IMG_2570.HEIC` | `IMG_2571.HEIC` | `review` localizado |
| `left_lower_detail` | `IMG_2616.HEIC` | `IMG_2617.HEIC` | `review` localizado |
| `rear_full` | `IMG_2624.HEIC` | `IMG_2625.HEIC` | `review` localizado |
| `left_oblique_full` | `IMG_2629.HEIC` | `IMG_2630.HEIC` | `review` localizado |
| `center_top_detail` | `IMG_2634.HEIC` | `IMG_2635.HEIC` | `review` localizado |
| `front_right_wide` | `IMG_2636.HEIC` | `IMG_2637.HEIC` | `review` localizado |
