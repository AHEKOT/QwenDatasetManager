# QDM QI2 Layers — ComfyUI

Инференс LoRA, обученных пресетом **Qwen Image 2.1 — Joint PSD layers LoRA**
в QwenDatasetManager. Плоское изображение → совместная генерация RGBA-слоёв → PSD.
Пак автономный: установленный trainer и его Python-окружение ему не нужны.

## Установка

1. Скопируйте всю папку `ComfyUI-QDM-QI2-Layers` в `ComfyUI/custom_nodes/`.
2. Установите зависимости **Python-ом ComfyUI**:

   ```powershell
   # Windows portable, из папки ComfyUI_windows_portable:
   .\python_embeded\python.exe -m pip install -r .\ComfyUI\custom_nodes\ComfyUI-QDM-QI2-Layers\requirements.txt

   # Обычная установка: используйте Python активного окружения ComfyUI:
   python -m pip install -r custom_nodes/ComfyUI-QDM-QI2-Layers/requirements.txt
   ```

3. Требуется ComfyUI с **нативной Qwen Image 2.1**, Qwen3-VL 8B и поддержкой
   выбранного формата весов. Перезапустите ComfyUI после установки пака.
   Обычная поддержка Qwen Image 1.x / Qwen-Image-Layered недостаточна.

## Модели

| Компонент | Папка ComfyUI | Файл по умолчанию в workflow |
|---|---|---|
| QI2 | `models/diffusion_models/` | `qwen_image_2.1_int8_convrot.safetensors` |
| Text encoder | `models/text_encoders/` | `qwen3vl_8b_bf16.safetensors` |
| Native RGBA VAE | `models/vae/` | `qwen_image_2.1_vae_bf16.safetensors` |
| Ваша обученная LoRA | `models/loras/` | Выбрать свой `.safetensors` |

Можно выбрать совместимые BF16/convrot-варианты базовых весов, поддерживаемые
вашей версией ComfyUI. В CLIPLoader оставьте тип **qwen_image**.
Существующие модели QDM можно подключить через `extra_model_paths.yaml`,
не копируя их. Пример для текущего расположения проекта:

```yaml
qdm:
  base_path: E:/Development/QwenDatasetManager/models
  diffusion_models: diffusion_models
  text_encoders: text_encoders
  vae: vae
```

Саму обученную LoRA скопируйте в `ComfyUI/models/loras/`.
Ноды ничего не скачивают и не запускают обучение.

## Готовые workflow

- [QI2_Image_to_PSD_4layers.json](workflows/QI2_Image_to_PSD_4layers.json) — адаптер, обученный с 4 слотами.
- [QI2_Image_to_PSD_20layers.json](workflows/QI2_Image_to_PSD_20layers.json) — адаптер, обученный с 20 слотами.
- Файлы с суффиксом `_api.json` предназначены для API; в интерфейс импортируйте
  обычные JSON без `_api`.

Перетащите workflow в ComfyUI, выберите свою LoRA и входную картинку. Проверьте
выбранные базовые веса, ширину/высоту результата и нажмите Queue. В workflow
оставлены явные имена-заглушки для будущей обученной LoRA и входного изображения;
до их выбора очередь закономерно не запустится.

Инструкция уже задана: **Create layered image from image 1**.
По умолчанию: 1024×1024, 40 шагов, Euler, CFG 1, LoRA strength 1.
CFG и steps можно менять; для сравнения с сэмплами тренера выставьте его значения.
`match_reference_area` должен соответствовать `Match target resolution` в обучении.
При выключении используется стандартный лимит reference 1024×1024 пикселей.

**Layer slots должны точно соответствовать обучению.** Они задаются в одной
ноде загрузки LoRA и передаются дальше автоматически. 20-слойный вариант
существенно дороже по VRAM. Уменьшайте разрешение, если модель не помещается;
смена числа слотов не является эквивалентной оптимизацией памяти.

## Ноды и результат

`QI2 Layer LoRA Loader` → `QI2 Joint Layer Sampler` → `QI2 Decode RGBA Layers`
→ `QI2 Save Layered PSD`. Нода `QI2 Layer Conditioning` подаёт текст и Control1
sampler-у. Загрузчики базовой модели, CLIP и VAE — штатные ComfyUI.

Sampler обрабатывает **один документ**. На каждом шаге все слои участвуют в
общем двунаправленном attention. VAE получает каждый слой отдельно, без
склейки в высокий холст и без временной компрессии. Batch на выходе decoder-а
содержит уже совместно сгенерированные слои снизу вверх.

Результат сохраняется в уникальную папку под `ComfyUI/output/QI2-Layers/`:

```text
layered_00001_<id>/
  layers.psd
  composite.png
  metadata.json
  layers/
    01.png
    02.png
    ...
```

PSD содержит Normal RGBA-слои `Layer 01`, `Layer 02`, … снизу вверх.
Пустые слоты сохраняются. PNG сохраняют альфа-канал; composite.png содержит
workflow-метаданные для повторного открытия. Выход `psd_path` сообщает полный
путь к PSD. `transparency_masks`: 1 означает прозрачность, 0 — непрозрачность.
Это растровые слои, без редактируемого текста, вектора и Photoshop-эффектов.

Для генерации без reference можно отсоединить `image` у conditioning-ноды и
написать описание сцены. Подготовленный PrismLayersPro-датасет обучает режим
**image-to-layers**; обучение text-to-layers требует соответствующих примеров.

## Совместимость и проверка

Реализация сверена с публичным ComfyUI:
`651ca296a73cd21c12a57eb8741d52e40dc6528f`,
[native QI2](https://github.com/Comfy-Org/ComfyUI/blob/651ca296a73cd21c12a57eb8741d52e40dc6528f/comfy/ldm/qwen_image21/model.py).
Пак проверяет семейство модели, сигнатуру sequence API, геометрию VAE,
отображение весов LoRA и число слотов, если оно есть в QDM metadata.
Если metadata отсутствует, число слотов необходимо указать по конфигу обучения:
из одних матриц LoRA восстановить его невозможно.

Меняется `build_sequence` только у клона ModelPatcher на время сэмплирования.
Глобальные классы и файлы ComfyUI не изменяются. Сохраняются штатные загрузка
LoRA, квантизация и управление памятью. Prefix cache отключён для этого пути.
Контракт v1 повторяет training RoPE, порядок токенов, общую target-маску,
нулевой timestep префикса и Euler schedule с dynamic shift и terminal 0.02.
Различия реализаций VAE/text encoder и вычислений ComfyUI могут давать
численные отличия от сэмплов тренера; побитовая идентичность не заявляется.

**Пройдены 11 CPU-проверок** в `tests/test_core.py` и `tests/test_nodes_contract.py`.
Проверяется attention-контракт,
RoPE относительно исходного класса тренера, schedule относительно Diffusers,
PSD round trip, границы нод и связи обоих workflow. Полная генерация с настоящими весами
и обученной LoRA ещё не проверена; качество разделения зависит от обучения.

Для разработчиков, из корня QDM:

```powershell
trainer/.venv/Scripts/python.exe -m unittest discover -s ComfyUI-QDM-QI2-Layers/tests -v
```

Внутренние зависимости ComfyUI этот requirements.txt не переустанавливает.
Diffusers нужен только для проверки schedule в dev-тесте, не для работы пака.
