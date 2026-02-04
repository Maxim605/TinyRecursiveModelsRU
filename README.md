# Меньше значит больше: Рекурсивное рассуждение с маленькими сетями

Это репозиторий для статьи: "Less is More: Recursive Reasoning with Tiny Networks". TRM — это подход рекурсивного рассуждения, который достигает впечатляющих результатов: 45% на ARC-AGI-1 и 8% на ARC-AGI-2, используя небольшую нейронную сеть всего с 7M параметров.

[Статья](https://arxiv.org/abs/2510.04871)

### Мотивация

Tiny Recursion Model (TRM) — это модель рекурсивного рассуждения, которая достигает впечатляющих результатов: 45% на ARC-AGI-1 и 8% на ARC-AGI-2, используя небольшую нейронную сеть всего с 7M параметров. Идея о том, что для успеха на сложных задачах необходимо полагаться на массивные базовые модели, обученные за миллионы долларов крупными корпорациями, является ловушкой. В настоящее время слишком много внимания уделяется использованию LLM, а не разработке и расширению новых направлений. С рекурсивным рассуждением оказывается, что "меньше значит больше": не всегда нужно увеличивать размер модели, чтобы она могла рассуждать и решать сложные задачи. Небольшая модель, предобученная с нуля, рекурсивно обрабатывающая саму себя и обновляющая свои ответы со временем, может достичь многого без больших затрат.

Эта работа появилась после того, как я узнал о недавней инновационной модели Hierarchical Reasoning Model (HRM). Я была поражена тем, что подход с использованием небольших моделей может так хорошо работать на сложных задачах, таких как соревнование ARC-AGI (достигая 40% точности, когда обычно только большие языковые модели могли конкурировать). Но я продолжала думать, что это слишком сложно, слишком полагается на биологические аргументы о человеческом мозге, и что этот процесс рекурсивного рассуждения можно значительно упростить и улучшить. Tiny Recursion Model (TRM) упрощает рекурсивное рассуждение до его сути, которая в конечном итоге не имеет ничего общего с человеческим мозгом, не требует никаких математических теорем (о неподвижной точке), ни какой-либо иерархии.

### Как работает TRM

<p align="center">
  <img src="https://AlexiaJM.github.io/assets/images/TRM_fig.png" alt="TRM"  style="width: 30%;">
</p>

Tiny Recursion Model (TRM) рекурсивно улучшает свой предсказанный ответ y с помощью небольшой сети. Она начинается с эмбеддинга входного вопроса x и начального эмбеддинга ответа y и латентного z. До K шагов улучшения она пытается улучшить свой ответ y. Это делается путем i) рекурсивного обновления n раз латентного z с учетом вопроса x, текущего ответа y и текущего латентного z (рекурсивное рассуждение), а затем ii) обновления ответа y с учетом текущего ответа y и текущего латентного z. Этот рекурсивный процесс позволяет модели постепенно улучшать свой ответ (потенциально исправляя любые ошибки из предыдущего ответа) чрезвычайно эффективным по параметрам способом, минимизируя переобучение.

### Требования

Установка должна занять несколько минут. Для самых маленьких экспериментов на Sudoku-Extreme (pretrain_mlp_t_sudoku) вам нужна 1 GPU с достаточным объемом памяти. С 1 L40S (48Gb RAM) это займет около 18 часов. Если у вас возникнут проблемы из-за версий библиотек, вот требования с точными версиями: [specific_requirements.txt](https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/specific_requirements.txt).

- Python 3.10 (или аналогичный)
- Cuda 12.6.0 (или аналогичный)

```bash
pip install --upgrade pip wheel setuptools
pip install --pre --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126 # установка torch в зависимости от вашей версии cuda
pip install -r requirements.txt # установка зависимостей
pip install --no-cache-dir --no-build-isolation adam-atan2 
wandb login YOUR-LOGIN # вход, если вы хотите, чтобы логирование синхронизировало результаты с вашим Weights & Biases (https://wandb.ai/)
```

### Подготовка датасетов

```bash
# ARC-AGI-1
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc1concept-aug-1000 \
  --subsets training evaluation concept \
  --test-set-name evaluation

# ARC-AGI-2
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir data/arc2concept-aug-1000 \
  --subsets training2 evaluation2 concept \
  --test-set-name evaluation2

## Примечание: Вы не можете обучать на обоих ARC-AGI-1 и ARC-AGI-2 и оценивать их оба, потому что обучающие данные ARC-AGI-2 содержат некоторые оценочные данные ARC-AGI-1

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 примеров, 1000 аугментаций

# Maze-Hard
python dataset/build_maze_dataset.py # 1000 примеров, 8 аугментаций
```

## Эксперименты

### Sudoku-Extreme (при условии 1 L40S GPU):

```bash
run_name="pretrain_mlp_t_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.mlp_t=True arch.pos_encodings=none \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True

Ожидаемый результат: Около 87% точной точности (+- 2%)

run_name="pretrain_att_sudoku"
python pretrain.py \
arch=trm \
data_paths="[data/sudoku-extreme-1k-aug-1000]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=6 \
+run_name=${run_name} ema=True
```

Ожидаемый результат: Около 75% точной точности (+- 2%)

*Время выполнения:* < 20 часов

### Maze-Hard (при условии 4 L40S GPU):

```bash
run_name="pretrain_att_maze30x30"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Время выполнения:* < 24 часов

На самом деле, вы можете запустить Maze-Hard с 1 L40S GPU, уменьшив размер батча без заметной потери производительности:

```bash
run_name="pretrain_att_maze30x30_1gpu"
python pretrain.py \
arch=trm \
data_paths="[data/maze-30x30-hard-1k]" \
evaluators="[]" \
epochs=50000 eval_interval=5000 \
lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True
```

*Время выполнения:* < 24 часов


### ARC-AGI-1 (при условии 4 H-100 GPU):

```bash
run_name="pretrain_att_arc1concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc1concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Время выполнения:* ~3 дня

### ARC-AGI-2 (при условии 4 H-100 GPU):

```bash
run_name="pretrain_att_arc2concept_4"
torchrun --nproc-per-node 4 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py \
arch=trm \
data_paths="[data/arc2concept-aug-1000]" \
arch.L_layers=2 \
arch.H_cycles=3 arch.L_cycles=4 \
+run_name=${run_name} ema=True

```

*Время выполнения:* ~3 дня


## Ссылки

Если вы найдете нашу работу полезной, пожалуйста, рассмотрите возможность цитирования:

```bibtex
@misc{jolicoeurmartineau2025morerecursivereasoningtiny,
      title={Less is More: Recursive Reasoning with Tiny Networks}, 
      author={Alexia Jolicoeur-Martineau},
      year={2025},
      eprint={2510.04871},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04871}, 
}
```

и иерархической модели рассуждения (HRM):

```bibtex
@misc{wang2025hierarchicalreasoningmodel,
      title={Hierarchical Reasoning Model}, 
      author={Guan Wang and Jin Li and Yuhao Sun and Xing Chen and Changling Liu and Yue Wu and Meng Lu and Sen Song and Yasin Abbasi Yadkori},
      year={2025},
      eprint={2506.21734},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2506.21734}, 
}
```

Этот код основан на иерархической модели рассуждения [код](https://github.com/sapientinc/HRM) и анализе иерархической модели рассуждения [код](https://github.com/arcprize/hierarchical-reasoning-model-analysis).
