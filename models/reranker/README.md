---
tags:
- sentence-transformers
- cross-encoder
- reranker
- generated_from_trainer
- dataset_size:17179
- loss:BinaryCrossEntropyLoss
base_model: cross-encoder/ms-marco-MiniLM-L6-v2
pipeline_tag: text-ranking
library_name: sentence-transformers
---

# CrossEncoder based on cross-encoder/ms-marco-MiniLM-L6-v2

This is a [Cross Encoder](https://www.sbert.net/docs/cross_encoder/usage/usage.html) model finetuned from [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) using the [sentence-transformers](https://www.SBERT.net) library. It computes scores for pairs of texts, which can be used for text reranking and semantic search.

## Model Details

### Model Description
- **Model Type:** Cross Encoder
- **Base model:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2) <!-- at revision c5ee24cb16019beea0893ab7796b1df96625c6b8 -->
- **Maximum Sequence Length:** 256 tokens
- **Number of Output Labels:** 1 label
- **Supported Modality:** Text
<!-- - **Training Dataset:** Unknown -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Documentation:** [Sentence Transformers Documentation](https://sbert.net)
- **Documentation:** [Cross Encoder Documentation](https://www.sbert.net/docs/cross_encoder/usage/usage.html)
- **Repository:** [Sentence Transformers on GitHub](https://github.com/huggingface/sentence-transformers)
- **Hugging Face:** [Cross Encoders on Hugging Face](https://huggingface.co/models?library=sentence-transformers&other=cross-encoder)

### Full Model Architecture

```
CrossEncoder(
  (0): Transformer({'transformer_task': 'sequence-classification', 'modality_config': {'text': {'method': 'forward', 'method_output_name': 'logits'}}, 'module_output_name': 'scores', 'architecture': 'BertForSequenceClassification'})
)
```

## Usage

### Direct Usage (Sentence Transformers)

First install the Sentence Transformers library:

```bash
pip install -U sentence-transformers
```

Then you can load this model and run inference.
```python
from sentence_transformers import CrossEncoder

# Download from the 🤗 Hub
model = CrossEncoder("cross_encoder_model_id")
# Get scores for pairs of inputs
pairs = [
    ['Вручают ли Нобелевкую премию посмертно?', 'Нобелевская премия\nТакже в §\xa04 указано, что премия не может быть присуждена посмертно. Однако, если претендент был жив в момент объявления о присуждении ему премии (обычно в октябре), но умер до церемонии вручения (10 декабря текущего года), то премия за ним сохраняется. Это правило принято в 1974 году, и до этого премия дважды присуждалась посмертно: Эрику Карлфельдту в 1931 году и Дагу Хаммаршёльду в 1961 году. Уже согласно принятому правилу премия сохранилась за Уильямом Викри, умершим в октябре 1996 года через несколько дней после присуждения ему премии. Однако в 2011 году правило было нарушено, когда по решению Нобелевского комитета Ральф Стейнман был награждён Нобелевской премией по физиологии или медицине посмертно, поскольку на момент присуждения премии нобелевский комитет считал его живым.'],
    ['На каком языке поет группа Би-2?', 'Би-2\nБи-2\xa0— российская альтернативная рок-группа белорусского происхождения, образованная в 1988 году в Бобруйске. Основатели и бессменные участники — Шура Би-2 (соло-гитара, вокал) и Лёва Би-2 (ведущий вокал, гитара, перкуссия). В состав команды также входят: Андрей Звонков (гитары), Макс Лакмус (бас-гитара), Борис Лифшиц (ударные, перкуссия) и Ян Николенко (бэк-вокал, клавишные, флейта, перкуссия). В данный момент Би-2 завершили работу над десятым студийным альбомом «Горизонт событий».'],
    ['Где находится штаб-квартира ООН?', 'Штаб-квартира ООН\nПомимо зданий в Нью-Йорке, ООН располагает ещё тремя вспомогательными региональными штаб-квартирами: в Женеве (Швейцария), Вене (Австрия) и Найроби (Кения). Однако наиболее важные решения организации принимаются именно в Нью-Йорке, так как здесь проходит большинство заседаний Генеральной Ассамблеи и Совета Безопасности ООН.'],
    ['Кто руководил вторжением шведов в Речь Посполитую в 1655−1660 годах?', 'Польско-шведская война (1621—1626)\nПоложение Речи Посполитой осложнялось враждой между гетманом польным литовским Христофором Радзивиллом и гетманом великим литовским Львом Сапегой: дело дошло до того, что их и без того немногочисленные войска воевали отдельно друг от друга. В октябре Радзивилл отбил несколько замков, а Сапега отразил шведскую атаку на Динабург. 12 октября Александр Гонсевский одержал победу в . Затем Радзивилл занялся обороной Курляндии, отправив Госевского организовывать партизанскую войну на захваченных шведами территориях Ливонии, а силы великого гетмана литовского, которыми командовал его сын Ян Станислав, стали в , между Кокенгаузеном и Бауском. Густав II Адольф решил воспользоваться разделением литовских войск, и разгромить их крупнейшую группировоку. 13 января 1626 года он собрал войска в и выступил через Кокенгаузен на Вальгоф. 17 января впервые в истории шведская армия в полевом сражении разгромила войска Речи Посполитой. После этого поражения литовское войско отступило и перешло к партизанской войне. Шведы также прекратили активные действия, сосредоточившись на обороне захваченной территории.'],
    ['Мари́я Румы́нская была последней королевой Румынии?', 'Мария Эдинбургская\nМария стала последней королевой Румынии, поскольку принцесса Елена была удостоена только титула королевы-матери между 1940 и 1947 годами. Мария также была одной из пяти коронованных внучек королевы Виктории и одной из трёх, сохранивших своё положение после окончания Первой мировой войны.'],
]
scores = model.predict(pairs)
print(scores)
# [ 1.211   2.1501  1.1047 -3.6451  1.7862]

# Or rank different texts based on similarity to a single text
ranks = model.rank(
    'Вручают ли Нобелевкую премию посмертно?',
    [
        'Нобелевская премия\nТакже в §\xa04 указано, что премия не может быть присуждена посмертно. Однако, если претендент был жив в момент объявления о присуждении ему премии (обычно в октябре), но умер до церемонии вручения (10 декабря текущего года), то премия за ним сохраняется. Это правило принято в 1974 году, и до этого премия дважды присуждалась посмертно: Эрику Карлфельдту в 1931 году и Дагу Хаммаршёльду в 1961 году. Уже согласно принятому правилу премия сохранилась за Уильямом Викри, умершим в октябре 1996 года через несколько дней после присуждения ему премии. Однако в 2011 году правило было нарушено, когда по решению Нобелевского комитета Ральф Стейнман был награждён Нобелевской премией по физиологии или медицине посмертно, поскольку на момент присуждения премии нобелевский комитет считал его живым.',
        'Би-2\nБи-2\xa0— российская альтернативная рок-группа белорусского происхождения, образованная в 1988 году в Бобруйске. Основатели и бессменные участники — Шура Би-2 (соло-гитара, вокал) и Лёва Би-2 (ведущий вокал, гитара, перкуссия). В состав команды также входят: Андрей Звонков (гитары), Макс Лакмус (бас-гитара), Борис Лифшиц (ударные, перкуссия) и Ян Николенко (бэк-вокал, клавишные, флейта, перкуссия). В данный момент Би-2 завершили работу над десятым студийным альбомом «Горизонт событий».',
        'Штаб-квартира ООН\nПомимо зданий в Нью-Йорке, ООН располагает ещё тремя вспомогательными региональными штаб-квартирами: в Женеве (Швейцария), Вене (Австрия) и Найроби (Кения). Однако наиболее важные решения организации принимаются именно в Нью-Йорке, так как здесь проходит большинство заседаний Генеральной Ассамблеи и Совета Безопасности ООН.',
        'Польско-шведская война (1621—1626)\nПоложение Речи Посполитой осложнялось враждой между гетманом польным литовским Христофором Радзивиллом и гетманом великим литовским Львом Сапегой: дело дошло до того, что их и без того немногочисленные войска воевали отдельно друг от друга. В октябре Радзивилл отбил несколько замков, а Сапега отразил шведскую атаку на Динабург. 12 октября Александр Гонсевский одержал победу в . Затем Радзивилл занялся обороной Курляндии, отправив Госевского организовывать партизанскую войну на захваченных шведами территориях Ливонии, а силы великого гетмана литовского, которыми командовал его сын Ян Станислав, стали в , между Кокенгаузеном и Бауском. Густав II Адольф решил воспользоваться разделением литовских войск, и разгромить их крупнейшую группировоку. 13 января 1626 года он собрал войска в и выступил через Кокенгаузен на Вальгоф. 17 января впервые в истории шведская армия в полевом сражении разгромила войска Речи Посполитой. После этого поражения литовское войско отступило и перешло к партизанской войне. Шведы также прекратили активные действия, сосредоточившись на обороне захваченной территории.',
        'Мария Эдинбургская\nМария стала последней королевой Румынии, поскольку принцесса Елена была удостоена только титула королевы-матери между 1940 и 1947 годами. Мария также была одной из пяти коронованных внучек королевы Виктории и одной из трёх, сохранивших своё положение после окончания Первой мировой войны.',
    ]
)
# [{'corpus_id': ..., 'score': ...}, {'corpus_id': ..., 'score': ...}, ...]
```

<!--
### Direct Usage (Transformers)

<details><summary>Click to see the direct usage in Transformers</summary>

</details>
-->

<!--
### Downstream Usage (Sentence Transformers)

You can finetune this model on your own dataset.

<details><summary>Click to expand</summary>

</details>
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Dataset

#### Unnamed Dataset

* Size: 17,179 training samples
* Columns: <code>sentence_0</code>, <code>sentence_1</code>, and <code>label</code>
* Approximate statistics based on the first 100 samples:
  |          | sentence_0                                                                         | sentence_1                                                                           | label                                                          |
  |:---------|:-----------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------|
  | type     | string                                                                             | string                                                                               | float                                                          |
  | modality | text                                                                               | text                                                                                 |                                                                |
  | details  | <ul><li>min: 16 tokens</li><li>mean: 37.75 tokens</li><li>max: 74 tokens</li></ul> | <ul><li>min: 66 tokens</li><li>mean: 243.92 tokens</li><li>max: 256 tokens</li></ul> | <ul><li>min: 0.0</li><li>mean: 0.62</li><li>max: 1.0</li></ul> |
* Samples:
  | sentence_0                                           | sentence_1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                | label            |
  |:-----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
  | <code>Вручают ли Нобелевкую премию посмертно?</code> | <code>Нобелевская премия<br>Также в § 4 указано, что премия не может быть присуждена посмертно. Однако, если претендент был жив в момент объявления о присуждении ему премии (обычно в октябре), но умер до церемонии вручения (10 декабря текущего года), то премия за ним сохраняется. Это правило принято в 1974 году, и до этого премия дважды присуждалась посмертно: Эрику Карлфельдту в 1931 году и Дагу Хаммаршёльду в 1961 году. Уже согласно принятому правилу премия сохранилась за Уильямом Викри, умершим в октябре 1996 года через несколько дней после присуждения ему премии. Однако в 2011 году правило было нарушено, когда по решению Нобелевского комитета Ральф Стейнман был награждён Нобелевской премией по физиологии или медицине посмертно, поскольку на момент присуждения премии нобелевский комитет считал его живым.</code> | <code>1.0</code> |
  | <code>На каком языке поет группа Би-2?</code>        | <code>Би-2<br>Би-2 — российская альтернативная рок-группа белорусского происхождения, образованная в 1988 году в Бобруйске. Основатели и бессменные участники — Шура Би-2 (соло-гитара, вокал) и Лёва Би-2 (ведущий вокал, гитара, перкуссия). В состав команды также входят: Андрей Звонков (гитары), Макс Лакмус (бас-гитара), Борис Лифшиц (ударные, перкуссия) и Ян Николенко (бэк-вокал, клавишные, флейта, перкуссия). В данный момент Би-2 завершили работу над десятым студийным альбомом «Горизонт событий».</code>                                                                                                                                                                                                                                                                                                                              | <code>0.0</code> |
  | <code>Где находится штаб-квартира ООН?</code>        | <code>Штаб-квартира ООН<br>Помимо зданий в Нью-Йорке, ООН располагает ещё тремя вспомогательными региональными штаб-квартирами: в Женеве (Швейцария), Вене (Австрия) и Найроби (Кения). Однако наиболее важные решения организации принимаются именно в Нью-Йорке, так как здесь проходит большинство заседаний Генеральной Ассамблеи и Совета Безопасности ООН.</code>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | <code>1.0</code> |
* Loss: [<code>BinaryCrossEntropyLoss</code>](https://sbert.net/docs/package_reference/cross_encoder/losses.html#binarycrossentropyloss) with these parameters:
  ```json
  {
      "activation_fn": "torch.nn.modules.linear.Identity",
      "pos_weight": null
  }
  ```

### Training Hyperparameters
#### Non-Default Hyperparameters

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 10
- `per_device_eval_batch_size`: 16

#### All Hyperparameters
<details><summary>Click to expand</summary>

- `per_device_train_batch_size`: 16
- `num_train_epochs`: 10
- `max_steps`: -1
- `learning_rate`: 5e-05
- `lr_scheduler_type`: linear
- `lr_scheduler_kwargs`: None
- `warmup_steps`: 0
- `optim`: adamw_torch_fused
- `optim_args`: None
- `weight_decay`: 0.0
- `adam_beta1`: 0.9
- `adam_beta2`: 0.999
- `adam_epsilon`: 1e-08
- `optim_target_modules`: None
- `gradient_accumulation_steps`: 1
- `average_tokens_across_devices`: True
- `max_grad_norm`: 1
- `label_smoothing_factor`: 0.0
- `bf16`: False
- `fp16`: False
- `bf16_full_eval`: False
- `fp16_full_eval`: False
- `tf32`: None
- `gradient_checkpointing`: False
- `gradient_checkpointing_kwargs`: None
- `torch_compile`: False
- `torch_compile_backend`: None
- `torch_compile_mode`: None
- `use_liger_kernel`: False
- `liger_kernel_config`: None
- `use_cache`: False
- `neftune_noise_alpha`: None
- `torch_empty_cache_steps`: None
- `auto_find_batch_size`: False
- `log_on_each_node`: True
- `logging_nan_inf_filter`: True
- `include_num_input_tokens_seen`: no
- `log_level`: passive
- `log_level_replica`: warning
- `disable_tqdm`: False
- `project`: huggingface
- `trackio_space_id`: None
- `trackio_bucket_id`: None
- `trackio_static_space_id`: None
- `per_device_eval_batch_size`: 16
- `prediction_loss_only`: True
- `eval_on_start`: False
- `eval_do_concat_batches`: True
- `eval_use_gather_object`: False
- `eval_accumulation_steps`: None
- `include_for_metrics`: []
- `batch_eval_metrics`: False
- `save_only_model`: False
- `save_on_each_node`: False
- `enable_jit_checkpoint`: False
- `push_to_hub`: False
- `hub_private_repo`: None
- `hub_model_id`: None
- `hub_strategy`: every_save
- `hub_always_push`: False
- `hub_revision`: None
- `load_best_model_at_end`: False
- `ignore_data_skip`: False
- `restore_callback_states_from_checkpoint`: False
- `full_determinism`: False
- `seed`: 42
- `data_seed`: None
- `use_cpu`: False
- `accelerator_config`: {'split_batches': False, 'dispatch_batches': None, 'even_batches': True, 'use_seedable_sampler': True, 'non_blocking': False, 'gradient_accumulation_kwargs': None}
- `parallelism_config`: None
- `dataloader_drop_last`: False
- `dataloader_num_workers`: 0
- `dataloader_pin_memory`: True
- `dataloader_persistent_workers`: False
- `dataloader_prefetch_factor`: None
- `remove_unused_columns`: True
- `label_names`: None
- `train_sampling_strategy`: random
- `length_column_name`: length
- `ddp_find_unused_parameters`: None
- `ddp_bucket_cap_mb`: None
- `ddp_broadcast_buffers`: False
- `ddp_static_graph`: None
- `ddp_backend`: None
- `ddp_timeout`: 1800
- `fsdp`: []
- `fsdp_config`: {'min_num_params': 0, 'xla': False, 'xla_fsdp_v2': False, 'xla_fsdp_grad_ckpt': False}
- `deepspeed`: None
- `debug`: []
- `skip_memory_metrics`: True
- `do_predict`: False
- `resume_from_checkpoint`: None
- `warmup_ratio`: None
- `local_rank`: -1
- `prompts`: None
- `batch_sampler`: batch_sampler
- `multi_dataset_batch_sampler`: proportional
- `router_mapping`: {}
- `learning_rate_mapping`: {}

</details>

### Training Logs
| Epoch  | Step  | Training Loss |
|:------:|:-----:|:-------------:|
| 0.4655 | 500   | 1.0373        |
| 0.9311 | 1000  | 0.6890        |
| 1.3966 | 1500  | 0.6772        |
| 1.8622 | 2000  | 0.6663        |
| 2.3277 | 2500  | 0.6556        |
| 2.7933 | 3000  | 0.6472        |
| 3.2588 | 3500  | 0.6448        |
| 3.7244 | 4000  | 0.6232        |
| 4.1899 | 4500  | 0.6236        |
| 4.6555 | 5000  | 0.6108        |
| 5.1210 | 5500  | 0.6084        |
| 5.5866 | 6000  | 0.5994        |
| 6.0521 | 6500  | 0.5793        |
| 6.5177 | 7000  | 0.5702        |
| 6.9832 | 7500  | 0.5752        |
| 7.4488 | 8000  | 0.5466        |
| 7.9143 | 8500  | 0.5594        |
| 8.3799 | 9000  | 0.5375        |
| 8.8454 | 9500  | 0.5362        |
| 9.3110 | 10000 | 0.5300        |
| 9.7765 | 10500 | 0.5235        |


### Training Time
- **Training**: 29.2 minutes

### Framework Versions
- Python: 3.14.0
- Sentence Transformers: 5.5.1
- Transformers: 5.7.0
- PyTorch: 2.11.0+cu130
- Accelerate: 1.13.0
- Datasets: 4.8.5
- Tokenizers: 0.22.2

## Additional Resources

- [Training and Finetuning Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-reranker): the end-to-end guide for training or finetuning Cross Encoder (reranker) models.
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers): use text, image, audio, and video reranker models through the same API.
- [Training and Finetuning Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/train-multimodal-sentence-transformers): training multimodal Cross Encoders.

## Citation

### BibTeX

#### Sentence Transformers
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->