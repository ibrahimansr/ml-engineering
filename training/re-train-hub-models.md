# Re-Train HF Hub Models From Scratch Using Finetuning Examples

HF Transformers has awesome finetuning examples  https://github.com/huggingface/transformers/tree/main/examples/pytorch, that cover pretty much any modality and these examples work out of box.

**But what if you wanted to re-train from scratch rather than finetune.**

Here is a simple hack to accomplish that.

We will use `facebook/opt-1.3b` and we will plan to use bf16 training regime as an example here. The trick is that `AutoConfig.from_pretrained` downloads only the model's architecture config (layer count, hidden size, vocab size, etc.), and `AutoModelForCausalLM.from_config` builds that architecture with freshly initialized random weights - unlike `AutoModelForCausalLM.from_pretrained`, which would also load the trained weights. The tokenizer is reused as is. (The `cat << EOT > prep-bf16.py ... EOT` wrapper is a shell heredoc that writes everything between the two `EOT` markers into `prep-bf16.py`.)

```bash
cat << EOT > prep-bf16.py
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch

mname = "facebook/opt-1.3b"

config = AutoConfig.from_pretrained(mname)
model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(mname)

path = "opt-1.3b-bf16"

model.save_pretrained(path)
tokenizer.save_pretrained(path)
EOT
```

now run:

```bash
python prep-bf16.py
```

This will create a folder: `opt-1.3b-bf16` with everything you need to train the model from scratch. In other words you have a pretrained-like model, except it only had its initializations done and none of the training yet.

Adjust the script above to use `torch.float16` or `torch.float32` if that's what you plan to use instead.

Now you can proceed with finetuning this saved model as normal. `run_clm.py` (causal language modeling) is one of the example scripts from the HF Transformers repo linked above, so run this from inside a clone of that repo (and point `--model_name_or_path` at wherever you created `opt-1.3b-bf16`). Its flags are the HF `Trainer`'s `TrainingArguments`: `--bf16` enables bf16 mixed precision, `--dataset_name`/`--dataset_config_name` pick a dataset from the HF Hub, `--report_to tensorboard` logs to TensorBoard, and the rest are the usual batch size, optimizer and LR-schedule hyper-parameters:

```bash
torchrun \
--nproc_per_node=1 --nnode=1 --node_rank=0 \
--master_addr=127.0.0.1 --master_port=9901 \
examples/pytorch/language-modeling/run_clm.py --bf16 \
--seed 42 --model_name_or_path opt-1.3b-bf16 \
--dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
--per_device_train_batch_size 12 --per_device_eval_batch_size 12 \
--gradient_accumulation_steps 1 --do_train --do_eval --logging_steps 10 \
--save_steps 1000 --eval_steps 100 --weight_decay 0.1 --num_train_epochs 1 \
--adam_beta1 0.9 --adam_beta2 0.95 --learning_rate 0.0002 --lr_scheduler_type \
linear --warmup_steps 500 --report_to tensorboard --output_dir save_dir
```

The key entry being:
```
--model_name_or_path opt-1.3b-bf16
```

where `opt-1.3b-bf16` is your local directory you have just generated in the previous step.

Sometimes it's possible to find the same dataset that the original model was trained on, sometimes you have to use an alternative dataset.

The rest of the hyper-parameters can often be found in the paper or documentation that came with the model.

To summarize, this recipe allows you to use finetuning examples to re-train whatever model you can find on [the HF hub](https://huggingface.co/models).
