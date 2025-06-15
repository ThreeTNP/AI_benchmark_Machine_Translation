from unsloth import FastModel
import torch
import pandas as pd
from tqdm import tqdm
from unsloth.chat_templates import get_chat_template
from unsloth.chat_templates import standardize_sharegpt
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth.chat_templates import train_on_responses_only


train_set = []
test_set = []
result = []
true_answer = []


model, tokenizer = FastModel.from_pretrained(
    model_name = "./med_gemma3",
    max_seq_length = 1024,   # Context length - can be longer, but uses more memory
    load_in_4bit = True,     # 4bit uses much less memory
    load_in_8bit = False,    # A bit more accurate, uses 2x memory
    full_finetuning = False, # We have full finetuning now!
    # token = "hf_...",      # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 128,           # Larger = higher accuracy, but might overfit
    lora_alpha = 128,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)


data_train = pd.read_csv('train_mt.csv')
data_val = pd.read_csv('val_mt.csv')
data_test = pd.read_csv('test_mt.csv')

all_data = pd.concat([data_train, data_val], axis=0)
all_data = all_data.reset_index()
all_data = all_data.drop(columns=['index'])


for i, row in tqdm(all_data.iterrows()):
    train_set.append({'conversations': [
        {'content':
f'''Translate the following Chinese sentence to Thai about Medical Domain Specific, the sentence is from doctor and user context:
{row['source']}
''', 'role': 'user'},
        {'content': row['translation'], 'role': 'assistant'}
    ]})


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

dataset = standardize_sharegpt(Dataset.from_list(train_set))
non_reasoning_conversations = tokenizer.apply_chat_template(
    dataset["conversations"],
    tokenize = False,
)

df = pd.Series(non_reasoning_conversations)
df.name = "text"

combined_dataset = Dataset.from_pandas(pd.DataFrame(df))
combined_dataset = combined_dataset.shuffle(seed=227)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset = None, # Can set up evaluation!
    args = SFTConfig(
        dataset_text_field = "text",
        per_device_train_batch_size = 32,
        gradient_accumulation_steps = 2, # Use GA to mimic batch size!
        warmup_steps = 5,
        num_train_epochs = 2, # Set this for 1 full training run.
        #max_steps = 30,
        learning_rate = 2e-4, # Reduce to 2e-5 for long training runs
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 227,
        report_to = "none", # Use this for WandB etc
        dataset_num_proc=2,
        output_dir = "outputs_mt_ver6",
    ),
)


trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer_stats = trainer.train()


for i, row in tqdm(data_test.iterrows()):
    test_set.append({'conversations': [
        {'content':
f'''Translate the following Chinese sentence to Thai about Medical Domain Specific, the sentence is from doctor and user context:
{row['source']}
''', 'role': 'user'}
    ]})

test_dataset = Dataset.from_list(test_set)
test_dataset = tokenizer.apply_chat_template(
        test_dataset['conversations'],
        tokenize = False,
        add_generation_prompt = True, # Must add for generation
        enable_thinking = False, # Disable thinking
    )


for text in tqdm(test_dataset):
    answer = model.generate(
        **tokenizer(text, return_tensors = "pt").to("cuda"),
        max_new_tokens = 512, # Increase for longer outputs!
        temperature = 0.1, top_p = 0.1, top_k = 1, # For non thinking
    )

    decoded = tokenizer.decode(answer[0], skip_special_tokens=True)
    result.append(decoded)

for answer in tqdm(result):
    true_answer.append(answer.split('model\n')[-1])

data_test['translation'] = true_answer

with open("submission_medgemma_ver3.txt", "w", encoding="utf-8") as f:
    for item in data_test["translation"]:
        f.write(item + "\n")

print("Sucessfull Pipeline")
