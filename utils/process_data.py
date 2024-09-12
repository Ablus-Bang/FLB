from datasets import load_dataset


def get_dataset(dataset_name):
    if dataset_name in ["gsm8k"]:
        dataset = load_dataset(dataset_name, split="train", name="main")
    elif dataset_name in ["lighteval/MATH"]:
        dataset = load_dataset(dataset_name, split="train", name="all")
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        dataset = load_dataset(dataset_name, split="train_sft")
    else:
        dataset = load_dataset(dataset_name, split="train")
    return dataset


def apply_chat_template(
    example,
    tokenizer,
):
    if "instruction" in example.keys():
        example["input"] = example["instruction"] + " " + example["input"]

    messages = [
        {"role": "user", "content": example["input"]},
        {"role": "assistant", "content": example["output"]},
    ]
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


def process_dataset_for_unified_format(dataset_name, dataset, tokenizer, seed=1234):
    if dataset_name in ["lucasmccabe-lmi/CodeAlpaca-20k",
                        "yahma/alpaca-cleaned",
                        "FinGPT/fingpt-sentiment-train",
                        "WizardLM/WizardLM_evol_instruct_70k",
                        "tatsu-lab/alpaca",
                        "vicgalle/alpaca-gpt4",
                        "gbharti/finance-alpaca",
                        "TIGER-Lab/MathInstruct",
                        "lighteval/MATH",
                        "gsm8k",
                        "medalpaca/medical_meadow_medical_flashcards"]:
        if dataset_name in ["WizardLM/WizardLM_evol_instruct_70k", "TIGER-Lab/MathInstruct"]:
            dataset = dataset.rename_column("instruction", "input")
        if dataset_name in ["lighteval/MATH"]:
            dataset = dataset.rename_column("solution", "output")
            dataset = dataset.rename_column("problem", "input")
        if dataset_name in ["gsm8k"]:
            dataset = dataset.rename_column("question", "input")
            dataset = dataset.rename_column("answer", "output")

    column_names = list(dataset.features)
    if 'input' not in column_names or 'output' not in column_names:
        raise ValueError(f"Invalid dataset format. The dataset {dataset_name} must contain 'input' and 'output' columns.")
    processed_dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )
    processed_dataset = processed_dataset.shuffle(seed=seed)
    return processed_dataset


def split_dataset(dataset, num_clients, seed=1234):
    dataset = dataset.shuffle(seed=seed)  # Shuffle the dataset
    local_datasets = []
    for i in range(num_clients):
        local_datasets.append(dataset.shard(num_clients, i))
    return local_datasets


def build_dataset(
    tokenizer, datasetname, num_clients, test_size=0.1, seed=1234, dataset_sample=200
):
    trainset_full = load_dataset(datasetname, split="train")
    train_test = trainset_full.train_test_split(test_size=test_size, seed=seed)
    train_dataset = train_test["train"]
    train_dataset = train_dataset.shuffle(seed=seed)
    if dataset_sample:
        num_sample = min(len(train_dataset), dataset_sample)
        train_dataset = train_dataset.select(range(num_sample))
    test_dataset = train_test["test"]
    column_names = list(train_dataset.features)
    processed_train_dataset = train_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to train_sft",
    )

    processed_test_dataset = test_dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=10,
        remove_columns=column_names,
        desc="Applying chat template to test_sft",
    )
    train_dataset_split = split_dataset(processed_train_dataset, num_clients, seed)
    return train_dataset_split, processed_test_dataset
