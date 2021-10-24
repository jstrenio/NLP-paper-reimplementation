# Group: John Strenio, Scott Klinn, Tuan Nguyen
# Commonsense QA Finetuning v4
# CS 510: Adventures in NLP
# Professor Ameeta Agrawal
# Contributors: John Strenio

# =========== Summary & Instructions ==========================
# This program is a modified version of the 
# commonsense_finetuned.ipynb notebook file specifically 
# designed to train 1 or 2 models and avoid the system crashes
# that I was experiencing attempting to train on the notebook.
# This program takes advantage of HuggingFace's AutoModels and
# AutoTokenizers so the only required input is the number of 
# models directly below, and the desired model checkpoints
# and directories for saving them. This program takes a pre-
# trained bert model and finetunes it on the commonsenseQA
# dataset using a custom training method of a single correct
# and single incorrect answer option for all the questions
# in CQA using a next sentence prediction head. Train:Valid 
# ratio is 9741:1221
# =============================================================

number_of_models = 1
# import datasets and model
for i in range(number_of_models):

    if i == 0:
        model_checkpoint = 'distilbert-base-uncased'
        save_directory = 'D:/project/saved_model1'
    else:
        model_checkpoint = 'bert-base-uncased'
        save_directory = 'D:/project/saved_model2'

    from datasets import load_dataset, load_metric, Dataset

    # loads the dataset (downloads it if you don't have it)
    dataset = load_dataset("commonsense_qa")

    # import a tokenizer the auto is just making sure it fits our model, it should just be selecting DistilBertTokenizerFast
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # make sure its an optimized one thats fast if available
    import transformers
    import random
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    def build_sentences(example):
        # each example is going to output 2 examples, 1 for correct, 1 for wrong (originally did one for each possible answer)
        first_sentences = []
        second_sentences = []

        # extract the question, choices and correct answer
        question = example['question']
        ans_text = [choice for choice in example['choices']['text']]
        choices = []

        # the correct answer will be first, the other answer will be selected randomly from whats left
        choices.append(ans_text.pop(ord(example['answerKey']) - 65))
        choices.append(random.choice(ans_text))

        # 1st sentence is question, 2nd is choices for NSP
        for i in range(len(choices)):
            first_sentences.append(question)
            second_sentences.append(choices[i])

        # we're choosing to take a correct (0 label) and an incorrect (1 label) from each example
        labels = [0, 1]

        return first_sentences, second_sentences, labels

    # collect the question/answer sentences and encode them into their representations
    def encode_dataset(dataset, dset_size):
        first_sentences_to_encode = []
        second_sentences_to_encode = []
        labels_to_encode = []

        for i in range(dset_size):
            first_sentences, second_sentences, labels = build_sentences(dataset[i])
            first_sentences_to_encode += first_sentences
            second_sentences_to_encode += second_sentences
            labels_to_encode += labels

        encodings = tokenizer(first_sentences_to_encode, second_sentences_to_encode, return_tensors='pt', padding="longest", truncation=True)
        
        return encodings, torch.LongTensor(labels_to_encode)

    import torch

    # define a custom dataset to convert our encodings into something bert can work with
    class QA_Dataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    dataset['train'] = Dataset.shuffle(dataset['train'])
    dataset['validation'] = Dataset.shuffle(dataset['validation'])
    dataset['test'] = Dataset.shuffle(dataset['test'])

    # I've been adjusting the dataset sizes to try to better loss without having to retrain for 6 hours.
    encodings, encoded_labels = encode_dataset(dataset['train'], 100) #len(dataset['train']))
    encodings2, encoded_labels2 = encode_dataset(dataset['validation'], 20) #len(dataset['validation']))

    train_dataset = QA_Dataset(encodings, torch.LongTensor(encoded_labels))
    val_dataset = QA_Dataset(encodings2, torch.LongTensor(encoded_labels2))

    # fine tuning (the warning just references that we're removing the classification head for finetuning)
    from transformers import BertForNextSentencePrediction, TrainingArguments, Trainer
    model =  BertForNextSentencePrediction.from_pretrained(model_checkpoint)

    # training arguments
    batch_size = 2

    if i == 0:
        args = TrainingArguments(
            f"model1_checkpoints",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=5
        )
    else:
        args = TrainingArguments(
            f"model2_checkpoints",
            evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=3,
            weight_decay=0.01,
            save_total_limit=5
        )

    # data collator
    from transformers import default_data_collator
    data_collator = default_data_collator

    # build trainer
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # this was the bandaid I found online for getting the batching to be accepted
    import torch
    torch.cuda.empty_cache()
    import gc
    #del variables
    gc.collect()

    # training
    output = trainer.train()
    print(output)

    model.save_pretrained(save_directory)


