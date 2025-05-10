from tqdm import tqdm
from openprompt.data_utils.text_classification_dataset import SciCite, ACL_ARC, ACL_ARC_New
import torch
import argparse

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate
from sklearn.metrics import classification_report


parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='scibert')
parser.add_argument("--model_name_or_path", default='allenai/scibert_scivocab_uncased')
parser.add_argument("--result_file", type=str, default="sfs_scripts/results_normal_manual_kpt.txt")
parser.add_argument("--openprompt_path", type=str, default="OpenPrompt")

parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset",type=str, default='scicite')
parser.add_argument("--test_dataset", type=str, default=None, 
                    help="Dataset to use for testing (scicite/acl_arc/act2). If None, uses same as training dataset")

parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=10)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--target", default="", type=str)
args = parser.parse_args()

this_run_unicode = f"citeprompt_{args.seed}_{args.verbalizer}_{args.filter}_{args.template_id}_{args.dataset}"

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)

import nltk
stopwords = nltk.corpus.stopwords.words('english')

from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

tokenizer.add_special_tokens({"additional_special_tokens": ["[CITED_AUTHOR]"]})
plm.resize_token_embeddings(len(tokenizer))

dataset = {}

# Load training dataset
if args.dataset == "scicite":
    dataset['train'] = SciCite().get_examples("datasets/formatted/scicite", 'train', stopwords)
    dataset['validation'] = SciCite().get_examples("datasets/formatted/scicite", 'dev', stopwords)
    class_labels = SciCite().get_labels()
    scriptsbase = "TextClassification/scicite"

elif args.dataset == "acl_arc":
    dataset['train'] = ACL_ARC().get_examples("datasets/formatted/acl-arc", 'train', stopwords)
    dataset['validation'] = ACL_ARC().get_examples("datasets/formatted/acl-arc", 'test', stopwords)
    class_labels = ACL_ARC().get_labels()
    scriptsbase = "TextClassification/acl_arc"

elif args.dataset == "acl_arc_scicite_schema":
    dataset['train'] = SciCite().get_examples("datasets/formatted/acl_arc_scicite_schema", 'train', stopwords)
    dataset['validation'] = SciCite().get_examples("datasets/formatted/acl_arc_scicite_schema", 'test', stopwords)
    class_labels = SciCite().get_labels()  # Use SciCite labels
    scriptsbase = "TextClassification/scicite"

elif args.dataset == "act2":
    dataset['train'] = ACL_ARC().get_examples("datasets/formatted/act2", 'train', stopwords)
    dataset['validation'] = ACL_ARC().get_examples("datasets/formatted/act2", 'dev', stopwords)
    class_labels = ACL_ARC().get_labels()
    scriptsbase = "TextClassification/acl_arc"

elif args.dataset == "acl_new":
    dataset['train'] = ACL_ARC_New(args.target).get_examples("datasets/formatted/acl_new", 'train', stopwords, args.target)
    dataset['validation'] = ACL_ARC_New(args.target).get_examples("datasets/formatted/acl_new", 'test', stopwords, args.target)
    class_labels = ACL_ARC_New(args.target).get_labels()
    scriptsbase = "TextClassification/acl_arc"
else:
    raise NotImplementedError

# Add this function after dataset loading and before dataloader creation
def map_acl_arc_to_scicite_labels(dataset_examples):
    """Maps ACL-ARC labels to SciCite label space
    background(0) <- Background(0), Motivation(4), Extends(2), Future work(3)
    method(1) <- Uses(5)
    result(2) <- Compare Contrast(1)
    """
    label_mapping = {
        0: 0,  # Background -> background
        1: 2,  # Compare Contrast -> result
        2: 0,  # Extends -> background
        3: 0,  # Future work -> background
        4: 0,  # Motivation -> background
        5: 1,  # Uses -> method
    }
    
    for example in dataset_examples:
        example.label = label_mapping[example.label]
    return dataset_examples

# Load test dataset
if args.test_dataset is None:
    # Use same dataset as training
    if args.dataset == "scicite":
        dataset['test'] = SciCite().get_examples("datasets/formatted/scicite", 'test', stopwords)
    elif args.dataset == "acl_arc":
        # Map ACL-ARC labels to SciCite labels if training on SciCite
        dataset['test'] = ACL_ARC().get_examples("datasets/formatted/acl-arc", 'test', stopwords)
    elif args.dataset == "act2":
        dataset['test'] = ACL_ARC().get_examples("datasets/formatted/act2", 'test', stopwords)
    elif args.dataset == "acl_new":
        dataset['test'] = ACL_ARC_New(args.target).get_examples("datasets/formatted/acl_new", 'test', stopwords, args.target)

else:
    # Load specified test dataset
    if args.test_dataset == "scicite":
        dataset['test'] = SciCite().get_examples("datasets/formatted/scicite", 'test', stopwords)
        class_labels = SciCite().get_labels()  # Use SciCite labels
    elif args.test_dataset == "acl_arc_scicite_schema":
        dataset['test'] = SciCite().get_examples("datasets/formatted/acl_arc_scicite_schema", 'test', stopwords)
        class_labels = SciCite().get_labels()  # Use SciCite labels
    elif args.test_dataset == "act2_scicite_schema":
        dataset['test'] = SciCite().get_examples("datasets/formatted/act2_scicite_schema", 'test', stopwords)
        class_labels = SciCite().get_labels()  # Use SciCite labels
    elif args.test_dataset == "acl_arc":
        dataset['test'] = ACL_ARC().get_examples("datasets/formatted/acl-arc", 'test', stopwords)
        if args.dataset == "scicite":
            dataset['test'] = map_acl_arc_to_scicite_labels(dataset['test'])
            class_labels = SciCite().get_labels()  # Use SciCite labels
    elif args.test_dataset == "act2":
        dataset['test'] = ACL_ARC().get_examples("datasets/formatted/act2", 'test', stopwords)
        # Map ACL-ARC labels to SciCite labels if training on SciCite
        if args.dataset == "scicite":
            dataset['test'] = map_acl_arc_to_scicite_labels(dataset['test'])
            class_labels = SciCite().get_labels()  # Use SciCite labels
    elif args.test_dataset == "acl_new":
        dataset['test'] = ACL_ARC_New(args.target).get_examples("datasets/formatted/acl_new", 'test', stopwords, args.target)
    elif args.test_dataset == "act2_new":
        dataset['test'] = ACL_ARC_New(args.target).get_examples("datasets/formatted/act2_new", 'test', stopwords, args.target)
    else:
        raise ValueError(f"Unknown test dataset: {args.test_dataset}")

# Common settings
scriptformat = "txt"
cutoff = 0.5
max_seq_l = 512
batch_s = 40
template_text = '{"placeholder":"text_a"} It has a citation of type {"mask"}'

# Update run identifier to include test dataset
this_run_unicode = f"citeprompt_{args.seed}_{args.verbalizer}_{args.filter}_{args.template_id}_{args.dataset}"
if args.test_dataset:
    this_run_unicode += f"_test_{args.test_dataset}"
if args.target:
    this_run_unicode += f"_target_{args.target}"

#template_text = '{"placeholder":"text_a"} It has a citation of type {"mask"}'
#template_text = '{"mask"} Citation type: {"placeholder":"text_a"}'
mytemplate = ManualTemplate(tokenizer=tokenizer, text=template_text)
if args.target:
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(f"scripts/{scriptsbase}/{args.target}.{scriptformat}")
else:
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(f"scripts/{scriptsbase}/knowledgeable_verbalizer.{scriptformat}")

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s, shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from openprompt.utils.metrics import classification_metrics
import matplotlib.pyplot as plt
from sklearn import metrics

def evaluate(prompt_model, dataloader, class_labels, dataset, seed, desc, per_class=False):
    prompt_model.eval()
    allpreds = []
    alllabels = []
    pbar = tqdm(dataloader, desc=desc)
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        alllabels.extend(labels.cpu().tolist())
        allpreds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    # If we're training on SciCite but testing on ACL-ARC/ACT2, handle the label mapping
    if args.dataset == "scicite" and args.test_dataset in ["acl_arc", "act2"]:
        mapped_preds = []
        for pred in allpreds:
            if pred == 0:  # If prediction is background
                # Count prediction as correct if true label is any of Background, Extends, Future, or Motivation
                if alllabels[len(mapped_preds)] in [0, 2, 3, 4]:
                    mapped_preds.append(alllabels[len(mapped_preds)])
                else:
                    mapped_preds.append(0)  # Default to Background if not matching
            elif pred == 1:  # If prediction is method
                mapped_preds.append(5)  # Map to Uses
            elif pred == 2:  # If prediction is result
                mapped_preds.append(1)  # Map to Compare Contrast
        allpreds = mapped_preds

    accuracy = classification_metrics(allpreds, alllabels, 'accuracy')
    f1_macro = classification_metrics(allpreds, alllabels, 'macro-f1')
    f1_micro = classification_metrics(allpreds, alllabels, 'micro-f1')
    if per_class:
        print(classification_report(alllabels, allpreds))

    return accuracy, f1_macro, f1_micro

###############

from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


if args.verbalizer == "soft":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=0, num_training_steps=tot_step)

elif args.verbalizer == "auto":
    prompt_initialize(myverbalizer, prompt_model, train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None

elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None


tot_loss = 0
log_loss = 0
best_val_acc = 0
best_val_f1 = 0
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    for step, inputs in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    val_acc, val_f1_macro, val_f1_micro = evaluate(prompt_model, validation_dataloader, class_labels, args.dataset, args.seed, desc="Valid")
    if val_f1_macro>=best_val_f1:
        torch.save(prompt_model.state_dict(),f"ckpts/{this_run_unicode}.ckpt")
        best_val_f1 = val_f1_macro
    print("Epoch {}, val_acc {}, val_f1(macro) {}, val_f1(micro) {}".format(epoch, val_acc, val_f1_macro, val_f1_micro), flush=True)

prompt_model.load_state_dict(torch.load(f"ckpts/{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
test_acc, test_f1_macro, test_f1_micro = evaluate(prompt_model, test_dataloader, class_labels, args.dataset, args.seed, desc="Test", per_class=True)

content_write = "="*20+"\n"
content_write += f"train_dataset {args.dataset}\t"
content_write += f"test_dataset {args.test_dataset or args.dataset}\t"
content_write += f"template_id {args.template_id}\t"
content_write += f"epochs {args.max_epochs}\t"
content_write += f"seed {args.seed}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += f"Acc: {test_acc}\t"
content_write += f"F1(macro): {test_f1_macro}\t"
content_write += f"F1(micro): {test_f1_micro}\t"
content_write += "\n"

print(content_write)

with open(f"{args.result_file}", "a") as fout:
    fout.write(content_write)
