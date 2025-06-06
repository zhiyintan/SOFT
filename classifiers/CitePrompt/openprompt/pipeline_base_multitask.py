from pickle import FALSE
from torch.utils.data.sampler import RandomSampler
from transformers.configuration_utils import PretrainedConfig
from transformers import GenerationMixin
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from typing import *
from openprompt.data_utils import InputExample, InputFeatures
from torch.utils.data._utils.collate import default_collate
from tqdm.std import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils.dummy_pt_objects import PreTrainedModel
from openprompt.plms.utils import TokenizerWrapper
from openprompt.prompt_base import Template, Verbalizer
from collections import defaultdict
from openprompt.utils import round_list, signature
import numpy as np
from torch.utils.data import DataLoader
from yacs.config import CfgNode
from openprompt.utils.logging import logger



class PromptDataLoader(object):
    r"""
    PromptDataLoader wraps the original dataset. The input data is firstly wrapped with the
    prompt's template, and then is tokenized by a wrapperd-tokenizer.

    Args:
        dataset (:obj:`Dataset` or :obj:`List`): Either a DatasetObject or a list containing the input examples.
        template (:obj:`Template`): A derived class of :obj:`Template`
        tokenizer (:obj:`PretrainedTokenizer`): The pretrained tokenizer.
        tokenizer_wrapper_class (:cls:`TokenizerWrapper`): The class of tokenizer wrapper.
        max_seq_length (:obj:`int`, optional): The max sequence length of the input ids. It's used to truncate sentences.
        batch_size (:obj:`int`, optional): The batch_size of data loader
        teacher_forcing (:obj:`bool`, optional): Whether to fill the mask with target text. Set to true in training generation model.
        decoder_max_length (:obj:`int`, optional): the decoder maximum length of an encoder-decoder model.
        predict_eos_token (:obj:`bool`, optional): Whether to predict the <eos> token. Suggest to set to true in generation.
        truncate_method (:obj:`bool`, optional): the truncate method to use. select from `head`, `tail`, `balanced`.
        kwargs  :Other kwargs that might be passed into a tokenizer wrapper.
    """
    def __init__(self,
                 dataset: Union[Dataset, List],
                 template: Template,
                 tokenizer_wrapper: Optional[TokenizerWrapper] = None,
                 tokenizer: PreTrainedTokenizer = None,
                 tokenizer_wrapper_class = None,
                 verbalizer: Optional[Verbalizer] = None,
                 max_seq_length: Optional[str] = 512,
                 batch_size: Optional[int] = 1,
                 shuffle: Optional[bool] = False,
                 teacher_forcing: Optional[bool] = False,
                 decoder_max_length: Optional[int] = -1,
                 predict_eos_token: Optional[bool] = False,
                 truncate_method: Optional[str] = "tail",
                 drop_last: Optional[bool] = False,
                 **kwargs,
                ):

        assert hasattr(dataset, "__iter__"), f"The dataset must have __iter__ method. dataset is {dataset}"
        assert hasattr(dataset, "__len__"), f"The dataset must have __len__ method. dataset is {dataset}"
        self.raw_dataset = dataset

        self.wrapped_dataset = []
        self.tensor_dataset = []
        self.template = template
        self.verbalizer = verbalizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.teacher_forcing = teacher_forcing

        if tokenizer_wrapper is None:
            if tokenizer_wrapper_class is None:
                raise RuntimeError("Either wrapped_tokenizer or tokenizer_wrapper_class should be specified.")
            if tokenizer is None:
                raise RuntimeError("No tokenizer specified to instantiate tokenizer_wrapper.")

            tokenizer_wrapper_init_keys = signature(tokenizer_wrapper_class.__init__).args
            prepare_kwargs = {
                "max_seq_length" : max_seq_length,
                "truncate_method" : truncate_method,
                "decoder_max_length" : decoder_max_length,
                "predict_eos_token" : predict_eos_token,
                "tokenizer" : tokenizer,
                **kwargs,
            }

            to_pass_kwargs = {key: prepare_kwargs[key] for key in prepare_kwargs if key in tokenizer_wrapper_init_keys}
            self.tokenizer_wrapper = tokenizer_wrapper_class(**to_pass_kwargs)
        else:
            self.tokenizer_wrapper = tokenizer_wrapper

        # check the satisfiability of each component
        assert hasattr(self.template, 'wrap_one_example'), "Your prompt has no function variable \
                                                         named wrap_one_example"

        # process
        self.wrap()
        self.tokenize()

        if self.shuffle:
            sampler = RandomSampler(self.tensor_dataset)
        else:
            sampler = None

        self.dataloader = DataLoader(
            self.tensor_dataset,
            batch_size = self.batch_size,
            sampler= sampler,
            collate_fn = InputFeatures.collate_fct,
            drop_last = drop_last,
        )


    def wrap(self):
        r"""A simple interface to pass the examples to prompt, and wrap the text with template.
        """
        if isinstance(self.raw_dataset, Dataset) or isinstance(self.raw_dataset, List):
            assert len(self.raw_dataset) > 0, 'The dataset to be wrapped is empty.'
            # for idx, example in tqdm(enumerate(self.raw_dataset),desc='Wrapping'):
            for idx, example in enumerate(self.raw_dataset):
                if self.verbalizer is not None and hasattr(self.verbalizer, 'wrap_one_example'): # some verbalizer may also process the example.
                    example = self.verbalizer.wrap_one_example(example)
                wrapped_example = self.template.wrap_one_example(example)
                self.wrapped_dataset.append(wrapped_example)
        else:
            raise NotImplementedError

    def tokenize(self) -> None:
        r"""Pass the wrapped text into a prompt-specialized tokenizer,
           the true PretrainedTokenizer inside the tokenizer is flexible, e.g. AlBert, Bert, T5,...
        """
        for idx, wrapped_example in tqdm(enumerate(self.wrapped_dataset),desc='tokenizing'):
        # for idx, wrapped_example in enumerate(self.wrapped_dataset):
            inputfeatures = InputFeatures(**self.tokenizer_wrapper.tokenize_one_example(wrapped_example, self.teacher_forcing), **wrapped_example[1]).to_tensor()
            self.tensor_dataset.append(inputfeatures)

    def __len__(self):
        return  len(self.dataloader)

    def __iter__(self,):
        return self.dataloader.__iter__()



class PromptModel(nn.Module):
    r'''``PromptModel`` is the encapsulation of ``Template`` and the ``pre-trained model``,
    with OpenPrompt, these modules could be flexibly combined. And this class is the base class of ``PromptForClassification`` and ``PromptForGeneration``

    Args:
        plm (:obj:`PreTrainedModel`): The pre-trained language model for the current prompt-learning task.
        template (:obj:`Template`): The ``Template`` object to warp the input data.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template_main: Template,
                 template_sca1: Template,
                 template_sca2: Template,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False,
                ):
        super().__init__()
        self.plm = plm
        self.template_main = template_main
        self.template_sca1 = template_sca1
        self.template_sca2 = template_sca2
        self.freeze_plm = freeze_plm
        self.plm_eval_mode = plm_eval_mode
        if freeze_plm:
            for param in self.plm.parameters():
                param.requires_grad = False
        if plm_eval_mode:
            self.plm.eval()
            for param in self.plm.parameters():
                param.requires_grad = False

        # get model's forward function's keywords
        self.forward_keys = signature(self.plm.forward).args

        self._prepare_main_input_name()

    def _prepare_main_input_name(self):
        model = self.plm
        if hasattr(model, "encoder") and hasattr(model.encoder, "main_input_name"):
            if model.encoder.main_input_name != model.main_input_name:
                main_input_name = model.encoder.main_input_name
            else:
                main_input_name = model.main_input_name
        else:
            main_input_name = getattr(model, "main_input_name", "input_ids")
        self.main_input_name = main_input_name

    def train(self, mode: bool = True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for name, module in self.named_children():
            if not (self.plm_eval_mode and 'plm' in name and mode):
                module.train(mode)
        return self

    def forward(self, batch_main: None, batch_sca1 = None, batch_sca2 = None):
        r"""
        This is a forward method to make wrapped input data go through the model, and return the output logits.
        Typically, this function aims to predict the ``<mask>`` position.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The input features of batchified data sequences.
        """
        if batch_main is not None and batch_sca1 is not None and batch_sca2 is not None:
            batch_main = self.template_main.process_batch(batch_main)
            input_batch_main = {key: batch_main[key] for key in batch_main if key in self.forward_keys}
            outputs_main = self.plm(**input_batch_main, output_hidden_states=True)
            outputs_main = self.template_main.post_processing_outputs(outputs_main)

            batch_sca1 = self.template_sca1.process_batch(batch_sca1)
            input_batch_sca1 = {key: batch_sca1[key] for key in batch_sca1 if key in self.forward_keys}
            outputs_sca1 = self.plm(**input_batch_sca1, output_hidden_states=True)
            outputs_sca1 = self.template_sca1.post_processing_outputs(outputs_sca1)

            batch_sca2 = self.template_sca2.process_batch(batch_sca2)
            input_batch_sca2 = {key: batch_sca2[key] for key in batch_sca2 if key in self.forward_keys}
            outputs_sca2 = self.plm(**input_batch_sca2, output_hidden_states=True)
            outputs_sca2 = self.template_sca2.post_processing_outputs(outputs_sca2)

            return outputs_main, outputs_sca1, outputs_sca2
        elif batch_main is None and batch_sca1 is not None and batch_sca2 is not None:
            batch_sca1 = self.template_sca1.process_batch(batch_sca1)
            input_batch_sca1 = {key: batch_sca1[key] for key in batch_sca1 if key in self.forward_keys}
            outputs_sca1 = self.plm(**input_batch_sca1, output_hidden_states=True)
            outputs_sca1 = self.template_sca1.post_processing_outputs(outputs_sca1)

            batch_sca2 = self.template_sca2.process_batch(batch_sca2)
            input_batch_sca2 = {key: batch_sca2[key] for key in batch_sca2 if key in self.forward_keys}
            outputs_sca2 = self.plm(**input_batch_sca2, output_hidden_states=True)
            outputs_sca2 = self.template_sca2.post_processing_outputs(outputs_sca2)

            return outputs_sca1, outputs_sca2
        elif batch_main is None and batch_sca1 is None and batch_sca2 is not None:

            batch_sca2 = self.template_sca2.process_batch(batch_sca2)
            input_batch_sca2 = {key: batch_sca2[key] for key in batch_sca2 if key in self.forward_keys}
            outputs_sca2 = self.plm(**input_batch_sca2, output_hidden_states=True)
            outputs_sca2 = self.template_sca2.post_processing_outputs(outputs_sca2)

            return outputs_sca2
        elif batch_main is None and batch_sca1 is not None and batch_sca2 is None:

            batch_sca1 = self.template_sca1.process_batch(batch_sca1)
            input_batch_sca1 = {key: batch_sca1[key] for key in batch_sca1 if key in self.forward_keys}
            outputs_sca1 = self.plm(**input_batch_sca1, output_hidden_states=True)
            outputs_sca1 = self.template_sca1.post_processing_outputs(outputs_sca1)

            return outputs_sca1
        else:
            batch_main = self.template_main.process_batch(batch_main)
            input_batch_main = {key: batch_main[key] for key in batch_main if key in self.forward_keys}
            outputs_main = self.plm(**input_batch_main, output_hidden_states=True)
            outputs_main = self.template_main.post_processing_outputs(outputs_main)

            return outputs_main


class PromptForClassification(nn.Module):
    r'''``PromptModel`` with a classification head on top. The classification head will map
    the logits in all position of the sequence (return value of a ``PromptModel``) into the
    logits of the labels, using a verbalizer.

    Args:
        plm (:obj:`PretrainedModel`): A pre-traiend model you decide to use for classification, e.g. BERT.
        template (:obj:`Template`): A ``Template`` object you use to wrap the input text for classification, e.g. ``ManualTemplate``.
        verbalizer (:obj:`Verbalizer`): A ``Verbalizer`` object you use to project the labels to label words for classification, e.g. ``ManualVerbalizer``.
        freeze_plm (:obj:`bool`): whether or not to freeze the pretrained language model
        plm_eval_mode (:obj:`bool`): this is a stronger freezing mode than freeze_plm, i.e. the dropout of the model is turned off. No matter whether the other part is set to train.
    '''
    def __init__(self,
                 plm: PreTrainedModel,
                 template_main: Template,
                 template_sca1: Template,
                 template_sca2: Template,
                 verbalizer_main: Verbalizer,
                 verbalizer_sca1: Verbalizer,
                 verbalizer_sca2: Verbalizer,
                 freeze_plm: bool = False,
                 plm_eval_mode: bool=False
                ):
        super().__init__()
        self.prompt_model = PromptModel(plm, template_main, template_sca1, template_sca2, freeze_plm, plm_eval_mode)
        self.verbalizer_main = verbalizer_main
        self.verbalizer_sca1 = verbalizer_sca1
        self.verbalizer_sca2 = verbalizer_sca2

    @property
    def plm(self):
        return self.prompt_model.plm

    @property
    def template_main(self):
        return self.prompt_model.template_main

    @property
    def template_sca1(self):
        return self.prompt_model.template_sca1

    @property
    def template_sca2(self):
        return self.prompt_model.template_sca2

    @property
    def device(self,):
        r"""Register the device parameter."""
        return self.plm.device

    def extract_at_mask(self,
                       outputs: torch.Tensor,
                       batch: Union[Dict, InputFeatures]):
        r"""Get outputs at all <mask> token
        E.g., project the logits of shape
        (``batch_size``, ``max_seq_length``, ``vocab_size``)
        into logits of shape (if num_mask_token > 1)
        (``batch_size``, ``num_mask_token``, ``vocab_size``)
        or into logits of shape (if ``num_mask_token`` = 1)
        (``batch_size``, ``vocab_size``).

        Args:
            outputs (:obj:`torch.Tensor`): The original outputs (maybe process by verbalizer's
                 `gather_outputs` before) etc. of the whole sequence.
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The extracted outputs of ``<mask>`` tokens.

        """
        outputs = outputs[torch.where(batch['loss_ids']>0)]
        outputs = outputs.view(batch['loss_ids'].shape[0], -1, outputs.shape[1])
        if outputs.shape[1] == 1:
            outputs = outputs.view(outputs.shape[0], outputs.shape[2])
        return outputs

    def forward(self, batch_main: None, batch_sca1 = None, batch_sca2 = None):
        r"""
        Get the logits of label words.

        Args:
            batch (:obj:`Union[Dict, InputFeatures]`): The original batch

        Returns:
            :obj:`torch.Tensor`: The logits of the label words (obtained by the current verbalizer).
        """
        if batch_main is not None and batch_sca1 is not None and batch_sca2 is not None:
            outputs_main, outputs_sca1, outputs_sca2 = self.prompt_model(batch_main, batch_sca1, batch_sca2)

            outputs_main = self.verbalizer_main.gather_outputs(outputs_main)
            if isinstance(outputs_main, tuple):
                outputs_at_mask_main = [self.extract_at_mask(output, batch_main) for output in outputs_main]
            else:
                outputs_at_mask_main = self.extract_at_mask(outputs_main, batch_main)
            label_words_logits_main = self.verbalizer_main.process_outputs(outputs_at_mask_main, batch=batch_main)

            outputs_sca1 = self.verbalizer_sca1.gather_outputs(outputs_sca1)
            if isinstance(outputs_sca1, tuple):
                outputs_at_mask_sca1 = [self.extract_at_mask(output, batch_sca1) for output in outputs_sca1]
            else:
                outputs_at_mask_sca1 = self.extract_at_mask(outputs_sca1, batch_sca1)
            label_words_logits_sca1 = self.verbalizer_sca1.process_outputs(outputs_at_mask_sca1, batch=batch_sca1)

            outputs_sca2 = self.verbalizer_sca2.gather_outputs(outputs_sca2)
            if isinstance(outputs_sca2, tuple):
                outputs_at_mask_sca2 = [self.extract_at_mask(output, batch_sca2) for output in outputs_sca2]
            else:
                outputs_at_mask_sca2 = self.extract_at_mask(outputs_sca2, batch_sca2)
            label_words_logits_sca2 = self.verbalizer_sca2.process_outputs(outputs_at_mask_sca2, batch=batch_sca2)

            return label_words_logits_main, label_words_logits_sca1, label_words_logits_sca2
        elif batch_main is None and batch_sca1 is not None and batch_sca2 is not None:
            outputs_sca1, outputs_sca2 = self.prompt_model(batch_main = None, batch_sca1 = batch_sca1, batch_sca2 = batch_sca2)

            outputs_sca1 = self.verbalizer_sca1.gather_outputs(outputs_sca1)
            if isinstance(outputs_sca1, tuple):
                outputs_at_mask_sca1 = [self.extract_at_mask(output, batch_sca1) for output in outputs_sca1]
            else:
                outputs_at_mask_sca1 = self.extract_at_mask(outputs_sca1, batch_sca1)
            label_words_logits_sca1 = self.verbalizer_sca1.process_outputs(outputs_at_mask_sca1, batch=batch_sca1)

            outputs_sca2 = self.verbalizer_sca2.gather_outputs(outputs_sca2)
            if isinstance(outputs_sca2, tuple):
                outputs_at_mask_sca2 = [self.extract_at_mask(output, batch_sca2) for output in outputs_sca2]
            else:
                outputs_at_mask_sca2 = self.extract_at_mask(outputs_sca2, batch_sca2)
            label_words_logits_sca2 = self.verbalizer_sca2.process_outputs(outputs_at_mask_sca2, batch=batch_sca2)

            return label_words_logits_sca1, label_words_logits_sca2
        elif batch_main is None and batch_sca1 is None and batch_sca2 is not None:
            outputs_sca2 = self.prompt_model(batch_sca2)

            outputs_sca2 = self.verbalizer_sca2.gather_outputs(outputs_sca2)
            if isinstance(outputs_sca2, tuple):
                outputs_at_mask_sca2 = [self.extract_at_mask(output, batch_sca2) for output in outputs_sca2]
            else:
                outputs_at_mask_sca2 = self.extract_at_mask(outputs_sca2, batch_sca2)
            label_words_logits_sca2 = self.verbalizer_sca2.process_outputs(outputs_at_mask_sca2, batch=batch_sca2)

            return label_words_logits_sca2
        elif batch_main is None and batch_sca1 is not None and batch_sca2 is None:
            outputs_sca1 = self.prompt_model( batch_sca1)

            outputs_sca1 = self.verbalizer_sca1.gather_outputs(outputs_sca1)
            if isinstance(outputs_sca1, tuple):
                outputs_at_mask_sca1 = [self.extract_at_mask(output, batch_sca1) for output in outputs_sca1]
            else:
                outputs_at_mask_sca1 = self.extract_at_mask(outputs_sca1, batch_sca1)
            label_words_logits_sca1 = self.verbalizer_sca1.process_outputs(outputs_at_mask_sca1, batch=batch_sca1)

            return label_words_logits_sca1
        elif batch_main is not None and batch_sca1 is None and batch_sca2 is None:
            outputs_main = self.prompt_model(batch_main = batch_main, batch_sca1 = None, batch_sca2 = None)

            outputs_main = self.verbalizer_main.gather_outputs(outputs_main)
            if isinstance(outputs_main, tuple):
                outputs_at_mask_main = [self.extract_at_mask(output, batch_main) for output in outputs_main]
            else:
                outputs_at_mask_main = self.extract_at_mask(outputs_main, batch_main)
            label_words_logits_main = self.verbalizer_main.process_outputs(outputs_at_mask_main, batch=batch_main)
            
            return label_words_logits_main

    def predict(self):
        pass

    @property
    def tokenizer(self):
        r'''Utility property, to get the tokenizer more easily.
        '''
        return self.verbalizer.tokenizer

    def parallelize(self, device_map=None):
        r"""Parallelize the model across device
        """
        if hasattr(self.plm, "parallelize"):
            self.plm.parallelize(device_map)
            self.device_map = self.plm.device_map
            self.template_main.cuda()
            self.template_sca1.cuda()
            self.template_sca2.cuda()
            self.verbalizer_main.cuda()
            self.verbalizer_sca1.cuda()
            self.verbalizer_sca2.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")

    def deparallelize(self):
        r"""Deparallelize the model across device
        """
        if hasattr(self.plm, "deparallelize"):
            self.plm.deparallelize()
            self.device_map = None
            self.template_main.cuda()
            self.template_sca1.cuda()
            self.template_sca2.cuda()
            self.verbalizer_main.cuda()
            self.verbalizer_sca1.cuda()
            self.verbalizer_sca2.cuda()
        else:
            raise NotImplementedError("parallelize method was not implemented for this plm.")
