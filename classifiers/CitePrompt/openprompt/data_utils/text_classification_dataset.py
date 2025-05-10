# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all TextClassification tasks.
"""

import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable
import pandas
from openprompt.utils.logging import logger
import pandas as pd

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

class SciCite(DataProcessor):
    """
    `SciCite <https://arxiv.org/pdf/1904.01608.pdf>`_ is a Citation Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/allenai/scicite>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SciCite"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["background", "method", "result"]
        self.label_mapping_csv = {
            "background information": 0,
            "background": 0,
            "method": 1,
            "results comparison": 2,
            "result": 2
        }

    def get_examples(self, data_dir, split, stopwords):
        path = os.path.join(data_dir, "{}.csv".format(split))
        df = pd.read_csv(path)
        examples = []
        df["citation_context"] = df["citation_context"].apply(
            lambda words: ' '.join(word.lower() for word in words.split() if word not in set(stopwords))
        )
        for i, row in df.iterrows():
            idx = row["unique_id"]
            text_cit = row["citation_context"]
            cit_label = self.label_mapping_csv[row["citation_class_label"]]
            example = InputExample(
                guid=idx, 
                text_a=text_cit, 
                label=int(cit_label)
            )
            examples.append(example)
        return examples
    

class ACL_ARC(DataProcessor):
    """
    `ACL_ARC <https://arxiv.org/pdf/1904.01608.pdf>`_ is a Citation Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/allenai/scicite>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SciCite"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["Background", "Compare Contrast", "Extends", "Future work", "Motivation", "Uses"]
        self.label_mapping_csv = {
            "BACKGROUND": 0,
            "COMPARES_CONTRASTS": 1,
            "EXTENSION": 2,
            "EXTENDS": 2,
            "FUTURE": 3,
            "MOTIVATION": 4,
            "USES": 5
        }
        
    def get_examples(self, data_dir, split, stopwords):
        path = os.path.join(data_dir, "{}.csv".format(split))
        df = pd.read_csv(path)
        examples = []
        df["citation_context"] = df["citation_context"].apply(
            lambda words: ' '.join(word.lower() for word in words.split() if word not in set(stopwords))
        )

        for i, row in df.iterrows():
            idx = row["unique_id"]
            text_cit = row["citation_context"]
            cit_label = self.label_mapping_csv[row["citation_class_label"]]
            example = InputExample(
                guid=idx, 
                text_a=text_cit, 
                label=int(cit_label)
            )
            examples.append(example)
        return examples
    

class ACL_ARC_New(DataProcessor):
    def __init__(self, target):
        super().__init__()
        if target == "citation_object":
            self.labels = ["Produced Resource", "Discovery", "Performed Work"]
            self.label_mapping_csv = {
                "Produced Resource": 0,
                "produced resource": 0,
                "Discovery": 1,
                "discovery": 1,
                "Performed Work": 2,
                "performed work": 2,
            }
        elif target == "citation_function":
            self.labels = ["Modify", "Use", "Signal Gap", "Justify Design Choice", "Evaluate Against", "Highlight Limitation", "Contextualize"]
            self.label_mapping_csv = {
                "Modify": 0,
                "modify": 0,
                "Use": 1,
                "use": 1,
                "Signal Gap": 2,
                "signal gap": 2,
                "Justify Design Choice": 3,
                "justify design choice": 3,
                "Evaluate Against": 4,
                "evaluate against": 4,
                "Highlight Limitation": 5,
                "highlight limitation": 5,
                "Highlight limitation": 5,
                "Contextualize": 6,
                "contextualize": 6,
            }
        else:
            raise ValueError("target should be either 'citation_object' or 'citation_function'")
        
    def get_examples(self, data_dir, split, stopwords, target):
        path = os.path.join(data_dir, "{}.csv".format(split))
        df = pd.read_csv(path)
        examples = []
        df["citation_context"] = df["citation_context"].apply(
            lambda words: ' '.join(word.lower() for word in words.split() if word not in set(stopwords))
        )

        for i, row in df.iterrows():
            idx = row["unique_id"]
            text_cit = row["citation_context"]
            cit_label = self.label_mapping_csv[row[target]]
            example = InputExample(
                guid=idx, 
                text_a=text_cit, 
                label=int(cit_label)
            )
            examples.append(example)
        return examples


class scaffold_data_citeworth(DataProcessor):
    """
    `ACL_ARC <https://arxiv.org/pdf/1904.01608.pdf>`_ is a Citation Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/allenai/scicite>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SciCite"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["true", "false"]
        
    def get_examples(self, data_dir, split, stopwords):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        k=0
        with open(path, 'r') as f:
            json_list = list(f)
        for i in json_list:
            #if k > 49999:
            #    break
            row = json.loads(i)
            if row["is_citation"] != "":
                cit_label = -1
                idx = "c" + str(k)
                text = row["cleaned_cite_text"].lower()
                tokens_without_stopwords = [t for t in text.split(" ") if t not in stopwords]
                text_cit = (" ").join(tokens_without_stopwords)
                if row["is_citation"] == False:
                    cit_label = 0
                elif row["is_citation"] == True:
                    cit_label = 1
                example = InputExample(guid=str(idx), text_a=text_cit, label=int(cit_label))
                examples.append(example)
                k = k + 1
        return examples


class scaffold_data_section(DataProcessor):
    """
    `ACL_ARC <https://arxiv.org/pdf/1904.01608.pdf>`_ is a Citation Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/allenai/scicite>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "SciCite"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["introduction", "related work", "method", "experiments", "conclusion"]
        
    def get_examples(self, data_dir, split, stopwords):
        path = os.path.join(data_dir, "{}.jsonl".format(split))
        examples = []
        k=0
        with open(path, 'r') as f:
            json_list = list(f)
        for i in json_list:
            #if k > 49999:
            #    break
            row = json.loads(i)
            if row["section_name"] != "":
                cit_label = -1
                idx = "c" + str(k)
                text = row["text"].lower()
                tokens_without_stopwords = [t for t in text.split(" ") if t not in stopwords]
                text_cit = (" ").join(tokens_without_stopwords)
                if row["section_name"] == "introduction":
                    cit_label = 0
                elif row["section_name"] == "related work":
                    cit_label = 1
                elif row["section_name"] == "method":
                    cit_label = 2
                elif row["section_name"] == "experiments":
                    cit_label = 3
                elif row["section_name"] == "conclusion":
                    cit_label = 4
                example = InputExample(guid=str(idx), text_a=text_cit, label=int(cit_label))
                examples.append(example)
                k = k + 1
        return examples


class ACT2(DataProcessor):
    """
    `ACT2 <https://aclanthology.org/2022.lrec-1.363/>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/oacore/act2>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "ACT2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["background", "uses", "compare_contrast", "motivation", "extension", "future"]

    def get_examples(self, data_dir, stopwords):
        dataset = pandas.read_csv(data_dir, sep="\t", index_col="unique_id")
        examples = []
        for i in dataset.index:
            if str(dataset["citation_context"][i]) != "nan" and str(dataset["citation_class_label"][i]) != "nan":
                cit_label = int(dataset["citation_class_label"][i])
                idx = i
                text = str(dataset["citation_context"][i]).lower()
                tokens_without_stopwords = [t for t in text.split(" ") if t not in stopwords]
                text_cit = (" ").join(tokens_without_stopwords)
            example = InputExample(guid=str(idx), text_a=text_cit, label=int(cit_label))
            examples.append(example)
        return examples


class ACT2_abstract(DataProcessor):
    """
    `ACT2 <https://aclanthology.org/2022.lrec-1.363/>`_ is a News Topic classification dataset

    we use dataset provided by `LOTClass <https://github.com/oacore/act2>`_

    Examples:

    ..  code-block:: python

        from openprompt.data_utils.text_classification_dataset import PROCESSORS

        base_path = "datasets/TextClassification"

        dataset_name = "ACT2"
        dataset_path = os.path.join(base_path, dataset_name)
        processor = PROCESSORS[dataset_name.lower()]()
        trainvalid_dataset = processor.get_train_examples(dataset_path)
        test_dataset = processor.get_test_examples(dataset_path)

        assert processor.get_num_labels() == 4
        assert processor.get_labels() == ["World", "Sports", "Business", "Tech"]
        assert len(trainvalid_dataset) == 120000
        assert len(test_dataset) == 7600
        assert test_dataset[0].text_a == "Fears for T N pension after talks"
        assert test_dataset[0].text_b == "Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
        assert test_dataset[0].label == 2
    """

    def __init__(self):
        super().__init__()
        self.labels = ["background", "uses", "compare_contrast", "motivation", "extension", "future"]

    def get_examples(self, data_dir, stopwords):
        dataset = pandas.read_csv(data_dir, sep="\t", index_col="unique_id")
        examples = []
        for i in dataset.index:
            if str(dataset["citation_context"][i]) != "nan" and str(dataset["citation_class_label"][i]) != "nan":
                cit_label = int(dataset["citation_class_label"][i])
                idx = i
                text = str(dataset["citing_abstract"][i]).lower()
                tokens_without_stopwords = [t for t in text.split(" ") if t not in stopwords]
                text_cit_a = (" ").join(tokens_without_stopwords)
                text = str(dataset["cited_abstract"][i]).lower()
                tokens_without_stopwords = [t for t in text.split(" ") if t not in stopwords]
                text_cit_b = (" ").join(tokens_without_stopwords)
            example = InputExample(guid=str(idx), text_a=text_cit_a, text_b=text_cit_b, label=int(cit_label))
            examples.append(example)
        return examples


PROCESSORS = {
    "scicite": SciCite,
    "act2abstract": ACT2_abstract,
    "ACT2" : ACT2,
    "acl_arc" : ACL_ARC,
    "scaffold_data_section": scaffold_data_section,
    "scaffold_data_citeworth": scaffold_data_citeworth,
}
