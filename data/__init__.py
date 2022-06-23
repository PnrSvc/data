# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
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

# Lint as: python3
"""The Sequence labellIng evaLuatIon benChmark fOr spoken laNguagE (SILICONE) benchmark."""


import textwrap

import pandas as pd

import datasets


_SILICONE_CITATION = """\
@inproceedings{chapuis-etal-2020-hierarchical,
    title = "Hierarchical Pre-training for Sequence Labelling in Spoken Dialog",
    author = "Chapuis, Emile  and
      Colombo, Pierre  and
      Manica, Matteo  and
      Labeau, Matthieu  and
      Clavel, Chlo{\'e}",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.239",
    doi = "10.18653/v1/2020.findings-emnlp.239",
    pages = "2636--2648",
    abstract = "Sequence labelling tasks like Dialog Act and Emotion/Sentiment identification are a
        key component of spoken dialog systems. In this work, we propose a new approach to learn
        generic representations adapted to spoken dialog, which we evaluate on a new benchmark we
        call Sequence labellIng evaLuatIon benChmark fOr spoken laNguagE benchmark (SILICONE).
        SILICONE is model-agnostic and contains 10 different datasets of various sizes.
        We obtain our representations with a hierarchical encoder based on transformer architectures,
        for which we extend two well-known pre-training objectives. Pre-training is performed on
        OpenSubtitles: a large corpus of spoken dialog containing over 2.3 billion of tokens. We
        demonstrate how hierarchical encoders achieve competitive results with consistently fewer
        parameters compared to state-of-the-art models and we show their importance for both
        pre-training and fine-tuning.",
}
"""

_SILICONE_DESCRIPTION = """\
The Sequence labellIng evaLuatIon benChmark fOr spoken laNguagE (SILICONE) benchmark is a collection
 of resources for training, evaluating, and analyzing natural language understanding systems
 specifically designed for spoken language. All datasets are in the English language and cover a
 variety of domains including daily life, scripted scenarios, joint task completion, phone call
 conversations, and televsion dialogue. Some datasets additionally include emotion and/or sentimant
 labels.
"""

_URL = "https://github.com/PnrSvc/data/find/master"


IEMOCAP_E_DESCRIPTION = {
    "1": "Disgust",
    "2": "Excitement",
    "3": "Fear",
    "4": "Frustration",
    "5": "Happiness",
}


class DatasetConfig(datasets.BuilderConfig):
    """BuilderConfig for SILICONE."""

    def __init__(
        self,
        text_features,
        label_column,
        data_url,
        citation,
        url,
        label_classes=None,
        **kwargs,
    ):
        """BuilderConfig for SILICONE.
        Args:
          text_features: `dict[string, string]`, map from the name of the feature
            dict for each text field to the name of the column in the tsv file
          label_column: `string`, name of the column in the csv/txt file corresponding
            to the label
          data_url: `string`, url to download the csv/text file from
          citation: `string`, citation for the data set
          url: `string`, url for information about the data set
          label_classes: `list[string]`, the list of classes if the label is
            categorical. If not provided, then the label will be of type
            `datasets.Value('float32')`.
          **kwargs: keyword arguments forwarded to super.
        """
        super(DatasetConfig, self).__init__(version=datasets.Version("1.0.0", ""), **kwargs)
        self.text_features = text_features
        self.label_column = label_column
        self.label_classes = label_classes
        self.data_url = data_url
        self.citation = citation
        self.url = url


class Dataset(datasets.GeneratorBasedBuilder):
    """The Sequence labellIng evaLuatIon benChmark fOr spoken laNguagE (SILICONE) benchmark."""

    BUILDER_CONFIGS = [
        DatasetConfig(
            name="pnr",
            description=textwrap.dedent(
                """\
            The DailyDialog Act Corpus contains multi-turn dialogues and is supposed to reflect daily
            communication by covering topics about daily life. The dataset is manually labelled with
             dialog act and emotions. It is the third biggest corpus of SILICONE with 102k utterances."""
            ),
            text_features={
                "sentiment": "sentiment",
                "label": "label",
            },
            label_classes=["1", "2", "3", "4", "5"],
            label_column="label",
            data_url={
                "train": _URL + "/data/pnr.csv",
            },
            citation=textwrap.dedent(
                """\
            @InProceedings{li2017dailydialog,
            author = {Li, Yanran and Su, Hui and Shen, Xiaoyu and Li, Wenjie and Cao, Ziqiang and Niu, Shuzi},
            title = {DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset},
            booktitle = {Proceedings of The 8th International Joint Conference on Natural Language Processing (IJCNLP 2017)},
            year = {2017}
            }"""
            ),
            url="http://yanran.li/dailydialog.html",
        ),
    ]

    def _info(self):
        features = {text_feature: datasets.Value("string") for text_feature in self.config.text_features.keys()}
        if self.config.label_classes:
            features["Label"] = datasets.features.ClassLabel(names=self.config.label_classes)
        features["Idx"] = datasets.Value("int32")
        return datasets.DatasetInfo(
            description=_SILICONE_DESCRIPTION,
            features=datasets.Features(features),
            homepage=self.config.url,
            citation=self.config.citation + "\n" + _SILICONE_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_files = dl_manager.download(self.config.data_url)
        splits = []
        splits.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_file": data_files["train"],
                    "split": "train",
                },
            )
        )
        return splits

    def _generate_examples(self, data_file, split):
        if self.config.name not in ():
            df = pd.read_csv(data_file, delimiter=",", header=0, quotechar='"', dtype=str)[
                self.config.text_features.keys()
            ]

        if self.config.name == "data":
            df = pd.read_csv(
                data_file,
                delimiter=",",
                header=0,
                quotechar='"',
                names=["sentiment", "label"],
                dtype=str,
            )[self.config.text_features.keys()]

        rows = df.to_dict(orient="records")

        for n, row in enumerate(rows):
            example = row
            example["Idx"] = n

            if self.config.label_column in example:
                label = example[self.config.label_column]
                example["Label"] = label

            yield example["Idx"], example