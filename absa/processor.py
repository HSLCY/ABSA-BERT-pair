# coding=utf-8

"""Processors for different tasks."""

import csv
import os

import pandas as pd

from absa import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class Sentihood_single_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train.tsv"), header=None, sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev.tsv"), header=None, sep="\t"
        ).values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test.tsv"), header=None, sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[2]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


class Sentihood_NLI_M_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_NLI_M.tsv"), sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_NLI_M.tsv"), sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_NLI_M.tsv"), sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            text_b = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[3]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Sentihood_QA_M_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_QA_M.tsv"), sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_QA_M.tsv"), sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_QA_M.tsv"), sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['None', 'Positive', 'Negative']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[1]))
            text_b = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[3]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Sentihood_NLI_B_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_NLI_B.tsv"), sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_NLI_B.tsv"), sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_NLI_B.tsv"), sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[3]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Sentihood_QA_B_Processor(DataProcessor):
    """Processor for the Sentihood data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_QA_B.tsv"), sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(os.path.join(data_dir, "dev_QA_B.tsv"), sep="\t").values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_QA_B.tsv"), sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[1]))
            label = tokenization.convert_to_unicode(str(line[3]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("text_b=", text_b)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Semeval_single_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['positive', 'neutral', 'negative', 'conflict', 'none']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label)
            )
        return examples


class Semeval_NLI_M_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_NLI_M.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev_NLI_M.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_NLI_M.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['positive', 'neutral', 'negative', 'conflict', 'none']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[3]))
            text_b = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Semeval_QA_M_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_QA_M.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev_QA_M.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_QA_M.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['positive', 'neutral', 'negative', 'conflict', 'none']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[3]))
            text_b = tokenization.convert_to_unicode(str(line[2]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Semeval_NLI_B_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_NLI_B.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev_NLI_B.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_NLI_B.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples


class Semeval_QA_B_Processor(DataProcessor):
    """Processor for the Semeval 2014 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        train_data = pd.read_csv(
            os.path.join(data_dir, "train_QA_B.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(train_data, "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        dev_data = pd.read_csv(
            os.path.join(data_dir, "dev_QA_B.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(dev_data, "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        test_data = pd.read_csv(
            os.path.join(data_dir, "test_QA_B.csv"), header=None, sep="\t"
        ).values
        return self._create_examples(test_data, "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            #  if i>50:break
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(str(line[2]))
            text_b = tokenization.convert_to_unicode(str(line[3]))
            label = tokenization.convert_to_unicode(str(line[1]))
            if i % 1000 == 0:
                print(i)
                print("guid=", guid)
                print("text_a=", text_a)
                print("label=", label)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
            )
        return examples
