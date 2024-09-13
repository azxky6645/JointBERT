import os
import re
from glob import glob
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset


class SpecificDomainNER(Dataset):
    def __init__(self, args, split, tokenizer, ner_to_index):
        super(SpecificDomainNER, self).__init__()
        self.args = args
        self.split = split
        self.tokenizer = tokenizer
        self.ner_to_index = ner_to_index
        self.samples = []

        self._load_data()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        inp = torch.tensor(example['input_ids']).to(self.args.device)
        attention_mask = inp.ne(self.tokenizer.pad_token_id).to(torch.uint8).to(self.args.device)
        out = torch.tensor(example['out']).to(self.args.device)
        return {'input_ids': inp, 'attention_mask': attention_mask}, out

    def _load_data(self):
        filenames = glob(os.path.join(self.args.data_root, self.split, '*.txt'))
        filenames = list(filter(lambda x: "논문" in x, filenames))
        filenames = tqdm(filenames)
        for f in filenames:
            filenames.set_description(f"parsing {os.path.basename(f)}")
            sample = self._parse(f)
            self.samples.extend(sample)

    def _tokenize(self, example):
        inp, tokens, prefix = self._transform_source_text(example['inp'])
        ner_ids, ner_labels = self._transform_target_text(example['out'], tokens, prefix)

        assert len(inp) == len(ner_ids)

        inp = self._pad_sequence(inp, self.tokenizer.pad_token_id)
        ner_ids = self._pad_sequence(ner_ids, self.tokenizer.pad_token_id)

        return {'input_ids': inp, 'out': ner_ids}

    # def _parse(self, filename):
    #     documents = []
    #     doc_id, doc_type, text, annotations = None, None, None, []
    #     with open(filename, 'r', encoding='utf-8') as f:
    #         for line in f.readlines():
    #             line = line.rstrip('\n')
    #             if not line:
    #                 documents.append(Document(doc_id, doc_type, text, annotations))
    #                 doc_id, doc_type, text, annotations = None, None, None, []
    #                 continue
    #
    #             if line.startswith("doc_id"):
    #                 _, doc_id = line.split('\t')
    #             elif line.startswith("doc_type"):
    #                 _, doc_type = line.split('\t')
    #             elif line.startswith("text"):
    #                 _, text = line.split('\t')
    #             else:
    #                 annotation = self._parse_annotation_line(line)
    #                 annotation.verify(text)
    #                 annotations.append(annotation)
    #
    #     for d in documents:
    #         d.verify_annotations()
    #     return documents

    def _parse(self, filename):
        documents = []
        with open(filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            doc_id, doc_type, inp, out = None, None, None, None
            for line in lines:
                line = line.rstrip('\n')
                if not line:
                    documents.append({'doc_id': doc_id, 'doc_type': doc_type, 'inp': inp, 'out': out})
                    doc_id, doc_type, inp, out = None, None, None, None
                    continue

                if line.startswith("#doc_id"):
                    _, doc_id = line.split('\t')
                elif line.startswith("#doc_type"):
                    _, doc_type = line.split('\t')
                elif line.startswith("#inp"):
                    _, inp = line.split('\t')
                elif line.startswith("#out"):
                    _, out = line.split('\t')
        documents = list(map(lambda x: self._tokenize(x), documents))
        return documents

    def _transform_source_text(self, txt):
        tokens = self.tokenizer.tokenize(txt)
        prefix_sum_of_token_start_index = []
        sum = 0
        for i, token in enumerate(tokens):
            if i == 0:
                prefix_sum_of_token_start_index.append(0)
                sum += len(token) - 1
            else:
                prefix_sum_of_token_start_index.append(sum)
                sum += len(token)
        return self.tokenizer.encode(txt), tokens, prefix_sum_of_token_start_index

    def _transform_target_text(self, txt, tokens, prefix_sum_of_token_start_index):
        regex_ner = re.compile('<(.+?):Term>')
        regex_filter_res = regex_ner.finditer(txt)

        ner_tags, ner_texts, start_end_indexes = [], [], []
        count = 0
        for i in regex_filter_res:
            ner_tag = i[0][-5:-1]
            ner_text = i[1]
            start_index = i.start() - 7 * count
            end_index = i.end() - 7 - 7 * count

            ner_tags.append(ner_tag)
            ner_texts.append(ner_text)
            start_end_indexes.append((start_index, end_index))
            count += 1

        ner_labels = []
        entity_index = 0
        is_entity_going = True

        for token, index in zip(tokens, prefix_sum_of_token_start_index):
            if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
                index += 1  # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

            if entity_index < len(start_end_indexes):
                start, end = start_end_indexes[entity_index]

                if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                    is_entity_going = True
                    entity_index = entity_index + 1 if entity_index + 1 < len(start_end_indexes) else entity_index
                    start, end = start_end_indexes[entity_index]

                if start <= index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                    entity_tag = ner_tags[entity_index]
                    if is_entity_going is True:
                        entity_tag = 'B-' + entity_tag
                        ner_labels.append(entity_tag)
                        is_entity_going = False
                    else:
                        entity_tag = 'I-' + entity_tag
                        ner_labels.append(entity_tag)
                else:
                    is_entity_going = True
                    entity_tag = 'O'
                    ner_labels.append(entity_tag)

            else:
                entity_tag = 'O'
                ner_labels.append(entity_tag)
        ner_ids = [self.ner_to_index['[CLS]']] + [self.ner_to_index[tag] for tag in ner_labels]
        ner_ids += [self.ner_to_index['[SEP]']]
        # ner_ids = [self.ignore_index] + [self.ner_to_index[tag] for tag in ner_labels]
        # ner_ids += [self.ignore_index]

        return ner_ids, ner_labels

    def pad_sequence(self, inputs, padding_token_idx):
        if len(inputs) < self.args.max_len:
            pad = np.array([padding_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.args.max_len]
        return inputs

    # @staticmethod
    # def _parse_annotation_line(line):
    #     fields = [f for f in line.split('\t')]
    #     id_ = fields[0]
    #     begin = int(fields[1])
    #     end = int(fields[2])
    #     type_ = fields[3]
    #     entity = fields[4]
    #
    #     ret = Annotation(id_, entity, type_, begin, end)
    #     if len(ret.entity) != ret.end - ret.begin:
    #         raise FormatError(f"Text {entity} has length {len(entity)}, end-start {end} - {begin} is {end - begin}")
    #
    #     return ret

    def _pad_sequence(self, sequence, _id):
        if len(sequence) < self.args.max_len:
            pad = [_id] * (self.args.max_len - len(sequence))
            sequence = sequence + pad
        else:
            sequence = sequence[:self.args.max_len]
        return sequence

    # def _tokenize(self, example: Document):
    #     text = example.text
    #     annotations = example.annotations
    #
    #     tokens = self.tokenizer.tokenize(text)
    #
    #     count = 0
    #     a_idx = 0
    #     ner_tags = []
    #     while a_idx < len(annotations):
    #         a = annotations[a_idx]
    #         if count >= a.begin:
    #             sub_tokens = self.tokenizer.tokenize(text[a.begin: a.end])
    #             # sub_ner_tags = []
    #             # for i, s_t in enumerate(sub_tokens):
    #             #     if a.type_ == 'TM' or a.type_ == 'TR':
    #             #         sub_tag_id = self.ner_to_index[f"B-Term"] if "▁" in s_t else self.ner_to_index[f"I-Term"]
    #             #         # sub_tag_id = self.ner_to_index[f"B-Term"] if sub_ner_tags[-1] == self.ner_to_index["B-Term"] else self.ner_to_index[f"I-Term"]
    #             #     else:
    #             #         sub_tag_id = self.ner_to_index["O"]
    #             #     sub_ner_tags.append(sub_tag_id)
    #
    #             if a.type_ == 'TM' or a.type_ == 'TR':
    #                 sub_ner_tags = [self.ner_to_index["B-Term"]] + ([self.ner_to_index["I-Term"]] * (len(sub_tokens) - 1))
    #             else:
    #                 sub_ner_tags = [self.ner_to_index["O"]] * len(sub_tokens)
    #             ner_tags.extend(sub_ner_tags)
    #
    #             count = a.end
    #             a_idx += 1
    #         else:
    #             sub_tokens = self.tokenizer.tokenize(text[count: a.begin])
    #             ner_tags.extend([self.ner_to_index["O"]] * len(sub_tokens))
    #
    #             count = a.begin
    #
    #     a = annotations[-1]
    #     if a.begin <= count:
    #         sub_tokens = self.tokenizer.tokenize(text[count: len(text)])
    #         ner_tags.extend([self.ner_to_index['O']] * len(sub_tokens))
    #
    #     tokens = [self.tokenizer.cls_token] + tokens + [self.tokenizer.sep_token]
    #     ner_tags = [self.ignore_index] + ner_tags + [self.ignore_index]
    #     input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
    #
    #     tokens = self._pad_sequence(tokens, self.tokenizer.pad_token)
    #     ner_tags = self._pad_sequence(ner_tags, self.ignore_index)
    #     input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
    #
    #     assert len(input_ids) == len(ner_tags) == len(tokens)
    #     return {'input_ids': input_ids, 'ner_tags': ner_tags, 'tokens': tokens}


class NamedEntityRecognitionDataset(Dataset):
    def __init__(self, args, split, tokenizer, ner_to_index):
        super(NamedEntityRecognitionDataset, self).__init__()
        self.args = args
        self.samples, self.labels = self.load_data(os.path.join(args.data_root, split))
        self.tokenizer = tokenizer
        self.ner_to_index = ner_to_index

    def __getitem__(self, idx):
        inp, tokens, prefix = self._transform_source_text(self.samples[idx])
        ner_ids, ner_labels = self._transform_target_text(self.labels[idx], tokens, prefix)

        inp = self.pad_sequence(inp, self.tokenizer.pad_token_id)
        ner_ids = self.pad_sequence(ner_ids, self.tokenizer.pad_token_id)

        inp = torch.tensor(inp).long().to(self.args.device)
        attention_mask = inp.ne(self.tokenizer.pad_token_id).float().to(self.args.device)
        out = torch.tensor(ner_ids).long().to(self.args.device)
        return {'input_ids': inp, 'attention_mask': attention_mask}, out

    def __len__(self):
        return len(self.samples)

    def load_data(self, data_dir):
        file_list = glob(os.path.join(data_dir, '*.txt'))

        source_texts, label_texts = [], []
        for f in file_list:
            inp, out = self._parsing(f)
            source_texts.extend(inp)
            label_texts.extend(out)
        return source_texts, label_texts

    @staticmethod
    def _parsing(file_name):
        with open(file_name, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

            count = 0
            inputs, outputs = [], []
            for line in lines:
                if line.startswith('##'):
                    count += 1
                else:
                    count = 0
                if count == 2:
                    inputs.append(line[3:].strip())
                elif count == 3:
                    outputs.append(line[3:].strip())
        return inputs, outputs

    def _transform_source_text(self, txt):
        tokens = self.tokenizer.tokenize(txt)
        prefix_sum_of_token_start_index = []
        sum = 0
        for i, token in enumerate(tokens):
            if i == 0:
                prefix_sum_of_token_start_index.append(0)
                sum += len(token) - 1
            else:
                prefix_sum_of_token_start_index.append(sum)
                sum += len(token)
        return self.tokenizer.encode(txt), tokens, prefix_sum_of_token_start_index

    def _transform_target_text(self, txt, tokens, prefix_sum_of_token_start_index):
        regex_ner = re.compile('<(.+?):[A-Z]{3}>')
        regex_filter_res = regex_ner.finditer(txt)

        ner_tags, ner_texts, start_end_indexes = [], [], []
        count = 0
        for i in regex_filter_res:
            ner_tag = i[0][-4:-1]
            ner_text = i[1]
            start_index = i.start() - 6 * count
            end_index = i.end() - 6 - 6 * count

            ner_tags.append(ner_tag)
            ner_texts.append(ner_text)
            start_end_indexes.append((start_index, end_index))
            count += 1

        ner_labels = []
        entity_index = 0
        is_entity_going = True

        for token, index in zip(tokens, prefix_sum_of_token_start_index):
            if '▁' in token:  # 주의할 점!! '▁' 이것과 우리가 쓰는 underscore '_'는 서로 다른 토큰임
                index += 1  # 토큰이 띄어쓰기를 앞단에 포함한 경우 index 한개 앞으로 당김 # ('▁13', 9) -> ('13', 10)

            if entity_index < len(start_end_indexes):
                start, end = start_end_indexes[entity_index]

                if end < index:  # 엔티티 범위보다 현재 seq pos가 더 크면 다음 엔티티를 꺼내서 체크
                    is_entity_going = True
                    entity_index = entity_index + 1 if entity_index + 1 < len(start_end_indexes) else entity_index
                    start, end = start_end_indexes[entity_index]

                if start <= index < end:  # <13일:DAT>까지 -> ('▁13', 10, 'B-DAT') ('일까지', 12, 'I-DAT') 이런 경우가 포함됨, 포함 안시키려면 토큰의 length도 계산해서 제어해야함
                    entity_tag = ner_tags[entity_index]
                    if is_entity_going is True:
                        entity_tag = 'B-' + entity_tag
                        ner_labels.append(entity_tag)
                        is_entity_going = False
                    else:
                        entity_tag = 'I-' + entity_tag
                        ner_labels.append(entity_tag)
                else:
                    is_entity_going = True
                    entity_tag = 'O'
                    ner_labels.append(entity_tag)

            else:
                entity_tag = 'O'
                ner_labels.append(entity_tag)
        ner_ids = [self.ner_to_index['[CLS]']] + [self.ner_to_index[tag] for tag in ner_labels]
        ner_ids += [self.ner_to_index['[SEP]']]

        return ner_ids, ner_labels

    def pad_sequence(self, inputs, padding_token_idx):
        if len(inputs) < self.args.max_len:
            pad = np.array([padding_token_idx] * (self.args.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.args.max_len]
        return inputs
