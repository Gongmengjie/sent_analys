import json
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")


# 从文件中读取数据
def read_data(file):
    with open(file, 'r', encoding="utf-8") as reader:
        data = json.load(reader)
    return data["questions"], data["paragraphs"]


def train_raw_data():
    train_questions, train_paragraphs = read_data("./data/train.json")
    return [train_questions, train_paragraphs]
def dev_raw_data():
    dev_questions, dev_paragraphs = read_data("./data/dev.json")
    return [dev_questions, dev_paragraphs]
def test_raw_data():
    test_questions, test_paragraphs = read_data("data./test.json")
    return [test_questions, test_paragraphs]


# 序列标志化
def tokenized_questions():
    train_questions_tokenized = tokenizer([train_question["question_text"] for train_question in train_raw_data()[0]],
                                              add_special_tokens=False)
    dev_questions_tokenized = tokenizer([dev_question["question_text"] for dev_question in dev_raw_data()[0]],
                                            add_special_tokens=False)
    test_questions_tokenized = tokenizer([test_question["question_text"] for test_question in test_raw_data()[0]],
                                             add_special_tokens=False)

    return [train_questions_tokenized, dev_questions_tokenized, test_questions_tokenized]

def tokenized_paragraphs():
    train_paragraphs_tokenized = tokenizer(train_raw_data()[1], add_special_tokens=False)
    dev_paragraphs_tokenized = tokenizer(dev_raw_data()[1], add_special_tokens=False)
    test_paragraphs_tokenized = tokenizer(train_raw_data()[1], add_special_tokens=False)

    return [train_paragraphs_tokenized, dev_paragraphs_tokenized, test_paragraphs_tokenized]


# batch itr
def load_data(args):
    train_set = QA_Dataset(args, "train", train_raw_data()[0], tokenized_questions()[0], tokenized_paragraphs()[0])
    dev_set = QA_Dataset(args, "dev", dev_raw_data()[0], tokenized_questions()[1], tokenized_paragraphs()[1])
    test_set = QA_Dataset(args, "test", test_raw_data()[0], tokenized_questions()[2], tokenized_paragraphs()[2])

    train_batch_size = args.train_batch_size

    train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, pin_memory=True)
    dev_loader = DataLoader(dev_set, batch_size=1, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, pin_memory=True)

    return train_loader, dev_loader, test_loader



class QA_Dataset(Dataset):

    def __init__(self, args, split, questions, tokenized_questions, tokenized_paragraphs):
        self.split = split
        self.questions = questions
        self.tokenized_questions = tokenized_questions
        self.tokenized_paragraphs = tokenized_paragraphs
        self.max_question_len = args.max_question_len
        self.max_paragraph_len = args.max_paragraph_len  # window的大小
        self.doc_stride = args.doc_stride

        # Input sequence length = [CLS] + question + [SEP] + paragraph + [SEP]
        self.max_seq_len = 1 + self.max_question_len + 1 + self.max_paragraph_len + 1

    def __len__(self):
        return len(self.questions)  # 有多少个问题，也就是样本的大小

    def __getitem__(self, idx):
        question = self.questions[idx]  # 获得第idx个样本
        tokenized_question = self.tokenized_questions[idx]
        tokenized_paragraph = self.tokenized_paragraphs[question["paragraph_id"]]

        if self.split == "train":

            answer_start_token = tokenized_paragraph.char_to_token(question["answer_start"])
            answer_end_token = tokenized_paragraph.char_to_token(question["answer_end"])
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_paragraph_len // 2,
                                         len(tokenized_paragraph) - self.max_paragraph_len))
            paragraph_end = paragraph_start + self.max_paragraph_len
            input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            input_ids_paragraph = tokenized_paragraph.ids[paragraph_start: paragraph_end] + [102]
            answer_start_token += len(input_ids_question) - paragraph_start
            answer_end_token += len(input_ids_question) - paragraph_start

            input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)
            return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(
                attention_mask), answer_start_token, answer_end_token
        # 测试集的数据处理方式与训练集不一样，需要设置多个窗口，防止跳过答案
        # 并且验证集要和测试集保持一直的处理方式
        else:
            input_ids_list, token_type_ids_list, attention_mask_list = [], [], []

            for i in range(0, len(tokenized_paragraph), self.doc_stride):

                input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
                input_ids_paragraph = tokenized_paragraph.ids[i: i + self.max_paragraph_len] + [102]
                input_ids, token_type_ids, attention_mask = self.padding(input_ids_question, input_ids_paragraph)

                input_ids_list.append(input_ids)
                token_type_ids_list.append(token_type_ids)
                attention_mask_list.append(attention_mask)

            return torch.tensor(input_ids_list), torch.tensor(token_type_ids_list), torch.tensor(attention_mask_list)

    def padding(self, input_ids_question, input_ids_paragraph):

        padding_len = self.max_seq_len - len(input_ids_question) - len(input_ids_paragraph)
        input_ids = input_ids_question + input_ids_paragraph + [0] * padding_len
        token_type_ids = [0] * len(input_ids_question) + [1] * len(input_ids_paragraph) + [0] * padding_len
        attention_mask = [1] * (len(input_ids_question) + len(input_ids_paragraph)) + [0] * padding_len

        return input_ids, token_type_ids, attention_mask

