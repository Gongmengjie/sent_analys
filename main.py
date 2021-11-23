import torch
from transformers import AdamW, BertForQuestionAnswering, BertTokenizerFast
from set_seed import set_seed
from utils import load_data, dev_raw_data, test_raw_data
from train_eval import train, test
import argparse

parser = argparse.ArgumentParser(description='QA_bert')

parser.add_argument('--num_warmup_steps', type=int, default=1000)
parser.add_argument('--num_epochs', type=int, default=1)            # 一轮训练就收敛了
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--train_batch_size', type=int, default=16)
parser.add_argument('--max_question_len', type=int, default=40)
parser.add_argument('--max_paragraph_len', type=int, default=150)
parser.add_argument('--doc_stride', type=int, default=150)
parser.add_argument('--do_scheduler', type=bool, default=False)


parser.add_argument('--save_model_path', type=str, default='./save_model')
parser.add_argument('--test_result_path', type=str, default='./test_result/est_result.csv')


args = parser.parse_args()

if __name__ == "__main__":

    set_seed(73)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 基于bert预训练模型
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    # 加载数据
    train_loader, dev_loader, test_loader = load_data(args)
    dev_questions = dev_raw_data()[0]
    test_questions = test_raw_data()[0]
    # 损失函数
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # 训练&验证
    train(args, model, optimizer, train_loader, dev_loader, dev_questions)

    # 测试
    model = BertForQuestionAnswering.from_pretrained("saved_model")
    test(args, model, test_loader, test_questions)
