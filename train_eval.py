from tqdm.auto import tqdm
import torch
from lr_warmup import get_cosine_schedule_with_warmup
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(args, model, optimizer, train_loader, dev_loader, dev_questions):

    # 1684是当batch_size = 16时候，训练数据总共的批次

    total_steps = args.num_epochs * 1684
    logging_step = 100
    validation = True
    scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                args.num_warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5,
                last_epoch=-1,
    )

    for epoch in range(args.num_epochs):

        model.train()
        print("Start Training ...")

        step = 1
        train_loss = train_acc = 0

        for data in tqdm(train_loader):

            data = [i.to(device) for i in data]
            # 模型输入: input_ids, token_type_ids, attention_mask, start_positions, end_positions
            # 模型输出: start_logits, end_logits, loss (当有start_positions/end_positions 时才能计算loss)
            # 因此只有在训练集有loss，在验证集、测试集上没有loss，只有acc
            output = model(input_ids=data[0], token_type_ids=data[1], attention_mask=data[2], start_positions=data[3],
                           end_positions=data[4])

            # 预测结果中选择最有可能的起始位置
            start_index = torch.argmax(output.start_logits, dim=1)
            end_index = torch.argmax(output.end_logits, dim=1)

            # 只有当起始位置都正确时，预测才正确
            train_acc += ((start_index == data[3]) & (end_index == data[4])).float().mean()
            train_loss += output.loss

            optimizer.step()
            if args.do_scheduler:
                scheduler.step()
            optimizer.zero_grad()
            step += 1
            if not scheduler:
                optimizer.param_groups[0]["lr"] -= args.learning_rate / (total_steps)

            if step % logging_step == 0:
                print(
                    f"Epoch {epoch + 1} | Step {step} | Train_loss = {train_loss.item() / logging_step:.3f}, Train_acc = {train_acc / logging_step:.3f}")
                train_loss = train_acc = 0
        if validation:
            print("Evaluating Dev Set ...")
            model.eval()
            with torch.no_grad():
                dev_acc = 0
                for i, data in enumerate(tqdm(dev_loader)):
                    output = model(input_ids=data[0].squeeze().to(device), token_type_ids=data[1].squeeze().to(device),
                                   attention_mask=data[2].squeeze().to(
                                       device))  # 之所以没有了loss因为output有可能为空，output.loss会报错
                    # 仅当答案文本完全匹配时预测才是正确的
                    dev_acc += evaluate(data, output) == dev_questions[i]["answer_text"]
                print(f"Validation | Epoch {epoch + 1} | acc = {dev_acc / len(dev_loader):.3f}")

    # 将模型及其配置文件保存到目录  args.saved_model
    # 目录下有两个文件 「pytorch_model.bin」 and 「config.json」
    # 可以使用重新加载保存的模型  model = BertForQuestionAnswering.from_pretrained("saved_model")
    print("Saving Model ...")
    model_save_dir = ars.save_model_path
    model.save_pretrained(model_save_dir)


def evaluate(data, output):
    answer = ''
    max_prob = float('-inf')
    num_of_windows = data[0].shape[1]

    for k in range(num_of_windows):
        # 通过选择最可能的开始位置/结束位置来获得答案
        start_prob, start_index = torch.max(output.start_logits[k], dim=0)
        end_prob, end_index = torch.max(output.end_logits[k], dim=0)

        # 答案的概率计算为 start_prob 和 end_prob 的总和
        prob = start_prob + end_prob

        # 更新答案(不同的窗口）
        if prob > max_prob:
            max_prob = prob
            # 把tokens解码为人能看懂的字符，例如 [1, 48, 29, 40] -> "深度学习"
            answer = tokenizer.decode(data[0][0][k][start_index: end_index + 1])
    # 除去字符的空格
    return answer.replace(' ', '')


def test(args, model, test_loader, test_questions):

    print("Evaluating Test Set ...")

    result = []

    model.eval()
    with torch.no_grad():
        for data in tqdm(test_loader):
            output = model(input_ids=data[0].squeeze(dim=0).to(device), token_type_ids=data[1].squeeze(dim=0).to(device),
                           attention_mask=data[2].squeeze(dim=0).to(device))
            result.append(evaluate(data, output))

    result_file = args.test_result_path
    with open(result_file, 'w') as f:
          # 生成文件的第一行
          f.write("ID,Answer\n")
          # 把问题的序号和预测答案写入文件中
          for i, test_question in enumerate(test_questions):
                f.write(f"{test_question['id']},{result[i].replace(',','')}\n")
    print(f"Completed! Result is in {result_file}")