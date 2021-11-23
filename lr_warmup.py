from torch.optim.lr_scheduler import LambdaLR
import math
# 效果没有学习率线性递减效果好（差了5个百分点）
def get_cosine_schedule_with_warmup(
  optimizer,
  num_warmup_steps,
  num_training_steps,
  num_cycles = 0.5,
  last_epoch = -1,
):
    def lr_lambda(current_step):
        # Warmup
        if current_step < num_warmup_steps:
          return float(current_step) / float(max(1, num_warmup_steps))
        # decadence
        progress = float(current_step - num_warmup_steps) / float(
          max(1, num_training_steps - num_warmup_steps)
        )
        return max(
          0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)