import matplotlib.pyplot as plot  # 导入matplotlib绘图库
import numpy as np  # 导入numpy库

# pretrain
test_pre_data = np.loadtxt('../logs/test_pretrain_set.log')
test_pre_x = test_pre_data[:, 0]
test_pre_y = test_pre_data[:, 1]

train_pre_data = np.loadtxt('../logs/train_0.1_pretrain_set.log')
train_pre_x = train_pre_data[:, 0]
train_pre_y = train_pre_data[:, 1]

fig1, ax1 = plot.subplots()  # 创建图实例

ax1.plot(test_pre_x, test_pre_y, color='xkcd:cornflower', linewidth=2, label='Pretrain Data 1M')
ax1.plot(train_pre_x, train_pre_y, color='xkcd:light orange', linewidth=2, label='Pretrain Data 50M')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
plot.title('PreTrain on WIKI of different sizes')
ax1.legend()
fig1.savefig("PreTrain_WIKI.png", dpi=300)


# finetune
test_finetune_data = np.loadtxt('../logs/test_finetune_set.log')
test_finetune_x = test_finetune_data[:, 0][:63]
test_finetune_y = test_finetune_data[:, 1][:63]

train_finetune_data = np.loadtxt('../logs/train_0.1_finetune_set.log')
train_finetune_x = train_finetune_data[:, 0][:63]
train_finetune_y = train_finetune_data[:, 1][:63]

# evaluate
test_finetune_eval_data = np.loadtxt('../logs/test_finetune_evaluate_set.log')
test_finetune_eval_x = test_finetune_eval_data[:, 0][:63]
test_finetune_eval_y = test_finetune_eval_data[:, 1][:63]

train_finetune_eval_data = np.loadtxt('../logs/train_0.1_finetune_evaluate_set.log')
train_finetune_eval_x = train_finetune_eval_data[:, 0][:63]
train_finetune_eval_y = train_finetune_eval_data[:, 1][:63]


fig2, ax2 = plot.subplots()  # 创建图实例

ax2.plot(test_finetune_x, test_finetune_y, color='xkcd:cornflower', linewidth=2, label='Finetune Model 1M')
ax2.plot(train_finetune_x, train_finetune_y, color='xkcd:light orange', linewidth=2, label='Finetune Model 50M')
ax2.plot(test_finetune_eval_x, test_finetune_eval_y, color='xkcd:cornflower', ls="--", linewidth=2, label='Eval Model 1M')
ax2.plot(train_finetune_eval_x, train_finetune_eval_y, color='xkcd:light orange', ls="--", linewidth=2, label='Eval Model 50M')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
plot.title('Finetune on IMDB')
ax2.legend()
fig2.savefig("Finetune_IMDB.png", dpi=300)


# acc
test_finetune_eval_acc_x = test_finetune_eval_data[:, 0][:63]
test_finetune_eval_acc_y = test_finetune_eval_data[:, 2][:63]

train_finetune_eval_acc_x = train_finetune_eval_data[:, 0][:63]
train_finetune_eval_acc_y = train_finetune_eval_data[:, 2][:63]

fig3, ax3 = plot.subplots()  # 创建图实例

ax3.plot(test_finetune_eval_acc_x, test_finetune_eval_acc_y, color='xkcd:cornflower', linewidth=2, label='Eval Model 1M')
ax3.plot(train_finetune_eval_acc_x, train_finetune_eval_acc_y, color='xkcd:light orange', linewidth=2, label='Eval Model '
                                                                                                             '50M')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
plot.title('Eval on IMDB')
ax3.legend()
plot.show()
fig3.savefig("Eval_IMDB.png", dpi=300)
