def logs2epochacc(input_path, save_path, status='pretrain'):
    # Load the logs
    data_logs = None
    with open(input_path, 'r', encoding='utf-8') as f:
        data_logs = f.readlines()

    # select the epoch or the acc
    epoch_logs = [l[63:-1].split(' ') for l in data_logs if 'Epoch' in l]
    epoch_logs = sorted(epoch_logs, key=lambda x: int(x[1]))
    if status == 'pretrain':
        epoch_logs = [' '.join([l[1], l[3]]) for l in epoch_logs]
    else:
        epoch_logs = [' '.join([l[1], l[3], l[6][:-1]]) for l in epoch_logs]

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(epoch_logs))


logs2epochacc('test_pretrain.log', 'test_pretrain_plot.log', status='pretrain')
logs2epochacc('test_finetune.log', 'test_finetune_plot.log', status='finetune')
logs2epochacc('test_finetune_evaluate.log', 'test_finetune_evaluate_plot.log', status='finetune')

logs2epochacc('train_50M_pretrain.log', 'train_50M_pretrain_plot.log', status='pretrain')
logs2epochacc('train_50M_finetune.log', 'train_50M_finetune_plot.log', status='finetune')
logs2epochacc('train_50M_finetune_evaluate.log', 'train_50M_finetune_evaluate_plot.log', status='finetune')
