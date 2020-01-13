import torch
import sys

exp_name = sys.argv[1]
in_file = f'/mnt/logdir/{exp_name}/checkpoints/ckpt-latest.dat'
out_file = f'/mnt/logdir/{exp_name}/checkpoints/weights_and_more-latest.dat'

ckpt = torch.load(in_file)
agent = ckpt['agent']
epoch = ckpt['epoch']
new_ckpt = {'optimizer': agent.optimizer, 'state_dict': agent.actor_critic.state_dict(), 'epoch': epoch }

torch.save(new_ckpt, out_file)


ckpt = torch.load(out_file)
agent.actor_critic.load_state_dict(ckpt['state_dict'])
agent.optimizer = ckpt['optimizer']

print('done')