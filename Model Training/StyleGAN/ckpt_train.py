import os
ckpt_list = sorted(os.listdir('./checkpoint/custom'))
if not ckpt_list:
    os.system(f'python train.py  ./Dataset')
else:
    target_ckpt = ckpt_list[-1]
    print(target_ckpt, 'train')
    os.system(f'python train.py --ckpt ./checkpoint/custom/{target_ckpt} ./Dataset')