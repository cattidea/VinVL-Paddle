import torch
import paddle

pretrained_torch_model = '/mnt/disk2T/Data/Research/Multi-Modal-Pretraining/2021-VinVL-CVPR/pretrained_model/base/checkpoint-1340000/pytorch_model.bin'
pretrained_torch_model = torch.load(pretrained_torch_model)

pretrained_paddle_model = {}

# List of parameters to be transposed
transpose_list = ['classifier.weight',
                  'bert.img_embedding.weight',
                  'bert.pooler.dense.weight']

for i in range(24):
    transpose_list.append(
        'bert.encoder.layer.{}.output.dense.weight'.format(str(i))
    )
    transpose_list.append(
        'bert.encoder.layer.{}.attention.self.query.weight'.format(str(i))
    )
    transpose_list.append(
        'bert.encoder.layer.{}.attention.self.key.weight'.format(str(i))
    )
    transpose_list.append(
        'bert.encoder.layer.{}.attention.self.value.weight'.format(str(i))
    )
    transpose_list.append(
        'bert.encoder.layer.{}.attention.output.dense.weight'.format(str(i))
    )
    transpose_list.append(
        'bert.encoder.layer.{}.intermediate.dense.weight'.format(str(i))
    )

for k, v in pretrained_torch_model.items():
    if k in transpose_list:
        v = v.transpose(1, 0)
    pretrained_paddle_model[k] = paddle.to_tensor(v.cpu().numpy(), place='cpu')

paddle.save(pretrained_paddle_model, 'paddle_model.bin')

