import torch.nn as nn


class PixelShuffle3D(nn.Module):
    """

    三维PixelShuffle模块
    """
    def __init__(self, upscale_factor):
        """

        :param upscale_factor: tensor的放大倍数
        """
        super(PixelShuffle3D, self).__init__()

        self.upscale_factor = upscale_factor

    def forward(self, inputs):

        batch_size, channels, in_depth, in_height, in_width = inputs.size()

        channels //= self.upscale_factor ** 3

        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor

        input_view = inputs.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width)

        shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return shuffle_out.view(batch_size, channels, out_depth, out_height, out_width)


import torch
from time import time

upscale_factor = 2

# cpu
# ps = PixelShuffle3D(upscale_factor)
# inputData = torch.rand(2, 16 * upscale_factor ** 3, 128, 128, 128)

# gpu
ps = PixelShuffle3D(upscale_factor).cuda()
inputData = torch.rand(2, 16 * upscale_factor ** 3, 128, 128, 128).cuda()

# 测试模块效率(运行时间)
start = time()
output = ps(inputData)

print(time() - start)
print(inputData.size(), output.size())

# cpu四次运行时间
# 0.4194648265838623
# 0.419353723526001
# 0.3946359157562256
# 0.3737030029296875

# gpu四次运行时间
# 0.003687620162963867
# 0.0037784576416015625
# 0.0014619827270507812
# 0.0014755725860595703

# gpu运行在毫秒级别,可以接受