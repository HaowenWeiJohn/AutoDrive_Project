# import __init__ as booger
from model.SalsaNext.SalsaNext import *
from model.ResLSTM_torch.ConvLSTM_pytorch import *
from model.ResLSTM_torch.TimeDis import *

class ResLSTM(nn.Module):

    def __init__(self, nclasses, time_seq=5):
        super(ResLSTM, self).__init__()

        self.tdconv1 = TimeDistributed(ResContextBlock(9, 32), time_steps=5)
        self.tdconv2 = TimeDistributed(ResContextBlock(32,32), time_steps=5)
        self.tdconv3 = TimeDistributed(ResContextBlock(32,32), time_steps=5)

        self.convlstm1 = ConvLSTM(input_dim=32, hidden_dim=32,
                            kernel_size=(3,3), num_layers=2,
                            batch_first=True, bias=True,
                            return_all_layers=False)

        self.salsanext = SalsaNext(nclasses=nclasses)


    def forward(self, x):
        x = self.tdconv1(x)
        x = self.tdconv2(x)
        # x = self.tdconv3(x)

        # x = self.convlstm1(x)
        layer_output_list, last_state_list = self.convlstm1(x) # list of layer output

        output = self.salsanext(layer_output_list[-1][:, -1, :, :, :])

        return output # logistic output