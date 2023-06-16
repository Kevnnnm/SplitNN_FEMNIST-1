from torch import nn

class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.input_size = 784
    self.hidden_sizes = [500, 128]
    self.output_size = 10
    self.cut_layer = 500

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[0], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[1]),
      nn.Linear(self.hidden_sizes[1], self.output_size),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.second_part(self.first_part(x))
  

class ShadowNN(nn.Module):
  def __init__(self):
    super(ShadowNN, self).__init__()
    self.output_size = 10
    self.input_size = 784
    self.hidden_sizes = [500, 128]
    self.cut_layer = 500

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[0], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[1]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[1], self.output_size),
      
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.second_part(self.first_part(x))
  

class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
        nn.Linear(500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 784),
    )

  def forward(self, x):
    return self.layers(x)