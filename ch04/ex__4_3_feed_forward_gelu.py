#%% Dependencies
import torch
import torch.nn as nn

#%% Test class
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        y = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2/torch.pi)) * 
                                      (x+0.044715 * torch.pow(x,3))))
        
        return y


#%% Test GELU
import matplotlib.pyplot as plt
gelu,relu = GELU(), nn.ReLU()

x=torch.linspace(-3,3,100)
y_gelu,y_relu=gelu(x),relu(x)

plt.figure(figsize=(8,3))

for i,(y,label) in enumerate(zip([y_gelu,y_relu],['GELU','RELU']),1):
    plt.subplot(1,2,i)
    plt.plot(x,y)
    plt.title(f'{label} Activation Function')
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.savefig("gelu_relu_activation_functions.png")



