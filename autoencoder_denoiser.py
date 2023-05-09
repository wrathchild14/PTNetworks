import torch
import torch.nn as nn
import torch.optim as optim

class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DenoisingAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        
    def forward(self, x, noise_std=0.1):
        # add noise
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise
        
        # encode the noisy sequence
        encoded, (h_n, c_n) = self.encoder(x_noisy)
        
        # repeat the last encoded hidden state for each time step in the output sequence
        h_n = h_n.repeat(x.shape[1], 1, 1)
        
        # decode the encoded hidden state
        decoded, _ = self.decoder(h_n, (h_n, c_n))
        
        # map the encoded hidden state to the final output
        output = self.linear(decoded)
        
        return output

input_size = 10
hidden_size = 20
num_layers = 2
lr = 0.001
num_epochs = 100

model = DenoisingAutoencoder(input_size, hidden_size, num_layers)
optimizer = optim.Adam(model.parameters(), lr=lr)

loss_fn = nn.MSELoss()

for epoch in range(num_epochs):
    # generate some random data
    x = torch.randn(32, 10, 10)
    
    output = model(x)
    loss = loss_fn(output, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")