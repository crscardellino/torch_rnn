require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'mltools'
require 'optim'

-- Load and split the dataset
print("Loading the dataset")
local newsgroups_sparse = mltools.SparseDataset('resources/newsgroups', 'torch.CudaTensor')
local train, test = mltools.crossvalidation.train_test_split(newsgroups_sparse:getTorchDataset())

-- Define the network parameters
local input_dim = newsgroups_sparse.shape.cols
local output_dim = newsgroups_sparse.num_classes

-- Define the network architecture
print("Defining network architecture")
local mlp = nn.Sequential()
mlp:add(nn.Linear(input_dim, 2500))
mlp:add(nn.Tanh())
mlp:add(nn.Dropout(0.5))
mlp:add(nn.Linear(2500, output_dim))
mlp:add(nn.LogSoftMax())

-- Move the network to the GPU
mlp:cuda()

-- Define the Criterion and move it to GPU
local loss = nn.ClassNLLCriterion()
loss:cuda()

-- Define and set the trainer and move it to GPU
local trainer = nn.StochasticGradient(mlp, loss)
trainer.learningRate = 0.01
trainer:cuda()

-- Train the Multilayer Perceptron
print("Training network")
trainer:train(train)

-- Evaluate the MLP
print("Evaluating network")
local matrix = optim.ConfusionMatrix(newsgroups_sparse.num_classes)

for i = 1, test:size() do
    matrix:add(mlp:forward(test[i][1]), test[i][2])
end

print(matrix)
