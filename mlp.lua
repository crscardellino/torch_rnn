--[[
	Copyright (c) 2016, Cristian Cardellino.
	This work is licensed under the "New BSD License". 
	See LICENSE for more information.
]]--

require 'torch'
require 'nn'
require 'mltools'

-- Load and split the dataset
print("Loading the dataset")
local newsgroups_sparse = mltools.SparseDataset('resources/newsgroups')
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

-- Define the Criterion
local criterion = nn.ClasNLLCriterion()

-- Define and set the trainer
local trainer = nn.StochasticGradient(mlp, criterion)
trainer.learningRate = 0.01

-- Train the Multilayer Perceptron
print("Training network")
trainer:train(train)

-- Evaluate the MLP
print("Evaluating network")
local correct_values = 0

for i = 1, test:size() do
    local y_hat = mlp:forward(test[i][1])
    if y_hat == test[i][2] then
        print(string.format("The result was the same %d - %d", test[i][2], y_hat))
        correct_values = correct_values + 1
    else
        print(string.format("The result was different %d - %d", test[i][2], y_hat))
    end
end

local accuracy = correct_values / test:size()
print(string.format("Final accuracy: %.2f", accuracy))