-- This program was written by Franciszek Jemioło
-- In this file I declare models of neural networks.

require('torch')
require('paths')
require('nn')

Model = {}

Model.__index = Model

-- Constructor
function Model.create()
    local mdl = {}
    setmetatable(mdl, Model)
    return mdl
end


function Model:__init(args)
    -- Our 10 classes
    -- Classes cant be zero with ClassNLLCriterion :(
    self.classes = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10'}

    -- The width and height of input images
    self.geometry = {28, 28}

    -- How many pixels are in the image = 28*28
    self.pixelCount = 784

    -- Default transfer function is ReLU, can be set to Tanh or Sigmoid
    self.transferFunction = args.transferFunction or 'ReLU'

    -- Use dropout layers
    self.useDropout = args.useDropout

    -- Use batch normalization
    self.useBatchNormalization = args.useBatchNormalization

    self.batchNormalizationRate = args.batchNormalizationRate or 0.001

    -- Number of hidden units
    self.hiddenNeurons = args.hiddenNeurons or 300

    -- Dropout rate
    self.dropoutRate = args.dropoutRate or 0.3

    self.neuralNetwork = nil
    -- Our model neural network
    self:createNeuralNetwork(args.networkType)
end

-- If we want to turn off dropout
function Model:useDropout(choice)
    self.useDropout = choice
end

-- If we want to turn on/off batch normalization
function Model:useBatchNormalization(choice)
    self.useBatchNormalization = choice
end

-- If we want other transfer function
function Model:setTransferFunction(transferF)
    self.transferFunction = transferF
end
    
-- Adds dropout layer to our network with given epsilon if dropout enabled
function Model:addDropoutLayer(epsilon)
    if self.useDropout then
        self.neuralNetwork:add(nn.Dropout(epsilon))
    end
end

-- Adds spatial batch normalization layer to our network with given parameters
function Model:addBatchNormalizationLayer(nInput)
    if self.useBatchNormalization then
        self.neuralNetwork:add(nn.BatchNormalization(nInput))
    end
end


-- Adds spatial batch normalization layer to our network with given parameters
function Model:addSpatialBatchNormalizationLayer(nInputPlanes, rate)
    if self.useBatchNormalization then
        self.neuralNetwork:add(
            nn.SpatialBatchNormalization(nInputPlanes, rate))
    end
end

-- Adds LogSoftMax layer at the end
function Model:addLogSoftMaxLayer()
    self.neuralNetwork:add(nn.LogSoftMax())
end


-- Returns the wanted transfer function
function Model:getTransferFunction()
    if self.transferFunction == 'ReLU' then
        return nn.ReLU()
    elseif transferFunction == 'Tanh' then
        return nn.Tanh()
    elseif transferFunction == 'Sigmoid' then
        return nn.Sigmoid()
    else
        print("Unknown transfer function, exiting...")
        os.exit()
    end
end

-- This function creates a linear model
function Model:createLinear()
    self.neuralNetwork = nn.Sequential()
    self.neuralNetwork:add(nn.View(self.pixelCount))
    self:addBatchNormalizationLayer(self.pixelCount)
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.Linear(self.pixelCount, #(self.classes)))
end


-- This function creates standard multilayer perceptron
function Model:createMLP()
    self.neuralNetwork = nn.Sequential()
    self.neuralNetwork:add(nn.View(self.pixelCount))
    self:addBatchNormalizationLayer(self.pixelCount)
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.Linear(self.pixelCount, self.hiddenNeurons))
    self:addBatchNormalizationLayer(self.hiddenNeurons)
    -- Using ReLU instead of Tanh (for default)
    self.neuralNetwork:add(self:getTransferFunction())
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.Linear(self.hiddenNeurons, #(self.classes)))
    return self.neuralNetwork
end

-- This function returns a convolutional network
-- No type means the default model
-- The extra model means we want a bigger conv net with more layers
function Model:createConvNet()--convNetType, hiddenNeurons)
    self.neuralNetwork = nn.Sequential()
    -- Producing 32 output planes, the kernel width and heigth is 5x5
    -- First convolution block
    self.neuralNetwork:add(nn.SpatialConvolutionMM(1, 32, 5, 5))
    self:addSpatialBatchNormalizationLayer(32, self.batchNormalizationRate)
    self.neuralNetwork:add(self:getTransferFunction())
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- Second convolution block
    self.neuralNetwork:add(nn.SpatialConvolution(32, 64, 5, 5))
    self:addSpatialBatchNormalizationLayer(64, self.batchNormalizationRate)
    self.neuralNetwork:add(self:getTransferFunction())
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.SpatialMaxPooling(2, 2, 2, 2))

    -- Standard 2 layer neural net
    self.neuralNetwork:add(nn.View(1024))
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.Linear(1024, self.hiddenNeurons))
    self:addBatchNormalizationLayer(self.hiddenNeurons)
    self.neuralNetwork:add(self:getTransferFunction())
    self:addDropoutLayer(self.dropoutRate)
    self.neuralNetwork:add(nn.Linear(self.hiddenNeurons, #(self.classes)))
end

-- Creates the neural network for the model
function Model:createNeuralNetwork(choice)
    if choice == "linear" then
        self:createLinear()
    elseif choice == "mlp" then
        self:createMLP()
    elseif choice == "convnet" then
        self:createConvNet()
    else
        print("Unknown type of neural networks --> exiting")
        os.exit()
    end
end

-- This function returns kept neural network
function Model:getNeuralNetwork()
    return self.neuralNetwork
end