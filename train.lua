-- This program was written by Franciszek Jemioło
-- In this file we train our network on the mnist data

require('torch')
require('paths')
require('nn')
require('mnistdataset')
require('model')
require('optim')
require('xlua')
-- Parse command line options
print '==> Processing options'

cmd = torch.CmdLine()
cmd:text()
cmd:text('MNIST training and testing program')
cmd:text()
cmd:text('Options:')
-- Do we use console version (lua train.lua) or we do we use it from itorch
cmd:option('-itorch', 1, 'Set to 1 for usage of itorch, or 0 for console')
-- Setting the seed
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 1, 'number of threads')
-- Using normalized or non normalized data
cmd:option('-normalized', 'yes', 'use normalized dataset? yes or no')
-- Model type, default - convnet
cmd:option('-model', 'convnet', 
    'type of model to construct: linear | mlp | convnet')
-- Model options
-- Using dropout
cmd:option('-dropout', 0, 'dropout rate [0,1), default not enabled')
cmd:option('-batch_normalization', 0, 'batch_normalization [0,1),' ..
    ' default not enabled')
-- Type of transfer function
cmd:option('-transfer', 'ReLU', 
    'type of transfer function: ReLU | Tanh | Sigmoid')
-- Count of hidden nodes in hidden layers
cmd:option('-hidden_count', 300, 'number of hidden neurons in hidden layers')
-- Defualt negative log loss
cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | margin')
-- Training options
-- Defaults to /results/
cmd:option('-save', 'results', 'subdirectory to save experiments in')
-- Defaults to stochastic gradient descent
cmd:option('-optimization', 'SGD', 'optimization methods: SGD | CG | LBFGS')
-- Learning rate, defaults to 0.001
cmd:option('-learning_rate', 1e-3, 'learning rate at beginning')
-- Batch size, defaults to 1
cmd:option('-batch_size', 10, 
    'the mini batch size, 1 is pure stochastic, batch size must be divider' .. 
    ' of 60k and 10k' .. ', for batch normalization > 1')
-- Weight decay, only for SGD, defualts to 0
cmd:option('-weight_decay', 0, 'weight decay only for SGD')
-- Momentum defaults to 0
cmd:option('-momentum', 0, 'momentum for SGD only')
-- Maximum number of iterations for CG and LBFGS, defaults to 5
cmd:option('-max_iter', 5, 'maximum number of iterations for CG and LBFGS')
cmd:text()
opt = cmd:parse(arg or {})

-- Setting float tensor as default for torch
torch.setdefaulttensortype('torch.FloatTensor')


-- Setting number of threads
torch.setnumthreads(opt.threads)
-- Setting manual seed
torch.manualSeed(opt.seed)

-- Setting our model of neural network
model = Model.create()

-- Setting args for our model 
args = {}

-- Set the usage of dropout layers
if opt.dropout == 0 then
    args.useDropout = false
else
    args.useDropout = true
    args.dropoutRate = opt.dropout
end

-- Set the usage of batch normalization
if opt.batch_normalization == 0 then
    args.useBatchNormalization = false
else
    args.useBatchNormalization = true
    args.batchNormalizationRate = opt.batch_normalization
end

-- Setting the transfer function
args.transferFunction = opt.transfer

-- Set number of hidde neurons
args.hiddenNeurons = opt.hidden_count

-- Model type
args.networkType = opt.model

-- Building our model
model:__init(args)

criterion = nil

-- Define loss criterion
if opt.loss == 'nll' then
    -- Adding the softmax layer
    model:addLogSoftMaxLayer()
    -- And of course the criterion
    criterion = nn.ClassNLLCriterion()
elseif opt.loss == 'margin' then
    criterion = nn.MultiMarginCriterion()
else
    print('Unknown loss criterion, aborting...')
    os.exit()
end

-- The matrix for recording current confusion across classes
confusion = optim.ConfusionMatrix(#model.classes)

-- Logging results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Getting the parameters from our model
parameters, gradParameters = model:getNeuralNetwork():getParameters()

-- We now configure optimizer
if opt.optimization == 'CG' then
    optimState = {
        maxIter = opt.maxIter
    }
    optimMethod = optim.cg
elseif opt.optimization == 'LBFGS' then
    optimState = {
        learningRate = opt.learning_rate,
        maxIter = opt.max_iter,
        -- This is magic number
        nCorrection = 10
    }
    optimMethod = optim.lbfgs
elseif opt.optimization == 'SGD' then
    optimState = {
        learningRate = opt.learning_rate,
        weightDecay = opt.weightDecay,
        momentum = opt.momentum,
        learningRateDecay = 1e-7
    }
    optimMethod = optim.sgd
else
    print('Unkown optimization method, aborting...')
    os.exit()
end

mnistDataset = MnistDataset.create()
mnistDataset:__init()

-- Download the data
mnistDataset:downloadAllData()

trainDataset = nil
testDataset = nil

-- Load the data to mem
if opt.normalized == 'yes' then
    trainDataset = mnistDataset:getNormalizedTrainDataset()
    testDataset = mnistDataset:getNormalizedTestDataset()
else
    trainDataset = mnistDataset:getTrainDataset()
    testDataset = mnistDataset:getTestDataset()
end


-- Now we define training function
function train(dataset)
    -- Just collecting some garbage
    collectgarbage()
    -- Epoch tracker
    epoch = epoch or 1
    
    -- Get current time
    local time = sys.clock()
    
    -- Setting model to training (for modules like Dropout)
    model:getNeuralNetwork():training()

    -- Shuffle the dataset at each epoch
    shuffle = torch.randperm(dataset.size)
    
    print('==> Doing online epoch # ' .. epoch .. 
        ' on training data [batchSize=' .. opt.batch_size .. ']')
    for t = 1, dataset.size,opt.batch_size do
        -- Display progress
        xlua.progress(t, dataset.size)

        local inputs = nil
        local targets = nil
        -- Creating mini batch of data
        inputs = torch.Tensor(opt.batch_size, 1, 28, 28)
        targets = torch.Tensor(opt.batch_size)
        j = 1
        for i = t, math.min(t + opt.batch_size - 1, dataset.size) do
            -- Loading new sample
            local sample = dataset[shuffle[i]]
            inputs[j] = sample.x
            targets[j] = sample.y
            j = j +1
        end

        -- Function to evaluate f(X) and df/dX
        local evaluate = function(x)
            -- Getting new parameters
            if x ~= parameters then
                parameters:copy(x)
            end
            
            -- Reseting gradients
            gradParameters:zero()
            
            -- F = average of all criterions
            local f = 0
            
            -- Evaluating function for complete mini batch
            local outputs = model:getNeuralNetwork():forward(inputs)
            
            -- Because errors == f (0 + errs = errs)
            f = criterion:forward(outputs, targets)
            local df_do = criterion:backward(outputs, targets)
            model:getNeuralNetwork():backward(inputs, df_do)

            -- Hack, otherwise doesn't work --> look in torch
            if opt.batch_size == 1 then
                confusion:add(outputs, targets)
            else
                confusion:batchAdd(outputs, targets)
            end           

            -- Returning f and df/dX
            return f, gradParameters
        end

        optimMethod(evaluate, parameters, optimState)
    end
    
    -- Time taken
    time = sys.clock() - time
    print("==> Time taken to learn whole dataset = " .. (time * 1000) .. 'ms')

    -- Log accuracy
    trainLogger:add{['% Mean class accuracy over train dataset'] = 
        confusion.totalValid * 100}

    -- Save current net
    local filename = paths.concat(opt.save, 'neuralnetwork.net')
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('==> Saving model to ' .. filename)
    torch.save(filename, model:getNeuralNetwork())
    
    -- Next epoch
    confusion:zero()
    epoch = epoch + 1
                    
end


-- Now we define test function
function test(dataset)
    -- Just collecting some garbage
    collectgarbage()
    -- Getting current time
    local time = sys.clock()
    
    -- Setting model to evaluate mode/ disabling dropout, batch normalization
    model:getNeuralNetwork():evaluate()
    
    -- Testing over the dataset
    print('==> Testing over given dataset:')
    -- Only convnet extra uses batch normalization
    for i = 1, dataset.size, opt.batch_size do
        -- Displaying progress
        xlua.progress(i, dataset.size)
        -- Getting new samples
        local inputs = torch.Tensor(opt.batch_size, 1, 28, 28)
        local targets = torch.Tensor(opt.batch_size)
        k = 1
        for j = i, math.min(i + opt.batch_size - 1, dataset.size) do
            -- Loading new sample
            local sample = dataset[j]
            inputs[k] = sample.x
            targets[k] = sample.y
            k = k + 1
        end

        -- Testing samples
        local predictions = model:getNeuralNetwork():forward(inputs)
        if opt.batch_size == 1 then
            confusion:add(predictions, targets)
        else
            confusion:batchAdd(predictions, targets)
        end
    end
    
    -- Time taken
    time = sys.clock() - time
    print("==> Time taken to test whole dataset = " .. (time * 1000) .. 'ms')
    
    -- Printing confusion matrix
    print(confusion)

    -- Logging accuracy
    testLogger:add{['% Mean class accuracy over test dataset'] = 
        confusion.totalValid * 100 }

    -- Zeroing the confusion on next iteration
    confusion:zero()
end


if opt.itorch == 0 then
    -- Now running our beautiful code
    print("==> Running beautiful code(training)")

    while true do
        train(trainDataset)
        test(testDataset)
    end
-- Else we are opening in itorch so we will call train and test ourselves
elseif opt.itorch == 1 then
    print("Using itorch for demo")
end
