-- This program was written by Franciszek Jemioło
-- It downloads the mnist dataset from remote location or retrieves one stored
-- on the computer in the given location

require('torch')
-- This is required for filename manipulation
require('paths')

MnistDataset = {}

MnistDataset.__index = MnistDataset

-- Constructor
function MnistDataset.create()
    mdat = {}
    setmetatable(mdat, MnistDataset)
    return mdat
end

-- Initializing our object
function MnistDataset:__init()
    -- Location set to the original source of mnist
    self.remoteLocation = 'yann.lecun.com/exdb/mnist/'
    -- Names of the files stored on LeCun's site.
    self.trainFileImages = 'train-images-idx3-ubyte'
    self.trainFileLabels = 'train-labels-idx1-ubyte'
    self.testFileImages = 't10k-images-idx3-ubyte'
    self.testFileLabels = 't10k-labels-idx1-ubyte'
    self.trainFileImagesGZ = self.trainFileImages .. ".gz"
    self.trainFileLabelsGZ = self.trainFileLabels .. ".gz"
    self.testFileImagesGZ = self.testFileImages .. ".gz"
    self.testFileLabelsGZ = self.testFileLabels .. ".gz"
    -- Name of the directory where we store our data
    self.storeDirectory = 'data'
end

-- Downloads the gz file from remote location and store in a directory 
function MnistDataset:downloadData(directoryName, fileName, fileNameGZ)
    if not paths.filep(paths.concat(directoryName, fileName)) then
        local file = (self.remoteLocation .. fileNameGZ)
        os.execute('wget -c ' .. file .. ' --random-wait; ' .. 'gunzip ' 
            .. fileNameGZ .. 
            '; mv ' .. fileName .. ' ' .. directoryName .. ';' )
    end
end

-- Reads raw mnist data and returns it in a format compatible for torch
function MnistDataset:readRawData(filename)
    local f = torch.DiskFile(filename)
    f:bigEndianEncoding()
    f:binary()
    -- Read the magic number.. (32 bit integer) 0x801 for label, 0x803 for data
    local magicNumber = f:readInt() - 0x800
    assert(magicNumber > 0)
    -- Because it is actually just number of dimensions (1 dim - label, 3 dim -
    -- number of images, number of rows, number of columns)
    local dims = torch.LongTensor(magicNumber)
    for i = 1, magicNumber do
        dims[i] = f:readInt()
        assert(dims[i] > 0)
    end
    local data = torch.ByteTensor(dims:storage())
    f:readByte(data:storage())
    f:close()
    return data
end

-- Function for creating our dataset
function MnistDataset:createDataset(dataFilename, labelsFilename)
    -- Getting the files
    local data = self:readRawData(dataFilename):float()
    local labels = self:readRawData(labelsFilename):float()
    -- Check if the data matches the labels
    assert(data:size(1) == labels:size(1))
    local dataset = {data = data, labels = labels, size = data:size(1)}
    -- Setting index function to access wanted elements
    setmetatable(dataset, {__index = function(self, idx)
        assert((idx > 0) and (idx <= self.size))
        -- 28 and 28 are the width and heigth, 1 - is the 1-channel
        -- The magic plus 1 at y is because ClassNLL criterion has to take
        -- classes numbers between 1 and N
        return {x = data[idx]:view(1,28,28), y = (labels[idx] + 1)}
    end})
    return dataset
end

-- Function for downloading all the data
function MnistDataset:downloadAllData()
    -- Create the directory for storage if one does not exist.
    if not paths.dirp(self.storeDirectory) then
        paths.mkdir(self.storeDirectory)
    end
    -- Check if the files are already there, if they are - we should not 
    -- download
    self:downloadData(self.storeDirectory, self.trainFileImages, 
        self.trainFileImagesGZ)
    self:downloadData(self.storeDirectory, self.trainFileLabels, 
        self.trainFileLabelsGZ)
    self:downloadData(self.storeDirectory, self.testFileImages, 
        self.testFileImagesGZ)
    self:downloadData(self.storeDirectory, self.testFileLabels, 
        self.testFileLabelsGZ)    

end

-- Function for getting the training dataset
function MnistDataset:getTrainDataset()
    -- We are under assumption that the data will be stored int the directory
    -- of the file of our script
    local path = paths.dirname(paths.thisfile())
    return self:createDataset(paths.concat(path, paths.concat(
        self.storeDirectory, self.trainFileImages)),
        paths.concat(path, paths.concat(self.storeDirectory, 
            self.trainFileLabels)))
end

-- Function for getting the test dataset
function MnistDataset:getTestDataset()
    -- We are under assumption that the data will be stored int the directory
    -- of the file of our script
    local path = paths.dirname(paths.thisfile())
    return self:createDataset(paths.concat(path, paths.concat(
        self.storeDirectory, self.testFileImages)),
        paths.concat(path, paths.concat(self.storeDirectory, 
            self.testFileLabels)))
end

-- Normalizing the dataset - 0 mean and 1 variance
function MnistDataset:normalize(dataset)
    local floatData = dataset.data:float()
    local mean = floatData:mean()
    local std = floatData:std()
    floatData:add(-mean)
    floatData:mul(1/std)
    local newDataset = {data = floatData, labels = dataset.labels, 
        size = floatData:size(1)}
    setmetatable(newDataset, {__index = function(self, idx)
        assert((idx > 0) and (idx <= self.size))
        return {x = floatData[idx]:view(1,28,28), y = (dataset.labels[idx] + 1)}
     end})
    return newDataset
end

-- Returns normalized training dataset
function MnistDataset:getNormalizedTrainDataset()
    return self:normalize(mnistDataset:getTrainDataset())
end

-- Returns normalized test dataset
function MnistDataset:getNormalizedTestDataset()
    return self:normalize(mnistDataset:getTestDataset())
end

    
    








