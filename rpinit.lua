local WT = require 'Math.WalshTransform'
local rpinit = {}


local function calcFan(module)
  local typename = torch.type(module)
  if typename:find('SpatialConvolution') or typename:find('SpatialFullConvolution') then
    return module.nInputPlane * module.kW * module.kH, module.nOutputPlane * module.kW * module.kH
  else
    error("Unsupported module")
  end
end

-- Add rpinitWeights to nn.Module
nn.Module.rpinitWeights = function(self, initMethodName, trainSize)
  local nninit = require 'nninit'
  local accessor = 'weight'
  local t = {
    kaiming   = function(module, trainSize) module:init(accessor, nninit.kaiming, {gain = 'relu'}) end,
    rp_gauss  = function(module, trainSize) module:init(accessor, rpinit.gauss) end,
    rp_sparse = function(module, trainSize) module:init(accessor, rpinit.sparse) end,
    rp_achl   = function(module, trainSize) module:init(accessor, rpinit.achl) end,
    rp_count  = function(module, trainSize) module:init(accessor, rpinit.count) end,
    rp_hada   = function(module, trainSize) module:init(accessor, rpinit.hada, trainSize) end,
  }
  if not t[initMethodName] then
    error("Undefined initialization scheme")
  end
  t[initMethodName](self, trainSize)
  return self
end


rpinit.gauss = function(module, tensor)
  local fanIn = calcFan(module)
  tensor:normal(0, math.sqrt(2 / fanIn))
  return module
end

li_matrix = function(d, k, s)
  s = s or math.sqrt(d)
  local p = 1 / (2 * s)
  local t = torch.multinomial(torch.Tensor{p, 1-2*p, p}, d * k, true):double()
  t = t - 2
  t = t * math.sqrt(s / k)
  return t
end

rpinit.sparse = function(module, tensor, transpose)
  transpose = transpose or false
  local fanIn = calcFan(module)
  local D = fanIn
  local K = module.nOutputPlane

  if transpose then
    local tmp = D
    D = K
    K = tmp
  end

  local R = li_matrix(D, K)
  local scaleFactor = math.sqrt(2/fanIn) / torch.std(R)
  R = R * scaleFactor
  tensor:copy(R)

  return module
end

rpinit.achl = function(module, tensor, transpose)
  transpose = transpose or false
  local fanIn = calcFan(module)
  local D = fanIn
  local K = module.nOutputPlane

  if transpose then
    local tmp = D
    D = K
    K = tmp
  end

  local R = li_matrix(D, K, 3)
  local scaleFactor = math.sqrt(2/fanIn) / torch.std(R)
  R = R * scaleFactor
  tensor:copy(R)

  return module
end

rpinit.count = function(module, tensor, transpose, sigma)
  transpose = transpose or false
  sigma = sigma or 1.0
  local fanIn = calcFan(module)
  local D = fanIn
  local K = module.nOutputPlane

  if transpose then
    local tmp = D
    D = K
    K = tmp
  end

  local R = torch.Tensor(K, D):double() -- transposed because of memory layout

  for i=1,D do
    R[torch.random(1, K)][i] = torch.random(0, 1)*2-1
  end

  R = R * sigma
  tensor:copy(R)
  return module
end

rpinit.hada = function(module, tensor, trainSize, transpose)
  transpose = transpose or false
  if trainSize == nil then
    error("rp_hada initialization requires to specify the training set size!")
  end
  local fanIn, fanOut = calcFan(module)
  local D = fanIn
  local K = module.nOutputPlane

  if transpose then
    local tmp = D
    D = K
    K = tmp
  end

  local d = torch.pow(2, torch.ceil(torch.log(D) / torch.log(2)))
  local q = 1
  local q2 = torch.pow(torch.log(trainSize), 2) / D
  if q2 < q then q = q2 end

  local R = torch.multinomial(torch.Tensor{1-q, q}, d * K, true):double()-1
  local Q = torch.Tensor(R:size()):normal(0, math.sqrt(1 / q))
  R:cmul(Q)

  R = R:view(K, d) -- transposed because of memory layout

  -- FHT
  for row=1,K do
      R[{ row, {}}] = math.sqrt(d) * torch.Tensor(WT.fht(torch.totable(R[{ row, {}}])))
  end

  -- flip cols signs
  for col=1,d do
    R[{ {}, col}] = (torch.random(0, 1)*2-1) * R[{ {}, col}]
  end

  R = R[{ {}, {1,D}}]

  local scaleFactor = math.sqrt(2/fanIn) / torch.std(R)
  R = R * scaleFactor
  tensor:copy(R)

  return module
end


return rpinit