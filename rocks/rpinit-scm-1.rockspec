package = "rpinit"
version = "scm-1"

source = {
  url = "git://github.com/piotriwojcik/rpinit",
  tag = "master"
}

description = {
  summary = "Random projection weight initialisation schemes for Torch7 neural network modules",
  detailed = [[
                Random projection weight initialisation schemes for Torch7 neural network modules.
  ]],
  homepage = "https://github.com/piotriwojcik/rpinit",
  license = "MIT"
}

dependencies = {
  "torch >= 7.0",
  "nn >= 1.0",
  "math-walshtransform >= 1.18"
}

build = {
  type = "builtin",
  modules = {
    rpinit = "rpinit.lua"
  }
}