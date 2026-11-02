module OptimizationOptimJLExt

import TimeVaryingParameters
import ForwardDiff          # `AutoForwardDiff()`, the default adtype, is unusable without it
using OptimizationOptimJL: OptimizationOptimJL as Optim

function __init__()
    return TimeVaryingParameters.DEFAULT_OPTIMIZER[] = Optim.BFGS()
end

end
