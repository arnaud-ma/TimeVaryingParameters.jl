module OptimizationOptimJLExt

import TimeVaryingParameters
using OptimizationOptimJL: OptimizationOptimJL as Optim

function __init__()
    return TimeVaryingParameters.DEFAULT_OPTIMIZER[] = Optim.BFGS()
end

end
