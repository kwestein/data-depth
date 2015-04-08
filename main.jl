include("helpers.jl")
include("experiments.jl")

function findAllDepths(filename)
    data, results = getDataAndResults(filename)

    allPointsAllAlgorithms(data, results)
end

function findDeepestPoint(filename)
    data = getData(filename)

    deepestPointExperiment(data)
end
