function randomSweepingHyperplane(S, numIterations)
    numPoints = size(S,1)
    numDimensions::Int = length(S)/numPoints -1
    depth = [numPoints for i=1:numPoints]

    depth = @sync @parallel min for i in 1:numIterations
        #random constraint
        coeff = 10*rand(1,numDimensions)-5

        #evaluate all point with constraint
        results = [sum([coeff[j]*S[i,j] for j = 1:numDimensions]) for i = 1:numPoints]

        #sorting ascending
        ascending_indices = sortperm(results)

        #sorting descending
        descending_indices = sortperm(results, rev=true)

        for i=1:numPoints
            ascending_depth = findin(ascending_indices, i) - 1
            descending_depth = findin(descending_indices, i) - 1
            depth[i] = minimum([ascending_depth, descending_depth])
        end
        depth
    end


    return depth
end
