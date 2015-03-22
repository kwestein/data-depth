@everywhere using JuMP
@everywhere using Clp
@everywhere using Cbc
@everywhere using Plotly

function chinnecksHeuristics(S, id, doPrint, doParallel)
    coverSet = {}
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
    n::Int = length(S)/size(S,1) -1
    model = Model(solver=ClpSolver())

    @defVar(model, a[1:n] )
    @defVar(model, e[1:length(S)] >= 0)

    @defConstrRef constraints[1:size(S,1)]

    for i = 1:size(S,1)
        constraint = 0
        for j = 1:n
           constraint = constraint + (S[i,j] - p[j])*a[j]
        end
        constraints[i] = @addConstraint(model, constraint + e[i] >= epsilon)
    end

    candidates = [false for i=1:length(constraints)]

    done = false
    while !done
        @setObjective(model, Min, sum{e[i], i=1:length(S)})
        if doPrint
            print(model)
        end

        solve(model)
        if getObjectiveValue(model) < 0.5
            done = true
            break
        end

        for i = 1:length(candidates)
            if getDual(constraints[i]) > 0.0
                candidates[i] = true
            else
                candidates[i] = false
            end
        end


        Z = [100.0 for i=1:length(candidates)]

        if(doParallel)
            @sync @parallel for j = 1:length(candidates)
                if candidates[j]
                    # candidate = constraints[j]
                    # chgConstrRHS(candidate, -999)

                    solve(model)
                    Z[j] = getObjectiveValue(model)

                    if Z[j] == 0.0
                        coverSet = [coverSet, {constraints[j]}]
                        done = true
                        break
                    else
                        # chgConstrRHS(candidate, epsilon)
                    end
                end
            end
        else
            for j = 1:length(candidates)
                if candidates[j]
                    candidate = constraints[j]
                    chgConstrRHS(candidate, -999)

                    solve(model)
                    Z[j] = getObjectiveValue(model)

                    if Z[j] == 0.0
                        coverSet = [coverSet, {constraints[j]}]
                        done = true
                        break
                    else
                        chgConstrRHS(candidate, epsilon)
                    end
                end
            end
        end

        minConstraint = null
        minIndex = 0
        if !done
            Znew = 9999999
            for i = 1:length(constraints)
                if candidates[i]
                    if Z[i] < Znew
                        Znew = Z[i]
                        minConstraint = constraints[i]
                        minIndex = i
                    end
                end
            end
            coverSet = [coverSet, {minConstraint}]
            chgConstrRHS(minConstraint, -999)
            candidates[minIndex] = false
        end
    end

    length(coverSet)
end

function MIP(S, id, doPrint)
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]

    epsilon = 1
    M = 1000
    n::Int = length(S)/size(S,1) -1

    model = Model(solver=CbcSolver())

    @defVar(model, a[1:n] )
    @defVar(model, y[1:length(S)], Bin)

    for i = 1:size(S,1)
        constraint = 0
        for j = 1:n
           constraint = constraint + (S[i,j] - p[j])*a[j]
        end
        @addConstraint(model, constraint >= epsilon - M*y[i])
    end

    @setObjective(model, Min, sum{y[i], i=1:length(S)})

    if doPrint
      print(model)
    end

    solve(model)
    depth = getObjectiveValue(model)
    convert(Int64, round(depth))
end

function projection(S, id, doPrint)
    #SS = S
    maxDepth = size(S,1)
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
    n::Int = length(S)/size(S,1) -1
    tempConstraint = Array(Any,size(S,1),n)

    #initialize gradientC and gradientCsquare
    gradientC = Array(Any,size(S,1),n)
    gradientCsquare = Array(Any,size(S,1),1)
    for i = 1:size(S,1)
        gradientCsquare[i] = 0
        for j = 1:n
           #temp = S[i,j] - p[j]
           gradientC[i,j] = S[i,j] - p[j]
           gradientCsquare[i] = gradientCsquare[i] + (gradientC[i,j])^2

        end
    end

    #select Random point
    currentP = 100*rand(1,n)-50

    #checking start here
    for maxLoop = 1:size(S,1)*10
        try
        result = 0
        #select Random constraint
        results = evaluatedAllConstraint(currentP,gradientC)
        randConstraint = getViolatedConstraintID(results)

        deno = sqrt(gradientCsquare[randConstraint])
        feaVecCoef = abs(epsilon-result)/(deno)

        for i = 1:n
            currentP[i] = currentP[i] + feaVecCoef*gradientC[randConstraint,i]
        end

        #checking number of violated constraint
        tempDepth = 0
        for i = 1:size(S,1)
            result = 0
            for j = 1:n
                result = result + currentP[j]*gradientC[i,j]
            end
            if result < 1
                tempDepth = tempDepth + 1
            end
        end

        #check if it has smaller depth and if it has smaller depth,record the constraint
        #if tempDepth< 10
        #    println(tempDepth)
        #end
        if tempDepth < maxDepth
            maxDepth = tempDepth
        end
        catch y
            continue
        end
    end
    return maxDepth
end

function cc(S, id, doPrint)
    maxDepth = size(S,1)
    p = S[id,:]
    SS = S
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
    n::Int = length(S)/size(S,1) -1
    tempConstraint = Array(Any,size(S,1),n)
    done = false

    #initialize gradientC and gradientCsquare
    gradientC = Array(Any,size(S,1),n)
    gradientCsquare = Array(Any,size(S,1),1)
    for i = 1:size(S,1)
        gradientCsquare[i] = 0
        for j = 1:n
           #temp = S[i,j] - p[j]
           gradientC[i,j] = S[i,j] - p[j]
           gradientCsquare[i] = gradientCsquare[i] + (gradientC[i,j])^2

        end

    end

    #select Random point
    currentP = 100*rand(1,n)-50

    currentConsId = 0
    for maxLoop = 1:1000
        try
        results = evaluatedAllConstraint(currentP,gradientC)

        r = Any[]
        for i = 1:3
            push!(r,getViolatedConstraintID(results))
            if i == 1
                continue
            end
            count = 0
            while r[i] == r[i-1] &&  count < 10000
                r[i] = getViolatedConstraintID(results)
                count = count + 1
            end
        end
        feaVecCoef = Any[]
        for i = 1:3
            push!(feaVecCoef,abs(epsilon-results[r[i]])/(sqrt(gradientCsquare[r[i]])))
        end

        for i = 1:n
            z = 0
            for j = 1:3
                z = z +feaVecCoef[j]*gradientC[r[j],i]
            end
            currentP[i] = currentP[i] + (z/3)
        end

        #checking number of violated constraint
        tempDepth = 0
        for i = 1:size(S,1)
            result = 0
            for j = 1:n
                result = result + currentP[j]*gradientC[i,j]
            end
            if result < 1
                tempDepth = tempDepth + 1
            end
        end

    #check if it has smaller depth and if it has smaller depth,record the constraint
        if tempDepth < maxDepth
            #tempConstraint = gradientC[randConstraint,1:end]
            maxDepth = tempDepth
        end
        catch y
            #continue
        end
    end
    #countViolated(SS,tempConstraint)
    return maxDepth
end

function randomSweepingHyperplane(S, numIterations, doParallel)
    numPoints = size(S,1)
    numDimensions::Int = length(S)/numPoints -1
    depth = [numPoints for i=1:numPoints]

    if(doParallel)
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
    else
        for i in 1:numIterations
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
                depth[i] = minimum([ascending_depth, descending_depth, depth[i]])
            end
        end
    end


    return depth
end

function evaluatedAllConstraint(p,constraints)
    s = size(constraints,1)
    n::Int = length(constraints)/s - 1
    results = Array(Any,s,1)
    for i = 1:size(constraints,1)
        result = 0
        for j = 1:n
            result = result + p[j]*constraints[i,j]
        end
        results[i] = result
    end
    return results
end

#Get a random violated constraints
function getViolatedConstraintID(results)
    violated = Any[]
    for i = 1:size(results,1)
        if results[i] < 1
            push!(violated,i)
        end
    end
    return violated[rand(1:size(violated,1))]
end

function importCSVFile(filename)
    importFile(filename, ',')
end

function importFile(filename, delim)
    readdlm(filename, delim)
end

function scatterPlotPoints(set)
    Plotly.signin("kirstenwesteinde", "bfod5kcm69")
    data = [[
        "x" => set[:, 1],
        "y" => set[:, 2],
        "type" => "scatter"
    ]]
    response = Plotly.plot(data, ["filename" => "depth-data-scatter", "fileopt" => "overwrite"])
    plot_url = response["url"]
end

#for every point in every model: known depth, depth by method 1, depth
#by method 2, depth by method 3, etc.
#for every point in every model: time for method 1, time for method 2,
#time for method 3, etc.

#You can summarize simple stats like averge difference from actual, std
#devn of difference from actual etc. And you can plot it in histograms,
#performance profiles etc.
function experiment1(data, results)
    tic()
    sweep_results = randomSweepingHyperplane(data, 1000)
    sweep_deepest_point = indmax(sweep_results)
    sweep_total_time = toc()
    sweep_error = [abs(results[i] - sweep_results[i]) for i=1:size(data,1)]

    projection_total_time = 0
    MIP_total_time = 0
    chinneck_total_time = 0

    projection_results = [0 for i=1:size(data,1)]
    MIP_results = [0 for i=1:size(data,1)]
    chinneck_results = [0 for i=1:size(data,1)]

    projection_error = [0 for i=1:size(data,1)]
    MIP_error = [0 for i=1:size(data,1)]
    chinneck_error = [0 for i=1:size(data,1)]

    println("id","\t", "sweep","\t","proj","\t","MIP","\t\t","chnck","\t", "proj time","\t","MIP time","\t","chnck time","\t",
        "sweep error","\t","proj error","\t","MIP error","\t","chnck error")

    for i=1:size(data, 1)
        tic()
        projection_results[i] = projection(data, i, false)
        proj_time = toq()
        projection_total_time += proj_time
        projection_error[i] = abs(results[i] - projection_results[i])

        tic()
        MIP_results[i] = MIP(data, i, false)
        MIP_time = toq()
        MIP_total_time += MIP_time
        MIP_error[i] = abs(results[i] - MIP_results[i])

        tic()
        chinneck_results[i] = chinnecksHeuristics(data, i,false)
        chnck_time = toq()
        chinneck_total_time += chnck_time
        chinneck_error[i] = abs(results[i] - chinneck_results[i])

        println(i,"\t", sweep_results[i],"\t\t",projection_results[i],"\t\t",MIP_results[i],"\t\t",chinneck_results[i],
          "\t\t",round(proj_time,3),"\t\t",round(MIP_time,3),"\t\t", round(chnck_time,3),"\t\t",
           "\t",sweep_error[i],"\t\t\t",projection_error[i],"\t\t\t",MIP_error[i],"\t\t\t", chinneck_error[i])
    end

    projection_deepest_point = indmax(projection_results)
    chinneck_deepest_point = indmax(chinneck_results)
    MIP_deepest_point = indmax(MIP_results)
    println("Deepest: ","\t",MIP_deepest_point,"\t",round(MIP_total_time,3),"\t",sum(MIP_error))
end

function parallelExperiment(data)
    tic()
    randomSweepingHyperplane(data, 1000, true)
    sweep_parallel_time = toc()

    tic()
    randomSweepingHyperplane(data, 1000, false)
    sweep_sequential_time = toc()
    println("Sweep parallel time: ",round(sweep_parallel_time,3)," Sweep Sequential time: ",round(sweep_sequential_time,3))

    println("Parallel time Sequential time")
    for i=1:size(data, 1)
        tic()
        chinnecksHeuristics(data, i,false, true)
        chnck_parallel_time = toq()

        tic()
        chinnecksHeuristics(data, i,false, false)
        chnck_time = toq()

        println(round(chnck_parallel_time,3),"\t\t",round(chnck_time,3))
    end

        tic()
        chinnecksHeuristics(data, 9,false, false)
        chnck_time = toq()

        println("Parallel time: ",round(chnck_parallel_time,3),"\t\tSequential time",round(chnck_time,3))
end

#You should set an upper time limit for each solution, say a few minutes
#The data to capture is:
# - for every model, statistics on size (number of features, number of
#instances)
# - for every point in every model: time for method 1, time for method 2,
#time for method 3 etc. plus an outcome (success or failure or timeout)
# - also run the complete set of models on 1 core, on 2 cores, on 3 cores,
#on 4 cores etc. and collect the data for each of these.
function experiment2(time_limit, num_cores, data, results)

end

function main(filename)
    data = importCSVFile(string("datasets/",filename))
    results = importCSVFile(string("results/",filename))
    parallelExperiment(data)
end
