using JuMP
using Clp
using Cbc
using Plotly

function chinnecksHeuristics(S, id, doPrint)
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

        @sync @parallel for j = 1:length(candidates)
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
    currentP = 10*rand(1,n)-5

    r1 = 1
    r2 = 1
    r3 = 1
    currentConsId = 0
    for maxLoop = 1:1000
        count = 0
        while !done
            result1 = 0
            result2 = 0
            result3 = 0

            #TODO : Make array of random number instead hardcode
            #select THREE Random constraint
            r1 = rand(1:size(S,1))
            r2 = rand(1:size(S,1))
            r3 = rand(1:size(S,1))
            while r1 == r2 || r2 == r3 || r1 == r3
                if r1 == r2
                    r2 = rand(1:size(S,1))
                else
                    r3 = rand(1:size(S,1))
                end
            end
            #evaluate constraints chosen
            for i = 1:n
                result1 = result1 + currentP[i]*gradientC[r1,i]
                result2 = result2 + currentP[i]*gradientC[r2,i]
                result3 = result3 + currentP[i]*gradientC[r3,i]
            end
            if result1 + result2 + result3 < 3
                break
            end
            count = count + 1
            if count > 10000
                done = true
            end
        end
        if done == true
            break
        end
        feaVecCoef1 = abs(epsilon-result1)/(sqrt(gradientCsquare[r1]))
        feaVecCoef2 = abs(epsilon-result2)/(sqrt(gradientCsquare[r2]))
        feaVecCoef3 = abs(epsilon-result3)/(sqrt(gradientCsquare[r3]))
        for i = 1:n
            currentP[i] = currentP[i] + ((feaVecCoef1*gradientC[r1,i]+feaVecCoef2*gradientC[r2,i]+feaVecCoef3*gradientC[r3,i])/3)
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
            tempConstraint = gradientC[randConstraint,1:end]
            maxDepth = tempDepth
        end
    end
    #countViolated(SS,tempConstraint)
    return maxDepth
end

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

function calculateTotalError(results, correctDepths)
    sum([abs(results[i] - correctDepths[i]) for i=1:length(results)])
end

function runAndPrintAllAlgorithms(data)
    println("id","\t", "proj","\t","sweep","\t","MIP","\t\t","chnck","\t", "proj time","\t","MIP time","\t","chnck time")
    tic()
    S = randomSweepingHyperplane(data, 5000)
    sweepTime = toq()
    projection_total_time = 0
    MIP_total_time = 0
    chinneck_total_time = 0

    for i = 1:size(data,1)
        tic()
        projection_result = projection(data, i, false)
        proj_time = toq()
        projection_total_time += proj_time
        tic()
        MIP_result = MIP(data, i, false)
        MIP_time = toq()
        MIP_total_time += MIP_time
        tic()
        chinnecks_result = chinnecksHeuristics(data, i,false)
        chnck_time = toq()
        chinneck_total_time += chnck_time
        println(i,"\t", projection_result,"\t\t",S[i],"\t\t",MIP_result,"\t\t",chinnecks_result,"\t\t",round(proj_time,3),"\t\t",round(MIP_time,3),"\t\t", round(chnck_time,3))
    end

    println("Sweep: ", round(sweepTime,3),", Projection: ",round(projection_total_time,3),", MIP: ",round(MIP_total_time,3),", Chinneck: ",round(chinneck_total_time,3))
end

function findAllDepths(S, results)
    tic()
    sweep_results = randomSweepingHyperplane(S, 1000)
    sweep_deepest_point = indmax(sweep_results)
    sweep_deepest_time = toc()

    tic()
    chinneck_results = [chinnecksHeuristics(S, i,false) for i=1:size(S,1)]
    chinneck_deepest_point = indmax(chinneck_results)
    chinneck_deepest_time = toc()

    tic()
    MIP_results = [MIP(S, i,false) for i=1:size(S,1)]
    MIP_deepest_point = indmax(MIP_results)
    MIP_deepest_time = toc()

    tic()
    projection_results = [projection(S, i,false) for i=1:size(S,1)]
    projection_deepest_point = indmax(projection_results)
    projection_deepest_time = toc()

    sweep_error = calculateTotalError(sweep_results, results)
    chinneck_error = calculateTotalError(chinneck_results, results)
    projection_error = calculateTotalError(projection_results, results)
    MIP_error = calculateTotalError(MIP_results, results)

    println("Sweeping hyperplane deepest point: ",sweep_deepest_point," took ",round(sweep_deepest_time,3)," seconds with ",sweep_error," total errors")
    println("Chinneck deepest point: ",chinneck_deepest_point," took ",round(chinneck_deepest_time,3)," seconds with ",chinneck_error," total errors")
    println("MIP deepest point: ",MIP_deepest_point," took ",round(MIP_deepest_time,3)," seconds with 0 total errors")
    println("Projection deepest point: ",projection_deepest_point," took ",round(projection_deepest_time,3)," seconds with ",projection_error," total errors")
end

function main(filename) #TODO: implement timeout
    data = importCSVFile(string("datasets/",filename))
    results = importCSVFile(string("results/",filename))
    findAllDepths(data, results)
end

main("DavidBremner.csv")
