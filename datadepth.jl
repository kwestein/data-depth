@everywhere using JuMP
@everywhere using Clp
@everywhere using Cbc

function chinnecksHeuristicsParallel(S, id)
    coverSet = {}
    coverSetIndices = {}
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
    n::Int = length(S)/size(S,1) -1
    model = Model(solver=ClpSolver())

    @defVar(model, a[1:n])
    @defVar(model, e[1:length(S)] >= 0)

    @defConstrRef constraints[1:size(S,1)]

    for i = 1:size(S,1)
        constraint = 0
        for j = 1:n
           constraint = constraint + (S[i,j] - p[j])*a[j]
        end
        constraints[i] = @addConstraint(model, constraint + e[i] >= epsilon)
    end

    done = false
    while !done

        @setObjective(model, Min, sum{e[i], i=1:length(S)})
        solve(model)

        if getObjectiveValue(model) < 0.5
            done = true
            break
        end

        candidates = [false for i=1:length(constraints)]
        for i = 1:length(candidates)
            if getDual(constraints[i]) > 0.0
                candidates[i] = true
            else
                candidates[i] = false
            end
        end


        Z = [100.0 for i=1:length(candidates)]

        np = nprocs()  # determine the number of processes available
        num_calls = length(candidates)
        completed = [false for i=1:length(candidates)]
        i = 1
        # function to produce the next work item from the queue.
        # in this case it's just an index.
        nextidx() = (idx=i; i+=1; idx)
        @sync begin
            for l = 1:length(workers())
                proc = workers()[l]
                if proc != myid() || np == 1
                    @async begin
                        while true
                            idx = nextidx()
                            if idx > num_calls
                                break
                            end

                            if candidates[idx]
                                 Z[idx], completed[idx] = remotecall_fetch(proc, evaluateCandidate, S, p, idx, coverSetIndices)
                                 if completed[idx]
                                     coverSet = [coverSet, {constraints[idx]}]
                                     coverSetIndices = [coverSetIndices, {idx}]
                                     break
                                 end
                             end
                         end
                    end
                end
            end
        end

        minConstraint = null
        minIndex = 0
        if !done
            minIndex = indmin(Z)
            minConstraint = constraints[minIndex]

            if (minConstraint != null)
                push!(coverSet, minConstraint)
                push!(coverSetIndices, minIndex)
                chgConstrRHS(minConstraint, -999)
            end
        end
    end

    length(coverSet)
end

function chinnecksHeuristics(S, id)
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

@everywhere function buildModel(S, p, coverSetIndices)
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
        if i in coverSetIndices
            constraints[i] = @addConstraint(model, constraint + e[i] >= -999)
        else
            constraints[i] = @addConstraint(model, constraint + e[i] >= epsilon)
        end
    end

    @setObjective(model, Min, sum{e[i], i=1:length(S)})

    return model, constraints
end

@everywhere function evaluateCandidate(S, p, idx, coverSetIndices)
    epsilon = 1
    local_model, local_constraints = buildModel(S, p, coverSetIndices)

    candidate = local_constraints[idx]
    chgConstrRHS(candidate, -999)

    solve(local_model)
    objective_value = getObjectiveValue(local_model)

    if objective_value == 0.0
        return objective_value, true
    else
        chgConstrRHS(candidate, epsilon)
    end

    return objective_value, false
end

function MIP(S, id)
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

    solve(model)
    depth = getObjectiveValue(model)
    convert(Int64, round(depth))
end

function cc(S, id, numConstraints)
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
           gradientC[i,j] = S[i,j] - p[j]
           gradientCsquare[i] = gradientCsquare[i] + (gradientC[i,j])^2
        end
    end
    
    #random point
    currentP = 100*rand(1,n)-50
    
    #loop start
    for maxLoop = 1:size(S,1)*1
        try
        result = 0
        results = evaluateAllConstraints(currentP,gradientC)
        
        #get violated constraint(s)
        r = unique([getViolatedConstraintID(results) for i=1:numConstraints])
        
        #update point with violated constraint(s)
        feasibilityVectorCoeffficient = [abs(epsilon-results[r[i]])/((gradientCsquare[r[i]])) for i=1:length(r)]
        z = [sum([feasibilityVectorCoeffficient[j]*gradientC[r[j],i] for j = 1:length(r)]) for i = 1:n]
        currentP = [currentP[i] + (z[i]/length(r)) for i=1:n]

        #checking number of violated constraints
        result = [sum([currentP[j]*gradientC[i,j] for j=1:n]) for i = 1:size(S,1)]
        tempDepth = length(find(x -> x < 1,result))

        #check if it has smaller depth and if it has smaller depth,record the constraint
        maxDepth = minimum([tempDepth, maxDepth])
        catch y
            continue
        end
    end
    return maxDepth
end

function cc(S, id)
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

function evaluateAllConstraints(p, constraints)
    s = size(constraints,1)
    numDims = length(constraints)/s - 1
    violatedAmount = [sum([p[j]*constraints[i,j] for j = 1:numDims]) for i = 1:s]
    return violatedAmount
end

#Get a random violated constraints
function getViolatedConstraintID(results)
    violated = find(x -> x < 1,results)
    return violated[rand(1:size(violated,1))]
end

function importCSVFile(filename)
    importFile(filename, ',')
end

function importFile(filename, delim)
    readdlm(filename, delim)
end

#You should set an upper time limit for each solution, say a few minutes
#The data to capture is:
# - for every model, statistics on size (number of features, number of
#instances)
# - for every point in every model: time for method 1, time for method 2,
#time for method 3 etc. plus an outcome (success or failure or timeout)
# - also run the complete set of models on 1 core, on 2 cores, on 3 cores,
#on 4 cores etc. and collect the data for each of these.

#for every point in every model: known depth, depth by method 1, depth
#by method 2, depth by method 3, etc.
#for every point in every model: time for method 1, time for method 2,
#time for method 3, etc.

#You can summarize simple stats like averge difference from actual, std
#devn of difference from actual etc. And you can plot it in histograms,
#performance profiles etc.
function experiment1(data, results)
        n = nprocs()
        tic()
        sweep_results = randomSweepingHyperplane(data, 1000)
        sweep_deepest_point = indmax(sweep_results)
        sweep_total_time = toc()
        sweep_error = [abs(results[i] - sweep_results[i]) for i=1:size(data,1)]

        projection_total_time = 0
        cc_total_time = 0
        MIP_total_time = 0
        chinneck_total_time = 0
        chinneck_parallel_total_time = 0

        projection_results = [0 for i=1:size(data,1)]
        cc_results = [0 for i=1:size(data,1)]
        MIP_results = [0 for i=1:size(data,1)]
        chinneck_results = [0 for i=1:size(data,1)]
        chinneck_parallel_results = [0 for i=1:size(data,1)]

        projection_error = [0 for i=1:size(data,1)]
        cc_error = [0 for i=1:size(data,1)]
        MIP_error = [0 for i=1:size(data,1)]
        chinneck_error = [0 for i=1:size(data,1)]
        chinneck_parallel_error = [0 for i=1:size(data,1)]

        println(n," workers results")
        println("---------------------")

        println("id","\t", "sweep","\t","proj","\t","cc","\t","MIP","\t\t","chnck","\t\t","chnckP","\t",
                "proj time","\t", "cc time","\t","MIP time","\t","chnck time","\t","chnckP time","\t",
                "sweep error","\t","proj error","\t","cc error","\t","MIP error","\t","chnck error","\t","chnckP error")

        for i=1:size(data, 1)
            tic()
            projection_results[i] = cc(data, i, 1)
            proj_time = toq()
            projection_total_time += proj_time
            projection_error[i] = abs(results[i] - projection_results[i])

            tic()
            cc_results[i] = cc(data, i, 3)
            cc_time = toq()
            cc_total_time += cc_time
            cc_error[i] = abs(results[i] - cc_results[i])

            tic()
            MIP_results[i] = MIP(data, i)
            MIP_time = toq()
            MIP_total_time += MIP_time
            MIP_error[i] = abs(results[i] - MIP_results[i])

            tic()
            chinneck_results[i] = chinnecksHeuristics(data, i)
            chnck_time = toq()
            chinneck_total_time += chnck_time
            chinneck_error[i] = abs(results[i] - chinneck_results[i])

            tic()
            chinneck_parallel_results[i] = chinnecksHeuristicsParallel(data, i)
            chnck_parallel_time = toq()
            chinneck_total_time += chnck_time
            chinneck_parallel_error[i] = abs(results[i] - chinneck_parallel_results[i])

            println(i,"\t", sweep_results[i],"\t\t",projection_results[i],"\t\t",cc_results[i],"\t\t",MIP_results[i],"\t\t",chinneck_results[i],"\t\t",chinneck_parallel_results[i],
            "\t\t",round(proj_time,3),"\t\t",round(cc_time,3),"\t\t",round(MIP_time,3),"\t\t", round(chnck_time,3),"\t\t", round(chnck_parallel_time,3),"\t\t",
            "\t",sweep_error[i],"\t\t\t",projection_error[i],"\t\t\t",cc_error[i],"\t\t\t",MIP_error[i],"\t\t\t", chinneck_error[i],"\t\t\t", chinneck_parallel_error[i])
        end
end

function main(filename)
    data = importCSVFile(string("datasets/",filename))
    results = importCSVFile(string("results/",filename))

    chinnecksHeuristics(data, 4)
    chinnecksHeuristicsParallel(data, 4)
    MIP(data, 4)

    #experiment1(data, results)
end
