using JuMP
using Clp
using Cbc
using Plotly

#type Point{T<:Real}
#  x::T
#  y::T
#end

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

        @parallel for j = 1:length(candidates)
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
    getObjectiveValue(model)
end

function projection(S, id, doPrint)
    #SS = S
    maxDepth = size(S,1)
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
    n::Int = length(S)/size(S,1) -1
    tempConstraint = Array(Any,size(S,1),n)

    #println("# of element : ", n)
    #println("total points : ", size(S,1) + 1)
    #print(p)

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

    r = 1
    randConstraint = 1
    currentConsId = 0
    for maxLoop = 1:5000
        result = 0
        #select Random constraint
        r = rand(1:size(S,1))
        while r == randConstraint
        #    print("oops")
            r = rand(1:size(S,1))
        end

        randConstraint = r

        #evaluate constraint chosen
        for i = 1:n
            result = result + currentP[i]*gradientC[randConstraint,i]
        end

        #if violate,update point
        if result < 1
            #feaVecCoef = (epsilon-result)/gradientCsquare[randConstraint]
            #println(epsilon-result)
            feaVecCoef = abs(epsilon-result)/(sqrt(gradientCsquare[randConstraint]))
            #fv = zeros(Any,n)
            for i = 1:n
                currentP[i] = currentP[i] + feaVecCoef*gradientC[randConstraint,i]
            end
        #else choose another point
        else
            #randConstraint = rand(1:size(S,1))
            continue
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
    end
    #println("Violated : ",countViolated(SS,tempConstraint))
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

function randomSweepingHyperplane(S)
depth = Array(Any,size(S,1),1)
        fill!(depth,size(S,1))
        n::Int = length(S)/size(S,1) -1

        for max = 1:1000
            results = zeros(size(S,1),1)
            #random constraint
            coeff = 10*rand(1,n)-5

            #evaluate all point with costraint
            for i = 1:size(S,1)
                result = 0
                for j = 1:n
                    result = result + coeff[j]*S[i,j]
                end
                results[i] = result
            end

            #sorting ascending
            tempArray = zeros(size(S,1),1)
            for i = 1: size(S,1)
                tempArray[i] = i
            end

            swapped = true
            while swapped
                swapped = false
                for i = 1:size(S,1)-1
                    if results[tempArray[i]]>results[tempArray[i+1]]
                        temp = tempArray[i]
                        tempArray[i] = tempArray[i+1]
                        tempArray[i+1] = temp
                        swapped = true
                    end
                end
            end

            for i = 1:size(S,1)
                if depth[i] > tempArray[i] -1
                    depth[i] = tempArray[i] - 1
                end
                #descending
                if depth[i] > size(S,1) - depth[i] - 1
                    depth[i] = size(S,1) - depth[i] - 1
                end
                depth[i] = round(depth[i])
            end

        end
        return depth
end

function importCSVFile(filename)
    importFile(filename, ',')
end

function importFile(filename, delim)
    readdlm(filename, delim)
end

function contourPlotResults(set, results)
    Plotly.signin("kirstenwesteinde", "bfod5kcm69")

    x = set[:, 1]
    y = set[:, 2]
    z = {results[1] => [Point(x[1], y[1])]}
    for i=2:length(results)
        if haskey(z, results[i])
            push!(z[results[i]], Point(x[i],y[i]))
        else
            z[results[i]] = [Point(x[i], y[i])]
        end
    end

    data = [["x" => Int64[], "y" => Int64[], "type" => "scatter"] for i=1:length(keys(z))]
    for k in keys(z)
        trace = [
            "x" => push!([z[k][i].x for i=1:length(z[k])], z[k][1].x),
            "y" => push!([z[k][i].y for i=1:length(z[k])], z[k][1].y),
            "type" => "scatter",
            "name" => k
        ]

        push!(data, trace)
    end

    response = Plotly.plot(data, ["filename" => "simple-contour", "fileopt" => "overwrite"])
    plot_url = response["url"]
    println(plot_url)
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

function findAllDepths(data)
    numPoints = size(data, 1)
    depths = [0 for i=1:numPoints]
    for i = 1:numPoints
        depths[i] = MIP(data, i, false)
    end
    if ndims(data) == 2
        contourPlotResults(data, depths)
    end
    depths
end

function main()
    data = importCSVFile("points2.csv")
    
    #tic()
    #projection_result = projection(data,17,false)
    #projection_time = toq()
    #tic()
    #chinnecks_result = chinnecksHeuristics(data, 17, false)
    #chinneck_time = toq()
    #tic()
    #mip_result = MIP(data, 17, false)
    #mip_time = toq()
    #tic()
    #all_depths_result = findAllDepths(data)
    #deepest_depth = maximum(all_depths_result)
    #all_depths_time = toq()
    
    #println("Projection result: ",projection_result," took ",projection_time," seconds")
    #println("Chinneck's result: ",chinnecks_result," took ",chinneck_time," seconds")
    #println("MIP result: ",mip_result," took ",mip_time," seconds")
    #println("Deepest point: ",deepest_depth," took ",all_depths_time," seconds")
    
    println("id","\t", "proj","\t","sweep","\t","MIP","\t","chnck")
    tic()
    S = randomHyperplane(data, false)
    sweepTime = toq()
    proj_time = 0
    MIP_time = 0
    chnck_time = 0

    for i = 1:size(data,1)
        tic()
        projection_result = projection(data, i, false)
        proj_time = proj_time + toq()
        tic()
        MIP_result = MIP(data, i, false)
        MIP_time = MIP_time + toq()
        tic()
        chinnecks_result = chinnecksHeuristics(data, i,false)
        chnck_time = chnck_time + toq()
        println(i,"\t", projection_result,"\t",S[i],"\t",MIP_result,"\t",chinnecks_result)
    end
    println("time\t",round(proj_time,3),"\t",round(sweepTime,3),"\t",round(MIP_time,3),"\t",round(chnck_time,3))
end

main()
