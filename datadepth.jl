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
    n = ndims(S)
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
    n = ndims(S)

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
    maxDepth = 0;
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
    n = ndims(S)
    #WRONG DIMENSION?-----------------------------------
    println("# of element : ", n)
    println("total points : ", size(S,1))

    model = Model()

    @defVar(model, a[1:n] )
    @defVar(model, e[1:length(S)] >= 0)

    @defConstrRef constraints[1:size(S,1)]
    gradientC = Array(Any,size(S,1),n)
    gradientCsquare = Array(Any,size(S,1),1)

    #initalizing constraints,gradient C and gradient C square
    for i = 1:size(S,1)
        constraint = 0
        gradientCsquare[i] = 0
        for j = 1:n
           gradientC[i,j] = (S[i,j] - p[j])
           gradientCsquare[i] = gradientCsquare[i] + (gradientC[i,j])^2
           constraint = constraint + (S[i,j] - p[j])*a[j]
        end
        constraints[i] = @addConstraint(model, constraint + e[i] >= epsilon)
    end

    #select Random point
    currentP = 10*rand(1,n)


    for maxLoop = 1:1000000
        result = 0
        #select Random constraint
        randConstraint = rand(1:size(S,1))
        #evaluate constraint chosen
        for i = 1:n
            result = result + currentP[i]*gradientC[randConstraint,i]
        end

        #if violate,update point
        if result < 1
            feaVecCoef = -(result-epsilon)/gradientCsquare[randConstraint]
            fv = zeros(Any,n)
            for i = 1:n
                currentP[i] = currentP[i] + feaVecCoef*gradientC[i]
            end
        #else choose another point
        else
            randConstraint = rand(1:size(S,1))
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

    #check if it has bigger depth
        if tempDepth > maxDepth
            maxDepth = tempDepth
        end
    end

    return maxDepth
end


function randomSweepingHyperplane(S)

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
    tic()
    projection_result = projection(data,17,false)
    projection_time = toq()
    tic()
    chinnecks_result = chinnecksHeuristics(data, 17, false)
    chinneck_time = toq()
    tic()
    mip_result = MIP(data, 17, false)
    mip_time = toq()
    tic()
    all_depths_result = findAllDepths(data)
    deepest_depth = maximum(all_depths_result)
    all_depths_time = toq()
    
    println("Projection result: ",projection_result," took ",projection_time," seconds")
    println("Chinneck's result: ",chinnecks_result," took ",chinneck_time," seconds")
    println("MIP result: ",mip_result," took ",mip_time," seconds")
    println("Deepest point: ",deepest_depth," took ",all_depths_time," seconds")
end

main()
