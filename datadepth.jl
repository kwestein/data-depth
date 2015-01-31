using JuMP
using Clp
using Cbc
using Plotly

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
    z = {results[1] => [1]}
    for i=2:length(results)
        if haskey(z, results[i])
            push!(z[results[i]], i)
        else
            z[results[i]] = [i]
        end
    end

    data = [["x" => Int64[], "y" => Int64[], "type" => "scatter"] for i=1:length(values(z))]
    for k in keys(z)
        x_for_trace = Int64[]
        y_for_trace = Int64[]

        for index=1:length(z[k])
            push!(x_for_trace, x[z[k][index]])
            push!(y_for_trace, y[z[k][index]])
        end

        push!(x_for_trace, x[z[k][1]])
        push!(y_for_trace, y[z[k][1]])

        trace = [
            "x" => x_for_trace,
            "y" => y_for_trace,
            "type" => "scatter",
            "name" => k
        ]

        push!(data, trace)
    end

    response = Plotly.plot(data, ["filename" => "simple-contour", "fileopt" => "overwrite"])
    plot_url = response["url"]
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
    data = importCSVFile("2dpointsordered.csv")
    tic()
    chinnecks_result = chinnecksHeuristics(data, 17, false)
    chinneck_time = toq()
    tic()
    mip_result = MIP(data, 17, false)
    mip_time = toq()
    tic()
    all_depths_result = findAllDepths(data)
    deepest_depth = maximum(all_depths_result)#indmax for index
    all_depths_time = toq()

    println("Chinneck's result: ",chinnecks_result," took ",chinneck_time," seconds")
    println("MIP result: ",mip_result," took ",mip_time," seconds")
    println("Deepest point: ",deepest_depth," took ",all_depths_time," seconds")
end

main()
