using JuMP
using Clp
using Cbc
using Plotly

#TODO specify that all variables are unconstrained
#TODO importing data sets (Dan Chen's)
function chinnecksHeuristics(S, id, n, doPrint)
    coverSet = {}
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]
    epsilon = 1
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

function MIP(S, id, n, doPrint)
    p = S[id,:]
    S = S[[1:id-1,id+1:end],:]

    epsilon = 1
    M = 1000

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

function contourPlotResults(set, results) #TODO
    Plotly.signin("kirstenwesteinde", "bfod5kcm69")
    data = [
        "x" => set[:,1],
        "y" => set[:,2],
        "z" => results,
        "type" => "contour"
    ]
    response = Plotly.plot(data)
    plot_url = response["url"]
end

function scatterPlotPoints(set) #TODO
    Plotly.signin("kirstenwesteinde", "bfod5kcm69")
    data = [
        "x" => set[:,1],
        "y" => set[:,2],
        "type" => "scatter"
    ]

    response = Plotly.plot(data)
    plot_url = response["url"]
end

data = importCSVFile("2dpoints.csv")
numDimensions = ndims(data)
chinnecksResults = chinnecksHeuristics(data, 25, numDimensions, false)
mipResults = MIP(data, 25, numDimensions, false)

println("Chinneck's results: ",chinnecksResults,", MIP results: ",mipResults) #TODO: chinneck's results are often incorrect

