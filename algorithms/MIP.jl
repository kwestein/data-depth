@everywhere using JuMP
@everywhere using Cbc

@everywhere function MIP(S, id)
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

@everywhere function MIP(args...)
    S = args[1][1]
    id = args[1][2]

    MIP(S, id)
end
