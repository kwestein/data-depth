@everywhere using JuMP
@everywhere using Clp

@everywhere function chinnecksHeuristicsParallel(S, id)
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

        np = nprocs() - 1  # determine the number of processes available
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

@everywhere function chinnecksHeuristicsParallel(args...)
    S = args[1][1]
    id = args[1][2]
    chinnecksHeuristicsParallel(S,id)
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
