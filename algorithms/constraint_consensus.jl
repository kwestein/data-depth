@everywhere function cc(S, id, numConstraints)
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
    for maxLoop = 1:size(S,1)*10
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

@everywhere function cc(args...)
    S = args[1][1]
    id = args[1][2]
    numConstraints = args[1][3]

    cc(S, id, numConstraints)
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
