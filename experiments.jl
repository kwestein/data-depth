include("algorithms/chinnecks_heuristic_parallel.jl")
include("algorithms/chinnecks_heuristic_sequential.jl")
include("algorithms/MIP.jl")
include("algorithms/constraint_consensus.jl")
include("algorithms/random_sweeping_hyperplane.jl")

function allPointsAllAlgorithms(data, results)
    n = nprocs()
    tic()
    sweep_results = randomSweepingHyperplane(data, 1000)
    sweep_deepest_point = indmax(sweep_results)
    sweep_total_time = toq()
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

    println(n-1," workers results")
    println("---------------------")

    println("id","\t", "sweep","\t","proj","\t","cc","\t","MIP","\t","chnck","\t","chnckP","\t",
            "proj time","\t", "cc time","\t","MIP time","\t","chnck time","\t","chnckP time","\t",
            "sweep error","\t","proj error","\t","cc error","\t","MIP error","\t","chnck error","\t","chnckP error")

    for i=1:size(data,1)
        try
            tic()
            projection_results[i] = runWithTimeout(120, cc, data, i, 1)
            proj_time = toq()
            projection_total_time += proj_time
            if (projection_results[i] >= 0)
                projection_error[i] = abs(results[i] - projection_results[i])
            end

            tic()
            cc_results[i] = runWithTimeout(120, cc, data, i, 3)
            cc_time = toq()
            cc_total_time += cc_time
            if (cc_results[i] >= 0)
                cc_error[i] = abs(results[i] - cc_results[i])
            end

            tic()
            x = runWithTimeout(120, MIP, data, i)
            MIP_results[i] = x
            MIP_time = toq()
            MIP_total_time += MIP_time
            if (MIP_results[i] >= 0)
                MIP_error[i] = abs(results[i] - MIP_results[i])
            end

            tic()
            chinneck_results[i] = runWithTimeout(120, chinnecksHeuristics, data, i)
            chnck_time = toq()
            chinneck_total_time += chnck_time
            if (chinneck_results[i] >= 0)
                chinneck_error[i] = abs(results[i] - chinneck_results[i])
            end

            tic()
            chinneck_parallel_results[i] = runWithTimeout(120, chinnecksHeuristicsParallel, data, i)
            chnck_parallel_time = toq()
            chinneck_parallel_total_time += chnck_parallel_time
            if (chinneck_parallel_results[i] != null)
                chinneck_parallel_error[i] = abs(results[i] - chinneck_parallel_results[i])
            end

            println(i,"\t", sweep_results[i],"\t",
                    (projection_results[i] < 0 ? "timeout" : projection_results[i]),"\t",
                    (cc_results[i] < 0 ? "timeout" : cc_results[i]),"\t",
                    (MIP_results[i] < 0 ? "timeout" : MIP_results[i]),"\t",
                    (chinneck_results[i] < 0 ? "timeout" : chinneck_results[i]),"\t",
                    (chinneck_parallel_results[i] < 0 ? "timeout" : chinneck_parallel_results[i]),"\t",
                    round(proj_time,3),"\t",round(cc_time,3),"\t",round(MIP_time,3),"\t", 
                    round(chnck_time,3),"\t", round(chnck_parallel_time,3),"\t\t",
                    sweep_error[i],"\t\t",projection_error[i],"\t\t",cc_error[i],"\t\t",
                    MIP_error[i],"\t\t", chinneck_error[i],"\t\t", chinneck_parallel_error[i])
        catch
            continue
        end
    end
    println("Sweep time: ",round(sweep_total_time, 3))
end

function deepestPointExperiment(data, i=1000, j=size(data,1)/2)
    # tic()
    # c = deepestPointMIP(data, i, j)
    # ct = toq()
    tic()
    c = deepestPointChnk(data, i, j)
    ct = toq()

    println(i," iterations and ",j," points")
    # println("MIP deepest: ",c," in ",round(ct,3))
    println("Chnk deepest: ",c," in ",round(ct,3))
end

function deepestPointChnk(data, numIterations, numPoints)
    sweep_results = copy(randomSweepingHyperplane(data, numIterations))
    deepest_points = [0 for i=1:numPoints]
    for i = 1:numPoints
        deepest_points[i] = indmax(sweep_results)
        splice!(sweep_results, deepest_points[i])
    end

    chinnecks_results = [0 for i=1:size(data,1)]
    if (nprocs() > 1)
        for i=1:length(deepest_points)
            chinnecks_results[deepest_points[i]] = runWithTimeout(10, chinnecksHeuristics, data, deepest_points[i])
        end
    else
        for i=1:length(deepest_points)
            chinnecks_results[deepest_points[i]] = chinnecksHeuristics(data, deepest_points[i])
        end
    end

    return indmax(chinnecks_results)
end

function deepestPointMIP(data, numIterations, numPoints)
    sweep_results = copy(randomSweepingHyperplane(data, numIterations))
    deepest_points = [0 for i=1:numPoints]
    for i = 1:numPoints
        deepest_points[i] = indmax(sweep_results)
        splice!(sweep_results, deepest_points[i])
    end

    MIP_results = [0 for i=1:size(data,1)]
    if (nprocs() > 1)
        for i=1:length(deepest_points)
            MIP_results[deepest_points[i]] = runWithTimeout(10, MIP, data, deepest_points[i])
        end
    else
        for i=1:length(deepest_points)
            MIP_results[deepest_points[i]] = MIP(data, deepest_points[i])
        end
    end

    return indmax(MIP_results)
end

function sweepingHyperplaneExperiment(numIterations = 1000)
    data, results = getDataAndResults("2dpoints.csv")
    tic()
    randomSweepingHyperplane(data, numIterations)
    twod_t = toq()

    data, results = getDataAndResults("DavidBremner.csv")
    tic()
    randomSweepingHyperplane(data, numIterations)
    db_t = toq()

    data, results = getDataAndResults("BupaLiver.csv")
    tic()
    randomSweepingHyperplane(data, numIterations)
    bl_t = toq()

    data, results = getDataAndResults("PimaIndiansDiabetes.csv")
    tic()
    randomSweepingHyperplane(data, numIterations)
    pid_t = toq()

    println(nprocs()," workers. 2d: ",round(twod_t,3)," DB: ",round(db_t,3)," BL: ",round(bl_t,3)," PID: ",round(pid_t,3))
end

function chinneckParallelExperiment()
    println(nprocs()," workers")
    println("==================================")

    data = getData("2dpoints.csv")
    println("\n\n2dpoints.csv")
    for i = 1:25
        tic()
        runWithTimeout(10,chinnecksHeuristicsParallel,data,i)
        println(round(toq(),3))
    end

    data = getData("DavidBremner.csv")
    println("\n\nDavid Bremner")
    for i = 1:25
        tic()
        runWithTimeout(10,chinnecksHeuristicsParallel,data,i)
        println(round(toq(),3))
    end

    data = getData("BupaLiver.csv")
    println("\n\nBupa Liver")
    for i = 1:25
        tic()
        runWithTimeout(10,chinnecksHeuristicsParallel,data,i)
        println(round(toq(),3))
    end

    data = getData("PimaIndiansDiabetes.csv")
    println("\n\nPima Indians Diabetes")
    for i = 1:25
        tic()
        runWithTimeout(10,chinnecksHeuristicsParallel,data,i)
        println(round(toq(),3))
    end
end
