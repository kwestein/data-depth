function importCSVFile(filename)
    importFile(filename, ',')
end

function importFile(filename, delim)
    readdlm(filename, delim)
end

#You must have at least 2 processors to run this function
function runWithTimeout(timeout, method, args...)
    try
        p = workers()
        response = RemoteRef()
        @async put!(response, remotecall_fetch(p[1], method, args))

        start=time()
        while !isready(response) && (time() - start) < timeout  
            sleep(0.1)
        end

        if !isready(response)
            interrupt(p[1]) 
            return -1
        else
            return fetch(response)
        end
    catch y
        return -1
    end
end

function getDataAndResults(filename)
    data = getData(filename)
    results = getResults(filename)

    return (data, results)
end

function getData(filename)
    importCSVFile(string("/Users/kirstenwesteinde/Documents/data-depth/datasets/",filename))
end

function getResults(filename)
    importCSVFile(string("/Users/kirstenwesteinde/Documents/data-depth/results/",filename))
end
