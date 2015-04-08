data-depth
==========

Carleton fourth year engineering project - SYSC4907 


This project uses multiple algorithms to (i) find the data depth of any given point, and (ii) find the deepest point in a multidimensional set

####Instructions for use: 
- [Install Julia](http://julialang.org/downloads/)
- Clone this repository to your local machine using ````git clone git@github.com:kwestein/data-depth.git````
- Open your instance of Julia or your Julia IDE
- Make sure that the required packages are installed (JuMP, Cbc, and Clp) 
  - ````Pkg.init()````
  - ````Pkg.add("JuMP")````
  - ````Pkg.add("Cbc")````
  - ````Pkg.add("Clp")````
- Include/build main.jl and then start using!
- You can see how many processors you're using by calling ````nprocs()```` and you can add processes using ````addprocs(1)````. After adding a processor you need to rebuild main.jl

