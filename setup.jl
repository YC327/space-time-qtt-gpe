using Pkg
using TOML

VERSION >= v"1.12" || error("Use Julia 1.12 or newer; the supplied dependency snapshots used Julia 1.12.2.")
Pkg.activate(@__DIR__)
config = TOML.parsefile(joinpath(@__DIR__, "Project.toml"))
specs = [Pkg.PackageSpec(name=name, url=config["sources"][name]["url"],
                         rev=config["sources"][name]["rev"])
         for name in ("MPSCore", "QTTCore")]
try
    # Install both unregistered packages together at their recorded revisions.
    Pkg.add(specs; preserve=Pkg.PRESERVE_ALL)
    Pkg.instantiate()
    Pkg.precompile()
catch
    println(stderr, "Setup failed. Check the error above and access to the two upstream repositories and exact commits listed in Project.toml. No fallback to a newer revision is performed.")
    rethrow()
end
println("Dependencies installed. See README.md for the required ground-state input and run command.")
