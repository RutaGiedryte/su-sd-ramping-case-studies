using BenchmarkTools
using TulipaEnergyModel
using TulipaIO
using DuckDB
using JuMP

case_studies_to_run = [
    "1h_basic",
    "1h_su_sd",
    "2h_basic",
    "2h_su_sd",
    "3h_basic",
    "3h_su_sd",
    "4h_basic",
    "4h_su_sd",
    "2+1h_basic",
    "2+1h_su_sd",
    "4+2h_basic",
    "4+2h_su_sd",
    "3+2h_basic",
    "3+2h_su_sd",
    "4+3h_basic",
    "4+3h_su_sd",
]

# DB connection helper
function input_setup(input_folder)
    connection = DBInterface.connect(DuckDB.DB)
    TulipaIO.read_csv_folder(
        connection,
        input_folder;
        schemas = TulipaEnergyModel.schema_per_table_name,
    )
    return connection
end

global energy_problem_solved = Dict()
global seeds = Dict()

# CREATE THE BENCHMARK SUITE
const SUITE = BenchmarkGroup()
SUITE["create_model"] = BenchmarkGroup()
SUITE["run_model"] = BenchmarkGroup()

for case in case_studies_to_run
    input_folder = joinpath(pwd(), "debugging/TulipaDataGeneration/$case")
    # input_folder = joinpath(pwd(), "test/inputs/$case")

    # Benchmark of creating the model
    SUITE["create_model"]["$case"] = @benchmarkable begin
        create_model!(energy_problem)
    end samples = 10 evals = 1 seconds = 864000 setup =
        (energy_problem = EnergyProblem(input_setup($input_folder)))

    key = "$case"
    seeds[case] = []
    # Benchmark of running the model
    SUITE["run_model"]["$case"] = @benchmarkable begin
        solve_model!(energy_problem)
    end samples = 10 evals = 1 seconds = 864000 setup = begin
        global seeds
        seed = rand(1:2e6)
        energy_problem = create_model!(EnergyProblem(input_setup($input_folder)))
        JuMP.set_attribute(energy_problem.model, "seed", seed)
        append!(seeds[$key], seed)
    end teardown = (global energy_problem_solved; energy_problem_solved[$key] = energy_problem)
end

results_of_run = run(SUITE; verbose = true)

debugFolder = mkpath(joinpath(pwd(), "debugging/results"))

# Save run times
BenchmarkTools.save("debugging/results/output.json", results_of_run)

# Save optimal solution
for (key, value) in energy_problem_solved
    exportFolder = mkpath(joinpath(debugFolder, key))

    # Variable Tables
    save_solution!(value)
    export_solution_to_csv_files(exportFolder, value)

    # Objective Value
    objValFile = joinpath(exportFolder, "objective_value.txt")
    write(objValFile, string(value.objective_value))

    # Seeds
    seedFile = joinpath(exportFolder, "seeds.txt")
    write(seedFile, join(string.(seeds[key], ";")))

    # Model.lp
    modelLpFile = joinpath(exportFolder, "model.lp")
    JuMP.write_to_file(value.model, modelLpFile)
end
