using TightBindingToolkit, LinearAlgebra
using LaTeXStrings, JLD2
using Plots
using PyPlot
pygui(true)

function GeoTensor(Ham::Hamiltonian, subset::Vector{Int64})

    Vx = conj.(permutedims.(Ham.states)) .* Ham.velocity[1] .* Ham.states
    Vy = conj.(permutedims.(Ham.states)) .* Ham.velocity[2] .* Ham.states

    geotensor = similar(Ham.states, Matrix{ComplexF64})

    for k in eachindex(Ham.bands)
        Es = Ham.bands[k]
        vx = Vx[k]
        vy = Vy[k]

        geotensor[k] = zeros(ComplexF64, 2, 2)

        for i in subset
            for j in setdiff(1:length(Es), subset)
                denom = ((Es[j] - Es[i])^2)
                geotensor[k][1, 1] += (vx[i, j] * vx[j, i] ) / denom
                geotensor[k][1, 2] += (vx[i, j] * vy[j, i] ) / denom
                geotensor[k][2, 1] += (vy[i, j] * vx[j, i] ) / denom
                geotensor[k][2, 2] += (vy[i, j] * vy[j, i] ) / denom
            end
        end

    end

    return geotensor/length(Ham.bands)

end

function Curvature(Ham::Hamiltonian, subset::Vector{Int64})::Matrix{Float64}

    Links   =   FindLinks(Ham, subset)
    Field   =   FieldStrength(Links)
    curvature = angle.(Field)
    return curvature
end

function KuboChern(Ham::Hamiltonian, bz::BZ, mu::Float64)

    Vx = conj.(permutedims.(Ham.states)) .* Ham.velocity[1] .* Ham.states
    Vy = conj.(permutedims.(Ham.states)) .* Ham.velocity[2] .* Ham.states

    chern = 0.0 + im * 0.0
    for k in eachindex(Ham.bands)
        Es = Ham.bands[k]
        vx = Vx[k]
        vy = Vy[k]

        ind = searchsortedfirst(Es, mu)
        if ind == 1 || ind == length(Es)
            continue
        else
            for i in 1:ind-1
                for j in ind:length(Es)
                    chern += (vx[i, j] * vy[j, i] - vx[j, i] * vy[i, j]) / ((Es[j] - Es[i])^2)
                end
            end
        end

    end

    b1 = [bz.basis[1]; 0.0]
    b2 = [bz.basis[2]; 0.0]
    bzUnitArea = cross(b1, b2)[3] / (4 * pi^2)

    return imag(chern) * bzUnitArea * 2 * pi / length(Ham.bands)
end


function get_metric_measures(Js::Vector{Float64}, UC::UnitCell, bz::BZ,
    HoppingParams::Vector{T}, jhParam::Param, band::Int64,
    filename::String="skyrmion_metric_measures.jld2" ; measures::Bool = true) where {T}

    gap_belows    =   Float64[]
    gap_aboves    =   Float64[]
    mus     =   Float64[]
    Cherns_wfns  =   Float64[]
    Cherns_metric =   Float64[]
    weights =   Float64[]
    volumes =   Float64[]
    bandwidths = Float64[]

    for J in Js
        jhParam.value = [J]
        UC.bonds = []
        CreateUnitCell!(UC, HoppingParams)

        H = Hamiltonian(UC, bz)
        DiagonalizeHamiltonian!(H)

        energies = getindex.(H.bands, band)
        energy_below = getindex.(H.bands, band-1)
        energy_above = getindex.(H.bands, band+1)
        gap_below =  minimum((energies-energy_below))
        gap_above =  minimum((energy_above-energies))
        bandwidth = maximum(energies) - minimum(energies)

        push!(gap_belows, gap_below)
        push!(gap_aboves, gap_above)
        push!(bandwidths, bandwidth)

        if measures
            GetVelocity!(H, bz)
            if gap_above>1e-3 && gap_below>1e-3
                chern = ChernNumber(H, [band])
                push!(Cherns_wfns, chern)

                geo = GeoTensor(H, [band])# #for (i, filling) in enumerate(filling_arr)
                metric = [real(mat) for mat in geo]
                berry = [-2*imag(mat) for mat in geo]
                metric_traces = [tr(mat) for mat in metric]
                berry_curvature = [mat[1, 2] for mat in berry]

                weight = sum(metric_traces) * bzUnitArea / (2*pi)
                push!(weights, weight)
                push!(Cherns_metric, sum(berry_curvature) * bzUnitArea / (2*pi))

                push!(volumes, sum(sqrt.(abs.(det.(metric))) * bzUnitArea / (pi)))
                println("W = $(round(weight, digits=3)), and C = $(round(chern, digits=3)) at Jh = $(J)")
            else
                push!(Cherns_wfns, 0.0)
                push!(weights, 0.0)
                push!(Cherns_metric, 0.0)
                push!(volumes, 0.0)
                println("W = 0.0, and C = 0.0 at Jh = $(J)")
            end
        end
    end

    data = Dict("Js" => Js, "Cherns_wfns" => Cherns_wfns, "Cherns_metric" => Cherns_metric,
        "weights" => weights, "volumes" => volumes, "bandwidths" => bandwidths, "gap_above"=>gap_aboves, "gap_below"=>gap_belows)
    # save(filename, data)
    return data
end

function hexagon(corner::Vector{Float64})
    RotMat = [cos(pi/3) sin(pi/3); -sin(pi/3) cos(pi/3)]
    lines = []
    for i in 1:6
        p1 = corner
        p2 = RotMat * corner

        if abs(p2[1]-p1[1])>1e-6
            slope = (p2[2] - p1[2]) / (p2[1] - p1[1])
            xs = LinRange(p1[1], p2[1], 100)
            ys = slope .* (xs .- p1[1]) .+ p1[2]
        else
            ys = LinRange(p1[2], p2[2], 100)
            xs = p1[1] .* ones(100)
        end
        push!(lines, (xs, ys))
        corner = RotMat * corner
    end

    return lines
end

function plot_data(data::Matrix{T},
    kxs::Vector{Float64}, kys::Vector{Float64},
    bz::BZ;
    colorbar_title::AbstractString = L"\Omega_n(\mathbf{k})",
    to_save::Tuple{Bool, String} = (false, ""),
    labels::Bool=true,
    clims::Tuple{T, T} = extrema(data),
    cmap::Symbol=:blues,
    annotation::AbstractString = "") where {T}

    font = "Computer Modern"

    ssf_plot = plot(framestyle=:box, grid=false, aspect_ratio=:equal,
        guidefontstyle = font, tickfont = font, legendfont = font,
        guidefontsize = 12, tickfontsize = 12, legendfontsize = 10)

    tick_start = round(Int, clims[1])
    tick_end = round(Int, clims[2])

    heatmap!(kxs, kys, data', c=cmap,
        clim=clims, colorbar_title=colorbar_title,
        colorbar_ticks = collect(range(tick_start, step=2, stop=tick_end)))

    xlims!(-2.025, 2.025)
    ylims!(-2.025, 2.025)
    if labels
        xlabel!(L"k_x")
        ylabel!(L"k_y")
        xticks!([0.0], [""])
        yticks!([0.0], [""])
        # xticks!([-2, -1, 0, 1, 2])
        # yticks!([-2, -1, 0, 1, 2])
    else
        xticks!([0.0], [""])
        yticks!([0.0], [""])
    end
    # annotate!()

    inner_hex = hexagon(bz.HighSymPoints["K1"])
    for edge in inner_hex
        xs, ys = edge
        plot!(xs, ys, label = "", lw=2.0, lc=:orange)
    end

    annotate!([(1.8, 1.8, text(annotation, "Computer Modern", 12, :black))])

    if to_save[1]
        savefig(to_save[2])
    end

    return ssf_plot
end


function pyplot_hexagonal(input::Dict, data::Matrix{T};
    colorbar_title::AbstractString=L"\Omega_n(\mathbf{k})",
    cmap::String="Blues", lim::Float64=2.025,
    annotation::AbstractString="",
    annotation_position::Tuple{Float64, Float64}=(0.0, 1.0),
    clims::Tuple{Float64, Float64}=(0.0, 7.0),
    savename::AbstractString="") where {T}

    PyPlot.cla()

    PyPlot.pcolormesh(input["kxs"], input["kys"], data',
        cmap=cmap, vmin=clims[1], vmax=clims[2])  # Adjust the size as needed

    PyPlot.colorbar(label=colorbar_title)

    inner_hex = input["inner_hex"]
    for edge in inner_hex
        xs, ys = edge
        PyPlot.plot(xs, ys, linewidth=2.0, c=:orange)
    end

    PyPlot.gca().set_aspect("equal")
    PyPlot.gca().set_xticks([])
    PyPlot.gca().set_yticks([])
    PyPlot.annotate(annotation, annotation_position, xycoords="axes fraction")
    PyPlot.gca().set_xlim(-lim, lim)
    PyPlot.gca().set_ylim(-lim, lim)
    PyPlot.gca().set_xlabel(L"k_x")
    PyPlot.gca().set_ylabel(L"k_y")

    if !isempty(savename)
        PyPlot.savefig(savename, bbox_inches="tight")
    end


end


##Triangular Lattice
# params = Dict()
# SkXSize = get!(params, "SkXSize", 2)
# SkX = get!(params, "SkX", "Bloch")
# SkX = "Bloch"
# a1 = SkXSize / 2 * [-3.0, sqrt(3)]
# a2 = SkXSize / 2 * [3.0, sqrt(3)]
# l1 = [1.0, 0]
# l2 = [-0.5, sqrt(3) / 2]
# UC = UnitCell([a1, a2], 2, 2)
# ##Parameters

# t = get!(params, "t", 1.0)
# jh = get!(params, "jh", 4.0)
# U = get!(params, "U", 0.0)
# ##### Thermodynamic parameters
# filling = get!(params, "filling", 12.5/24)
# T = get!(params, "T", 0.0)
# t1 = -t
# t1Param = Param(t1, 2)
# jhParam = Param(jh, 2)
# HoppingParams = [t1Param, jhParam]
# su2spin = SpinMats(1 // 2)

# ##Adding inner-hexagon structure
# for j = 0:(SkXSize-1)
#     for i = 0:(SkXSize*3-1)
#         AddBasisSite!(UC, i .* l1 + j .* l2)
#     end
# end
# AddIsotropicBonds!(t1Param, UC, 1.0, su2spin[4], "t1", checkOffsetRange=1)
# ##Functions that will be useful for adding anisotropic bonds
# weiss_neel(v) = [sin(pi * (norm(v) / (SkXSize))) * v[1] / norm(v), sin(pi * (norm(v) / (SkXSize))) * v[2] / norm(v), cos(pi * (norm(v) / (SkXSize)))]
# weiss_bloch(v) = [sin(pi * (norm(v) / (SkXSize))) * v[2] / norm(v), sin(pi * (norm(v) / (SkXSize))) * -v[1] / norm(v), cos(pi * (norm(v) / (SkXSize)))]
# weiss = Dict("Neel" => weiss_neel, "Bloch" => weiss_bloch)
# sigmav(i, j) = 2 .* [su2spin[1][i, j], su2spin[2][i, j], su2spin[3][i, j]]
# s11 = sigmav(1, 1)
# s12 = sigmav(1, 2)
# s21 = sigmav(2, 1)
# s22 = sigmav(2, 2)

# intermat(s) = [dot(s, s11) dot(s, s12); dot(s, s21) dot(s, s22)]


# ##Adding anisotropic bonds and normalizing if needed
# for (ind, bas) in enumerate(UC.basis)
#     closest = [bas, bas - a1, bas - a2, bas - a1 - a2, bas + a1, bas + a2, bas + a1 + a2, bas + a1 - a2, bas - a1 + a2]
#     minimal = findmin(x -> norm(x), closest)[2]
#     if (SkXSize - 1) < norm(closest[minimal]) < SkXSize
#         mat = intermat(normalize(weiss[SkX](closest[minimal]) + weiss[SkX](-closest[minimal])))
#     else
#         spn = weiss[SkX](closest[minimal])
#         replace!(spn, NaN => 0.0)
#         mat = intermat(normalize(spn))
#     end
#     AddAnisotropicBond!(jhParam, UC, ind, ind, [0, 0], mat, 0.0, "Hunds")
# end
# CreateUnitCell!(UC, HoppingParams)


# ##Creating BZ and Hamiltonian Model
# n = 50
# kSize = 6 * n + 3
# bz = BZ(kSize)
# FillBZ!(bz, UC)

# b1 = [bz.basis[1]; 0.0]
# b2 = [bz.basis[2]; 0.0]
# bzUnitArea = abs(cross(b1, b2)[3])

# H = Hamiltonian(UC, bz)
# DiagonalizeHamiltonian!(H)
# GetVelocity!(H, bz)

# Mdl = Model(UC, bz, H; filling=filling)
# # SolveModel!(Mdl; get_gap=true)

# kxs = collect(LinRange(-2, 2, 101))
# kys = collect(LinRange(-2, 2, 101))
# ks = [[kx, ky] for kx in kxs, ky in kys]
# indices = GetQIndex.(ks, Ref(bz) ; nearest=true)
# indices = Tuple.(indices)
# indices = CartesianIndex.(indices)


# band = 13
# geo = GeoTensor(H, [band])

# metric = [real(mat) for mat in geo]
# berry = [-2*imag(mat) for mat in geo]
# berry_curvature = [mat[1, 2] for mat in berry]
# metric_traces = [tr(mat) for mat in metric]
# metric_sqrtDets = [sqrt(abs(det(mat))) for mat in metric]

# curvature = Curvature(H, [band])
# curvature = curvature * prod(bz.gridSize)/bzUnitArea

# berry_curvature = berry_curvature * prod(bz.gridSize)
# metric_traces = metric_traces * prod(bz.gridSize)
# metric_sqrtDets = metric_sqrtDets * prod(bz.gridSize)

# dA = bzUnitArea / (2*pi*prod(bz.gridSize))

# weight = sum(metric_traces) * dA
# volume = sum(metric_sqrtDets) * dA
# chern = sum(berry_curvature) * dA
# chern_wfn = sum(curvature) * dA

# inner_hex = hexagon(bz.HighSymPoints["K1"])
# save("J=$(jh)_band=$(band)_metric.jld2",
#     Dict("metric"=>metric,
#         "berry"=>berry[indices],
#         "curvature"=>curvature[indices],
#         "metric_traces"=>metric_traces[indices],
#         "metric_sqrtDets"=>metric_sqrtDets[indices],
#         "berry_curvature"=>berry_curvature[indices],
#         "weight"=>weight,
#         "volume"=>volume,
#         "chern"=>chern,
#         "chern_wfn"=>chern_wfn,
#         "inner_hex"=>inner_hex,
#         "bzUnitArea"=>bzUnitArea,
#         "dA"=>dA,
#         "band"=>band,
#         "kxs" => kxs,
#         "kys" => kys,
#         "indices" => indices,
#         ))


input = load("/home/anjishnubose/Research/Repos/skyrmion_bilayer/CQIQC-Research/J=4.0_band=1_metric.jld2")

data = abs.(input["curvature"])
pyplot_hexagonal(input, data, cmap="Blues", clims=(0.0, 7.0), savename="")

# berry_plot = plot_data(abs.(curvature[indices]), kxs, kys, bz ;
#     clims=(0.0, 18.0), annotation="(a)")
# volume_plot = plot_data(metric_sqrtDets[indices] , kxs, kys, bz ;
#     clims=(0.0, 12.0), cmap=:algae, colorbar_title=L"V_{n}(\mathbf{k})",
#     annotation="(b)")

# p =plot(berry_plot, volume_plot, layout=grid(2, 1), size=(400, 600))

# pNormal(x, mu, sigma) = 1/(sigma*sqrt(2*pi)) * exp(-0.5*((x-mu)/sigma)^2)
