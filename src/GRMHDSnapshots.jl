module GRMHDSnapshots

using HDF5
using StaticArrays
using Unitful, UnitfulAstro
using AxisKeys: NamedDimsArray, dimnames
using DataManipulation: mapview
using OhMyThreads: tmap
import StructArrays
import Adapt
using Zarr: zopen, ZGroup
using ZarrZfp

export GRMHDSnapshot, load_snapshot, load_koral, load_iharm, load_grid, r_min, r_max,
    plasma_state, map_plasma, fluid_ucon, fluid_velocity, flow_speed, proper_velocity, lapse, grav_redshift, volume_ratio, lunit, rhounit, bunit,
    comoving_bsq, comoving_B_gauss, bfield_magnitude, magnetization

include("coordinates.jl")

# `M_unit`/`MBH` carry their own type parameter `U`: `Float64` when physical units are known,
# `Missing` for scale-free dumps (e.g. iharm3d/KHARMA) loaded without them — then the unit scalars
# `lunit`/`rhounit`/`bunit` throw rather than silently returning `missing`.
struct GRMHDSnapshot{Ar, Av, Ab, Rp, Tg, T, U}
    rho::Ar
    velrel::Av
    bfield::Ab
    r_prof::Rp
    th_grid::Tg
    φ0::T
    dφ::T
    spin::T
    gam::T
    t::T
    M_unit::U
    MBH::U
end

"""
    load_snapshot(path; kwargs...)

Load a single GRMHD snapshot, auto-detecting the on-disk format:
- `.h5` with a `prims` dataset → iharm3d/KHARMA FMKS ([`load_iharm`](@ref)), read in memory;
- `.h5` with a `quants/` group → koral ([`load_koral`](@ref)), memory-mapped;
- a `.../RUN.zarr/snapNNNN` path → koral Zarr ([`load_koral`](@ref)).

Memory-mapping is chosen for the caller: koral `.h5` is mmapped, iharm reads in memory (its
velocity/field components are reconstructed, not stored, so there is nothing to map). `kwargs...`
go to the selected loader (e.g. `M_unit`/`MBH` for iharm); use the explicit loaders for full control.
"""
function load_snapshot(path::AbstractString; kwargs...)
    p = rstrip(path, '/')
    snapshot_format(p) === :iharm && return load_iharm(p; kwargs...)
    endswith(p, ".h5") ? load_koral(p; mmap = true, kwargs...) : load_koral(p; kwargs...)
end

"""
    snapshot_format(path) -> Symbol

Detect a snapshot's on-disk format: `:iharm` for an iharm3d/KHARMA `.h5` dump (a `prims` dataset),
or `:koral` for the koral layout (a `.h5` with a `quants/` group, or a Zarr snapshot path).
"""
function snapshot_format(path::AbstractString)
    p = rstrip(path, '/')
    endswith(p, ".h5") || return :koral        # Zarr snapshot/run path
    h5open(p, "r") do h5
        haskey(h5, "quants") ? :koral :
        haskey(h5, "prims")  ? :iharm :
        throw(ArgumentError("unrecognized HDF5 snapshot format (no `quants/` or `prims`): $p"))
    end
end

"""
    load_koral(path; kwargs...)

Load a raw `.h5` snapshot or a Zarr snapshot path.

# Arguments
- `path`: `.h5` file, or `.../RUN.zarr/snapNNNN` store snapshot path.
- `kwargs...`: Forwarded to the selected loader.

# Returns
- `GRMHDSnapshot`.
"""
function load_koral(path::AbstractString; kwargs...)
    p = rstrip(path, '/')
    endswith(p, ".h5") ?
        load_koral_h5(p; kwargs...) :
        load_koral(zopen(dirname(p), "r")::ZGroup, basename(p); kwargs...)
end

function load_koral_h5(path::AbstractString; mmap=false)
    h5open(path, "r") do h5
        nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
        if mmap
            # mmap bytes keep file endianness.
            order = HDF5.API.h5t_get_order(HDF5.datatype(h5["quants/rho"]).id)
            swap = order == HDF5.API.H5T_ORDER_BE ? ntoh : ltoh
            mm(name) = HDF5.readmmap(h5["quants/$name"])
            rho = nd(mapview(swap, mm("rho")))
            svf(a, b, c) = nd(mapview(v -> swap.(v),
                StructArrays.StructArray{SVector{3,Float64}}((mm(a), mm(b), mm(c)))))
            velrel = svf("U1", "U2", "U3"); bfield = svf("B1", "B2", "B3")
        else
            rd(name) = read(h5["quants/$name"])
            rho = nd(rd("rho"))
            sv3(a, b, c) = nd(SVector{3,Float64}.(a, b, c))
            velrel = sv3(rd("U1"), rd("U2"), rd("U3")); bfield = sv3(rd("B1"), rd("B2"), rd("B3"))
        end

        r = read(h5["grid_out/r"]); th = read(h5["grid_out/th"]); ph = read(h5["grid_out/ph"])
        n1 = size(r, 3); n2 = size(r, 2); n3 = size(r, 1)
        r_prof  = Float64[r[1, 1, i] for i in 1:n1]
        th_grid = Float64[th[1, j, i] for i in 1:n1, j in 1:n2]
        φ0_file = Float64(ph[1, 1, 1]); dφ = (Float64(ph[n3, 1, 1]) - φ0_file)/(n3 - 1)
        # ipole convention: file φ + π.
        φ0 = φ0_file + π

        rdscalar(name) = Float64(read(h5[name]))
        GRMHDSnapshot(rho, velrel, bfield, r_prof, th_grid, φ0, dφ,
            rdscalar("header/bhspin"), rdscalar("header/gam"), rdscalar("t"),
            rdscalar("header/units/M_unit"), ustrip(u"g", rdscalar("header/units/M_bh") * u"Msun"))
    end
end

"""
    load_grid(store)

Load shared grid arrays from a Zarr run store.

# Arguments
- `store`: Open Zarr run store containing `r`, `th`, and `ph`.

# Returns
- Named tuple with `r_prof`, `th_grid`, `φ0`, and `dφ`.
"""
function load_grid(store::ZGroup)
    rd(name) = store[name][:, :, :]
    r = rd("r"); th = rd("th"); ph = rd("ph")
    n1 = size(r, 3); n2 = size(r, 2); n3 = size(r, 1)
    φ0_file = Float64(ph[1, 1, 1])
    (; r_prof  = Float64[r[1, 1, i] for i in 1:n1],
       th_grid = Float64[th[1, j, i] for i in 1:n1, j in 1:n2],
       φ0 = φ0_file + π,
       dφ = (Float64(ph[n3, 1, 1]) - φ0_file)/(n3 - 1))
end

"""
    load_koral(store, snapname; fields=(:rho, :velrel, :bfield), grid=load_grid(store))

Load one snapshot from an open Zarr run store.

# Arguments
- `store`: Open Zarr run store.
- `snapname`: Snapshot group name, e.g. `"snap002000"`.
- `fields`: Fluid fields to read; subset of `(:rho, :velrel, :bfield)`.
- `grid`: Cached grid from `load_grid(store)`.

# Returns
- `GRMHDSnapshot`; unselected fields are `nothing`.
"""
function load_koral(store::ZGroup, snapname::AbstractString;
                    fields = (:rho, :velrel, :bfield), grid = load_grid(store))
    fields ⊆ (:rho, :velrel, :bfield) || throw(ArgumentError("unknown fluid field(s) in $fields"))
    snap = store[snapname]::ZGroup
    nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
    rd(name) = Threads.@spawn snap[name][:, :, :]                # each zfp chunk decompresses independently
    svf(ts) = nd(StructArrays.StructArray{SVector{3,Float32}}(map(fetch, ts)))
    # launch every requested component up front so the reads decompress concurrently, then assemble
    trho = :rho    in fields ? rd("rho")                   : nothing
    tvel = :velrel in fields ? map(rd, ("U1", "U2", "U3")) : nothing
    tbf  = :bfield in fields ? map(rd, ("B1", "B2", "B3")) : nothing
    rho    = isnothing(trho) ? nothing : nd(fetch(trho))
    velrel = isnothing(tvel) ? nothing : svf(tvel)
    bfield = isnothing(tbf)  ? nothing : svf(tbf)

    sc(x) = Float64(x)
    hdr = store.attrs; units = hdr["units"]
    GRMHDSnapshot(rho, velrel, bfield, grid.r_prof, grid.th_grid, grid.φ0, grid.dφ,
        sc(hdr["bhspin"]), sc(hdr["gam"]), sc(snap.attrs["t"]),
        sc(units["M_unit"]), ustrip(u"g", sc(units["M_bh"]) * u"Msun"))
end

"""
    load_iharm(path; fields=(:rho, :velrel, :bfield), M_unit=missing, MBH=missing)

Load an iharm3d/KHARMA `.h5` dump in FMKS coordinates (`header/metric` == `"FMKS"`/`"MMKS"`).

The packed `prims` array holds the primitives in *native* FMKS components; the relative velocity
`ũ^i` and lab field `B^i` are transformed to Kerr-Schild `(r, θ, φ)` components at load via the
spatial Jacobian `J₃(r, θ)` (matching ipole's reconstruction), so the result is an ordinary
[`GRMHDSnapshot`] that all downstream physics/interpolation handles unchanged.

# Arguments
- `fields`: fluid fields to read; subset of `(:rho, :velrel, :bfield)`. Unselected are `nothing`.
- `M_unit`, `MBH`: physical units in grams. These dumps are scale-free (no units stored), so pass
  both to enable `lunit`/`rhounit`/`bunit`, or omit both (default `missing`) — those scalars then throw.

# Returns
- `GRMHDSnapshot`.
"""
function load_iharm(path::AbstractString; fields = (:rho, :velrel, :bfield), M_unit = missing, MBH = missing)
    fields ⊆ (:rho, :velrel, :bfield) || throw(ArgumentError("unknown fluid field(s) in $fields"))
    (M_unit === missing) == (MBH === missing) || throw(ArgumentError("pass both M_unit and MBH, or neither"))
    h5open(path, "r") do h5
        rd(name) = read(h5[name])
        metric = strip(c -> c == '\0' || c == ' ', rd("header/metric"))
        metric in ("FMKS", "MMKS") ||
            throw(ArgumentError("only FMKS iharm/KHARMA dumps are supported, got metric = $(repr(metric))"))
        n1 = Int(rd("header/n1")); n2 = Int(rd("header/n2")); n3 = Int(rd("header/n3"))
        # grid start/step under header/geom/; FMKS shape params under header/geom/mmks/ (ipole's dir for FMKS).
        geom(name) = rd("header/geom/$name"); fmks(name) = rd("header/geom/mmks/$name")
        sx1, sx2, sx3 = geom("startx1"), geom("startx2"), geom("startx3")
        dx1, dx2, dx3 = geom("dx1"), geom("dx2"), geom("dx3")
        coords = FMKS(startx1 = sx1, hslope = fmks("hslope"), mks_smooth = fmks("mks_smooth"),
                      poly_xt = fmks("poly_xt"), poly_alpha = fmks("poly_alpha"))

        # zone-centered native coords (Julia index i → 0-based i-1). r depends on X1 only, θ on (X1, X2).
        X1(i) = sx1 + (i - 0.5)*dx1
        X2(j) = sx2 + (j - 0.5)*dx2
        r_prof  = Float64[first(bl_coord(coords, X1(i), X2(1))) for i in 1:n1]
        th_grid = Float64[last(bl_coord(coords, X1(i), X2(j)))  for i in 1:n1, j in 1:n2]

        # prims: HDF5 (n1,n2,n3,n_prim) row-major → Julia (n_prim,n3,n2,n1). Slots: 1=RHO, 3-5=U1-3, 6-8=B1-3.
        prims = rd("prims")
        nd(A) = NamedDimsArray(A, (:φ, :θ, :r))
        rho = :rho in fields ? nd([prims[1, k, j, i] for k in 1:n3, j in 1:n2, i in 1:n1]) : nothing
        # native contravariant 3-vector → KS: v_KS = J₃·v_nat, J₃(r,θ) built once per (i,j) and reused over φ.
        J = [jacobian_spatial(coords, X1(i), X2(j)) for i in 1:n1, j in 1:n2]
        tovec(s1, s2, s3) = nd([J[i, j] * SVector(prims[s1, k, j, i], prims[s2, k, j, i], prims[s3, k, j, i])
                                for k in 1:n3, j in 1:n2, i in 1:n1])
        velrel = :velrel in fields ? tovec(3, 4, 5) : nothing
        bfield = :bfield in fields ? tovec(6, 7, 8) : nothing

        sc(x) = Float64(x)   # heterogeneous header scalars → uniform Float64 struct fields
        # φ = X3 (zone-centered); no +π offset (that is specific to the koral converter).
        GRMHDSnapshot(rho, velrel, bfield, r_prof, th_grid, sc(sx3 + 0.5*dx3), sc(dx3),
            sc(fmks("a")), sc(rd("header/gam")), sc(rd("t")), M_unit, MBH)
    end
end

"""
    r_min(snapshot)

Minimum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
"""
r_min(s::GRMHDSnapshot) = @inbounds s.r_prof[1]

"""
    r_max(snapshot)

Maximum radial grid coordinate.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
"""
r_max(s::GRMHDSnapshot) = @inbounds s.r_prof[end]

# GPU-safe searchsortedlast.
@inline function bsearchlast(a, x)
    lo = 1; hi = length(a)
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        a[mid] ≤ x ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Avoids a device view.
@inline function bsearchlast_row(a, ir, x, n)
    lo = 1; hi = n
    @inbounds while lo < hi
        mid = (lo + hi + 1) >> 1
        a[ir, mid] ≤ x ? (lo = mid) : (hi = mid - 1)
    end
    return lo
end

# Trilinear stencil at Cartesian (x,y,z): bracket indices + weights, field-independent.
# `n3` = φ count (same for every field). Returns `nothing` outside the radial grid.
@inline function _sample_stencil(s::GRMHDSnapshot, n3, x, y, z)
    r = sqrt(x^2 + y^2 + z^2)
    (r < r_min(s) || r > r_max(s)) && return nothing
    θ = acos(clamp(z/r, -one(r), one(r)))
    φ = atan(y, x)
    n2 = size(s.th_grid, 2)

    i = clamp(bsearchlast(s.r_prof, r), 1, length(s.r_prof) - 1)
    ti = (r - s.r_prof[i])/(s.r_prof[i+1] - s.r_prof[i])
    # Keep NamedDims indices homogeneous for Metal.
    fk = (φ - s.φ0)/s.dφ
    k0 = unsafe_trunc(Int, floor(fk)); tk = fk - k0
    k = mod(k0, n3) + 1; kp = mod(k0 + 1, n3) + 1

    θbracket(ir) = @inbounds begin
        j = clamp(bsearchlast_row(s.th_grid, ir, θ, n2), 1, n2 - 1)
        a = s.th_grid[ir, j]
        (j, clamp((θ - a)/(s.th_grid[ir, j+1] - a), zero(θ), one(θ)))
    end
    ji, tji = θbracket(i); jp, tjp = θbracket(i + 1)
    return (; i, ti, k, kp, tk, ji, tji, jp, tjp)
end

# Apply a precomputed stencil to a single field array.
@inline function _apply_stencil(st, A)
    face(ir, jj, tj) = @inbounds (A[φ=st.k, θ=jj, r=ir]*(1-tj) + A[φ=st.k, θ=jj+1, r=ir]*tj)*(1-st.tk) +
                                 (A[φ=st.kp, θ=jj, r=ir]*(1-tj) + A[φ=st.kp, θ=jj+1, r=ir]*tj)*st.tk
    face(st.i, st.ji, st.tji)*(1 - st.ti) + face(st.i + 1, st.jp, st.tjp)*st.ti
end

"""
    sample_field(snapshot, field_array, x, y, z)

Interpolate a single fluid field array (e.g. `snap.velrel`) at Cartesian `(x, y, z)`.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot`.
- `field_array`: One `(φ, θ, r)` field array, e.g. `snap.rho`, `snap.velrel`, or `snap.bfield`.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Interpolated value; `zero(eltype(field_array))` outside the grid.
"""
@inline function sample_field(s::GRMHDSnapshot, A, x, y, z)
    st = _sample_stencil(s, size(A, :φ), x, y, z)
    # `* one(weight)` promotes the out-of-grid zero to the interpolated type (Float32 field → Float64).
    st === nothing ? zero(eltype(A)) * one(eltype(s.r_prof)) : _apply_stencil(st, A)
end

"""
    sample(snapshot, x, y, z)

Interpolate `(rho, velrel, bfield)` at Cartesian position `(x, y, z)`.

# Arguments
- `snapshot`: Loaded `GRMHDSnapshot` with all fluid fields present.
- `x`, `y`, `z`: Cartesian coordinates in gravitational radii.

# Returns
- Tuple `(rho, velrel, bfield)`.
"""
@inline function sample(s::GRMHDSnapshot, x, y, z)
    st = _sample_stencil(s, size(s.rho, :φ), x, y, z)
    if st === nothing
        # Match the interpolation result type.
        T = promote_type(eltype(s.rho), eltype(s.r_prof)); z3 = zero(SVector{3,T})
        return (zero(T), z3, z3)
    end
    return (_apply_stencil(st, s.rho), _apply_stencil(st, s.velrel), _apply_stencil(st, s.bfield))
end

include("physics.jl")
include("synchrotron.jl")
include("radiative_volume.jl")
export radiative_volume

Adapt.adapt_structure(to, x::NamedDimsArray) = NamedDimsArray(Adapt.adapt(to, parent(x)), dimnames(x))

Adapt.adapt_structure(to, s::GRMHDSnapshot) = GRMHDSnapshot(
    Adapt.adapt(to, s.rho), Adapt.adapt(to, s.velrel), Adapt.adapt(to, s.bfield),
    Adapt.adapt(to, s.r_prof), Adapt.adapt(to, s.th_grid),
    s.φ0, s.dφ, s.spin, s.gam, s.t, s.M_unit, s.MBH)

end
