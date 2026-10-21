# Cartesian snapshot preparation shared by file-backed editor inputs and scenes.
# Coordinates and extent are in Schwarzschild radii (2 r_g).
function radiative_volume(path::AbstractString; resolution = 192, extent = 15f0, normalization=nothing)
    resolution>=2 || throw(ArgumentError("volume resolution must be at least two"))
    isfinite(extent) && extent>1.025 || throw(ArgumentError("invalid volume extent"))
    G = @__MODULE__
    isfile(path) || error("Missing GRMHD snapshot: $path; fetch the declared sequence before rendering")
    snap = G.load_snapshot(path; M_unit = 3e24, MBH = 6.5e9 * 1.98847e33)
    iszero(snap.spin) || error("This medium's transport requires a spin-zero snapshot")
    ep = G.EmissionParams()
    native_j = G.emissivity_grid(snap, ep)
    native_a = G.absorption_grid(snap, ep)
    j = zeros(Float32, resolution, resolution, resolution)
    alpha = similar(j)
    velocity = fill(zero(SVector{3,Float32}), size(j))
    rs_cm = 2G.lunit(snap)
    Threads.@threads for k in 1:resolution
        for iy in 1:resolution, ix in 1:resolution
            p = SVector{3,Float32}(ix-1, iy-1, k-1) .* (2extent/(resolution-1)) .- extent
            r = sqrt(sum(abs2,p))
            if r <= 1.025f0 || r >= 14f0
                alpha[ix,iy,k] = 0f0
                continue
            end
            x, y, z = 2 .* Float64.(p)
            rg = 2Float64(r)
            theta = acos(clamp(z/rg, -1, 1))
            phi = atan(y,x)
            taper = 1 - clamp((r-11f0)/3f0, 0f0, 1f0)^2 * (3 - 2clamp((r-11f0)/3f0, 0f0, 1f0))
            j[ix,iy,k] = Float32(max(G.sample_field(snap, native_j, x,y,z), 0) * taper)
            alpha[ix,iy,k] = Float32(max(G.sample_field(snap, native_a, x,y,z), 0) * taper * rs_cm)
            u = G.sample_field(snap, snap.velrel, x,y,z)
            # Relative velocity -> orthonormal Eulerian frame (a=0), then a
            # radial Lorentz boost into the local STATIC Schwarzschild frame.
            ur, ut, up = u[1]*sqrt(1+2/rg), rg*u[2], rg*sin(theta)*u[3]
            gamma = sqrt(1+ur^2+ut^2+up^2)
            vr, vt, vp = ur/gamma, ut/gamma, up/gamma
            bn = -2/rg
            den = 1+bn*vr
            vr = (vr+bn)/den
            vt *= sqrt(1-bn^2)/den
            vp *= sqrt(1-bn^2)/den
            st, ct = sincos(theta); sp, cp = sincos(phi)
            v = SVector{3,Float32}(vr*st*cp + vt*ct*cp - vp*sp,
                      vr*st*sp + vt*ct*sp + vp*cp, vr*ct - vt*st)
            velocity[ix,iy,k] = v * min(1f0, 0.999f0/max(sqrt(sum(abs2,v)), 1f-8))
        end
    end
    # Exposure is a display choice, not a fitted flux measurement. Preserve
    # relative physical emissivity; absorption retains the stated CGS units.
    peak = maximum(j)
    peak > 0 || error("GRMHD snapshot has no emitting plasma in the scene")
    reference = isnothing(normalization) ? peak : normalization
    isfinite(reference) && reference > 0 || error("Invalid emissivity normalization")
    j ./= reference
    # Bound each trilinear cell jointly. Multiplying the GLOBAL densest cell
    # by the GLOBAL fastest cell is safe but grossly pessimistic: the fastest
    # infall is not where absorption peaks. Convex interpolation bounds both
    # alpha and |v| by the maxima of the same eight corners.
    slab_bounds=zeros(Float32,resolution-1)
    Threads.@threads for k in 1:resolution-1
        bound=0f0
        @inbounds for iy in 1:resolution-1, ix in 1:resolution-1
            amax=0f0; v2max=0f0
            for dz in 0:1, dy in 0:1, dx in 0:1
                v=velocity[ix+dx,iy+dy,k+dz]
                amax=max(amax,alpha[ix+dx,iy+dy,k+dz])
                v2max=max(v2max,sum(abs2,v))
            end
            vmax=sqrt(v2max)
            bound=max(bound,amax*((1+vmax)/(1-vmax))^Float32((ep.p+2)/4))
        end
        slab_bounds[k]=bound
    end
    majorant = maximum(slab_bounds) * 1.0001f0 + 1f-6
    result = (; density=j, absorption=alpha, velocity, path=abspath(path), normalization,
              max_absorption=Float32(majorant), emissivity_peak=peak,
              extent=Float32(extent))
    return result
end
