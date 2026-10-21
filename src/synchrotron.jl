# Synchrotron physics moved from PlasmaScope/src/field.jl.
# Shared headless model: no viewer, image generation or display transform.
using Unitful
using SpecialFunctions: gamma

# ── Local power-law synchrotron emission model (plain CGS: ne cm⁻³, B Gauss, ν Hz) ──
# Physical constants stripped to plain CGS Float64 (matching the downstream synchrotron code).
const MEC2 = ustrip(u"erg", Unitful.me*Unitful.c0^2)   # electron rest energy mₑc²
const MPME = ustrip(u"g",   Unitful.mp + Unitful.me)   # ion + electron mass

# Pluggable nonthermal electron-density model.
abstract type NeModel end
struct MagneticEtaNe{T} <: NeModel; η::T; end          # fraction η of B²/(8π), B in Gaussian CGS
struct RhoNe{T} <: NeModel; scale::T; end              # ne ∝ thermal number density ρ·ρunit/(mₚ+mₑ)

electron_density(m::MagneticEtaNe, B_gauss, ρ_code, ρunit, p, γmin) =
    m.η * B_gauss^2/(8π) * (p - 2)/((p - 1)*MEC2*γmin)
electron_density(m::RhoNe, B_gauss, ρ_code, ρunit, p, γmin) =
    m.scale * ρ_code * ρunit / MPME

# Power-law normalization N(γ)=Kγ^(-p) on [γmin,γmax]: K/ne = (p-1)/(γmin^(1-p) - γmax^(1-p)).
K(p, γmin, γmax) = (p - 1)/(γmin^(1 - p) - γmax^(1 - p))

# Ordered-field CGS emissivity coefficient c₅(p) (Rybicki–Lightman power-law synchrotron).
function c₅(p)
    e = 4.8032068e-10                    # electron charge [statC]
    me = ustrip(u"g", Unitful.me)
    c = ustrip(Float64, u"cm/s", Unitful.c0)   # Float64: c0 is exact-integer, c² would overflow Int64
    A = 2π*me*c/(3e)
    pref0 = √3*e^3/(4π*me*c^2)
    pref0/(p + 1)*A^(-(p - 1)/2)*gamma((3p - 1)/12)*gamma((3p + 19)/12)
end

# Local emissivity in the "maximum direction" (B_perp = |B_comoving|, isotropic electrons, φ=1).
_jν(Cj, ne, B_perp, p, ν) = Cj*ne*B_perp^((p + 1)/2)*ν^(-(p - 1)/2)
synchrotron_jν(ne, B_perp, p, ν, γmin, γmax) = _jν(c₅(p)*K(p, γmin, γmax), ne, B_perp, p, ν)

# Stokes-I power-law absorption, Pandya et al. (2016), equations 30/34;
# same expression as ipole's symphony/power_law_fits.c, in plain CGS.
function synchrotron_αν(ne, B_perp, p, ν, γmin, γmax)
    e = 4.8032068e-10
    me = ustrip(u"g", Unitful.me)
    c = ustrip(Float64, u"cm/s", Unitful.c0)
    νc = e * B_perp / (2π * me * c)
    ne * e^2 / (ν * me * c) * 3^((p+1)/2) * K(p, γmin, γmax) / 4 *
        gamma((3p+2)/12) * gamma((3p+22)/12) * (νc / ν)^((p+2)/2)
end

# Emission parameters carried by the field source; UI drives these.
@kwdef struct EmissionParams
    ne::NeModel = MagneticEtaNe(0.01)
    p = 2.5
    ν = 2.3e11
    γmin = 10.0
    γmax = 1e5
    σcut = 1.0
end

# x^E for a quarter-integer E (the p slider snaps to ½ steps → emission exponents are quarters), else the
# exact `x^E`. E is a compile-time `Val`, so this folds to a few `sqrt`s — no libm `pow` in the hot loop.
@inline function _powq(x, ::Val{E}) where {E}
    i = floor(Int, E); f = E - i
    f == 0    ? x^i :
    f == 0.25 ? x^i * sqrt(sqrt(x)) :
    f == 0.5  ? x^i * sqrt(x) :
    f == 0.75 ? x^i * sqrt(x * sqrt(x)) :
                x^E                                       # non-quarter exponent → exact fallback
end

# Per-cell synchrotron emissivity jν over the snapshot grid (:φ,:θ,:r), σ-cut applied — fused into one
# threaded pass over `map_plasma` (physics + electron density + jν + mask, no intermediate arrays). The
# cell-independent ν^(-(p-1)/2) is folded into the prefactor. The magnetization cut keeps emission where
# b² ≤ σcut·ρ (Synchray's form): a Bool factor that strong-zeros the funnel and any ρ≤0 cells.
function emissivity_grid(snap, ep::EmissionParams)
    (; ne, p, ν, γmin, γmax, σcut) = ep
    Cj = c₅(p)*K(p, γmin, γmax)*ν^(-(p - 1)/2)
    _emissivity_grid(snap, ne, Cj, bunit(snap), rhounit(snap), p, γmin, σcut, Val((p + 1)/2))
end

function absorption_grid(snap, ep::EmissionParams)
    (; ne, p, ν, γmin, γmax, σcut) = ep
    Bu, ρu = bunit(snap), rhounit(snap)
    # Evaluate only the cell-independent coefficient on the host.
    Cα = synchrotron_αν(1.0, 1.0, p, ν, γmin, γmax)
    map_plasma(snap, snap.rho) do ps, _, (ρ,)
        Bg = sqrt(max(ps.bsq, 0.0)) * Bu
        ps.bsq <= σcut * ρ && ρ > 0 ?
            Cα * electron_density(ne, Bg, ρ, ρu, p, γmin) * Bg^((p+2)/2) : 0.0
    end
end
@inline function _emissivity_grid(snap, ne, Cj, Bunit, ρunit, p, γmin, σcut, ::Val{EX}) where {EX}
    map_plasma(snap, snap.rho) do ps, _, (ρ,)
        Bg = sqrt(ps.bsq)*Bunit
        Cj*electron_density(ne, Bg, ρ, ρunit, p, γmin)*_powq(Bg, Val(EX)) * (ps.bsq <= σcut*ρ)
    end
end

# Emissivity × gravitational-redshift boost g^((p+5)/2), g=√(−g_tt) (Liouville g³ × frequency shift
# g^((p-1)/2)), fused into the same single pass. `zg` is the per-cell `grav_redshift` from `map_plasma`;
# it is 0 inside the ergosphere (no static emitter), which zeroes that region's contribution.
function emission_contribution_grid(snap, ep::EmissionParams)
    (; ne, p, ν, γmin, γmax, σcut) = ep
    Cj = c₅(p)*K(p, γmin, γmax)*ν^(-(p - 1)/2)
    _emission_contribution_grid(snap, ne, Cj, bunit(snap), rhounit(snap), p, γmin, σcut, Val((p + 1)/2), Val((p + 5)/2))
end
@inline function _emission_contribution_grid(snap, ne, Cj, Bunit, ρunit, p, γmin, σcut, ::Val{EX}, ::Val{GEX}) where {EX,GEX}
    map_plasma(snap, snap.rho) do ps, zg, (ρ,)
        Bg = sqrt(ps.bsq)*Bunit
        Cj*electron_density(ne, Bg, ρ, ρunit, p, γmin)*_powq(Bg, Val(EX)) * (ps.bsq <= σcut*ρ) * _powq(zg, Val(GEX))
    end
end

# Contribution to the observed image per *displayed* (Cartesian) volume: emission_contribution × the
# proper-to-coordinate volume ratio √γ/(r²sinθ). Integrating this over the drawn dx dy dz box recovers
# the observed flux (optically thin, no ray bending, gravitational redshift only) — unlike
# emission_contribution_grid, which is a density per *proper* volume.
image_contribution_grid(snap, ep::EmissionParams) = emission_contribution_grid(snap, ep) .* volume_ratio(snap)
