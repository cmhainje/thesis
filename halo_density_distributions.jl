### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ 43180e7a-802c-11eb-22f9-f90a6e9a526e
using Plots

# ╔═╡ 167e2fba-8037-11eb-2897-3f52b70f7f83
md"
# Two-power models

Define a two-power density model as having the following density distribution:

\begin{equation}
    \rho(r) = \frac{\rho_0}{(r/a)^\alpha(1+r/a)^{\beta-\alpha}}.
\end{equation}

Some special cases:
- NFW: $(\alpha, \beta) = (1, 3)$
- Hernquist: $(\alpha, \beta) = (1, 4)$
"

# ╔═╡ 4d663dfc-802c-11eb-1272-49a4f2026d04
function two_power_density(r, ρ_0, a, α, β)
	x = r ./ a
	return ρ_0 ./ (x.^α .* (1 .+ x).^(β - α))
end

# ╔═╡ 937e5b6c-802c-11eb-2deb-3737d72d6620
function nfw_density(r, ρ_0, a)
	two_power_density(r, ρ_0, a, 1, 3)
end

# ╔═╡ 7cba71e0-802c-11eb-3fdf-2916082f3d3d
function hernquist_density(r, ρ_0, a)
	two_power_density(r, ρ_0, a, 1, 4)
end

# ╔═╡ 9448d948-8040-11eb-05f4-1f982783556a
md"
We can also write down an equation for the mass enclosed within a given radius.

\begin{equation}
	M(r) = 4 \pi \rho\_0 a^3 \int\_0^{r/a} ds \,
           \frac{s^{2-\alpha}}{(1+s)^{\beta-\alpha}}
\end{equation}

We can consider the special cases considered above:

\begin{equation}
    M\_{\text{NFW}} (r) = 4 \pi \rho\_0 a^3 
                      \left[ \ln(1 + r/a) - \frac{r/a}{1 + r/a} \right]
\end{equation}

\begin{equation}
    M\_{\text{H}} (r) = 2 \pi \rho\_0 a^3 \frac{(r/a)^2}{(1 + r/a)^2}
\end{equation}

Taking the limit as $r \to \infty$ shows that the total mass of the NFW distribution will diverge. The Hernquist model, however, gives a total mass of 

\begin{equation}
    M\_{\text{H}} (r \to \infty) = 2 \pi \rho\_0 a^3
\end{equation}
"

# ╔═╡ 133cd1c6-802d-11eb-15f7-ed66d2529095
function nfw_enclosed(r, ρ_0, a)
	x = r / a
	return 4 * pi * ρ_0 * a^3 * (log.(1 .+ x) .- x ./ (1 .+ x))
end

# ╔═╡ 9e74402c-802c-11eb-1109-5173362ae07a
function hernquist_enclosed(r, ρ_0, a)
	x = r / a
	return 2 * pi * ρ_0 * a^3 * x.^2 ./ (1 .+ x).^2
end

# ╔═╡ 4011ba22-8041-11eb-1113-5dd249b0c2f1
md"Let's plot these distributions for some physical parameters. We'll use $\rho_0$ and $a$ corresponding to the Milky Way halo from Dierickx and Loeb 2017."

# ╔═╡ af62af46-802e-11eb-2ff2-3f88382136be
begin
	M_200 = 1e12      # virial mass [M_sol]
	R_200 = 206       # virial radius [kpc]
	c = 10            # concentration
	a = R_200 / c     # NFW scale length
	f = log(1 + c) - c / (1 + c) # where does this come from?
	ρ_0 = M_200 / (4 * pi * f * a^3)
	
	a_H = 38.35       # Hernquist scale length [kpc]
	M_halo = 1.25e12  # Hernquist total mass [M_sol]
	ρ_H = M_halo / (2 * pi * a_H^3)
	
	radii = collect(range(1, R_200, length=200))
end

# ╔═╡ 5ffff3d0-802d-11eb-307c-750e23925857
begin
	plot(
		radii,
		[nfw_enclosed(radii, ρ_0, a), hernquist_enclosed(radii, ρ_H, a_H)],
		title = "Enclosed mass",
		line = [:solid :dash],
		label = ["NFW" "Hernquist"],
		legend = :topleft
	)
end

# ╔═╡ 07c8acf6-802e-11eb-3b41-3d7395e803bd
begin
	plot(
		radii,
		[nfw_density(radii, ρ_0, a), hernquist_density(radii, ρ_H, a_H)],
		title = "Mass density",
		line = [:solid :dash],
		label = ["NFW" "Hernquist"],
		scale = :log10,
		legend = :topright
	)
end

# ╔═╡ Cell order:
# ╟─167e2fba-8037-11eb-2897-3f52b70f7f83
# ╠═4d663dfc-802c-11eb-1272-49a4f2026d04
# ╠═937e5b6c-802c-11eb-2deb-3737d72d6620
# ╠═7cba71e0-802c-11eb-3fdf-2916082f3d3d
# ╟─9448d948-8040-11eb-05f4-1f982783556a
# ╠═133cd1c6-802d-11eb-15f7-ed66d2529095
# ╠═9e74402c-802c-11eb-1109-5173362ae07a
# ╟─4011ba22-8041-11eb-1113-5dd249b0c2f1
# ╠═af62af46-802e-11eb-2ff2-3f88382136be
# ╠═5ffff3d0-802d-11eb-307c-750e23925857
# ╠═07c8acf6-802e-11eb-3b41-3d7395e803bd
# ╠═43180e7a-802c-11eb-22f9-f90a6e9a526e
