"""Patch 37: scientific navigation and interpretation for the curated educational API.

This is explanatory metadata, NOT a registry of new computation endpoints.
Research and planned problems remain non-executable through the educational API.
Actual numerical values always come from educational_contract.run_experiment.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

from .educational_contract import SCHEMA_VERSION, catalog, run_experiment

IDEAL = 'ideal-boson-addition-v1'
RECORDED = 'softcore-m8-baseline-recorded-v1'
OSC = 'harmonic-oscillator-vmc-v1'
STRUCT = 'softcore-structure-vmc-v1'

# A small, finite, inspectable educational ontology: which questions are
# answerable now, and which are only present in the research engine.
_PROBLEMS = [
    dict(id='thermal-bosons', title='What does temperature change in a Bose gas?',
         area='Quantum statistics', status='interactive', preset_id=IDEAL,
         question='How does changing inverse temperature or box size affect adding a boson to the zero-momentum state?',
         method='Canonical ideal-gas partition sum', observable='Particle-addition Green function at k=0',
         visual='momentum-levels', level='Foundation'),
    dict(id='interacting-addition', title='What does repulsion do to particle addition?',
         area='Interacting quantum matter', status='recorded', preset_id=RECORDED,
         question='How does an interacting two-boson sector compare with a free-boson control and a spectral reference?',
         method='Baseline continuum worm Monte Carlo', observable='G₀(β/2) with replica SEM',
         visual='worldlines', level='Intermediate'),
    dict(id='ground-state', title='How can we find ground-state energies?',
         area='Variational methods', status='interactive', preset_id=OSC,
         question='How close can a Gaussian trial wavefunction come to a known ground-state energy?',
         method='Live Metropolis VMC, Gaussian trial and an exact oscillator reference',
         observable='Variational energy, exact E₀ and Gaussian expectation',
         visual='energy-landscape', level='Foundation'),
    dict(id='spatial-order', title='How can a quantum fluid develop structure?',
         area='Many-body correlations', status='interactive', preset_id=STRUCT,
         question='What can pair correlation g(r) and structure factor S(k) tell us about arrangements of particles?',
         method='Live finite-N Jastrow VMC (trial-state diagnostics)',
         observable='Radial g(r), directional S(k), variational energy',
         visual='particle-cloud', level='Intermediate'),
    dict(id='exchange-cycles', title='Why does bosonic exchange matter?',
         area='Quantum statistics', status='research_only', preset_id=None,
         question='What is a permutation of indistinguishable bosons in imaginary time?',
         method='Canonical path-integral Monte Carlo (research modules)',
         observable='Permutation-cycle frequencies', visual='worldlines', level='Future laboratory'),
]

_METHODS = [
    dict(id='analytic', title='Analytical thermal reference', availability='interactive',
         principle='Write the canonical one-boson partition function as a sum over allowed momentum states, then evaluate a known addition-correlation identity.',
         procedure=['Define a periodic cubic box and free-boson Hamiltonian.',
                    'Enumerate momentum modes with εₙ = D(2π/L)²|n|².',
                    'Sum Boltzmann weights to obtain Z₁; compute G₀⁺ = 1 + 1/Z₁.'],
         limitation='The partition sum is truncated numerically (|nᵢ| ≤ 10); the last-shell gap is an indicator, not a rigorous error bound.'),
    dict(id='worm', title='Baseline continuum worm Monte Carlo', availability='recorded',
         principle='Sample both closed host configurations and open particle-addition configurations in imaginary time; use their properly normalized ratio to estimate G₀⁺.',
         procedure=['Represent the finite-temperature host and added particle with periodic imaginary-time links.',
                    'Propose open/close and configuration updates satisfying the intended sampling weights.',
                    'Accumulate correlated numerator/denominator measurements; assess uncertainty using independent replicas.'],
         limitation='Only a recorded N=1→2, M=8 benchmark is curated. Finite timestep and mixing errors are not proven absent.'),
    dict(id='vmc', title='Variational Monte Carlo', availability='limited_live' ,
         principle='Choose a trial state, sample its probability distribution, and estimate an energy upper bound under the usual variational assumptions.',
         procedure=['Choose a trial wavefunction and its parameters.',
                    'Sample configurations proportional to |Ψₜ|².',
                    'Average the local energy and evaluate statistical uncertainty.'],
         limitation='Only fixed small educational presets run live. An unconstrained VMC trial is not necessarily the exact ground state; finite-size and sampling biases matter.'),
    dict(id='dmc', title='Diffusion Monte Carlo', availability='research_only',
         principle='Approximate imaginary-time projection toward a ground state with branching or resampling and importance sampling.',
         procedure=['Initialize walkers, often from a VMC distribution.',
                    'Advance with short-time drift and diffusion plus weights.',
                    'Assess timestep, population-control, and trial-state biases.'],
         limitation='The existing experimental split/RK2 method is not certified second-order.'),
    dict(id='pimc', title='Bosonic path-integral Monte Carlo', availability='research_only',
         principle='At finite temperature, a quantum partition function can be represented with closed paths and boson permutations in imaginary time.',
         procedure=['Discretize inverse temperature into links.',
                    'Sample closed paths with periodic boundary conditions and permutation sectors.',
                    'Measure permutation statistics and suitable diagonal observables.'],
         limitation='Permutation frequency alone does not prove superfluidity or condensation.'),
]

# Parameters are identified by their JSON contract paths, not copied numeric
# literals. This prevents the UI from silently showing stale values.
_PROFILES = {
    IDEAL: dict(
        problem_id='thermal-bosons', method_id='analytic',
        question='How do temperature and confinement change the probability amplitude for adding a zero-momentum boson?',
        model='One ideal boson is in a 3D periodic cubic box at inverse temperature β. The added sector has two identical, noninteracting bosons.',
        observable='G₀⁺(β/2), the canonical imaginary-time particle-addition Green function at zero momentum (dimensionless).',
        interpretation='For an ideal N=1 thermal host at zero momentum, G₀⁺ = 1 + 1/Z₁. The box and temperature change Z₁, not the interaction.',
        source_kind='live analytical control with a numerically truncated partition sum',
        parameters=[
            dict(path='parameters.beta', label='Inverse temperature β', category='physical', explanation='β = 1/(kBT) in these units; increasing β cools the system.'),
            dict(path='parameters.cell_length', label='Cubic cell length L', category='physical', explanation='Sets the box volume L³ and spacing between allowed momenta.'),
            dict(path='parameters.kinetic_coefficient', label='Kinetic coefficient D', category='physical', explanation='The kinetic Hamiltonian is −D∇²; D = ℏ²/(2m).'),
            dict(path='parameters.n_background', label='Background bosons N', category='physical', explanation='One thermal boson before a second is added.'),
            dict(path='parameters.interaction_height', label='Interaction height', category='physical', explanation='Exactly zero for the ideal gas.'),
            dict(path='parameters.momentum_mode', label='Reciprocal mode', category='observable', explanation='Zero momentum k=0, not an arbitrary position.'),
            dict(path='parameters.t_fraction', label='Imaginary-time fraction', category='observable', explanation='Measured at τ=β/2; the ideal k=0 expression is τ-independent.'),
            dict(path='parameters.momentum_cutoff', label='Momentum cutoff |nᵢ|', category='numerical', explanation='The analytic thermal sum enumerates integer modes through |nᵢ|≤10.'),
            dict(path='parameters.truncation_check_cutoff', label='Check cutoff |nᵢ|', category='numerical', explanation='A second sum at |nᵢ|≤8 checks sensitivity to truncation.'),
            dict(path='observables.0.numerical_truncation_indicator', label='Last-shell sensitivity', category='numerical', explanation='Difference between momentum cutoffs 8 and 10; not a rigorous truncation bound.'),
        ],
        comparison=dict(title='Vary one parameter within the same ideal model',
            target='Two values of the same analytical Green function',
            held_fixed='Hamiltonian, N=1, zero momentum and whichever parameter is not being swept',
            changed='Either inverse temperature β or box length L',
            cautions='This is a controlled analytical parameter response, not a noisy Monte Carlo run.'),
        theory_ids=['hamiltonian', 'quantum-statistics', 'green-function', 'periodic-boundaries'],
    ),
    RECORDED: dict(
        problem_id='interacting-addition', method_id='worm',
        question='How does a soft-core repulsion affect the measured particle-addition Green function, and how trustworthy is the estimate?',
        model='N=1 thermal free host in a 3D periodic cubic box; the added N+1=2 sector has a finite-range repulsive pair potential.',
        observable='G₀⁺(β/2), a dimensionless particle-addition correlation estimated from open/closed worm sectors.',
        interpretation='The recorded mean and its between-replica SEM are compared with a K=2 finite-basis spectral reference for the same interacting Hamiltonian.',
        source_kind='fixed historical 8-replica Monte Carlo measurement, not a new live calculation',
        parameters=[
            dict(path='parameters.beta', label='Inverse temperature β', category='physical', explanation='Fixed at 2.4 in the recorded run.'),
            dict(path='parameters.cell_length', label='Cubic cell length L', category='physical', explanation='The periodic box has side length 2.8.'),
            dict(path='parameters.kinetic_coefficient', label='Kinetic coefficient D', category='physical', explanation='D=0.5, corresponding to the reduced-unit particle mass.'),
            dict(path='parameters.n_background', label='Background bosons N', category='physical', explanation='N=1 host and N+1=2 particle-added sector.'),
            dict(path='parameters.potential_height', label='Repulsion strength V₀', category='physical', explanation='The potential has maximum height 3.0 at coincidence.'),
            dict(path='parameters.potential_cutoff', label='Interaction range r꜀', category='physical', explanation='Soft-core interaction vanishes beyond r꜀=1.2.'),
            dict(path='parameters.M', label='Imaginary-time links M', category='numerical', explanation='Eight links give Δτ=β/M=0.3; discretization has systematic error.'),
            dict(path='parameters.method', label='Sampler', category='numerical', explanation='Baseline worm sampler (not replica exchange).'),
            dict(path='parameters.validation_steps', label='Steps per chain', category='numerical', explanation='Each of eight held-out runs used 30,689 updates.'),
            dict(path='parameters.burn_in', label='Warm-up updates', category='numerical', explanation='400 initial updates discarded in each chain.'),
            dict(path='parameters.thin', label='Recording stride', category='numerical', explanation='Record one measurement every two updates after warm-up.'),
            dict(path='parameters.sector_fugacity', label='Worm-sector fugacity', category='numerical', explanation='Sampling normalization η=0.9; not a physical chemical potential.'),
            dict(path='parameters.momentum_mode', label='Reciprocal mode', category='observable', explanation='Zero-momentum particle-addition observable.'),
            dict(path='parameters.t_fraction', label='Imaginary-time fraction', category='observable', explanation='Evaluated at the midpoint β/2.'),
            dict(path='provenance.replicas', label='Independent replicas', category='statistical', explanation='Eight independent validation chains determine the reported SEM.'),
            dict(path='provenance.validation_seeds', label='Validation seeds', category='statistical', explanation='Independent held-out validation seed identifiers.'),
            dict(path='provenance.cpu_target_seconds', label='CPU target per chain (s)', category='statistical', explanation='Pilot-calibrated soft target, not a stopping rule.'),
            dict(path='provenance.mean_cpu_seconds', label='Measured mean CPU (s)', category='statistical', explanation='Actual process CPU averaged across the eight chains.'),
            dict(path='observables.0.uncertainty', label='Replica SEM', category='statistical', explanation='Between-independent-chain standard error, not a proof of convergence.'),
        ],
        comparison=dict(title='Monte Carlo benchmark versus spectral reference',
            target='Same interacting model and G₀⁺(β/2), estimated with different numerical methods',
            held_fixed='β, cubic cell, kinetic coefficient, pair potential, boson sector and zero momentum',
            changed='Worm: primitive imaginary-time links M=8; reference: finite plane-wave basis K=2',
            cautions='Agreement within SEM is a useful cross-check, not a proof of zero timestep bias or spectral-basis convergence.'),
        theory_ids=['hamiltonian', 'imaginary-time', 'green-function', 'uncertainty'],
    ),
}


# Patch 38: two tightly bounded live VMC laboratories. Parameter paths refer to
# real response values; the question/model/method descriptions are not guesses.
_PROFILES[OSC] = dict(
    problem_id='ground-state', method_id='vmc',
    question='How does choosing a different Gaussian trial wavefunction change a ground-state energy estimate?',
    model='One particle in a 3D isotropic harmonic trap; H=−(1/2)∇²+(1/2)ω²r², with a Gaussian Ψα∝exp(−αr²/2). No periodic boundaries.',
    observable='Variational energy from live Metropolis sampling of |Ψα|², with independent-replica SEM; exact ground energy and exact trial expectation.',
    interpretation='At α=ω the Gaussian is the exact eigenstate and the local energy has zero variance. Away from α=ω the trial expectation rises above E₀, while finite Monte Carlo means can fluctuate below.',
    source_kind='live, bounded educational VMC using the existing numerical engine',
    parameters=[
        dict(path='parameters.trial_alpha', label='Trial width α', category='variational', explanation='A variational wavefunction parameter; NOT a change to the harmonic trap Hamiltonian.'),
        dict(path='parameters.omega', label='Trap angular frequency ω', category='physical', explanation='Sets the actual harmonic potential strength and exact energy.'),
        dict(path='parameters.kinetic_coefficient', label='Kinetic coefficient D', category='physical', explanation='Fixed D=0.5.'),
        dict(path='parameters.dimensions', label='Spatial dimensions', category='physical', explanation='Exactly three.'),
        dict(path='parameters.n_walkers', label='Walkers', category='numerical', explanation='Independent walker coordinates within each Markov run.'),
        dict(path='parameters.n_steps', label='Steps per run', category='numerical', explanation='Metropolis updates including burn-in.'),
        dict(path='parameters.burn_in', label='Warm-up updates', category='numerical', explanation='Updates before measured energy samples.'),
        dict(path='parameters.sample_every', label='Measurement stride', category='numerical', explanation='Collect energies every this many steps after warm-up.'),
        dict(path='provenance.replicas', label='Independent replicas', category='statistical', explanation='Replicated seeds; SEM is across run means.'),
        dict(path='provenance.seeds', label='Fixed seeds', category='statistical', explanation='Reproducible distinct random seeds.'),
        dict(path='observables.0.uncertainty', label='Energy SEM', category='statistical', explanation='Sampling uncertainty, not variational bias.'),
        dict(path='observables.0.exact_ground_energy', label='Exact ground energy', category='observable', explanation='Exact benchmark E₀=3ω/2 in the chosen units.'),
        dict(path='observables.0.analytic_trial_expectation', label='Exact Gaussian trial expectation', category='observable', explanation='⟨H⟩α=3(α+ω²/α)/4, not a Monte Carlo estimate.'),
    ],
    comparison=dict(title='VMC versus two mathematically distinct references',
        target='Live VMC estimate; known Gaussian variational expectation; exact ground energy',
        held_fixed='For a fixed ω, the same harmonic Hamiltonian and sampling rules',
        changed='Trial parameter α changes the trial state, not the trap; ω changes the Hamiltonian',
        cautions='Monte Carlo noise affects the measured VMC value, while the exact trial expectation captures the trial-state bias.'),
    theory_ids=['hamiltonian', 'uncertainty', 'variational-principle'],
)
_PROFILES[STRUCT] = dict(
    problem_id='spatial-order', method_id='vmc',
    question='How do correlations of a symmetric trial state appear in g(r) and S(k)?',
    model='Six bosons, cubic periodic cell L=5, fixed finite-range soft-core repulsion V₀=3, cutoff 1.2. Sample an exchange-symmetric Jastrow trial.',
    observable='Radial pair correlation g(r) for complete shells and directional S(k) on four reciprocal modes, plus variational energy per particle.',
    interpretation='At amplitude=0, Ψ is constant and the positional distribution is uniform. Increasing the Jastrow parameter suppresses close pairs in the TRIAL distribution; this is not changing V₀.',
    source_kind='live, bounded educational VMC using existing g(r) and S(k) accumulators',
    parameters=[
        dict(path='parameters.jastrow_amplitude', label='Trial Jastrow strength a', category='variational', explanation='Variational wavefunction parameter; does NOT change the physical pair potential.'),
        dict(path='parameters.n_particles', label='Boson count N', category='physical', explanation='Six indistinguishable bosons in the finite box.'),
        dict(path='parameters.cell_length', label='Box length L', category='physical', explanation='Fixed periodic cubic length.'),
        dict(path='parameters.potential_height', label='Soft-core repulsion height V₀', category='physical', explanation='Held fixed even when the Jastrow amplitude changes.'),
        dict(path='parameters.cutoff', label='Pair cutoff r꜀', category='physical', explanation='Pair interaction and trial factor vanish beyond 1.2.'),
        dict(path='parameters.kinetic_coefficient', label='Kinetic coefficient D', category='physical', explanation='Fixed at 0.5.'),
        dict(path='parameters.n_walkers', label='Walkers per run', category='numerical', explanation='Parallel configurations within one run, not independent replica counts.'),
        dict(path='parameters.n_steps', label='Metropolis sweeps', category='numerical', explanation='One sweep attempts one move per particle.'),
        dict(path='parameters.burn_in', label='Warm-up sweeps', category='numerical', explanation='Initial sweeps discarded.'),
        dict(path='parameters.sample_every', label='Measurement stride', category='numerical', explanation='Sampling the structural observables every five sweeps.'),
        dict(path='provenance.replicas', label='Independent replicas', category='statistical', explanation='Three independently seeded runs.'),
        dict(path='provenance.seeds', label='Seeds', category='statistical', explanation='Fixed random seeds.'),
        dict(path='observables.0.uncertainty', label='Energy SEM', category='statistical', explanation='Independent-replica SEM for variational energy per particle.'),
    ],
    comparison=dict(title='Controlled trial-state comparison, not an interaction scan',
        target='g(r) and S(k) for chosen Jastrow amplitude versus the uniform ideal positional reference 1',
        held_fixed='N, box, repulsive Hamiltonian and VMC settings',
        changed='Only the symmetric trial wavefunction amplitude a',
        cautions='The reference g=1, S=1 is for uniform independent positions, not the ground state of the interacting Hamiltonian. Pointwise error bars do not imply phase boundaries.'),
    theory_ids=['hamiltonian', 'periodic-boundaries', 'uncertainty', 'variational-principle', 'structural-observables'],
)

_THEORY = [
    dict(id='hamiltonian', title='The Hamiltonian: what the system is',
         concept='The Hamiltonian describes the physical problem, not the accuracy of a numerical method.',
         physics='Kinetic energy favors spreading of the wavefunction; the chosen soft-core repulsion penalizes pairs at short distances. Setting V=0 produces an ideal gas.',
         equation='H = −D Σᵢ ∇ᵢ² + Σᵢ<ⱼ V(|rᵢ−rⱼ|)',
         interpretation='D=ℏ²/(2m). For the interacting preset, V(r)=V₀ max(1−r²/r꜀²,0)³ with periodic minimum-image distances.'),
    dict(id='quantum-statistics', title='Why identical bosons are different',
         concept='Identical bosons cannot be distinguished by permanent particle labels. Their quantum amplitudes are symmetric under exchange.',
         physics='Even without interactions, bosonic statistics changes occupation of momentum modes. Temperature changes how the one-boson background populates allowed states.',
         equation='Z₁ = Σₙ exp[−β D (2π/L)² |n|²]',
         interpretation='In a cubic periodic box n is a triple of integers. At lower temperature (larger β), excited momentum states become less populated.'),
    dict(id='green-function', title='What the Green function measures',
         concept='A correlation function asks how a prepared quantum disturbance is related to a later operation—not where a particle travels in real time.',
         physics='The particle-addition Green function inserts a momentum-k boson and later removes it in imaginary time, thermally averaging over the N-boson host.',
         equation='Gₖ⁺(τ) = ⟨aₖ(τ) aₖ†(0)⟩ₙ; ideal k=0: G₀⁺ = 1 + 1/Z₁',
         interpretation='The extra “1” reflects bosonic creation algebra. The recorded interacting value is a different Hamiltonian, not an exact analytical control.'),
    dict(id='imaginary-time', title='Imaginary time and the worm method',
         concept='Imaginary time turns parts of a quantum problem into statistical weights that can be sampled.',
         physics='At finite temperature the Euclidean time extent is β. The worm algorithm switches between closed and open sectors to estimate off-diagonal particle-addition correlations.',
         equation='τ_step = β/M; T(τ_step) ≈ exp(−τ_step K/2) exp(−τ_step V) exp(−τ_step K/2)',
         interpretation='M controls numerical discretization. A sampled imaginary-time path is neither a measured real-time trajectory nor a literal particle orbit.'),
    dict(id='periodic-boundaries', title='The periodic box is not a wall',
         concept='A particle leaving one side of a periodic simulation box returns through the opposite side.',
         physics='Periodic boundaries replace a finite physical container with a repeating computational cell. The permitted wavevectors are discrete.',
         equation='k = (2π/L) (nₓ, nᵧ, n_z), nᵢ ∈ ℤ',
         interpretation='The neighboring boxes in the 3D view are repeated images, not additional independent particles.'),
    dict(id='variational-principle', title='Why a trial state matters',
         concept='VMC samples what a chosen wavefunction predicts. The trial state is an approximation, not the actual ground state by definition.',
         physics='For a normalized trial state, the exact energy expectation cannot be below the true ground energy. A finite Monte Carlo estimate can fall below because it fluctuates.',
         equation='E[Ψ] = ⟨Ψ|H|Ψ⟩/⟨Ψ|Ψ⟩ ≥ E₀;  E_Gaussian(α)=3(α+ω²/α)/4',
         interpretation='For D=0.5 and harmonic frequency ω, the optimum Gaussian is α=ω. It has E₀=3ω/2 and zero local-energy variance.'),
    dict(id='structural-observables', title='Correlations and spatial order',
         concept='A snapshot gives positions, but structural observables summarize statistical patterns across many configurations.',
         physics='g(r) measures the density of particle pairs at separation r relative to uncorrelated uniform positions. S(k) measures Fourier-space density fluctuations at specified directions.',
         equation='g(r)∝⟨Σ_{i<j} δ(r−|rᵢ−rⱼ|)⟩; S(k)=⟨|Σⱼ exp(ik·rⱼ)|²⟩/N (k≠0)',
         interpretation='For independently uniform positions, g(r)=1 (within full spherical shells) and S(k)=1 for nonzero reciprocal k. Finite-N trial-state variations are not phase transitions.'),
    dict(id='imaginary-time-projection', title='Imaginary-time projection and the ground state',
         concept='Imaginary-time projection can filter excited states to reveal the ground state if the starting state overlaps it.',
         physics='Each energy eigenstate amplitude is weighted by exp(−τE). After normalization, higher-energy contributions decay more rapidly than the ground-state contribution.',
         equation='|ψ(τ)⟩ ∝ exp(−τH)|ψ(0)⟩; pₑ(τ) = pₑ(0) exp(−2Δτ) / [1−pₑ(0)+pₑ(0) exp(−2Δτ)]',
         interpretation='This is a two-level ground-state projector with E₀=0 and Δ>0. It is distinct from the finite-temperature imaginary-time worm correlation sampled over period β; τ is not real time.'),
    dict(id='uncertainty', title='What does ± mean?',
         concept='A numerical result has statistical uncertainty and may also have systematic error. These are distinct.',
         physics='Independent Monte Carlo chains estimate the between-chain spread. A spectral comparison checks agreement with a truncated independent solver, not the exact continuum target.',
         equation='SEM_replica = s(chain means) / √R; z = |G_MC−G_ref| / SEM_replica',
         interpretation='An apparently small z does not prove convergence. Sampling correlations, timestep bias, and reference truncation still matter.'),
]

def science_map() -> dict[str, Any]:
    """Serializable, read-only educational navigation and experiment profiles."""
    return deepcopy(dict(schema_version=SCHEMA_VERSION, problems=_PROBLEMS,
                         methods=_METHODS, profiles=_PROFILES, theory=_THEORY))


def validate_science_map() -> None:
    """Fail tests if the copy of supported scope or parameter contracts drifts."""
    presets = {p['id']: p for p in catalog()}
    assert set(_PROFILES) == set(presets)
    methods = {m['id'] for m in _METHODS}
    problems = {p['id']: p for p in _PROBLEMS}
    theory = {t['id'] for t in _THEORY}
    for problem in _PROBLEMS:
        if problem['preset_id'] is not None:
            assert problem['preset_id'] in presets
            assert problem['status'] in ('interactive', 'recorded')
        else:
            assert problem['status'] == 'research_only'
    for preset_id, profile in _PROFILES.items():
        assert profile['method_id'] in methods
        assert problems[profile['problem_id']]['preset_id'] == preset_id
        assert set(profile['theory_ids']) <= theory
        result = run_experiment(preset_id)
        for spec in profile['parameters']:
            node: Any = result
            for segment in spec['path'].split('.'):
                node = node[int(segment)] if isinstance(node, list) else node[segment]
            assert node is not None, (preset_id, spec['path'])
