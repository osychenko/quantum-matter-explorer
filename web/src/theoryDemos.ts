/** Patch 40: exact small-model calculations for teaching only.
 * These are not sampled Monte Carlo measurements.
 */
function positive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive and finite`);
}

/** Normalized excited-state weight after two-level imaginary-time projection.
 * H has E0=0 and E1=gap; the trial state has initial excited-state weight p.
 * Squared amplitudes decay as exp(-2 tau E).
 */
export function projectedExcitedWeight(tau: number, gap: number, p: number): number {
  if (!Number.isFinite(tau) || tau < 0) throw new RangeError('tau must be nonnegative');
  positive(gap, 'gap');
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new RangeError('p must be in [0,1]');
  const attenuation = Math.exp(-2 * tau * gap);
  const denominator = 1 - p + p * attenuation;
  // The case p=1 has no overlap with the ground state and never projects onto it.
  return denominator === 0 ? 1 : p * attenuation / denominator;
}

export function oscillatorExactEnergy(omega: number): number {
  positive(omega, 'omega');
  return 1.5 * omega;
}

/** Exact expectation for a Gaussian in the 3D oscillator, NOT a VMC result.
 * H = -∇²/2 + omega² r²/2, ψ ∝ exp(-alpha*r²/2).
 */
export function oscillatorTrialEnergy(alpha: number, omega: number): number {
  positive(alpha, 'alpha'); positive(omega, 'omega');
  return 0.75 * (alpha + omega * omega / alpha);
}

/** Analytical local-energy standard deviation, not a statistical SEM. */
export function oscillatorLocalEnergyStd(alpha: number, omega: number): number {
  positive(alpha, 'alpha'); positive(omega, 'omega');
  return Math.sqrt(3 / 8) * Math.abs(omega * omega - alpha * alpha) / alpha;
}

/** Exchanged term divided by (direct + exchanged) in the diagonal two-free-boson
 * 1D density matrix. This is a ratio of terms, NOT a path-swap probability.
 * Free kernel ρ(x,x';β) ∝ exp[-(x-x')²/(4 D β)].
 */
export function twoBosonExchangeTermFraction(distance: number, beta: number, D=0.5): number {
  if (!Number.isFinite(distance) || distance < 0) throw new RangeError('distance must be nonnegative');
  positive(beta, 'beta'); positive(D, 'D');
  const ratio = Math.exp(-distance * distance / (2 * D * beta));
  return ratio / (1 + ratio);
}
