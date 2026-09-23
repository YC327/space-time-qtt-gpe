# Initial ground state

- `ground_state_state.jls`: serialized state and grid metadata, loaded by the 07 driver.
- `ground_state.csv`: scaled complex wavefunction and physical density.
- `summary.txt`: ground-state parameters and diagnostics.
- `picard_history.csv`: nonlinear solver convergence history.
- `ground_state.png`: ground-state density plot.

The provided summary reports g=12.5, a 32 x 32 periodic grid on [-6,6)^2,
`validation_pass = true`, norm squared 0.9999999999999997,
and stationary residual 3.7909558617859326e-5.

The CSV norm, RMS radius, boundary-density summary and quench-adjusted energy
agree with the recorded breathing run at t=0. The serialized file is preserved
byte for byte; it has not been deserialized or used for a new Julia run here.
The default run command finds `inputs/ground_state_state.jls` automatically.
