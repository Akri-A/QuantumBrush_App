# Quantum Pointillism Brush

**Bradford Quantum Hackathon 2025**
**Team**: Khrystian Koci & Benjamin Thomas

---

## Status: 75% Complete ✅

### ✅ What's Working (Classical Infrastructure)

**Fully functional brush** creating pointillism effects with neighbor-correlated colors:

- **Poisson disk sampling** - Uniform dot distribution
- **Neighbor graph** - Spatial connectivity between dots
- **Adaptive algorithms** - Dynamic sizing and distance calculations
- **Classical color blending** - Iterative neighbor-aware color evolution
  - Ferromagnetic mode (J > 0): Smooth gradients
  - Antiferromagnetic mode (J < 0): High contrast
- **Visualization** - Optional display of quantum interaction graph
- **6 user parameters** - Full control over appearance and behavior

**Try it now**: Draw with the brush, adjust Coupling Strength to see ferromagnetic vs antiferromagnetic behavior!

---

### 📋 Quantum Placeholders (Ready for Implementation)

**4 functions** with detailed implementation guides in docstrings:

1. `encode_color_to_qubit()` - RGB → quantum angles (5 min)
2. `spherical_to_rgb()` - Quantum angles → RGB (10 min)
3. `create_ising_pointillism_circuit()` - Build Ising Hamiltonian circuit (1-1.5 hrs)
4. `measure_all_qubits_to_colors()` - Measure & decode to colors (1-1.5 hrs)

**Total quantum implementation time**: 2-3 hours

**Architecture**: When quantum functions return valid data (not `None`), brush automatically switches from classical to quantum mode.

---

## The Innovation

**Classical pointillism**: Random dot colors
**Our approach**: Dot colors determined by quantum Ising model interactions

**Result**: Neighbor correlations create emergent patterns impossible with classical randomness - visual demonstration of quantum many-body physics!

---
