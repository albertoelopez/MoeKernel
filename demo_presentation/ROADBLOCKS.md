# What roadblocks did you run into?

### Technical Challenges

1.  **NumPy Version Compatibility**: MAX required NumPy < 2.0, which conflicted with other modern packages. This was resolved by pinning the version in our requirements.
    *   **Lesson**: Ecosystem version compatibility is crucial.

2.  **TensorType Initialization**: The `device` parameter required for MAX graph creation was not clearly documented, leading to trial-and-error.
    *   **Lesson**: Clear API documentation for edge cases is needed.

3.  **Cross-Platform Testing**: Performance benchmarks varied significantly across different hardware.
    *   **Lesson**: Standardized hardware baselines are necessary for comparable results.

### Development and Integration Challenges

1.  **Simulation vs. Reality Gap**: Python simulations could not fully predict the performance of the final Mojo kernel.
    *   **Lesson**: Better profiling tools for hybrid Python-Mojo development are needed.

2.  **Visualization Dependencies**: Cross-platform compatibility issues with `seaborn` and `matplotlib` required implementing graceful fallbacks.
    *   **Lesson**: Robust dependency management is critical for demos.

3.  **Demo Accessibility**: To make the project accessible without a full Mojo environment, we created Python-based simulations for immediate validation.
    *   **Lesson**: Multiple entry points and demos increase project adoption.
