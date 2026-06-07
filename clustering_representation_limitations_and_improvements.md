# Clustering Feature Representation: Limitations and Potential Improvements

## 1. Current Pipeline

The current Python pipeline follows the original MATLAB-style representation.
Continuous Harp IMU data are converted into fixed-length behavior windows before clustering.

Current flow:

```text
raw Harp IMU data
-> cleaned motion data
-> processed sensor signals
-> 300 ms windows, 60 samples per window
-> normalized histogram features
-> clustering
```

Each 300 ms behavior window is represented using four sensor-derived features:

```text
y_GA
z
z_gyro
log(total body acceleration)
```

For each feature, the pipeline builds a 99-bin normalized histogram.
The four histograms are concatenated:

```text
4 features x 99 bins = 396-dimensional feature vector
```

This representation is useful because it is compatible with the original MATLAB pipeline, computationally efficient, and suitable for EMD/Wasserstein-style distance calculations.

However, this representation also has several important limitations, especially for Parkinson's disease progression analysis.

## 2. Main Limitations

### 2.1 Fixed 300 ms Window Size

The pipeline uses a fixed window size:

```text
60 samples at 200 Hz = 300 ms
```

Different behaviors may have different natural time scales:

```text
fast head movement: around 50 ms
one walking step: around 500 ms
turning: around 1 second
```

A fixed 300 ms window can split one complete behavior into multiple windows or merge multiple behaviors into one window.
This may be especially problematic for Parkinson's mouse models, because movement rhythm and action duration may differ from healthy controls.

### 2.2 Temporal Order Is Lost

Histogram features describe the distribution of values inside a window, but they do not preserve temporal order.

For example:

```text
fast -> slow
slow -> fast
```

These two windows may produce very similar histograms, even though they may represent different behaviors.

This is a major concern for Parkinson's analysis.
Tremor is a high-frequency, small-amplitude temporal pattern.
It may have a similar amplitude distribution to other small movements, but a very different time structure.

### 2.3 Cross-Channel Relationships Are Weakened

The four sensor features are histogrammed separately and then concatenated.
The distance calculation can be interpreted as:

```text
D = EMD_y_GA + EMD_z + EMD_z_gyro + EMD_log_accel
```

This captures within-channel distribution differences, but it weakens cross-channel relationships.

For example:

```text
running: high acceleration + high gyroscope activity
head turning: low acceleration + high gyroscope activity
```

These behaviors may have similar channel-wise distance totals but different cross-channel structures.
Behavior is often defined by coordinated changes across multiple sensor channels, so this information may be important.

### 2.4 Fixed Histogram Bin Edges

The histogram bin edges are hardcoded from the original MATLAB pipeline.

Example:

```python
_EDGES_3D = [
    np.linspace(-1.0,  0.8,  100),   # y_GA
    np.linspace(-1.5,  2.0,  100),   # z
    np.linspace(-2e4,  2e4,  100),   # z_gyro
    np.linspace(-8.0,  0.7,  100),   # log(total acceleration)
]
```

These ranges are not learned from the current dataset.
They were inherited from the MATLAB pipeline.

If Parkinson's mice have smaller movement amplitudes or tremor-like motion, their values may concentrate in a narrow region of these fixed ranges.
As a result:

```text
many bins may be zero
low-amplitude behaviors may collapse into the same few bins
disease-related movement differences may become harder to detect
```

This could make it difficult to distinguish normal immobility from Parkinson's-related motor impairment.

### 2.5 Sparse Histograms

Each 300 ms window contains only 60 samples, but each feature has 99 bins.
Therefore, many bins are expected to be zero.

This is not a bug, but sparse histograms can make distance estimates less stable.
Small changes in bin assignment can affect the resulting feature vector, especially for short windows.

### 2.6 The Four Features May Not Be Optimal for Parkinson's Phenotypes

The current four features were inherited from the MATLAB pipeline:

```text
y_GA
z
z_gyro
log(total body acceleration)
```

These are useful general movement features, but they may not be optimal for Parkinson's-related motor phenotypes.

Potentially important disease-related features include:

```text
high-frequency acceleration components
tremor-band power
movement rhythmicity
gait periodicity
step regularity
pause duration
movement initiation latency
```

These features are not directly captured by the current histogram representation.

### 2.7 Normalization May Weaken Intensity or Duration Information

Each histogram is normalized by the number of samples in the window.
This makes windows comparable, but it also emphasizes relative distribution shape rather than raw count or duration information.

Some intensity-related or duration-related movement changes may be weakened after normalization.

## 3. Most Important Concerns for Parkinson's Disease Progression

The two most important concerns for Parkinson's disease analysis are likely:

```text
1. fixed bin edges may not fit Parkinson's mouse movement distributions
2. temporal information is lost inside each 300 ms window
```

If Parkinson's mice gradually show reduced movement amplitude, but the bins are calibrated for normal movement ranges, disease-related low-amplitude behaviors may fall into the same few bins.
The algorithm may then struggle to distinguish normal stillness from Parkinson's-related movement difficulty.

Similarly, tremor is fundamentally a temporal and frequency-domain pattern.
A histogram representation that ignores time order may not be sufficient to detect tremor-like movement.

## 4. Potential Improvements

### 4.1 Test Multiple Window Sizes

Instead of using only 300 ms windows, test multiple time scales:

```text
50 ms
150 ms
300 ms
600 ms
1 second
```

This can help determine whether disease-related behaviors are better captured at shorter or longer time scales.

### 4.2 Add Temporal Features

Keep the current histogram features as a baseline, but add temporal summaries within each window.

Possible temporal features:

```text
first-half vs second-half difference
slope
peak timing
number of zero crossings
autocorrelation
movement burst duration
```

These features can help distinguish behavior sequences with similar distributions but different order.

### 4.3 Add Frequency-Domain Features

Parkinson's-related tremor and gait changes may be better captured in the frequency domain.

Possible features:

```text
tremor-band power
dominant frequency
spectral entropy
high-frequency acceleration power
rhythmicity score
```

These may be especially useful for MitoPark or other Parkinson's model datasets.

### 4.4 Recalibrate Histogram Bin Edges

Compare the current MATLAB hardcoded bins against alternative binning methods:

```text
fixed MATLAB bins
global quantile bins
arena-specific bins
control/mouse-population calibrated bins
reduced-bin versions, such as 30 or 50 bins
```

For cross-dataset comparison, bins should be calibrated globally rather than separately for each mouse.
Otherwise, the same bin index may represent different physical values in different datasets.

### 4.5 Reduce Histogram Sparsity

Possible strategies:

```text
reduce bins from 99 to 50 or 30
apply histogram smoothing
increase window size
add pseudo-counts for numerical stability
```

These changes should be tested against the current MATLAB-style baseline.

### 4.6 Add Cross-Channel Features

To capture coordination between sensor channels, add features such as:

```text
correlation between z and z_gyro
correlation between acceleration and gyroscope channels
cross-channel covariance
2D histograms, such as z vs z_gyro
```

This may help distinguish behaviors with similar marginal distributions but different multi-channel structure.

### 4.7 Test Raw Stage2 Window Representations

The current histogram representation converts each window into a distribution.
An alternative is to use the processed stage2 window directly:

```text
60 samples x 4 channels
```

Possible approaches:

```text
flatten 60 x 4 into a 240-dimensional vector
use DTW-style distances
use sliced Wasserstein on 60 x 4 point clouds
use sequence models
```

This would preserve more temporal information but may be more sensitive to noise, alignment, and scaling.

## 5. Recommended Priorities

### Priority 1: Keep Current Pipeline as Baseline

The current MATLAB-style histogram pipeline should remain the baseline because it is reproducible and compatible with previous analysis.

### Priority 2: Add Disease-Relevant Features

Add temporal and frequency-domain features focused on Parkinson's phenotypes:

```text
tremor-related features
movement rhythmicity
pause duration
movement initiation features
```

### Priority 3: Test Bin Edge Sensitivity

Compare clustering results using:

```text
current fixed MATLAB bins
reduced bin counts
global quantile bins
```

This directly tests whether Parkinson's mouse data are being compressed into too few bins.

### Priority 4: Validate With Behavior and Biology

Cluster metrics alone are not enough.
Validation should include:

```text
video examples
cluster usage by condition
control vs Parkinson's comparison
July vs October progression
2D vs 3D arena usage
known tremor or gait phenotypes
```

## 6. Main Discussion Question

Is the current MATLAB-style histogram representation sufficient for detecting Parkinson's-related motor phenotypes, or should we extend the feature representation to preserve more temporal, frequency-domain, and cross-channel movement structure?

## Short Summary

The current histogram-based representation is fast, stable, and compatible with the original MATLAB pipeline.
However, it may lose temporal order, weaken cross-channel relationships, depend on fixed bin edges, and miss disease-specific features such as tremor or gait rhythmicity.

For Parkinson's disease progression analysis, the most important next steps are to test bin-edge sensitivity and add temporal/frequency-domain features while keeping the current pipeline as a baseline.
