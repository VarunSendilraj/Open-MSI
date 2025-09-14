# Open-MSI

**Open-Source GUI Tool for Personalized Visualization and Streamlined Analysis of Mass Spectrometry Data**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()

Open-MSI is a Python-based GUI application that converts Time of Flight Mass Spectrometry (TOF-MS) data into image data for visualization and analysis. The application provides an interactive interface for optimizing mass alignment values, noise parameters, and start times to enhance data interpretation through computer vision and machine learning algorithms.

![Open-MSI Interface](https://github.com/VarunSendilraj/Open-MSI/assets/57602146/bb355e20-7c51-4449-9d5f-2062123b1989)

## Features

- **Interactive Parameter Tuning**: Real-time adjustment of mass alignment, noise reduction, and ablation time parameters
- **Multiple Visualization Modes**: Standard imaging, heatmaps, 3D visualization, and clustering analysis
- **Data Export**: Save processed images in various formats (BMP, PNG, etc.)
- **Machine Learning Integration**: Built-in clustering and pattern recognition capabilities
- **User-Friendly Interface**: Intuitive GUI with sliders and real-time preview

## Background

Time of Flight Mass Spectrometry (TOF-MS) is an analytical technique that measures the mass-to-charge ratio of ions by measuring their flight time through a drift region. The resulting data forms a two-dimensional matrix where:
- **X-axis**: Time of flight
- **Y-axis**: Mass-to-charge ratio (m/z)
- **Intensity**: Ion abundance at each time/mass coordinate

Converting this data to image format enables advanced visualization and analysis techniques, helping researchers identify patterns, structures, and compositions in their samples.

## Installation

### Option 1: Binary Distribution
1. Download the pre-built executable from the [releases page](../../releases)
2. Extract and run the application directly (no installation required)

### Option 2: From Source

**Prerequisites:**
- Python 3.7 or higher
- Git (for cloning the repository)

**Installation Steps:**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/VarunSendilraj/Open-MSI.git
   cd Open-MSI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python src/GuiCode/ToFMSGui.py
   ```

## Quick Start

1. **Launch the application** using one of the installation methods above
2. **Load your data** by clicking "Open File" and selecting a TOF-MS data file (`.txt` format)
3. **Adjust parameters** using the interactive sliders:
   - **Ablation Time**: Controls the time interval for data collection
   - **Mass Alignment**: Aligns peaks for improved accuracy
   - **Noise Threshold**: Reduces background noise for clearer images
4. **Preview changes** in real-time as you modify parameters
5. **Save your results** by clicking "Save Image"

## Project Structure

```
Open-MSI/
├── src/                          # Source code
│   ├── GuiCode/                  # Main GUI application
│   └── fileFormatting/           # Data formatting utilities
├── visualization/                # Additional visualization tools
│   └── AdditionalVisualizationTechniques/
├── examples/                     # Sample data and outputs
│   ├── data/                     # Sample TOF-MS datasets
│   ├── outputs/                  # GUI-generated results
│   └── sample_outputs/           # Reference outputs
├── legacy/                       # Original code (for reference)
│   └── OriginalCode/
├── docs/                         # Documentation and images
│   └── images/
├── build/                        # Build artifacts
├── dist/                         # Distribution files
└── requirements.txt              # Python dependencies
```

## Data Format

Open-MSI accepts TOF-MS data in tab-separated text format (`.txt`):
```
time1   mass1   intensity1
time2   mass2   intensity2
...
```

Sample data files are available in the `examples/data/` directory.

## Advanced Features

### Visualization Techniques
- **3D Visualization**: Explore data in three dimensions
- **Heatmap Generation**: Create intensity-based color maps
- **Overlay Analysis**: Compare multiple datasets

### Machine Learning
- **K-means Clustering**: Automatic pattern detection
- **PCA Analysis**: Dimensionality reduction and feature extraction

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Open-MSI in your research, please cite:
```
[Citation format to be added]
```

## Support

- **Issues**: Report bugs or request features on our [GitHub Issues](../../issues) page
- **Documentation**: Find detailed guides in the [wiki](../../wiki)
- **Discussions**: Join the community in [GitHub Discussions](../../discussions)

## Acknowledgments

- Built with Python, Tkinter, OpenCV, and scikit-learn
- Special thanks to the mass spectrometry community for feedback and testing
