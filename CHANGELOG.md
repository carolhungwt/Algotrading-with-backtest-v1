# Changelog

## [v1.1.0] - 2025-03-22

### Added

#### New Strategy: Triple Moving Average with Slope Confirmation
- Implemented `TripleMASlope` strategy that uses three moving averages (fast, mid, slow) to identify trends
- Added slope calculation to confirm trend direction and strength
- Incorporated equivocal market detection to avoid trading in sideways markets
- Configurable parameters for moving average periods, slope threshold, and equivocal threshold

#### Debug Mode
- Added comprehensive debugging system for strategy analysis
- Implemented visualization of moving averages and slopes when signals are triggered
- Added tracking and saving of detailed trade information to CSV files
- Created method to save snapshots of market conditions at signal points
- Included `analyze_triple_ma.py` script for analyzing debug data with statistics and visualizations
- Added command-line parameters `--debug` and `--debug-dir` for controlling debug mode

#### Scanning Mode
- Implemented scanning mode to identify buy or sell signals without executing trades
- Added ability to scan multiple tickers to find the best trading opportunities
- Created signal strength quantification for better decision making
- Added forward performance testing to evaluate signal accuracy
- Included `scan_signals.py` helper script with:
  - `scan` command for running signal scans with various strategies
  - `analyze` command for generating reports and visualizations from scan results
- Comprehensive visualization including signal count comparison, accuracy metrics, and price distributions

#### Utility Scripts
- Added `test_features.py` script to demonstrate and test the new features
- Improved error handling and logging throughout the codebase

### Fixed
- Fixed parameter passing when using separate buy and sell strategies
- Improved error handling in data processing and strategy initialization
- Enhanced documentation with examples for all new features

### Changed
- Updated README.md with detailed instructions for using new features
- Improved signal generation logic for better market adaptation
- Enhanced visualization capabilities for better analysis of results 