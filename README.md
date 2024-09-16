# Flood Warning Clustering Analysis

## Project Objective
This project aims to analyze flood warning data from Malaysian weather stations using unsupervised machine learning techniques, specifically K-means clustering. The goal is to identify patterns and group similar flood warning stations based on their geographical locations and water level characteristics.

## Problems Aimed to Solve
1. Identify geographical clusters of flood-prone areas in Malaysia.
2. Recognize stations with similar flood risk profiles.
3. Provide insights for more targeted flood management and prevention strategies.
4. Assist in the allocation of resources for flood mitigation efforts.

## Key Processes Involved
1. Data Fetching: Retrieve real-time flood warning data from the Malaysian government API.
2. Data Preprocessing: Clean and prepare the data for analysis, including handling missing values and invalid coordinates.
3. Feature Scaling: Normalize the data to ensure all features contribute equally to the clustering process.
4. K-means Clustering: Apply the K-means algorithm to group stations based on their characteristics.
5. Visualization: Create a scatter plot to visualize the geographical distribution of clusters.
6. Analysis: Compute and display summary statistics for each cluster.

## How to Run the Project

### Prerequisites
- Python 3.7 or higher
- Poetry (for dependency management)

### Setup
1. Clone the repository:
   ```
   git clone [your-repo-url]
   cd [your-repo-name]
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

### Running the Script
1. Activate the Poetry shell:
   ```
   poetry shell
   ```

2. Run the main script:
   ```
   python open-moh-kmeans.py
   ```

### Expected Output
- A matplotlib plot showing the geographical distribution of clustered flood warning stations.
- Printed summary statistics for each cluster, including average water levels and thresholds.
- Information on the total number of stations and those used in the clustering analysis.

## Data Source
This project uses real-time data from the Malaysian government's open data API:
https://api.data.gov.my/flood-warning/?meta=true

## Future Improvements
- Implement dynamic cluster number selection (e.g., elbow method).
- Add more visualizations to represent water level data.
- Incorporate time-series analysis for temporal patterns in flood warnings.
- Develop a web interface for interactive data exploration.

## Contributing
Contributions to improve the analysis or extend the project's capabilities are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## License
[Specify your license here]
