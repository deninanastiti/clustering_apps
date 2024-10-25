# Importing Important Libraries
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

# Set a wider layout for a better viewing experience
st.set_page_config(layout="wide", page_title="Clustering Application")

# Add a title and description with enhanced styling
st.title('🔍 Clustering Application: K-Means, Hierarchical, DBSCAN, Mean Shift, & GMM')
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Welcome to the **Clustering Application**! 🎨  
Choose from **K-Means**, **Hierarchical Clustering**, **DBSCAN**, **Mean Shift**, or **Gaussian Mixture Model (GMM)**.
Input your data points and see the results visualized instantly! 🪄
""")

# Sidebar for user inputs
st.sidebar.header("Input Options")
num_points = st.sidebar.number_input('Number of data points:', min_value=1, value=5)
clustering_method = st.sidebar.selectbox(
    'Choose Clustering Method:', 
    ('K-Means', 'Hierarchical', 'DBSCAN', 'Mean Shift', 'Gaussian Mixture Model (GMM)')
)
n_clusters = st.sidebar.number_input('Number of clusters (for K-Means, Hierarchical, and GMM):', min_value=1, value=2)
eps = st.sidebar.slider('DBSCAN Epsilon (radius):', 0.1, 5.0, 0.5)
min_samples = st.sidebar.number_input('DBSCAN Minimum Samples:', min_value=1, value=2)

# Input data points using Streamlit's expander for a cleaner look
st.write(f'Enter {num_points} data points in **x, y** format:')
data = []

with st.expander("Input Data Points", expanded=True):
    for i in range(int(num_points)):
        col1, col2 = st.columns(2)
        with col1:
            x = st.number_input(f'Point {i+1} - X:', key=f'x{i}')
        with col2:
            y = st.number_input(f'Point {i+1} - Y:', key=f'y{i}')
        data.append([x, y])

data = np.array(data)

# Display the entered data in a neat table
st.subheader('Data Points:')
st.dataframe(data, use_container_width=True)

# Perform clustering when button is pressed
if st.button('🔮 Perform Clustering'):
    if len(data) > 0:
        if clustering_method == 'K-Means':
            # K-Means Clustering
            model = KMeans(n_clusters=n_clusters, random_state=42)
            model.fit(data)
            labels = model.labels_
            centroids = model.cluster_centers_

            # Display cluster results
            st.subheader('Cluster Labels for Each Data Point (K-Means):')
            results = {'Data Point': [f'Point {i+1}' for i in range(len(labels))], 'Cluster': labels}
            st.table(results)
            
            st.subheader('Centroids of Each Cluster:')
            centroids_df = {'Cluster': [f'Cluster {i}' for i in range(len(centroids))], 'Centroid Coordinates': centroids}
            st.table(centroids_df)

            # Visualize clustering
            st.subheader('Clustering Visualization (K-Means):')
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8, edgecolors='k', label='Data Points')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids', edgecolors='k')
            ax.set_title('K-Means Clustering', fontsize=16)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend(title='Legend', loc='upper right', fontsize=10)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster Label')
            st.pyplot(fig)

        elif clustering_method == 'Hierarchical':
            # Hierarchical Clustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
            model.fit(data)
            labels = model.labels_

            # Display cluster results
            st.subheader('Cluster Labels for Each Data Point (Hierarchical):')
            results = {'Data Point': [f'Point {i+1}' for i in range(len(labels))], 'Cluster': labels}
            st.table(results)

            # Visualize clustering
            st.subheader('Clustering Visualization (Hierarchical):')
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8, edgecolors='k', label='Data Points')
            ax.set_title('Hierarchical Clustering', fontsize=16)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster Label')
            st.pyplot(fig)

            # Create and show the dendrogram
            st.subheader('Dendrogram:')
            linked = linkage(data, 'ward')
            fig, ax = plt.subplots(figsize=(10, 7))
            dendrogram(linked, labels=[f'Point {i+1}' for i in range(num_points)], ax=ax)
            ax.set_title('Dendrogram', fontsize=16)
            ax.set_xlabel('Data Points', fontsize=12)
            ax.set_ylabel('Euclidean Distance', fontsize=12)
            st.pyplot(fig)

        elif clustering_method == 'DBSCAN':
            # DBSCAN Clustering
            model = DBSCAN(eps=eps, min_samples=min_samples)
            model.fit(data)
            labels = model.labels_

            # Display cluster results
            st.subheader('Cluster Labels for Each Data Point (DBSCAN):')
            results = {'Data Point': [f'Point {i+1}' for i in range(len(labels))], 'Cluster': labels}
            st.table(results)

            # Visualize clustering
            st.subheader('Clustering Visualization (DBSCAN):')
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8, edgecolors='k', label='Data Points')
            ax.set_title('DBSCAN Clustering', fontsize=16)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster Label')
            st.pyplot(fig)

        elif clustering_method == 'Mean Shift':
            # Mean Shift Clustering
            model = MeanShift()
            model.fit(data)
            labels = model.labels_
            centroids = model.cluster_centers_

            # Display cluster results
            st.subheader('Cluster Labels for Each Data Point (Mean Shift):')
            results = {'Data Point': [f'Point {i+1}' for i in range(len(labels))], 'Cluster': labels}
            st.table(results)
            
            st.subheader('Centroids of Each Cluster:')
            centroids_df = {'Cluster': [f'Cluster {i}' for i in range(len(centroids))], 'Centroid Coordinates': centroids}
            st.table(centroids_df)

            # Visualize clustering
            st.subheader('Clustering Visualization (Mean Shift):')
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8, edgecolors='k', label='Data Points')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids', edgecolors='k')
            ax.set_title('Mean Shift Clustering', fontsize=16)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend(title='Legend', loc='upper right', fontsize=10)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster Label')
            st.pyplot(fig)

        elif clustering_method == 'Gaussian Mixture Model (GMM)':
            # GMM Clustering
            model = GaussianMixture(n_components=n_clusters, random_state=42)
            labels = model.fit_predict(data)
            centroids = model.means_

            # Display cluster results
            st.subheader('Cluster Labels for Each Data Point (GMM):')
            results = {'Data Point': [f'Point {i+1}' for i in range(len(labels))], 'Cluster': labels}
            st.table(results)
            
            st.subheader('Centroids of Each Cluster:')
            centroids_df = {'Cluster': [f'Cluster {i}' for i in range(len(centroids))], 'Centroid Coordinates': centroids}
            st.table(centroids_df)

            # Visualize clustering
            st.subheader('Clustering Visualization (GMM):')
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', s=100, alpha=0.8, edgecolors='k', label='Data Points')
            ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids', edgecolors='k')
            ax.set_title('GMM Clustering', fontsize=16)
            ax.set_xlabel('X Coordinate', fontsize=12)
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.legend(title='Legend', loc='upper right', fontsize=10)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Cluster Label')
            st.pyplot(fig)
    else:
        st.error('❗ Please enter valid data points to perform clustering.')
