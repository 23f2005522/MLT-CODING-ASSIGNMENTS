# 🐦‍🔥 KMeans CSV Analyser

A fully interactive **Streamlit-based KMeans Clustering Visualizer** built using **pure NumPy** (without scikit-learn).  
Upload any CSV dataset, choose numeric columns, and visualize clustering results in both **2D** and **3D**.

Built for learning how KMeans works internally using manual implementation of:

- Cluster assignment
- Centroid recomputation
- Euclidean distance calculation
- Iterative optimization
- Inertia (SSE) calculation

---

## 🚀 Features

-  Upload any CSV dataset
-  Select 2D or 3D clustering visualization
-  Pure NumPy implementation (No sklearn KMeans)
-  Interactive cluster selection
-  Elbow Curve for optimal `k`
-  Iteration-by-iteration visualization
-  Download clustered CSV output
-  Educational implementation for understanding KMeans internals

---

## 🛠️ Technologies Used

- Python
- Streamlit
- NumPy
- Pandas
- Matplotlib

---

## 📁 Project Structure

```bash
project/
│
├── app.py
├── README.md
└── requirements.txt