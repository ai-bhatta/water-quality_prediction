# 💧 Water Quality Prediction and Monitoring using Unsupervised Machine Learning

This project focuses on **predicting and monitoring water quality** using **unsupervised learning techniques**.  
By analyzing key water parameters, the model identifies potential anomalies or pollution trends — helping monitor river water quality efficiently and proactively.

---

## 📊 Project Overview

Water pollution is a major concern, and real-time analysis is essential for sustainable water resource management.  
This project uses **unsupervised learning** (specifically, **K-Means Clustering**) to group and monitor water samples based on their chemical properties.

By doing so, we can:
- Detect abnormal readings that indicate pollution.
- Track water quality trends across different monitoring stations.
- Provide insights for environmental policy and management.

---

## 🧠 Techniques & Approach

1. **Exploratory Data Analysis (EDA)**  
   - Visualized parameter distributions and relationships.  
   - Identified missing values, outliers, and normalization requirements.  
   - Key tools: `pandas`, `matplotlib`, `seaborn`.

2. **Preprocessing**  
   - Handled missing data with imputation.  
   - Standardized numerical features using `StandardScaler`.  
   - Removed outliers to improve clustering quality.

3. **Unsupervised Learning (K-Means Clustering)**  
   - Used **K-Means** to group similar water quality samples.  
   - Determined optimal clusters using **Elbow Method** and **Silhouette Score**.  
   - Interpreted clusters based on mean parameter values.

4. **Visualization & Insights**  
   - Plotted clusters and correlations between features.  
   - Used PCA (Principal Component Analysis) for 2D visualization.  
   - Derived insights on which stations had potential contamination.

---

## 🧾 Dataset

- **Source:** [Central Pollution Control Board (CPCB), India – River Water Quality 2023](https://cpcb.nic.in/wqm/2023/WQuality_River-Data-2023.pdf)  
- **Format:** CSV file converted from CPCB PDF dataset  
- **Attributes:**  
  Includes parameters like:
  - pH  
  - Dissolved Oxygen (DO)  
  - Biochemical Oxygen Demand (BOD)  
  - Nitrate (NO₃⁻)  
  - Conductivity  
  - Total Coliform, Fecal Coliform, etc.

---

## ⚙️ Project Workflow

| Step | Task | Description |
|------|------|-------------|
| 1️⃣ | Data Loading | Load and preview dataset in Google Colab |
| 2️⃣ | EDA | Analyze structure, missing values, and variable distributions |
| 3️⃣ | Data Cleaning | Handle missing values, outliers, and normalization |
| 4️⃣ | Clustering | Apply K-Means and determine optimal `k` |
| 5️⃣ | Visualization | Plot cluster distribution and interpret |
| 6️⃣ | Deployment | Deploy using Streamlit on Hugging Face or GitHub Pages |

---

## 🚀 Running the Project on Google Colab

1. **Upload the dataset** to your Google Colab working directory.
2. **Upload the notebook file**:  
   `Water_Quality_RUN_1.ipynb`
3. Run the notebook cells step by step.
4. The results (EDA, Clustering, Visualization) will appear inline.

---

## 🌐 Deployment (Optional)

You can deploy this model using:
- **Streamlit** (recommended for ML dashboards)  
- **Hugging Face Spaces** (for free hosting)
  
### Example:
```bash
pip install streamlit
streamlit run app.py
