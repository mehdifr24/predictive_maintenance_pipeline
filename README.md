# ✈️ Predictive Maintenance for NASA Jet Engines: Remaining Useful Life (RUL) Estimation

## 📖 About
This project demonstrates a production-grade, end-to-end Machine Learning workflow and software deployment for predicting the Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS (FD001) time-series sensor dataset[cite: 11]. The pipeline handles core industrial challenges such as multi-sensor telemetry processing, non-linear degradation modeling, strict data leakage prevention, and robust model serialization[cite: 11]. Finally, the optimized standalone model is deployed to a public-facing web application via Streamlit Community Cloud and GitHub Releases[cite: 11].

🔗 **[Live Streamlit Web Application](https://predictivemaintenancepipeline-gdf6tjoqmwt25mjelftkus.streamlit.app)**[cite: 11]  
📊 **[Weights & Biases Public Dashboard](https://wandb.ai/mehdifr24-/nasa-jet-engine-rul)**[cite: 20]

---

## 🚀 Technologies
* Python[cite: 11]
* Pandas & NumPy[cite: 11]
* Scikit-Learn (ExtraTreesRegressor, Pipelines)[cite: 11]
* Weights & Biases (WandB) for Experiment Tracking[cite: 11]
* Joblib (Model & Bundle Serialization)[cite: 11]
* Streamlit Community Cloud & GitHub Releases[cite: 11]

---

## 📂 Dataset
The dataset is derived from NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation simulation (FD001 subset)[cite: 11]. It contains multi-variate time-series sensor measurements and operational settings recorded until system failure, allowing supervised regression models to learn progressive wear patterns and estimate remaining cycles[cite: 11].

---

## 📊 Project Workflow
1. **Data Preprocessing & Telemetry Cleaning:** Handling missing values, normalizing sensor readings, and structuring operational settings safely[cite: 11].
2. **Target Engineering:** Computing the Remaining Useful Life (RUL) for each engine cycle based on operational failure thresholds[cite: 11].
3. **Model Training & Hyperparameter Tuning:** Evaluating regression models and optimizing an **Extra Trees Regressor** to capture complex non-linear feature interactions[cite: 11].
4. **Experiment Tracking:** Logging all metrics, parameters, and training runs systematically using **Weights & Biases (WandB)**[cite: 11].
5. **Production Serialization (MLOps):** Bundling the standalone estimator along with expected feature signatures into a single robust `.pkl` artifact using `joblib` and hosting large artifacts via **GitHub Releases** (>25MB Git limit workaround)[cite: 11].
6. **Cloud Deployment:** Serving predictions interactively via a clean, zero-setup **Streamlit Cloud** web application[cite: 11].

---

## 📈 Model Performance & Evaluation
To evaluate regression robustness, RMSE (Root Mean Squared Error) and $R^2$ Score were prioritized during model selection and tracking[cite: 11].

| Strategy / Configuration | Key Hyperparameters | Tracking Metric / Result |
| :--- | :--- | :--- |
| Baseline Regressor | Default Settings | Moderate RMSE[cite: 11] |
| Optimized Extra Trees | n_estimators=100, random_state=42 | High $R^2$ & Minimized RMSE[cite: 11] |

---

## 📊 Visual Artifacts & Feature Insights

### 📈 Model Evaluation & Experiment Tracking
- View live tracking metrics, runs, and system telemetry interactively via the [Weights & Biases Public Dashboard](https://wandb.ai/mehdifr24-/nasa-jet-engine-rul)[cite: 20].

### 📉 Feature Importance
<img width="734" height="468" alt="image" src="https://github.com/user-attachments/assets/f150ed57-a23f-4303-b226-ee84f97e910a" />


- **Streamlit App Preview:**  
  ![Streamlit Dashboard Preview](https://predictivemaintenancepipeline-gdf6tjoqmwt25mjelftkus.streamlit.app)[cite: 11]

---

## 🏆 Why Extra Trees & MLOps Architecture?
For high-dimensional industrial sensor data, tree-based ensemble methods like Extra Trees provide exceptional non-linear mapping capabilities and robustness against overfitting[cite: 11].
- **Leakage Prevention:** Clean separation of train/test telemetry cycles[cite: 11].
- **Production-Ready Artifacts:** Bundling expected feature signatures ensures zero runtime mismatch during inference[cite: 11].
- **Scalable Hosting:** Leveraging GitHub Releases for large artifact storage bridges the gap between local research and cloud-native deployment[cite: 11].

---

## 👨‍💻 Author
**Mehdi Ferdosi**  
Computer Science Student | Machine Learning Enthusiast[cite: 11]  
GitHub: [mehdifr24](https://github.com/mehdifr24)[cite: 11]
