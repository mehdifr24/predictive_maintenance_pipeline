# ✈️ Predictive Maintenance for NASA Jet Engines: Remaining Useful Life (RUL) Estimation

## 📖 About
This project demonstrates a production-grade, end-to-end Machine Learning workflow and software deployment for predicting the Remaining Useful Life (RUL) of turbofan engines using NASA's C-MAPSS (FD001) time-series sensor dataset. The pipeline handles core industrial challenges such as multi-sensor telemetry processing, non-linear degradation modeling, strict data leakage prevention, and robust model serialization. Finally, the optimized standalone model is deployed to a public-facing web application via Streamlit Community Cloud and GitHub Releases.

🔗 **[Live Streamlit Web Application](https://predictivemaintenancepipeline-gdf6tjoqmwt25mjelftkus.streamlit.app)**  
📊 **[Weights & Biases Public Dashboard](https://wandb.ai/mehdifr24-/nasa-jet-engine-rul)**

---

## 🚀 Technologies
* Python
* Pandas & NumPy
* Scikit-Learn (ExtraTreesRegressor, Pipelines)
* Weights & Biases (WandB) for Experiment Tracking
* Joblib (Model & Bundle Serialization)
* Streamlit Community Cloud & GitHub Releases

---

## 📂 Dataset
The dataset is derived from NASA's C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) turbofan engine degradation simulation (FD001 subset). It contains multi-variate time-series sensor measurements and operational settings recorded until system failure, allowing supervised regression models to learn progressive wear patterns and estimate remaining cycles.

---

## 📊 Project Workflow
1. **Data Preprocessing & Telemetry Cleaning:** Handling missing values, normalizing sensor readings, and structuring operational settings safely.
2. **Target Engineering:** Computing the Remaining Useful Life (RUL) for each engine cycle based on operational failure thresholds.
3. **Model Training & Hyperparameter Tuning:** Evaluating regression models and optimizing an **Extra Trees Regressor** to capture complex non-linear feature interactions.
4. **Experiment Tracking:** Logging all metrics, parameters, and training runs systematically using **Weights & Biases (WandB)**.
5. **Production Serialization (MLOps):** Bundling the standalone estimator along with expected feature signatures into a single robust `.pkl` artifact using `joblib` and hosting large artifacts via **GitHub Releases** (>25MB Git limit workaround).
6. **Cloud Deployment:** Serving predictions interactively via a clean, zero-setup **Streamlit Cloud** web application.

---

## 📈 Model Performance & Evaluation
To evaluate regression robustness, RMSE (Root Mean Squared Error) and $R^2$ Score were prioritized during model selection and tracking.

| Strategy / Configuration | Key Hyperparameters | Tracking Metric / Result |
| :--- | :--- | :--- |
| Baseline Regressor | Default Settings | Moderate RMSE |
| Optimized Extra Trees | n_estimators=100, random_state=42 | High $R^2$ & Minimized RMSE |

---

## 📊 Visual Artifacts & Feature Insights

### 📈 Model Evaluation & Experiment Tracking
- View live tracking metrics, runs, and system telemetry interactively via the [Weights & Biases Public Dashboard](https://wandb.ai/mehdifr24-/nasa-jet-engine-rul).

### 📉 Feature Importance
<img width="734" height="468" alt="image" src="https://github.com/user-attachments/assets/f150ed57-a23f-4303-b226-ee84f97e910a" />

- **Streamlit App Preview:**  
  ![Streamlit Dashboard Preview](https://predictivemaintenancepipeline-gdf6tjoqmwt25mjelftkus.streamlit.app)

---

## 🏆 Why Extra Trees & MLOps Architecture?
For high-dimensional industrial sensor data, tree-based ensemble methods like Extra Trees provide exceptional non-linear mapping capabilities and robustness against overfitting.
- **Leakage Prevention:** Clean separation of train/test telemetry cycles.
- **Production-Ready Artifacts:** Bundling expected feature signatures ensures zero runtime mismatch during inference.
- **Scalable Hosting:** Leveraging GitHub Releases for large artifact storage bridges the gap between local research and cloud-native deployment.

---

## 👨‍💻 Author
**Mehdi Ferdosi**  
Computer Science Student | Machine Learning Enthusiast  
GitHub: [mehdifr24](https://github.com/mehdifr24)
