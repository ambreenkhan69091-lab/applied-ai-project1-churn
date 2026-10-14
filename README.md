# Blueprint Specification: Subscriber Attrition Intelligence System (SAIS)

**Core Engineering Lead:** Ambreen Khan  
**Course Framework:** Applied Machine Learning Systems (BS 8th Semester)  
**Hosted Digital Endpoint:** [PASTE AMBREEN'S LIVE STREAMLIT APP LINK HERE]

---

### 🗂️ PHASE 1 ARCHITECTURE: RECOVERY, INGESTION & PIPELINE TRACING

#### Operational Framework Profiles
* **Foundation Matrix Population:** 7,043 client transaction records across 21 input vectors.
* **Target Objective Element:** Class binary label tracking customer departure.
* **Imbalance Disparity Index:** 26.54% true signal distribution base.

#### Phase 1 Engineering Takeaways
- **Contract Duration Friction:** Account profiles running on short-term rolling cycles show rapid decay, identifying them as a primary source of operational revenue loss.
- **Premium Infrastructure Paradox:** High-speed Fiber Optic lines show higher risk scores than traditional copper setups. This points to billing pain points or connection instability shortly after installation.
- **Clearing Pipeline Vulnerabilities:** Manual paper check clearing methods show elevated user drop-off velocities compared to automated bank routing profiles.
- **Type Incompatibility Fix:** Trapped a hidden parsing bug where total billing accumulations were stored as text. Forcing numeric conversion exposed 11 null values belonging to new profiles with zero months of history.

---

### ⚙️ PHASE 2 ARCHITECTURE: EVALUATION MATRIX & OPERATIONAL COST TUNING

#### Out-of-Fold Performance Blueprints (Test Split Independent Verification)

* **Naive Status Quo Loop**
  - Accuracy Mapping: 73.5% | Diagnostic Precision: 0.0% | System Sensitivity (Recall): 0.0% | F1 Balance: 0.0% | Area Under ROC: 0.500
* **Base Linear Classifier Pipeline**
  - Accuracy Mapping: 80.7% | Diagnostic Precision: 65.8% | System Sensitivity (Recall): 56.7% | F1 Balance: 0.609 | Area Under ROC: 0.842
* **Frequency-Shifted Balanced Linear Pipeline**
  - Accuracy Mapping: 74.0% | Diagnostic Precision: 50.7% | System Sensitivity (Recall): **78.6%** | F1 Balance: 0.616 | Area Under ROC: 0.841
* **Default Ensemble Tree Pipeline**
  - Accuracy Mapping: 78.5% | Diagnostic Precision: 61.8% | System Sensitivity (Recall): 49.7% | F1 Balance: 0.551 | Area Under ROC: 0.825

#### Production Optimization Takeaways
- **The Accuracy Delusion:** A status-quo dummy model achieves an attractive 73.5% general score while completely failing to flag a single customer departure. This proves that standard accuracy metrics are highly deceptive on imbalanced datasets.
- **Sensitivity Correction:** Applying inverse data frequency adjustments via class-balancing scripts successfully forces the system to prioritize catching at-risk users, driving model Recall up to 78.6%.
- **Financial Threshold Tuning:** Configured a dedicated operational loss matrix utilizing specific financial weights (Missed Churn Penalty = PKR 6,000 vs. False Positive Incentive Cost = PKR 1,000). The optimal decision threshold drops to 0.15, minimizing corporate cash leakage.

---

### 👥 PHASE 3 ARCHITECTURE: BEHAVIORAL COHORTS & DECOMPOSITION

#### Customer Behavior Strata (k=4 K-Means Grouping Engine)
- **Strata Node 0 (The Volatile High-Value Group):** Early-stage accounts (~18.4 months avg. lifetime) dealing with premium high-tier bills (~\$80.41/mo). Displays an aggressive 43.0% churn concentration, identifying the main target for proactive retention efforts.
- **Strata Node 2 (New, Cheap Tryouts):** Short-tenure users (~9 months) on cheap base packages (~\$37.71). Displays a high 32.0% churn rate.
- **Strata Nodes 1 & 3 (The Legacy Base):** Highly stable accounts spanning 53 to 60 months. Despite high service costs in Strata Node 3 (~\$92.09/mo), multiple active add-ons heavily insulate them from risk vectors (retaining an 85.5% loyalty rate).

#### Dimensionality Subspace Decomposition (PCA)
- Structural variance scanning reveals substantial input vector overlap. 15 out of 30 principal components successfully retain 90% of total system variance, proving heavy redundancy across the raw categorical features space.

#### Production Deployment Specifications
- **Validation Framework:** 5-Fold Stratified Cross-Validation loop execution.
- **Single-Run Test Matrix Performance:** Final Test ROC-AUC locks at 0.841 (fully stable and confirming optimal generalization limits without overfitting).
- **Exported Asset:** Serialized the finished, production-ready pipeline architecture directly into `churn_model.joblib`.

---

### 🛡️ PHASE 4 ARCHITECTURE: STREAMLIT SERVICE WEB APP

- Re-engineered the serialized pipeline binaries into a clean, responsive web interface via Streamlit. The system includes an automated What-If Optimization Matrix for individual profiles and a batch upload portal for bulk spreadsheet processing.
