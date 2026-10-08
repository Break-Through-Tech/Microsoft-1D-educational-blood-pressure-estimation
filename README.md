# CufflessAI: An Educational Blood Pressure Estimation

See the [project roadmap](ROADMAP.md) for the current implementation assessment, research-informed next steps, proposed October–November phases, and completion criteria. Dataset findings are documented in [data notes](data/DATA_NOTES.md) and the [EDA summary](eda/EDA-readme.md).

For modeling, follow [dataset setup](data/README.md), run `python data/load_files.py`, then consume `data.load_files.iter_split_windows("train")`. The prepared archive includes stable IDs and saved recording-based splits; exact duplicate copies are excluded by this loader. [Data notes](data/DATA_NOTES.md) explain validation/test use and ID preservation through preprocessing.

> 💡 **Note for the team:** This is just a template. Update the above title with your AI Studio Challenge Project name. Remove all guidance notes and example text in this template and populate this README with your own content. You can work on this README throughout AI Studio, and get feedback from your AI Studio Coach and Challenge Advisor before finalizing it.  

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Kushagra Dhall    | @KR-0000 | EDA, Data setup            |
| Ingrid Escalante-Hernandez   | @imehtn     | Data collection, exploratory data analysis (EDA), dataset documentation  |
| Katherine Wu     | @wukatherine  | EDA, Data visualization                 |
| Joanna Liu      | @lte24       | Missing values  |
| Sarah To       | @aizhenii    | Outliers           |
| Kien Nguyen       | @tkien17    |   placeholder         |
| Harshitha Venkateswaran       | @ (add username)    |       placeholder     |

---

## 🎯 **Project Highlights**

**Example:**

- Developed a machine learning model using `[model type/technique]` to address `[challenge project task]`.
- Achieved `[key metric or result]`, demonstrating `[value or impact]` for `[host company]`.
- Generated actionable insights to inform business decisions at `[host company or stakeholders]`.
- Implemented `[specific methodology]` to address industry constraints or expectations.

---

## 👩🏽‍💻 **Setup and Installation**

**Provide step-by-step instructions so someone else can run your code and reproduce your results. Depending on your setup, include:**

* How to clone the repository
* How to install dependencies
* How to set up the environment
* How to access the dataset(s)
* How to run the notebook or scripts

---

## 🏗️ **Project Overview**

**Describe:**

- How this project is connected to the Break Through Tech AI Program
- Your AI Studio host company and the project objective and scope
- The real-world significance of the problem and the potential impact of your work

---

## 📊 **Data Exploration**

### Dataset(s) used

The project uses the **UCI Cuff-Less Blood Pressure Estimation dataset**, distributed as four MATLAB files: `Part_1.mat` through `Part_4.mat`. Although the project calls these files “raw,” UCI describes the source dataset as preprocessed.
- **Format:** MATLAB v7.3 files, read as HDF5 using `h5py`. The project also creates a local processed archive, `processed_dataset.npz`.
- **Size:** 12,000 records, each containing synchronized signals with variable lengths of 1,000–74,000 samples. The three channels are sampled at **125 Hz**:
  - **PPG:** optical pulse waveform; intended model input.
  - **ABP:** arterial blood-pressure waveform in mmHg; source of reference SBP/DBP labels.
  - **ECG:** electrocardiogram; retained for possible future work, but outside the initial model scope.
- **Processed dataset:** 528,828 complete, non-overlapping five-second windows, each 625 samples long, with aligned PPG, ABP, and ECG arrays. That is 330,517,500 retained samples per channel, or about 734.5 hours of signal.

## Data exploration and preprocessing

The data was inspected record by record to check record shapes, channel alignment, sample lengths, and invalid values. The loader reads the three synchronized channels, divides each record into complete 625-sample windows, and drops any remaining samples shorter than a full window rather than padding them. It saves the aligned windows to `processed_dataset.npz`.

For the current baseline, SBP and DBP are derived from the **maximum and minimum ABP sample in each five-second window**, respectively. ECG remains in the processed archive, but the initial modeling task uses PPG to predict the ABP-derived labels.

The EDA also checked PPG/ABP numeric validity, counted numeric zeros in PPG, searched for exact repeated recordings, and compared whole-window ABP extrema with exploratory beat-level extrema. 

## EDA insights

- All 12,000 records had the expected three-channel structure, and channels were aligned.
- Record lengths varied substantially. Windowing produced **528,828 complete windows**; 10,082 records had a short trailing segment discarded. In total, 3,172,500 sample positions per channel were dropped—about 7.05 hours across the dataset.
- PPG and ABP had no observed NaN or infinite samples. ECG had 16 NaN values in one record.
- **5,407 PPG windows (1.02%)** contained at least one numeric zero. The EDA did not establish whether these zeros indicate artifacts or valid signal values.
- The audit identified **14 exact duplicate recordings**, including 13 matches across different part files. The later duplicate copies account for 700 windows. This creates a risk of train/test leakage if matching records or neighboring windows are split independently.
- The median of the baseline whole-window labels was **131.10 mmHg for SBP** and **62.03 mmHg for DBP**. In a sample of 20,263 windows, whole-window maxima were a median 2.35 mmHg above detected beat-peak medians, while minima were 1.76 mmHg below detected beat-trough medians. These comparisons are exploratory, not clinical validation.
- The processed PPG and ABP arrays matched windows reconstructed from the source files exactly and in order. This verifies loading consistency, not signal quality.

## Challenges and assumptions

- **Label quality:** A single maximum and minimum can be affected by noise or artifacts. Beat-based peak/valley detection and aggregation may produce more robust labels, but the method and quality criteria are not finalized.
- **Leakage and provenance:** Source-record IDs should be preserved so duplicates and related windows can be kept together across data splits. The files do not provide verified patient-level identity, so a record should not be assumed to equal a unique patient.
- **Signal quality:** Numeric zeros were observed, but their cause is unknown. Rules for noisy-window rejection and physiologically plausible values still need to be established. Feature engineering to address this is in progress.
- **Model preparation:** PPG normalization and the choice between raw windows, engineered features, or both remain open.
- **Scope:** PPG is treated as the model input and synchronized ABP as the training reference. PPG amplitude is not itself a blood-pressure measurement, and the current whole-window ABP labels are a baseline rather than finalized ground truth.

**Potential visualizations to include:**

* Plots, charts, heatmaps, feature visualizations, sample dataset images

---

## 🧠 **Model Development**

**You might consider describing the following (as applicable):**

* Model(s) used (e.g., CNN with transfer learning, regression models)
* Feature selection and Hyperparameter tuning strategies
* Training setup (e.g., % of data for training/validation, evaluation metric, baseline performance)


---

## 📈 **Results & Key Findings**

**You might consider describing the following (as applicable):**

* Performance metrics (e.g., Accuracy, F1 score, RMSE)
* How your model performed
* Insights from evaluating model fairness

**Potential visualizations to include:**

* Confusion matrix, precision-recall curve, feature importance plot, prediction distribution, outputs from fairness or explainability tools

---

## 🚀 **Next Steps**

**You might consider addressing the following (as applicable):**

* What are some of the limitations of your model?
* What would you do differently with more time/resources?
* What additional datasets or techniques would you explore?

---

## 📝 **License**

Specify how your project can be used by others. Choose an appropriate license and link it here (e.g., MIT, Apache 2.0). Make sure your Challenge Advisor approves of the selected license type. 

**Example:**
This project is licensed under the MIT License.

---

## 📄 **References** (Optional but encouraged)

Cite relevant papers, articles, or resources that supported your project.

---

## 🙏 **Acknowledgements** (Optional but encouraged)

Thank your Challenge Advisor, host company representatives, TA, and others who supported your project.
