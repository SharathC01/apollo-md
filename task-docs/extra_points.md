**✨ EXAMPLE OF OTHER USE CASES — EXTRA POINTS**  

**Supported Query Types**  
• biomarker/score → mortality association  
• predictor comparison across studies  
• phenotype extraction from papers  

---

**🧠 EXAMPLE QUERY**  

*"What is the relationship between lactate levels and 28-day mortality in septic shock?"*  

The following examples illustrate how structured evidence extraction from clinical literature can support real-world research workflows. These use cases are provided to guide participants and demonstrate the potential applications of their systems.  

---

**🟢 USE CASE 2: SEPSIS PHENOTYPE EXTRACTION**  

**Context**  
Sepsis is a highly heterogeneous condition. Multiple studies propose different patient phenotypes based on clustering of latent structure.  

**Objective**  
Identify published sepsis phenotypes and assess whether they can be applied to an external patient cohort.  

**Role of the AI System**  
The system should extract:  
• phenotype identification methods *(e.g., k-means, latent class analysis)*  
• variables used for clustering  
• number of clusters  
• cluster characteristics *(means, medians, distributions)*  
• clinical interpretation of clusters  
• outcomes per phenotype  

**Expected Output**  
A structured representation of phenotype definitions, for example:  
• Cluster A: low severity, low mortality  
• Cluster B: high inflammation, high mortality  
• Cluster C: organ failure dominant  

with associated quantitative descriptions.

```
STUDY-LEVEL SUMMARY
---------------------------------------------------------------------------------------------------------------
| Study           | Country | Setting | Sample Size | Sepsis Def | Method            | Clusters | Variables |
|-----------------|---------|---------|-------------|------------|-------------------|----------|-----------|
| Donzelli 2019   | Norway  | ICU     | N=1476      | Sepsis-3   | k-means clustering| 4 (A–D)  | 18 vars   |
| ...             | ...     | ...     | ...         | ...        | ...               | ...      | ...       |
---------------------------------------------------------------------------------------------------------------


PHENOTYPE (CLUSTER-LEVEL) TABLE
----------------------------------------------------------------------------------------------------------------------------------
| Study         | Cluster | Key Features                      | Clinical Description        | Outcomes              | Notes                  |
|---------------|---------|-----------------------------------|-----------------------------|----------------------|------------------------|
| Donzelli 2019 | A       | Platelets↓, Lactate↓, SOFA↓       | Low severity phenotype      | ICU mortality ~12%   | Mild inflammation      |
| Donzelli 2019 | B       | Mixed markers                     | Moderate severity           | Mortality ...        | Mixed inflammation     |
| Donzelli 2019 | C       | Lactate↑, Procalcitonin↑          | High inflammation phenotype | Mortality ...        | Elevated biomarkers    |
| Donzelli 2019 | D       | SOFA↑, Lactate↑                   | Severe organ dysfunction    | Highest mortality    | High SOFA, lactate     |
| ...           | ...     | ...                               | ...                         | ...                  | ...                    |
----------------------------------------------------------------------------------------------------------------------------------
```

**⚠️ IMPORTANT NOTE**  

In many studies, phenotype assignment rules may not be fully reproducible.  
The system should explicitly indicate:  

• whether assignment is possible  
• or whether information is insufficient  

---

**📊 OUTCOME**  

Supports feasibility assessment and downstream modeling for phenotype-based analysis.

**🟢 USE CASE 3: BIOMARKER SELECTION FOR RISK STRATIFICATION**  

**Context**  
A clinical trial requires selecting a single biomarker or score to stratify patients by mortality risk.  

**Objective**  
Identify which clinical variables have the strongest prognostic value for 28-day mortality.  

**Role of the AI System**  
The system should extract and compare across studies:  
• biomarkers *(e.g., lactate, IL-6, CRP)*  
• clinical scores *(e.g., SOFA, APACHE)*  
• effect sizes *(AUC, OR, HR)*  
• statistical models  
• validation methods  
• cohort characteristics  

**Expected Output**  
A structured comparison table enabling:  
• ranking of predictors  
• comparison across studies  
• filtering by population relevance

