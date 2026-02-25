# Clinical Knowledge Book: Medical Imaging Agent Specifications

> [!IMPORTANT]
> **Clinical Safety Warning**: This document serves as the ground-truth knowledge base for the MedSAM-3/MedGemma Agent. All decision logic must strictly adhere to radiological standards (ACR/Lung-RADS/LI-RADS/CAP).

## 0. General Radiology Assessment (Universal Protocols)
**Keywords**: detect, pathology, pathologies, abnormality, scan, general, check, screen, find, localize.

### 0.1 Systematic Search Pattern (ABCDE)
For any Chest X-Ray or General Scan request ("Detect pathologies"):
-   **A (Airway)**: Check trachea for deviation (tension pneumothorax/mass effect).
-   **B (Breathing)**: Inspect lung fields for consolidation (pneumonia), nodules, or collapse (atelectasis). Compare symmetry.
-   **C (Circulation)**: Assess heart size (Cardiothoracic Ratio > 0.5 = Cardiomegaly). Check mediastinal width.
-   **D (Diaphragm)**: Look for costophrenic angle blunting (Pleural Effusion) or free air (Pneumoperitoneum).
-   **E (Extras)**: Check bones (fractures), tubes, lines, and soft tissues (subcutaneous emphysema).

---

## 1. Chest Radiology (X-Ray & CT)

### 1.1 Pulmonary Nodules (Lung-RADS Logic)
**Keywords**: nodule, mass, lung, opacities, cancer, coin lesion.
-   **Ground Glass Opacity (GGO)**: Hazy increased opacity preserving bronchial/vascular margins.
    -   *Risk*: Persistent GGOs >10mm have high malignancy probability (Adenocarcinoma spectrum).
-   **Part-Solid Nodule**: Contains both GGO and solid soft-tissue components.
    -   *Risk*: **Highest Malignancy Rate**. The solid component size is the key prognostic factor.
-   **Spiculation**: "Sunburst" appearance. Highly specific for malignancy.
-   **Calcification Patterns**:
    -   *Benign*: Diffuse, Central, Laminated, Popcorn (Hamartoma).
    -   *Malignant*: Eccentric, Amorphous, Punctate.

### 1.2 Pneumothorax Quantification
**Keywords**: pneumothorax, collapse, air, pleural.
-   **Small**: <2cm distance between lung margin and chest wall at apex.
-   **Large**: >2cm separation. Requires considering tube thoracostomy.
-   **Tension**: Tracheal deviation away from side of pneumothorax + hemodynamic instability. **Critical Alert**.

### 1.3 Pneumonia & Consolidation
**Keywords**: pneumonia, consolidation, infiltration, opacity, infection.
-   **Lobar**: Solid opacity adhering to lobar fissures (e.g., Strep pneumoniae).
-   **Interstitial**: Reticular/net-like markings (e.g., Viral, Mycoplasma).
-   **Air Bronchogram**: Dark, air-filled bronchi visible within dense consolidation. confirms alveolar pathology.

---

## 2. Neuroimaging (MRI & fMRI)

### 2.1 Acute Ischemic vs. Hemorrhagic Stroke Protocols
**Keywords**: stroke, brain, infarct, bleed, hemorrhage, mri, head.
**Decision Tree**:
1.  **Hemorrhage Exclusion (NCCT)**:
    -   Scan for high-attenuation material (50-100 HU).
    -   Differentiate calcification (>100 HU, focal) from acute blood.
2.  **Early Ischemic Signs (NCCT)**:
    -   **Insular Ribbon Sign**: Loss of gray-white differentiation in insular cortex (early MCA infarct).
    -   **Obscuration of Lentiform Nucleus**: Loss of edges due to edema.
    -   **Dense MCA Sign**: Focal hyperdensity indicating thrombus.
3.  **MRI Confirmation (DWI/ADC)**:
    -   **True Infarct**: Hyperintense on DWI + Hypointense on ADC (Restricted Diffusion).
    -   **T2 Shine-through**: Hyperintense on DWI + Iso/Hyperintense on ADC.
    -   **PWI/DWI Mismatch**: Large PWI defect (penumbra) vs small DWI core = Salvageable tissue (Thrombectomy candidate).

### 2.2 Brain Tumor Characterization
**Keywords**: tumor, glioma, gbm, meningioma, mass, brain.
**Differential**: Glioblastoma (GBM) vs. Metastasis vs. Meningioma.

| Feature | Glioblastoma (GBM) | Metastasis | Meningioma |
| :--- | :--- | :--- | :--- |
| **Enhancement** | Thick, irregular, rim-enhancing. | Thin, uniform ring. | Homogeneous, intense. |
| **Necrosis** | Common (Central). | Common (Central). | Rare. |
| **Margination** | Infiltrative (fuzzy). | Well-defined. | Well-defined (Dural Tail). |
| **Edema** | Infiltrative (Tumor cells present). | Vasogenic (Pure fluid). | Variable. |
| **Perfusion (rCBV)** | >1.75x in edema (Infiltration). | Low/Normal in edema. | Extremely High (Hypervascular). |

### 2.3 functional MRI (fMRI) & Presurgical Mapping
**Keywords**: fmri, functional, motor, language, mapping.
-   **Task-Based fMRI**:
    -   **Language Mapping**: Identify Broca’s (IFG) and Wernicke’s (STG) areas. Calculate Lateralization Index (LI). LI > 0.2 indicates Left Dominance.
    -   **Motor Mapping**: Locate Primary Motor Cortex via "Omega Sign" in precentral gyrus.
-   **Resting-State fMRI (rs-fMRI)**:
    -   **Default Mode Network (DMN)**: Seed-Based Analysis using Posterior Cingulate Cortex (PCC) [MNI: 0, -53, 26].
    -   **Alzheimer's Biomarker**: Reduced PCC-Hippocampus connectivity.

---

## 3. Cardiovascular Imaging (CT/MRI/Ultrasound)

### 3.1 Coronary CT Angiography (CCTA) - Plaque Analysis
**Keywords**: heart, coronary, vessel, plaque, stenosis.
-   **Calcified Plaque**: >400 HU. Stable but stenotic.
-   **Fibrous Plaque**: 60-150 HU. Intermediate.
-   **Low-Attenuation Plaque (LAP)**: <30 HU. **High Risk** (Lipid-Rich Core).
-   **Napkin-Ring Sign**: Central low attenuation (necrotic) surrounded by high attenuation (cap). Specific for high-risk vulnerable plaque.

### 3.2 Cardiac MRI - Viability & Cardiomyopathy
**Keywords**: cardiac, mri, myocardium, heart failure.
-   **Late Gadolinium Enhancement (LGE)**:
    -   *Ischemic*: Subendocardial to Transmural enhancement matching vascular territory.
    -   *Non-Ischemic (Dilated CM)*: Mid-wall septal striae.
    -   *Myocarditis*: Epicardial or patchy mid-wall enhancement.
-   **T1 Mapping**: Low T1 indicates Iron Overload; High T1 indicates Fibrosis/Edema.

---

## 4. Abdominal & Pelvic Imaging

### 4.1 Liver Masses (LI-RADS)
**Keywords**: liver, hepatic, mass, hcc.
-   **Hepatocellular Carcinoma (HCC)**:
    -   Arterial Phase Hyperenhancement (APHE) (Non-rim).
    -   Portal Venous "Washout".
    -   Enhancing "Capsule".
-   **Hemangioma**: Peripheral discontinuous nodular enhancement with progressive centripetal fill-in.

### 4.2 Kidney Masses (Bosniak Classification)
**Keywords**: kidney, renal, cyst, mass.
-   **Category I**: Simple cyst (water density, hair-thin wall). *Benign*.
-   **Category II**: Minimally complex (thin septa). *Benign*.
-   **Category III**: Indeterminate (thick/irregular septa). *Surgical*.
-   **Category IV**: Clearly malignant (enhancing solid components). *Surgical*.

---

## 5. Digital Pathology (Whole Slide Imaging)

### 5.1 Prostate Cancer (Gleason Grading)
**Keywords**: prostate, gleason, cancer, pathology, slide.
-   **Gleason 3**: Individual, well-formed glands.
-   **Gleason 4**: Fused glands, cribriform (sieve-like) pattern, poorly defined lumens.
-   **Gleason 5**: Solid sheets of cells, necrosis, no gland formation.

### 5.2 Breast Cancer
**Keywords**: breast, cancer, mitosis, tubule.
-   **Mitotic Index**: Count mitotic figures (dark, hairy, irregular chromatin) in High-Power Fields. Key aggression indicator.
-   **Tubule Formation**: % of tumor forming tubules.

