# README

## Soldier Defection Risk Synthetic Dataset Generation

This repository contains a Python script that generates a synthetic dataset of 10,000 soldiers to model and predict the likelihood of military defection based on various individual and external factors. The dataset is designed to simulate real-world scenarios and is useful for research, analysis, and modeling purposes.

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
  - [Individual-Level Factors](#individual-level-factors)
  - [External-Level Factors](#external-level-factors)
- [Dataset Generation Process](#dataset-generation-process)
  - [1. Import Libraries](#1-import-libraries)
  - [2. Initialize the Dataset](#2-initialize-the-dataset)
  - [3. Generate Features](#3-generate-features)
    - [Security Clearance Level](#security-clearance-level)
    - [Morale Score](#morale-score)
    - [Family Military History](#family-military-history)
    - [Punishment Policy](#punishment-policy)
    - [Regime Type](#regime-type)
    - [Trust in Leadership Score](#trust-in-leadership-score)
    - [Military Structure](#military-structure)
    - [Defector Capture Rate](#defector-capture-rate)
    - [Promotion Fairness Score](#promotion-fairness-score)
    - [Communication Quality Score](#communication-quality-score)
    - [Opportunity Cost](#opportunity-cost)
    - [Has Enemy Connections](#has-enemy-connections)
    - [Time of Measurement](#time-of-measurement)
- [Defection Risk Calculation](#defection-risk-calculation)
  - [Feature Weights](#feature-weights)
  - [Normalization](#normalization)
  - [Defection Risk Score](#defection-risk-score)
  - [Threshold Determination](#threshold-determination)
  - [Final Classification](#final-classification)
- [Usage](#usage)
- [Conclusion](#conclusion)
- [Next Steps](#next-steps)

---

## Introduction

Military defection is a complex phenomenon influenced by a multitude of factors at both the individual and external levels. Understanding these factors is crucial for predicting defection risk and implementing strategies to maintain military cohesion. This project simulates a dataset that captures these factors, allowing for analysis and modeling of defection behaviors.

---

## Features

The dataset includes a variety of features that are divided into individual-level and external-level factors. Each feature is carefully generated to reflect realistic distributions and relationships with the likelihood of defection.

### Individual-Level Factors

1. **Opportunity Cost (`opportunity_cost_norm`)**: The potential benefits a soldier foregoes by remaining in service. Higher values increase defection risk.

2. **Family Military History (`family_military_history_bin`)**: Indicates if a soldier has family members who served in the military. Contrary to intuition, in this model, it positively correlates with defection risk.

3. **Security Clearance Level (`security_clearance_level_num`)**: Access to sensitive information. Higher clearance levels increase defection risk due to the value of the information.

4. **Has Enemy Connections (`has_enemy_connections_bin`)**: Indicates connections with enemy forces. Presence of such connections increases defection risk.

5. **Morale Score (`morale_score_norm`)**: Overall satisfaction and morale. Higher morale decreases defection risk.

6. **Trust in Leadership Score (`trust_in_leadership_score_norm`)**: Trust in military leadership. Higher trust decreases defection risk.

### External-Level Factors

1. **Punishment Policy (`punishment_policy_num`)**: Perceived strictness of anti-defection policies. Lenient policies increase defection risk.

2. **Regime Type (`regime_type_num`)**: Type of political regime. Personalist regimes have higher defection risk.

3. **Military Structure (`military_structure_num`)**: Level of institutionalization. Institutionalized structures have higher defection risk due to formal systems.

4. **Defector Capture Rate (`defector_capture_rate_norm`)**: Rate at which defectors are captured. Higher rates decrease defection risk.

5. **Promotion Fairness Score (`promotion_fairness_score_norm`)**: Fairness of promotion systems. Fair systems decrease defection risk.

6. **Communication Quality Score (`communication_quality_score_norm`)**: Quality of communication within the military. Better communication decreases defection risk.

---

## Dataset Generation Process

The dataset is generated through a series of steps, each responsible for creating and processing different features.

### 1. Import Libraries

```python
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(0)
```

### 2. Initialize the Dataset

```python
# Number of soldiers
n = 10000

# Initialize empty DataFrame
df = pd.DataFrame()
```

### 3. Generate Features

#### Security Clearance Level

- **Description**: Represents the security clearance of each soldier.
- **Values**: 'low' (60%), 'medium' (30%), 'high' (10%).

```python
df['security_clearance_level'] = np.random.choice(
    ['low', 'medium', 'high'], size=n, p=[0.6, 0.3, 0.1]
)
# Map to numerical values
security_clearance_mapping = {'low': 1, 'medium': 2, 'high': 3}
df['security_clearance_level_num'] = df['security_clearance_level'].map(security_clearance_mapping)
```

#### Morale Score

- **Description**: Represents the morale level of each soldier.
- **Values**: Random float between 0 and 100.

```python
df['morale_score'] = np.random.uniform(0, 100, size=n)
```

#### Family Military History

- **Description**: Indicates if a soldier has family members who served in the military.
- **Values**: Binary (0 or 1), with a 10% chance of being 1.

```python
df['family_military_history_bin'] = np.random.binomial(1, 0.1, size=n)
```

#### Punishment Policy

- **Description**: Perceived strictness of anti-defection policies.
- **Values**: 'strict' (80%), 'lenient' (20%).

```python
df['punishment_policy'] = np.random.choice(
    ['strict', 'lenient'], size=n, p=[0.8, 0.2]
)
# Encode to numerical values
df['punishment_policy_num'] = df['punishment_policy'].map({'strict': 0, 'lenient': 1})
```

#### Regime Type

- **Description**: Type of political regime.
- **Values**: 'personalist'.

```python
df['regime_type'] = 'personalist'
# Encode to numerical values
df['regime_type_num'] = df['regime_type'].map({'personalist': 1, 'party-based': 0})
```

#### Trust in Leadership Score

- **Description**: Level of trust in military leadership.
- **Values**: Random float between 0 and 100.

```python
df['trust_in_leadership_score'] = np.random.uniform(0, 100, size=n)
```

#### Military Structure

- **Description**: Level of institutionalization in the military.
- **Values**: 'patrimonial'.

```python
df['military_structure'] = 'patrimonial'
# Encode to numerical values
df['military_structure_num'] = df['military_structure'].map({'patrimonial': 0, 'institutionalized': 1})
```

#### Defector Capture Rate

- **Description**: Rate at which defectors are captured.
- **Values**: Constant at 0.8.

```python
df['defector_capture_rate'] = 0.8
```

#### Promotion Fairness Score

- **Description**: Fairness of the promotion system.
- **Values**: 80% between 0 and 40, 20% between 40 and 100.

```python
df['promotion_fairness_score'] = np.where(
    np.random.rand(n) < 0.8,
    np.random.uniform(0, 40, size=n),
    np.random.uniform(40, 100, size=n)
)
```

#### Communication Quality Score

- **Description**: Quality of communication within the military.
- **Values**: Random float between 0 and 100.

```python
df['communication_quality_score'] = np.random.uniform(0, 100, size=n)
```

#### Opportunity Cost

- **Description**: Potential benefits a soldier foregoes by remaining in service.
- **Values**: Random float between 0 and 5000, adjusted based on security clearance.

```python
df['opportunity_cost'] = np.random.uniform(0, 5000, size=n)
# Adjust based on security clearance
df['opportunity_cost'] = df.apply(
    lambda row: row['opportunity_cost'] if row['security_clearance_level'] == 'low' else
                row['opportunity_cost'] * 0.5 if row['security_clearance_level'] == 'medium' else
                row['opportunity_cost'] * 0.2,
    axis=1
)
# Normalize
df['opportunity_cost_norm'] = (
    df['opportunity_cost'] - df['opportunity_cost'].min()
) / (df['opportunity_cost'].max() - df['opportunity_cost'].min())
```

#### Has Enemy Connections

- **Description**: Indicates if a soldier has connections with enemy forces.
- **Values**: 10% 'Yes', 90% 'No', with 95% of values set to NaN to reflect missing data.

```python
df['has_enemy_connections'] = np.where(
    np.random.binomial(1, 0.1, size=n) == 1, 'Yes', 'No'
)
# Set 95% to NaN
missing_mask = np.random.rand(n) < 0.95
df.loc[missing_mask, 'has_enemy_connections'] = np.nan
# Encode to binary
df['has_enemy_connections_bin'] = df['has_enemy_connections'].map({'Yes': 1, 'No': 0}).fillna(0)
```


---

## Defection Risk Calculation

### Feature Weights

Weights are assigned to each feature based on their positive or negative relationship with defection risk.

- **Positive Relationship (increase defection risk):**

  | Feature                      | Weight |
  |------------------------------|--------|
  | opportunity_cost_norm        | +0.4   |
  | security_clearance_level_num | +0.3   |
  | has_enemy_connections_bin    | +0.5   |
  | punishment_policy_num        | +0.2   |
  | regime_type_num              | +0.3   |
  | military_structure_num       | +0.2   |

- **Negative Relationship (decrease defection risk):**

  | Feature                           | Weight |
  |-----------------------------------|--------|
  | family_military_history_bin       | -0.2   |
  | morale_score_norm                 | -0.3   |
  | trust_in_leadership_score_norm    | -0.3   |
  | promotion_fairness_score_norm     | -0.2   |
  | communication_quality_score_norm  | -0.1   |
  | defector_capture_rate_norm        | -0.4   |

### Normalization

Features are normalized to ensure they are on the same scale (0 to 1).

```python
# Normalize negative relationship features
df['morale_score_norm'] = df['morale_score'] / 100
df['trust_in_leadership_score_norm'] = df['trust_in_leadership_score'] / 100
df['promotion_fairness_score_norm'] = df['promotion_fairness_score'] / 100
df['communication_quality_score_norm'] = df['communication_quality_score'] / 100
df['defector_capture_rate_norm'] = df['defector_capture_rate']  # Already between 0 and 1

# Normalize security clearance level
df['security_clearance_level_norm'] = df['security_clearance_level_num'] / df['security_clearance_level_num'].max()
```

### Defection Risk Score

The defection risk score is calculated by summing the weighted contributions of each feature.

```python
df['defection_risk_score'] = (
    weights['opportunity_cost_norm'] * df['opportunity_cost_norm'] +
    weights['family_military_history_bin'] * df['family_military_history_bin'] +
    weights['security_clearance_level_num'] * df['security_clearance_level_norm'] +
    weights['has_enemy_connections_bin'] * df['has_enemy_connections_bin'] +
    weights['morale_score_norm'] * df['morale_score_norm'] +
    weights['trust_in_leadership_score_norm'] * df['trust_in_leadership_score_norm'] +
    weights['promotion_fairness_score_norm'] * df['promotion_fairness_score_norm'] +
    weights['communication_quality_score_norm'] * df['communication_quality_score_norm'] +
    weights['defector_capture_rate_norm'] * df['defector_capture_rate_norm'] +
    weights['punishment_policy_num'] * df['punishment_policy_num'] +
    weights['regime_type_num'] * df['regime_type_num'] +
    weights['military_structure_num'] * df['military_structure_num']
)
```

### Threshold Determination

A threshold is set based on the median defection risk score.

```python
threshold = df['defection_risk_score'].median()
```

### Final Classification

Soldiers are classified as 'yes' (will defect) or 'no' (will not defect) based on whether their defection risk score exceeds the threshold.

```python
df['will_defect'] = np.where(df['defection_risk_score'] > threshold, 'yes', 'no')
```

---

## Usage

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/defection-risk-dataset.git
   cd defection-risk-dataset
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.x installed along with `numpy` and `pandas`.

   ```bash
   pip install numpy pandas
   ```

3. **Run the Script:**

   ```bash
   python generate_dataset.py
   ```

4. **Explore the Dataset:**

   The script will generate `soldier_defection_dataset.csv` containing the synthetic data.

---

## Conclusion

This project provides a comprehensive synthetic dataset that models the complex factors influencing military defection. By assigning weights based on domain knowledge and normalizing features, we create a realistic simulation useful for analytical and predictive purposes.

---

## Next Steps

- **Model Training:** Use the dataset to train machine learning models to predict defection.

- **Validation:** Validate the synthetic data and model predictions against real-world data if available.

- **Feature Refinement:** Adjust feature weights and distributions based on additional research or data.

- **Temporal Analysis:** Incorporate the `time_of_measurement` feature to study how defection risk changes over time.

- **Scenario Simulation:** Modify constant features like `defector_capture_rate` and `regime_type` to simulate different scenarios and their impact on defection risk.

---

## Contact

For questions or suggestions, please contact [official.tanmay1306@gmail.com](mailto:official.tanmay1306@gmail.com).

---

**Disclaimer:** This dataset is synthetic and created for educational and research purposes. It does not represent real individuals or events.
