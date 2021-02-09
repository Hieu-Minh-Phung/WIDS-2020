# WIDS_2020
 The WiDS Datathon 2020 focuses on patient health through data from MIT’s GOSSIS (Global Open Source Severity of Illness Score) initiative. Brought to you by the Global WiDS team, the West Big Data Innovation Hub, and the WiDS Datathon Committee.

# Overview
 The challenge is to create a model that uses data from the first 24 hours of intensive care to predict patient survival. MIT's GOSSIS community initiative, with privacy certification from the Harvard Privacy Lab, has provided a dataset of more than 130,000 hospital Intensive Care Unit (ICU) visits from patients, spanning a one-year timeframe. This data is part of a growing global effort and consortium spanning Argentina, Australia, New Zealand, Sri Lanka, Brazil, and more than 200 hospitals in the United States.
Labeled training data are provided for model development; you will then upload your predictions for unlabeled data to Kaggle and these predictions will be used to determine the public leaderboard rankings, as well as the final winners of the competition.
Acknowledgements\
 The WiDS Datathon 2020 is a collaboration led by the Global WiDS team at Stanford, the West Big Data Innovation Hub, and the WiDS Datathon Committee. Special thanks to the MIT GOSSIS Initiative, the University of Toronto, and the Harvard Data Privacy Lab, as well as our growing community of sponsors and supporters.

Team members: <br/>
Linh Nguyen - Github https://github.com/linhnk597 <br/>
Hieu Phung- Github https://github.com/Hieu-Minh-Phung <br/>
Winnie Nguyen- Github: https://github.com/zdnguyen18 <br/>

Technology: <br/>
Tools: Python- Google colab/ Jupyter Notebook\

Analytical Techniques: <br/>
Problem identification: supervised learning – binary classification problem, predicting the probability of 2 output classes, Hospital_death Yes or No
Machine learning modeling & analysis: logistic regression, KNN, Naive Bayes, Decision Tree

# Exploratory Data Analysis
DATASET FEATURE\
TRAIN SET\
![image](https://user-images.githubusercontent.com/59891364/107321452-27b13500-6a68-11eb-8714-1a726445a37e.png)

Target distribution
![image](https://user-images.githubusercontent.com/59891364/107321284-e1f46c80-6a67-11eb-9f87-5fe7ef41b549.png)
  
The patient survival prediction dataset has heavily imbalance target as 91.4% of hospital_death are classified as “Alive” (Coded: 0)
With this situation, even without any model applied and we set all values to “Alive”, we can still get a high accuracy score of more than 91%. Therefore, in next few steps, we will employ OVERSAMPLING method using SMOTE to increase the “Death” class to achieve a 30:70 ration, meaning- for every 3 “deaths”, there will be 7 “Alive”.

TEST SET

![image](https://user-images.githubusercontent.com/59891364/107321507-431c4000-6a68-11eb-9b4f-673dcd82f859.png)
	
Data Cleansing and Transformation: <br/>
Dropping columns<br/>
ID columns: <br/>
Drop 3 ID columns -> 183 columns left<br/>
df.drop(['encounter_id','patient_id','hospital_id'],inplace=True,axis=1)<br/>

Columns with only 1 class: <br/>
As column “read” has only one value ‘0’ in the column, we decided to drop it.<br/>
-> 182 columns left\
df.drop(['readmission_status'], axis=1,inplace=True)<br/>

Columns which has more than 60% of missing data<br/>
df = df.loc[:,df.isnull().mean() < .6]<br/>

Drop columns including:<br/>
Index(['bilirubin_apache', 'fio2_apache', 'paco2_apache', 'paco2_for_ph_apache', 'pao2_apache', 'ph_apache', 'd1_diasbp_invasive_max', 'd1_diasbp_invasive_min', 'd1_mbp_invasive_max', 'd1_mbp_invasive_min', 'd1_sysbp_invasive_max', 'd1_sysbp_invasive_min', 'h1_diasbp_invasive_max', 'h1_diasbp_invasive_min', 'h1_mbp_invasive_max', 'h1_mbp_invasive_min', 'h1_sysbp_invasive_max', 'h1_sysbp_invasive_min', 'd1_inr_max', 'd1_inr_min', 'd1_lactate_max', 'd1_lactate_min', 'h1_albumin_max', 'h1_albumin_min', 'h1_bilirubin_max', 'h1_bilirubin_min', 'h1_bun_max', 'h1_bun_min', 'h1_calcium_max', 'h1_calcium_min', 'h1_creatinine_max', 'h1_creatinine_min', 'h1_hco3_max', 'h1_hco3_min', 'h1_hemaglobin_max', 'h1_hemaglobin_min', 'h1_hematocrit_max', 'h1_hematocrit_min', 'h1_inr_max', 'h1_inr_min', 'h1_lactate_max', 'h1_lactate_min', 'h1_platelets_max', 'h1_platelets_min', 'h1_potassium_max', 'h1_potassium_min', 'h1_sodium_max', 'h1_sodium_min', 'h1_wbc_max', 'h1_wbc_min', 'd1_arterial_pco2_max',
       'd1_arterial_pco2_min', 'd1_arterial_ph_max', 'd1_arterial_ph_min', 'd1_arterial_po2_max', 'd1_arterial_po2_min', 'd1_pao2fio2ratio_max', 'd1_pao2fio2ratio_min', 'h1_arterial_pco2_max', 'h1_arterial_pco2_min', 'h1_arterial_ph_max', 'h1_arterial_ph_min', 'h1_arterial_po2_max', 'h1_arterial_po2_min', 'h1_pao2fio2ratio_max', 'h1_pao2fio2ratio_min'],
      dtype='object')

There are 66 columns which has more than 60% of missing data -> 116 columns left<br/>
	
Categorical Feature Manipulation<br/>
-Fill the missing data with the most popular class<br/>
-Using Label encoder to change the categorical data -> numeric<br/>

Numerical Feature Manipulation<br/>
Fill the missing data with the median<br/>

Data Visualization<br/>
Analyze categorical variables<br/>
![image](https://user-images.githubusercontent.com/59891364/107322593-5d571d80-6a6a-11eb-9396-ca496a5e53c0.png)

Correlataion matrix for numerical (heatmap style)
![image](https://user-images.githubusercontent.com/59891364/107322666-811a6380-6a6a-11eb-8006-3da4ff68227b.png)

hospital_death correlation matrix for top 10 numericals
![image](https://user-images.githubusercontent.com/59891364/107322699-91cad980-6a6a-11eb-8d27-a5525521e466.png)

Apache_4a_hospital_death_prob, apache_4a_icu_death_prob,ventilated_apache have the highest correlation with the target.

# Modelling and Predicting
Keras Functional API was used to build the model.Details of model architectures are described below\
![image](https://user-images.githubusercontent.com/59891364/107322950-09006d80-6a6b-11eb-8864-e6072adcdf91.png)

This architecture consists of an input layer that has 119 nodes, then 3 dense layers of 250 nodes. Relu is chosen as the activation function for the first. Sigmoid function is chosen for the output layer. The optimizer and loss function are “ADAS” - Adaptive Step Size, a new optimization function that overpowers traditional Adam x9 times. To train the
model(s), dataset is divided into 2 subsets: training(70%) and validation (30%) after using SMOTE and Standard Scaler to solve imbalance issue.\

This is the result of model training:<br/>
![image](https://user-images.githubusercontent.com/59891364/107323356-bc696200-6a6b-11eb-9e75-900df567a993.png)

ROC CURVE\
![image](https://user-images.githubusercontent.com/59891364/107323473-f175b480-6a6b-11eb-9922-6203ca4d256a.png)

CONFUSION MATRIX\
![image](https://user-images.githubusercontent.com/59891364/107323510-018d9400-6a6c-11eb-9ad5-0a4171f22a9f.png)

Alas, our final result on unlabeled TEST SET\
![image](https://user-images.githubusercontent.com/59891364/107323626-3699e680-6a6c-11eb-9db1-90ea4855015f.png)





