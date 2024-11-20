# Libraries used for the analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import streamlit as st

# Loading the dataset
data_path = r"C:\Users\keval\OneDrive\Desktop\UTA dataset.xlsx"
data = pd.read_excel(data_path)

# Streamlit App Title and Heading
st.title("UTA Student Retention Analysis App")
st.write("### Analyzing Key Factors Affecting Student Retention Using Correlation, Logistic Regression, and Random Forest Models")

# Convert categorical variables to numeric using one-hot encoding
data_encoded = pd.get_dummies(data, drop_first=True)

# Check  and convert any boolean columns to integers
data_encoded = data_encoded.astype(int)

# Handle missing values by dropping rows with NaN if any present, this is for checks and balances
data_encoded = data_encoded.dropna()

# Prepare the independent variables (X) and the target variable (y)
X = data_encoded.drop('OneYearRetention', axis=1)
y = data_encoded['OneYearRetention']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------- Feature Importance Calculation -------------------
# Step 1: Correlation Matrix with Heatmap
st.write("#### Correlation Heatmap for All Variables")
correlation_matrix = data_encoded.corr()

# Plotting the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title('Correlation Heatmap')
st.pyplot(plt)

# Correlation with One Year Retention
st.write("#### Feature Importance Based on Correlation with OneYearRetention")
correlation_with_target = correlation_matrix['OneYearRetention'].abs().drop('OneYearRetention')
st.dataframe(correlation_with_target.sort_values(ascending=False))

# Interpretation for Correlation:
st.write("""
- **Correlation** measures the linear relationship between a feature and the target variable.
- Features with higher absolute correlation values have a stronger linear relationship with retention.
""")

# Step 2: Logistic Regression to Get Coefficients
X_const = sm.add_constant(X)  # Add constant for logistic regression
logit_model = sm.Logit(y, X_const)
logit_result = logit_model.fit()

# Display the logistic regression results
st.write("#### Logistic Regression Model Summary")
st.text(logit_result.summary())

# Extracting the logistic regression coefficients
logit_coefficients = logit_result.params.abs().drop('const')  # Get absolute coefficients
st.write("#### Feature Importance Based on Logistic Regression Coefficients")
st.dataframe(logit_coefficients.sort_values(ascending=False))

# Interpretation for Logistic Regression:
st.write("""
- **Logistic Regression Coefficients** show how much a unit increase in a feature changes the likelihood of retention.
- Positive coefficients increase the chance of retention, and negative ones decrease it.
- Larger absolute coefficients imply more influence on retention.
""")

# Step 3: Random Forest to Get Feature Importance
st.write("#### Feature Importance Based on Random Forest")
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
st.dataframe(rf_importance.sort_values(ascending=False))

# Interpretation for Random Forest:
st.write("""
- **Random Forest Feature Importance** ranks features based on how much they help the model split the data to make accurate predictions.
- It captures complex relationships that other methods might miss.
""")

# Step 4: Normalize the importance scores from all three methods, Min max scaler makes more sense
scaler = MinMaxScaler()
#scaler = StandardScaler()

# Normalize Correlation scores
correlation_norm = pd.Series(scaler.fit_transform(correlation_with_target.values.reshape(-1, 1)).flatten(),
                             index=correlation_with_target.index)

# Normalize Logistic Regression coefficients
logit_norm = pd.Series(scaler.fit_transform(logit_coefficients.values.reshape(-1, 1)).flatten(),
                       index=logit_coefficients.index)

# Normalize Random Forest feature importances
rf_norm = pd.Series(scaler.fit_transform(rf_importance.values.reshape(-1, 1)).flatten(), index=rf_importance.index)

# Combine the actual values and the normalized scores side by side for each method
importance_combined = pd.DataFrame({
    'Correlation (Actual)': correlation_with_target,
    'Correlation (Scaled)': correlation_norm,
    'Logistic Regression (Actual)': logit_coefficients,
    'Logistic Regression (Scaled)': logit_norm,
    'Random Forest (Actual)': rf_importance,
    'Random Forest (Scaled)': rf_norm
})
# Step 5: Combine the normalized importance scores by averaging
final_importance_score = (correlation_norm + logit_norm + rf_norm) / 3

# Step 6: Display combined actual and normalized scores side-by-side
st.write("#### Actual and Scaled Feature Importance (Correlation, Logistic Regression, Random Forest)")
st.dataframe(importance_combined)

# Step 7: Display final importance score
st.write("#### Combined Feature Importance (Correlation, Logistic Regression, Random Forest)")
st.dataframe(final_importance_score.sort_values(ascending=False))

# Interpretation for Combined Feature Importance:
st.write("""
- **Combined Feature Importance** gives a more balanced view by averaging the importance scores from correlation, logistic regression, and random forest.
- Features with higher combined scores are likely to be the most important predictors for retention.
""")

# ------------------- Model Evaluation and Prediction App -------------------
# Evaluate the Random Forest model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Display accuracy and classification report
st.write(f"#### Model Accuracy: {accuracy * 100:.2f}%")
st.write("#### Classification Report")
st.text(classification_rep)

# Save the trained model using pickle for future predictions
with open('student_retention_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Streamlit-based Prediction App
st.write("## Student Retention Prediction App")

# Collect user inputs for the prediction model
admit_year = st.number_input('Admit Year', min_value=1990, max_value=2024, value=2020)
sat_score = st.number_input('SAT Total Score', min_value=400, max_value=1600, value=1000)
gpa = st.number_input('First Term GPA', min_value=0.0, max_value=4.0, value=2.5)
family_income = st.number_input('Total Family Income', min_value=0, value=50000)
hs_gpa = st.number_input('High School GPA', min_value=0.0, max_value=4.0, value=3.0)

# Gender input
gender = st.selectbox('Gender', ['Male', 'Female'])
gender_encoded = 1 if gender == 'Male' else 0

# CapFlag input
cap_flag = st.selectbox('Cap Flag', ['No', 'Yes'])
cap_flag_encoded = 1 if cap_flag == 'Yes' else 0

# Extra-curricular activities input
extra_curr = st.selectbox('Extra-Curricular Activities', ['Yes', 'No'])
extra_curr_encoded = 1 if extra_curr == 'Yes' else 0

# Pell Eligibility input
pell_elig = st.selectbox('Pell Eligibility', ['Yes', 'No'])
pell_elig_encoded = 1 if pell_elig == 'Yes' else 0

# College input (handle one-hot encoding for colleges)
college = st.selectbox('First Term Enrolled College', [
    'College of Business',
    'College of Engineering',
    'College of Liberal Arts',
    'College of Science',
    'Division of Student Success'
])

# Create dummy variables for college (one-hot encoding)
college_business = 1 if college == 'College of Business' else 0
college_engineering = 1 if college == 'College of Engineering' else 0
college_liberal_arts = 1 if college == 'College of Liberal Arts' else 0
college_science = 1 if college == 'College of Science' else 0
college_success = 1 if college == 'Division of Student Success' else 0

# Prediction button
if st.button('Predict Retention'):
    # Load the trained model
    with open('student_retention_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # Collect all inputs into a DataFrame (with matching feature names)
    features = pd.DataFrame({
        'AdmitYear': [admit_year],
        'SatTot02': [sat_score],
        'FirstTermGPA': [gpa],
        'TotalFamilyIncome': [family_income],
        'HsGPA': [hs_gpa],
        'Gender_Male': [gender_encoded],
        'CapFlag_Yes': [cap_flag_encoded],
        'ExtraCurricularActivities_Yes': [extra_curr_encoded],
        'PellEligibilitYes_Yes': [pell_elig_encoded],
        'FirstTermEnrolledCollege_College of Business': [college_business],
        'FirstTermEnrolledCollege_College of Engineering': [college_engineering],
        'FirstTermEnrolledCollege_College of Liberal Arts': [college_liberal_arts],
        'FirstTermEnrolledCollege_College of Science': [college_science],
        'FirstTermEnrolledCollege_Division of Student Success': [college_success]
    })

    # Make prediction and display the result
    prediction = model.predict(features)

    if prediction == 1:
        st.success('The student is likely to be retained.')
    else:
        st.warning('The student is at risk.')



