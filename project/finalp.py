import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import xgboost as xgb

from scipy.stats import chi2_contingency, f_oneway
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from itertools import product

df1 = pd.read_excel(r"C:\Users\Dell\Desktop\project\case_study1.xlsx")
df2 = pd.read_excel(r"C:\Users\Dell\Desktop\project\case_study2.xlsx")

#Remove null values
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]

columns_to_be_removed = []

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed .append(i)



df2 = df2.drop(columns_to_be_removed, axis =1)

df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )

# check how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
   


# Chi-square test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)


# Since all the categorical features have pval <=0.05, we will accept all




# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)



# VIF sequentially check

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0



for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)
    
    
    if vif_value <= 6:
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)

   





# check Anova for columns_to_be_kept 

from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)




# feature selection is done for cat and num features




# listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]






# Label encoding for the categorical features
#['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']



df['MARITALSTATUS'].unique()    
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()



# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3


# Others has to be verified by the business end user 




df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3




df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()






def build_and_evaluate_models(df, features):
    X = df[features]
    y = df['Approved_Flag']

    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    numeric_feat = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_feat = [col for col in features if col not in numeric_feat]

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_feat),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_feat)
    ])

    models = {
        'RandomForest': RandomForestClassifier(n_estimators=200, random_state=42),
        'DecisionTree': DecisionTreeClassifier(max_depth=20, min_samples_split=10),
        'XGBoost': xgb.XGBClassifier(objective='multi:softmax', num_class=4, use_label_encoder=False, eval_metric='mlogloss')
    }

    results = {}
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
        results[name] = {
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    return results, label_enc, X_train, y_train, X_test, y_test


def hyperparameter_tune_xgboost(X_train, y_train, X_test, y_test):
    le_y = LabelEncoder()
    y_train_encoded = le_y.fit_transform(y_train)
    y_test_encoded = le_y.transform(y_test)

    # One-hot encode categorical variables
    X_train_proc = pd.get_dummies(X_train.copy(), drop_first=True)
    X_test_proc = pd.get_dummies(X_test.copy(), drop_first=True)

    # Ensure same columns in train and test after encoding
    X_train_proc, X_test_proc = X_train_proc.align(X_test_proc, join='left', axis=1, fill_value=0)

    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }

    best_model = None
    best_params = None
    best_test_acc = 0
    best_train_acc = 0

    for idx, (n_estimators, max_depth, learning_rate) in enumerate(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['learning_rate']
    ), start=1):
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(np.unique(y_train_encoded)),
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate
        )

        model.fit(X_train_proc, y_train_encoded)

        y_train_pred = model.predict(X_train_proc)
        y_test_pred = model.predict(X_test_proc)

        train_acc = accuracy_score(y_train_encoded, y_train_pred)
        test_acc = accuracy_score(y_test_encoded, y_test_pred)

        print(f"Combo {idx}: Estimators={n_estimators}, Depth={max_depth}, LR={learning_rate}")
        print(f"     Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Update best model based on test accuracy, then train accuracy
        if (test_acc > best_test_acc) or (test_acc == best_test_acc and train_acc > best_train_acc):
            best_model = model
            best_params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate
            }
            best_test_acc = test_acc
            best_train_acc = train_acc

    print(f"\nBest Parameters: {best_params}")
    print(f"Best Train Accuracy: {best_train_acc:.4f}")
    print(f"Best Test Accuracy: {best_test_acc:.4f}")

    return best_params, best_model



# Data load example (replace with actual file paths)


results, label_enc, X_train, y_train, X_test, y_test = build_and_evaluate_models(df,features)
best_params, best_xgb_model = hyperparameter_tune_xgboost(X_train, y_train,X_test,y_test)

print(results)
print("Best XGBoost Params:", best_params)

# ---------------- SHAP EXPLAINABILITY SECTION ------------------



