import streamlit as st
import pyreadstat
import numpy as np
import pandas as pd
import pyreadstat
from zipfile import ZipFile
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap 
import streamlit.components.v1 as components
import holoviews as hv

df, meta = pyreadstat.read_sav("./ATP W63.sav")
column_filter = ['ACCEPTOWN_a_W63',
                 'DOV_KIDS0TO11_W63', 
                 'PARENTTIME_W63',
                 'PARENTJOB_W63',
                 'F_CREGION',
                 'F_SEX',
                 'F_EDUCCAT2',
                 'F_MARITAL',
                 'F_PARTY_FINAL']


df = df[column_filter]

df = df.replace(99, np.nan)
df = df.dropna()


kid_own_map = {1:1,
               2:1,
               3:1,
               4:1,
               5:2,
               6:2,
               7:2}
df['kid_own'] = df['ACCEPTOWN_a_W63'].map(kid_own_map)

X = df.drop(['kid_own', 'ACCEPTOWN_a_W63'], axis=1)
y = df['kid_own'] 

X_train, X_test = train_test_split(X, test_size=.66, train_size=.34, random_state=101)
y_train, y_test = train_test_split(y, test_size=.66, train_size=.34, random_state=101)

rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train, y_train)

st.title("At what age would you let a child have a smartphone?")


kids = st.sidebar.slider(
    'How many kids do you have?',0,10
)

time_map ={i:x+1 for x,i in 
         enumerate(['Not enough time together.',
                    'Too little time together',
                    'The right amount of time.'])}
time = st.sidebar.selectbox(
    'Thinking about the amount of time you spend with your children, in general, do you think you spend...',
    ('Not enough time together.',
     'Too little time together',
     'The right amount of time.')
)
time = time_map[time]

job_map ={i:x+1 for x,i in 
         enumerate(["A very good job",
                    "A good job",
                    "Only a fair job",
                    "A poor job"])}
job = st.sidebar.selectbox('When thinking about your role as a parent, how would you rate the job you do?',
  ( "A very good job",
    "A good job",
    "Only a fair job",
    "A poor job")) 
job = job_map[job]

region_map={i:x+1 for x,i in enumerate(['Northeast','Midwest','South','West'])}
region = st.sidebar.selectbox(
    'Where do you live?',
    ("Northeast","Midwest","South","West")
)
region = region_map[region]

sex = st.sidebar.selectbox(
    'What sex are you?',
    ('Male', 'Female')
)
sex = 1 if sex == 'Male' else 2

educ_map = {i:x+1 for x,i in enumerate(("Less than high school", 
    "High school graduate",
    "Some college, no degree",
    "Associate’s degree", 
    "College graduate/some postgrad",
    "Postgraduate degree"))}

education = st.sidebar.selectbox(
    'What level of education do you have?',
    ("Less than high school", 
    "High school graduate",
    "Some college, no degree",
    "Associate’s degree", 
    "College graduate/some postgrad",
    "Postgraduate degree")
)
education = educ_map[education]

married_map = {i:x+1 for x,i in enumerate(("Married",
    "Living with a partner",
    "Divorced",
    "Separated",
    "Widowed",
    "Never been married"))}

married = st.sidebar.selectbox(
    "Which of these best describes you?",
    ("Married",
    "Living with a partner",
    "Divorced",
    "Separated",
    "Widowed",
    "Never been married")
)
married = married_map[married]
polt_map = {i:x+1 for x,i in enumerate(("Republican",
    "Democrat",
    "Independent",
    "Something else"))}

party = st.sidebar.selectbox(
    "In politics today what would you consider yourself?",
    ("Republican",
    "Democrat",
    "Independent",
    "Something else")
)
party = polt_map[party]

new_df = pd.DataFrame(columns=column_filter)
new_df.drop('ACCEPTOWN_a_W63', axis=1, inplace=True)

new_df.loc[0] = [kids,   
             time,
             job,
             region,
             sex,
             education,
             married,
             party]
responses = new_df.iloc[[0]]
print(responses)
pred_val = rf.predict(responses)[0]
pred_prob = rf.predict_proba(responses)[0][pred_val-1]


pred_map = {1:"allow an under 12 year old to own a phone.",
            2:"NOT allow an under 12 year old to own a phone."}
st.write(f"You are {pred_prob * 100:.2f}% likely to {pred_map[pred_val]}")


explainer = shap.explainers.Tree(rf)
shap_vals_yes = shap.explainers.Tree(rf).shap_values(responses)[pred_val-1]

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


st_shap(shap.force_plot(explainer.expected_value[pred_val-1], shap_vals_yes, responses))

def bar_with_highlight(feature, selection):
    amt = 0 if feature == 'DOV_KIDS0TO11_W63' else 1
    col = 'ACCEPTOWN_a_W63'
    print(df.groupby(
        feature
    ).count())
    bars = hv.Bars(df.groupby(
        feature
    ).count()[
        col
    ])
    sel = bars.iloc[selection - amt]
    return bars * sel

col1, col2, col3, col4 =  st.columns((1, 1, 1, 1))

kids_plot =  bar_with_highlight('DOV_KIDS0TO11_W63', kids)
time_plot = bar_with_highlight('PARENTTIME_W63', time)
job_plot = bar_with_highlight('PARENTJOB_W63', job)
region_plot = bar_with_highlight('F_CREGION', region)
sex_plot = bar_with_highlight('F_SEX', sex)
education_plot = bar_with_highlight('F_EDUCCAT2', education)
married_plot = bar_with_highlight('F_MARITAL', married)
party_plot = bar_with_highlight('F_PARTY_FINAL', party)
accept_plot = bar_with_highlight('kid_own', pred_val)

st.bokeh_chart(hv.render(kids_plot + time_plot + job_plot, backend='bokeh'))
st.bokeh_chart(hv.render(region_plot + sex_plot + education_plot, backend='bokeh'))
st.bokeh_chart(hv.render(married_plot + party_plot + accept_plot, backend='bokeh'))