import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import category_encoders as ce
import xgboost as xgb



data = pd.read_csv('mushrooms.csv')
data.drop('veil-type',axis=1,inplace=True)
def convert(col):
    if col == 'e':
        return 0
    else:
        return 1
X = data.drop('class',axis=1)
y = data['class'].apply(convert)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.30, random_state=101)

cat_cols = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
    'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
    'stalk-surface-below-ring', 'stalk-color-above-ring',
    'stalk-color-below-ring', 'veil-color', 'ring-number', 'ring-type',
    'spore-print-color', 'population', 'habitat']

cat_enc = ce.CatBoostEncoder(cols=cat_cols,random_state=101)
cat_enc.fit(X_train, y_train)
encX_train = X_train.join(cat_enc.transform(X_train).add_suffix('_cb'))
encX_test = X_test.join(cat_enc.transform(X_test).add_suffix('_cb'))
encX_train.drop(cat_cols,axis=1,inplace=True)
encX_test.drop(cat_cols,axis=1,inplace=True)

my_model = xgb.XGBClassifier(max_depth=300, learning_rate=0.05,random_state=101)
my_model.fit(encX_train, y_train, 
        early_stopping_rounds=5, 
        eval_set=[(encX_test, y_test)], 
        verbose=False)




st.write("""
# Mushroom Type Prediction App
This app predicts if the **Mushroom** is poisonous or not! 🍄 
""")

st.write("""
cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s

cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s

cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y

bruises: bruises=t,no=f

odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s

gill-attachment: attached=a,descending=d,free=f,notched=n

gill-spacing: close=c,crowded=w,distant=d

gill-size: broad=b,narrow=n

gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y

stalk-shape: enlarging=e,tapering=t

stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?

stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s

stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s

stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y

veil-type: partial=p,universal=u

veil-color: brown=n,orange=o,white=w,yellow=y

ring-number: none=n,one=o,two=t

ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z

spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y

population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y

habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d  """)

st.sidebar.header('User Input Parameters')

def user_input_features():
    cap_shape = st.sidebar.selectbox('Cap Shape', ('b', 'c', 'x','f','k','s'))
    cap_surface = st.sidebar.selectbox('Cap Surface', ('f','g','y','s'))
    cap_color = st.sidebar.selectbox('Cap Color', ('n','b','c','g','r','p','u','e','w','y'))
    bruises = st.sidebar.selectbox('Bruises', ('t','f'))
    odor = st.sidebar.selectbox('Odor', ('a','l','c','y','f','m','n','p','s'))
    gill_attachment = st.sidebar.selectbox('Gill Attachment', ('a','d','f','n'))
    gill_spacing = st.sidebar.selectbox('Gill Spacing', ('c','w','d'))
    gill_size = st.sidebar.selectbox('Gill Size', ('b','n'))
    gill_color = st.sidebar.selectbox('Gill Color', ('k','n','b','h','g','r','o','p','u','e','w','y'))
    stalk_shape = st.sidebar.selectbox('Stalk Shape', ('e','t'))
    stalk_root = st.sidebar.selectbox('Stalk Root', ('b','c','u','e','z','r','?'))
    stalk_surface_above_ring = st.sidebar.selectbox('Stalk surface above ring', ('f','y','k','s'))
    stalk_surface_below_ring = st.sidebar.selectbox('Stalk surface below ring', ('f','y','k','s'))
    stalk_color_above_ring = st.sidebar.selectbox('Stalk color above ring', ('n','b','c','g','o','p','e','w','y'))
    stalk_color_below_ring = st.sidebar.selectbox('Stalk color below ring', ('n','b','c','g','o','p','e','w','y'))
    veil_color = st.sidebar.selectbox('Veil Color', ('n','o','w','y'))
    ring_number = st.sidebar.selectbox('Ring Number', ('n','o','t'))
    ring_type = st.sidebar.selectbox('Ring Type', ('c','e','f','l','n','p','s','z'))
    spore_print_color = st.sidebar.selectbox('Spore Color', ('k','n','b','h','r','o','u','w','y'))
    population = st.sidebar.selectbox('Population',('a','c','n','s','v','y'))
    habitat = st.sidebar.selectbox('Habitat', ('g','l','m','p','u','w','d'))
    
    data = {'cap-shape': cap_shape,
            'cap-surface': cap_surface,
            'cap-color': cap_color,
            'bruises': bruises,
            'odor': odor,
            'gill-attachment': gill_attachment,
            'gill-spacing': gill_spacing,
            'gill-size': gill_size,
            'gill-color': gill_color,
            'stalk-shape': stalk_shape,
            'stalk-root': stalk_root,
            'stalk-surface-above-ring': stalk_surface_above_ring,
            'stalk-surface-below-ring': stalk_surface_below_ring,
            'stalk-color-above-ring': stalk_color_above_ring,
            'stalk-color-below-ring': stalk_color_below_ring,
            'veil-color': veil_color,
            'ring-number': ring_number,
            'ring-type': ring_type,
            'spore-print-color': spore_print_color,
            'population': population,
            'habitat': habitat}
    global features
    features = pd.DataFrame(data, index=[0])
    nfeatures = features.join(cat_enc.transform(features).add_suffix('_cb'))
    nfeatures.drop(cat_cols,axis=1,inplace=True)
    return nfeatures

df = user_input_features()

st.subheader('User Input parameters')
st.write(features)

st.subheader('Class labels and their corresponding index number')
preddf = pd.DataFrame({'Type':['Edible','Poisonous']},index=[0,1])
st.write(preddf)

prediction = my_model.predict(df)

st.subheader('Prediction')
st.write(preddf['Type'][prediction])
prediction_proba = my_model.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)