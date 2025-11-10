import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ================================
# 1️⃣ Judul Aplikasi
# ================================
st.title("🎬 Netflix Type Classifier")
st.write("Prediksi apakah tayangan Netflix merupakan **Movie** atau **TV Show** berdasarkan fitur tertentu.")

# ================================
# 2️⃣ Baca Dataset
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df = df.dropna(subset=['type', 'release_year', 'rating', 'duration', 'listed_in', 'country'])
    return df

df = load_data()
st.write("### Contoh Data:")
st.dataframe(df.head())

# ================================
# 3️⃣ Persiapan Data
# ================================
target = 'type'
features = ['release_year', 'rating', 'duration', 'listed_in', 'country']

label_encoders = {}
for col in features + [target]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ================================
# 4️⃣ Latih Model
# ================================
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ================================
# 5️⃣ Tampilkan Evaluasi
# ================================
st.subheader("Evaluasi Model")
st.write("Akurasi Model:", accuracy_score(y_test, y_pred))
st.text("Laporan Klasifikasi:")
st.text(classification_report(y_test, y_pred))

# ================================
# 6️⃣ Input Data Baru
# ================================
st.subheader("🎥 Coba Prediksi Data Baru")

release_year = st.number_input("Tahun Rilis", min_value=1900, max_value=2025, value=2021)
rating = st.text_input("Rating (misal: PG-13, TV-MA, R)", "PG-13")
duration = st.text_input("Durasi (misal: 90 min, 1 Season)", "90 min")
listed_in = st.text_input("Genre (misal: Dramas, Comedies)", "Dramas")
country = st.text_input("Negara Produksi", "United States")

if st.button("Prediksi"):
    contoh_data = {
        'release_year': [release_year],
        'rating': [rating],
        'duration': [duration],
        'listed_in': [listed_in],
        'country': [country]
    }
    contoh_df = pd.DataFrame(contoh_data)

    # transform data baru
    for col in contoh_df.columns:
        if col in label_encoders:
            contoh_df[col] = contoh_df[col].astype(str)
            contoh_df[col] = contoh_df[col].map(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
            contoh_df[col] = label_encoders[col].transform(contoh_df[col])

    hasil_pred = model.predict(contoh_df)
    pred_label = label_encoders[target].inverse_transform(hasil_pred)
    st.success(f"Prediksi: **{pred_label[0]}**")

# ================================
# 7️⃣ Footer
# ================================
st.markdown("---")
st.caption("Dibuat dengan menggunakan Streamlit & Scikit-learn")
