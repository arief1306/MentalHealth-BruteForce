import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import time

# Memuat dataset dari file CSV
url = 'https://storage.googleapis.com/kagglesdsdata/datasets/2168167/5012788/Student%20Mental%20health.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240613%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240613T152639Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=8c55b3013166a0df27bc1dfc8c7fedc399a13c28b6d026758761827a1376e3a3d281b1b39f266ce765e950401ecaa6cb978f09497693388674a76218bebacbc533b8b82652ce5e590cf0853c2a1c90d31ebeb8efa01a5e48fc6312d0dbdde518ca781a786668584675d5fec707b5530fb2f7387b839a99276521e0c6c917162e292aff15624a9e47c21f417a99a604cc77f095316631c86191ed43f509683165290500fb12dfe5585058f2416d358923ab00d40a2f61180180b33007500146f5671987abdeed4ba9a967e16624dd6494b57e91edf6178ee68ae984d4e10a67722c129382b15e8169f7d86ae2e49fb48ba99a6be19725231b52453fe2e42f6b4c'

start_time = time.time()
df = pd.read_csv(url)
print(f"Time taken to load dataset: {time.time() - start_time:.2f} seconds")

# Eksplorasi distribusi CGPA
plt.figure(figsize=(10, 6))
sns.histplot(df['What is your CGPA?'].dropna(), kde=True)
plt.title('Distribution of CGPA')
plt.xlabel('CGPA')
plt.ylabel('Frequency')
plt.show()

# Konversi CGPA ke nilai numerik
df['CGPA_numeric'] = df['What is your CGPA?'].apply(lambda x: float(x.split(' ')[0]))

# Korelasi antara CGPA dan variabel independen
mental_health_cols = ['Do you have Depression?', 'Do you have Anxiety?', 'Do you have Panic attack?']

# Mengkonversi variabel independen ke bentuk numerik
mental_health_numeric = df[mental_health_cols].applymap(lambda x: 1 if x == 'Yes' else 0)

# Menggabungkan CGPA numerik dengan variabel independen yang sudah diubah ke bentuk numerik
X = sm.add_constant(mental_health_numeric)
y = df['CGPA_numeric']

# Inisialisasi variabel untuk menyimpan hasil analisis
selected_features = []
max_corr_values = []

# Melakukan greedy selection untuk memilih fitur berdasarkan korelasi tertinggi dengan CGPA/IPK
for i in range(len(mental_health_numeric.columns)):
    max_corr = 0
    best_feature = None
    for feature in mental_health_numeric.columns:
        if feature not in selected_features:
            X_subset = X[['const'] + selected_features + [feature]]
            corr = X_subset.corrwith(y).abs()[feature]
            if corr > max_corr:
                max_corr = corr
                best_feature = feature
    selected_features.append(best_feature)
    max_corr_values.append(max_corr)

# Menyimpan hasil analisis untuk output
max_corr_feature = selected_features[0]  # Misalnya, kita mengambil fitur pertama dari seleksi greedy
max_corr_value = max_corr_values[0]

# Cetak hasil analisis
print("\n=== Kesimpulan ===")
print(f"Faktor yang paling terpengaruh oleh CGPA adalah: {max_corr_feature}, dengan korelasi sebesar {max_corr_value:.2f}")
print(f"Time taken for Linear Regression with Greedy Method: {time.time() - start_time:.2f} seconds")
