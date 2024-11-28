"""
run modul analisis io

@author: fawdywahyu18
"""

import pandas as pd
from modul_analisis_io_indonesia import *

# Load data awal
df_io = pd.read_excel('Tabel Input Ouput 2016.xlsx', sheet_name='BCD 185')
df_io.shape

matrix_clean = cleaning_matrix(df_io)

source_file = 'Tabel Input Ouput 2016.xlsx'

# PCT = Transaksi Total atas dasar harga pembeli
# BCT = Transkasi Total atas dasar harga dasar
# BCD = Transaksi Domestik atas dasar harga dasar

# Analisis IO
sheet = 'PCT 185'
industry_name = list(nama_industri_IO(source_file, sheet))
industri = 'Alat Kedokteran' # Contoh nama Industri


analisis_io = io_analysis(clean_matrix=matrix_clean,
                          list_industry=industry_name,
                          input_industry_name=industri,
                          delta_fd_input=1,
                          delta_va_input=1,
                          first_round_id='Demand')


analisis_io.to_excel('Export Tabel/Analisis IO Alat Kedokteran.xlsx',
                     index=False)

# Update tabel IO
matrix_z = matrix_clean['Matrix Z']
final_demand = matrix_clean['Vector Final Demand']
gdp_growth = 0.05  # Pertumbuhan GDP sebesar 5%
np.random.seed(42)  # Agar hasil acak konsisten
sector_growth = np.random.uniform(0.01, 0.05, 185)  # Pertumbuhan sektor antara 1% hingga 5%

hasil_update = update_io_table(matrix_z, final_demand, gdp_growth, sector_growth)

# Green Input-Output
matrix_z_updated = hasil_update['Updated Intermediate Demand (Z)']
final_demand_updated = hasil_update['Updated Final Demand (f1)']
green_indices = [3, 7, 15]  # Contoh indeks untuk industri hijau
green_growth = 0.1  # Pertumbuhan 10% untuk industri hijau
value_added_vector = matrix_clean['Vector Value Add']
income_vector = matrix_clean['Vector Kompensasi Tenaga Kerja']  # Saya berasumsi income adalah kompensasi tenaga kerja
employment_vector = np.random.randint(50, 500, 185)  # Contoh data pekerja per industri


green_result = expand_green_industries(
    matrix_z_updated, 
    final_demand_updated, 
    employment_vector, 
    green_indices, 
    green_growth, 
    value_added_vector, 
    income_vector
)
