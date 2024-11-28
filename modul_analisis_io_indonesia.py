"""
Module untuk Analisis Tabel Input-Output di Indonesia
@author: fawdywahyu
"""

import pandas as pd
import numpy as np
from ipfn import ipfn


# Cleaning Matrix
def cleaning_matrix(tabel_input=None):
    # tabel_input = df_io
    # shape_id = '52 sectors'
    
    # tabel_input = pd.read_excel('Tabel Input Ouput 2016.xlsx', sheet_name='PCT 185')
    
    df_to_clean = tabel_input.copy()
    
    shape_mat = df_to_clean.shape
    max_row_mat = shape_mat[0]-9
    max_col_mat = shape_mat[1]-21
    
    # Demand Side
    mat_Z = df_to_clean.iloc[5:max_row_mat, 3:max_col_mat].to_numpy(dtype=float)
    vec_fd = df_to_clean.iloc[5:max_row_mat, 196].to_numpy(dtype=float)
    vec_output = df_to_clean.iloc[5:max_row_mat, 207].to_numpy(dtype=float)
    
    # vec_output = np.sum(mat_Z, axis=1) + vec_fd
    # mat_output = vec_output * np.identity(len(vec_output))
    mat_output = vec_output * np.identity(len(vec_output))
    inv_x = np.linalg.pinv(mat_output)
    
    mat_A = np.matmul(mat_Z, inv_x)
    mat_L = np.identity(len(mat_A)) - mat_A
    mat_L_inv = np.linalg.inv(mat_L)
    
    # Supply Side
    mat_B = np.matmul(inv_x, mat_Z) # allocation coefficient
    mat_G = np.identity(len(mat_B)) - mat_B
    mat_G_inv = np.linalg.inv(mat_G) # the output inverse
    
    vec_wage = df_to_clean.iloc[-5,3:max_col_mat].to_numpy(dtype=float)
    vec_sub = df_to_clean.iloc[-4,3:max_col_mat].to_numpy(dtype=float)
    vec_nt = df_to_clean.iloc[-3,3:max_col_mat].to_numpy(dtype=float)
    vec_va = vec_wage + vec_sub + vec_nt
    # vec_input = df_to_clean.iloc[-1, 3:max_col_mat].to_numpy(dtype=float)
    
    cleaning_result = {
        'Matrix Z': mat_Z,
        'Matrix A': mat_A,
        'Matrix L': mat_L,
        'Matrix Leontief Inverse': mat_L_inv,
        'Matrix B': mat_B,
        'Matrix G': mat_G,
        'Matrix G Inverse': mat_G_inv,
        'Vector Output': vec_output,
        'Vector Value Add': vec_va,
        'Vector Final Demand': vec_fd,
        'Vector Kompensasi Tenaga Kerja': vec_wage
        }
    
    return cleaning_result

def io_analysis(clean_matrix=None, list_industry=None, 
                input_industry_name=None,
                delta_fd_input=None,
                delta_va_input=None,
                first_round_id=None):
    
    # clean_matrix = matrix_clean_papbar
    # list_industry = industry_name_papbar
    # input_industry_name = sektor_unggulan_lq_papbar[2]
    # delta_fd_input = 1
    # delta_va_input = 1
    # first_round_id='Demand'
    # first_round_id='Demand' or 'Supply'
    
    # Multiplier and Linkage Analysis
    # Demand-Side Multiplier
    indeks_bb = list_industry.index(input_industry_name)
    delta_fd = pd.DataFrame([0] * len(clean_matrix['Vector Final Demand']))
    vec_dfd = delta_fd.to_numpy(dtype=int)
    vec_dfd[indeks_bb] = delta_fd_input #disesuaikan dengan nilai impor industri
    new_output = np.matmul(clean_matrix['Matrix Leontief Inverse'], vec_dfd)
    multiplier_demand = np.sum(new_output)
    
    # Supply Side Multiplier
    delta_va = pd.DataFrame([0] * len(clean_matrix['Vector Value Add']))
    vec_dva = delta_va.to_numpy(dtype=int)
    vec_dva[indeks_bb] = delta_va_input #disesuaikan dengan tambahan value added idustri
    new_output_supply = np.matmul(clean_matrix['Matrix G Inverse'], vec_dva)
    multiplier_supply = np.sum(new_output_supply)
    
    # Linkage (Source: Miller and Blair)
    # Backward Linkage
    colsum_A = np.sum(clean_matrix['Matrix A'], axis=0)
    rowsum_A = np.sum(clean_matrix['Matrix A'], axis=1)
    nrow_A = clean_matrix['Matrix A'].shape[0]
    # ncol_A = mat_A.shape[1]
    denominator_BL = np.matmul(colsum_A, rowsum_A)
    numerator_BL = nrow_A * colsum_A
    backward_linkage = numerator_BL/denominator_BL # Normalized backward linkage
    bl_j = backward_linkage[indeks_bb] # Normalized Direct Backward Linkage for [Industry_Name]
    # Angka BL menunjukkan bahwa [Industry_Name] memliki kategori (above average atau below average)
    # atau linkage yang kuat/lemah sebagai pembeli
    
    # Direct Backward Linkage
    direct_bl_j = colsum_A[indeks_bb]
    
    # Indirect Backward Linkage
    colsum_L = np.sum(clean_matrix['Matrix L'], axis=0)
    total_bl_j = colsum_L[indeks_bb]
    indirect_bl_j = total_bl_j - direct_bl_j
    
    
    # Direct Forward Linkage
    colsum_B = np.sum(clean_matrix['Matrix B'], axis=0)
    rowsum_B = np.sum(clean_matrix['Matrix B'], axis=1)
    nrow_B = clean_matrix['Matrix B'].shape[0]
    denominator_FL = np.matmul(colsum_B, rowsum_B)
    numerator_FL = nrow_B * colsum_B
    forward_linkage = numerator_FL/denominator_FL # Normalized forward linkage
    fl_j = forward_linkage[indeks_bb] # Normalized Direct Forward Linkage for [Industry_Name]
    # Angka BL menunjukkan bahwa [Industry_Name] memliki kategori (above average/below average)
    # atau linkage yang kuat/lemah sebagai suplier
    
    # Direct Forward Linkage
    direct_fl_j = rowsum_B[indeks_bb]
    
    # Indirect Forward Linkage
    rowsum_G = np.sum(clean_matrix['Matrix G'], axis=1)
    total_fl_j = rowsum_G[indeks_bb]
    indirect_fl_j = total_fl_j - direct_fl_j
    
    
    # Indeks Daya Penyebaran dan Indeks Derajat Kepekaan
    colsum_L = np.sum(clean_matrix['Matrix Leontief Inverse'], axis=0)
    rowsum_L = np.sum(clean_matrix['Matrix Leontief Inverse'], axis=1)
    sumcolsum_L = np.sum(colsum_L)
    sumrowsum_L = np.sum(rowsum_L)
    jumlah_industri = len(list_industry)
    idk = (jumlah_industri/sumrowsum_L)*rowsum_L # indeks daya penyebaran
    idp = (jumlah_industri/sumcolsum_L)*colsum_L # indeks derajat kepekaan
    idp_j = idp[indeks_bb]
    idk_j = idk[indeks_bb]
    
    # Industry Effect
    list_result_multiplier_output = [np.round(multiplier_demand,2), 
                                     np.round(multiplier_supply,2)]
    initial_effect = 1
    if first_round_id=='Demand':
        sum_indikator = rowsum_A
        indeks_multiplier = 0
    elif first_round_id=='Supply':
        sum_indikator = rowsum_B
        indeks_multiplier = 1
    first_round_effect = sum_indikator[indeks_bb]
    industrial_support_effect = list_result_multiplier_output[indeks_multiplier] - initial_effect - first_round_effect
    production_induced_effect = first_round_effect + industrial_support_effect
    
    
    analysis_result = {
        'Multiplier Output Demand': np.round(multiplier_demand,4),
        'Multiplier Output Supply': np.round(multiplier_supply,4),
        'Normalized Direct Backward Linkage': np.round(bl_j, 4),
        'Indirect Backward Linkage': np.round(indirect_bl_j, 4),
        'Direct Backward Linkage': np.round(direct_bl_j, 4),
        'Total Backward Linkage' : np.round(total_bl_j, 4),
        'Normalized Direct Forward Linkage': np.round(fl_j, 4),
        'Indirect Forward Linkage': np.round(indirect_fl_j, 4),
        'Direct Forward Linkage': np.round(direct_fl_j, 4),
        'Total Forward Linkage' : np.round(total_fl_j, 4),
        'Indeks Daya Penyebaran': np.round(idp_j, 4),
        'Indeks Derajat Kepekaan': np.round(idk_j, 4),
        'First Round Effect': np.round(first_round_effect, 4),
        'Industrial Support Effect': np.round(industrial_support_effect, 4),
        'Production Induced Effect': np.round(production_induced_effect, 4)
        }
    
    df_analisis_io = pd.DataFrame(analysis_result, index=[0]).reset_index()
    df_melt = pd.melt(df_analisis_io, id_vars=['index'], value_vars=df_analisis_io.columns)
    list_rename = ['Nama Industri', 'Jenis Analisis', 'Nilai']
    df_melt.columns = list_rename
    df_melt['Nama Industri'] = input_industry_name

    return df_melt

def struktur_io(mat_Z=None, list_industri = None, lab_input='input', 
                nama_industri=None, top_berapa=10):
    # mat_Z = matrix, hasil dari cleaning process
    # list_industri = list, hasil dari function initial identification
    # lab_input= str, default='input'
    # nama_industri = str, default= 'Besi dan Baja Dasar'
    # top_berapa = int, default=20
    
    # mat_Z = matrix_clean_sulsel['Matrix Z']
    # list_industri = industry_name
    # lab_input='input'
    # nama_industri='Jasa Kesehatan dan Kegiatan Sosial'
    
    indeks_bb = list_industri.index(nama_industri)
    
    if lab_input=='input':
        input_komoditas = pd.DataFrame(mat_Z[:,indeks_bb])
        total_value_input = input_komoditas.sum()[0]
        persentase_input = [i*100/total_value_input for i in mat_Z[:,indeks_bb]]
        df_input_komoditas = pd.concat([input_komoditas, pd.Series(list_industri), 
                                        pd.Series(persentase_input)], axis=1)
        df_input_komoditas.columns = ['Nilai Input', 'Nama Produk', 'Persentase Input']
        df_input_komoditas_sort = df_input_komoditas.sort_values(by=['Nilai Input'],
                                                                 ascending=False)
        top_n_input = df_input_komoditas_sort.iloc[:top_berapa,:]
        
        return top_n_input
    elif lab_input=='output':
        output_komoditas = pd.DataFrame(mat_Z[indeks_bb,:])
        total_value_output = output_komoditas.sum()[0]
        persentase_output = [o*100/total_value_output for o in mat_Z[indeks_bb,:]]
        df_output_komoditas = pd.concat([output_komoditas, pd.Series(list_industri),
                                         pd.Series(persentase_output)], axis=1)
        df_output_komoditas.columns = ['Nilai Output', 'Nama Produk', 'Persentase Output']
        df_output_komoditas_sort = df_output_komoditas.sort_values(by=['Nilai Output'],
                                                                   ascending=False)
        top_n_output = df_output_komoditas_sort.iloc[:top_berapa,:]

        return top_n_output
    else:
        raise ValueError('unrecognize label for argument "input"')

def nama_industri_IO(sumber_file, nama_sheet):
    df = pd.read_excel(sumber_file, sheet_name=nama_sheet)
    df_extracted = df.iloc[3:,2:].reset_index().drop(columns=['index'])
    nama_produk = df_extracted.iloc[2:,0].reset_index().iloc[:185,1]
    nama_produk_array = nama_produk.to_numpy(dtype=str)
    return nama_produk_array

def update_io_table(matrix_z, final_demand, gdp_growth, sector_growth):
    """
    Mengupdate tabel IO menggunakan IPF berdasarkan data terbaru.
    
    Args:
        matrix_z (np.array): Matriks teknologi awal (intermediate demand).
        final_demand (np.array): Permintaan akhir (vector).
        gdp_growth (float): Pertumbuhan GDP secara total.
        sector_growth (np.array): Pertumbuhan sektor individu.
        
    Returns:
        dict: Matriks teknologi baru (A1) dan Leontief model (x1).
    """
    # Step 1: Hitung output total (x1) dengan sektor pertumbuhan
    total_output = matrix_z.sum(axis=1) + final_demand
    x1 = total_output * (1 + sector_growth)  # Perbarui berdasarkan sektor

    # Step 2: Perbarui permintaan akhir (f1) dengan pertumbuhan GDP
    f1 = final_demand * (1 + gdp_growth)

    # Step 3: Hitung konsumsi intermediate (ic1) dan permintaan intermediate (id1)
    ic1 = x1 - f1
    id1 = x1 - x1.sum()

    # Step 4: Terapkan IPF untuk menyeimbangkan intermediate consumption & demand
    aggregates = [ic1, id1]
    dimensions = [[0], [1]]  # Baris (konsumsi) dan kolom (permintaan)

    # Gunakan IPF untuk memperbarui matrix_z
    ipf = ipfn.ipfn(matrix_z, aggregates, dimensions, convergence_rate=1e-6)
    matrix_z_updated = ipf.iteration()

    # Step 5: Hitung matriks teknologi baru (A1)
    A1 = matrix_z_updated / x1[:, None]  # Kolom dimensi sebagai pembagi

    # Step 6: Hitung matriks Leontief untuk tahun ke-1
    identity_matrix = np.identity(A1.shape[0])
    L1 = np.linalg.inv(identity_matrix - A1)
    x1_new = np.matmul(L1, f1)

    # Hasil
    return {
        "Updated Technology Matrix (A1)": A1,
        "Updated Leontief Inverse (L1)": L1,
        "Updated Output (x1)": x1_new,
        "Updated Intermediate Demand (Z)": matrix_z_updated,
        'Updated Final Demand (f1)': f1
    }

def expand_green_industries(matrix_z, final_demand, employment_vector, green_indices, green_growth, value_added_vector, income_vector):
    """
    Ekspansi Tabel IO untuk memasukkan Green Industries dan menghitung multipliers.
    
    Args:
        matrix_z (np.array): Matriks teknologi awal.
        final_demand (np.array): Vektor permintaan akhir.
        employment_vector (np.array): Vektor jumlah pekerja per industri.
        green_indices (list): Indeks industri hijau.
        green_growth (float): Tingkat pertumbuhan sektor hijau.
        value_added_vector (np.array): Vektor value added per industri.
        income_vector (np.array): Vektor income per industri.
        
    Returns:
        dict: Ekspansi green industries pada matriks IO dan multipliers terkait.
    """
    
    # Ekstrak data green industries
    matrix_z_green_col = matrix_z[:, green_indices]
    matrix_z_green_row = matrix_z[green_indices, :]
    final_demand_green = final_demand[green_indices]
    employment_green = employment_vector[green_indices]
    value_added_green = value_added_vector[green_indices]
    income_green = income_vector[green_indices]

    # Perbarui untuk green growth
    final_demand_green_updated = final_demand_green * (1 + green_growth)
    employment_green_updated = employment_green * (1 + green_growth)
    value_added_green_updated = value_added_green * (1 + green_growth)
    income_green_updated = income_green * (1 + green_growth)

    # Gabungkan kembali dengan matrix asli
    matrix_z_expanded_col = np.hstack([matrix_z, matrix_z_green_col])
    additional_columns = np.zeros((matrix_z_green_row.shape[0], len(green_indices)))  # Matriks nol berukuran 3x3
    matrix_z_green_row_expanded = np.hstack([matrix_z_green_row, additional_columns])
    matrix_z_expanded = np.vstack([matrix_z_expanded_col, matrix_z_green_row_expanded])
    final_demand_expanded = np.concatenate([final_demand, final_demand_green_updated])
    employment_expanded = np.concatenate([employment_vector, employment_green_updated])
    value_added_expanded = np.concatenate([value_added_vector, value_added_green_updated])
    income_expanded = np.concatenate([income_vector, income_green_updated])

    # Hitung matriks teknologi baru (A_1)
    total_output_expanded = matrix_z_expanded.sum(axis=1) + final_demand_expanded
    matrix_a_expanded = matrix_z_expanded / total_output_expanded[:, None]

    # Hitung Green Leontief Model (L = (I - A)^-1)
    identity_matrix = np.identity(matrix_a_expanded.shape[0])
    leontief_inverse = np.linalg.inv(identity_matrix - matrix_a_expanded)
    green_output = np.matmul(leontief_inverse, final_demand_expanded)

    # Employment/Output Ratio
    employment_coefficient = employment_expanded / total_output_expanded

    # Value Added/Output Ratio
    value_added_coefficient = value_added_expanded / total_output_expanded

    # Income/Output Ratio
    income_coefficient = income_expanded / total_output_expanded

    # Hitung Employment, Value Added, dan Income Multipliers
    green_employment_multiplier = np.matmul(employment_coefficient, leontief_inverse)
    green_value_added_multiplier = np.matmul(value_added_coefficient, leontief_inverse)
    green_income_multiplier = np.matmul(income_coefficient, leontief_inverse)

    # Labour Productivity Rates
    labour_productivity = total_output_expanded / employment_expanded

    return {
        "Expanded Matrix Z": matrix_z_expanded,
        "Expanded Final Demand": final_demand_expanded,
        "Expanded Employment Vector": employment_expanded,
        "Green Leontief Inverse": leontief_inverse,
        "Green Output": green_output,
        "Employment Coefficient": employment_coefficient,
        "Value Added Coefficient": value_added_coefficient,
        "Income Coefficient": income_coefficient,
        "Green Employment Multiplier": green_employment_multiplier,
        "Green Value Added Multiplier": green_value_added_multiplier,
        "Green Income Multiplier": green_income_multiplier,
        "Labour Productivity": labour_productivity
    }

