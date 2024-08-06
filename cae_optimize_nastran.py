from collections import defaultdict
# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import logging
# import matplotlib.pyplot as plt
# import math
import numpy as np
import os
# import pandas as pd
import re
import sys
import shutil
import time

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler("C:\\work\\github\\q-annealing-d-wave-test\\cae_opti_info.log"),
                        logging.StreamHandler()
                    ])

class OptimizeManager:
    def __init__(self, initial_condition_data_dict=None, mat_data_dict=None):
        if initial_condition_data_dict is None:
            print("Please set initial_condition_data_dict")
            initial_condition_data_dict = {}
        if mat_data_dict is None:
            mat_data_dict = {}
        self._initial_condition_data_dict = initial_condition_data_dict
        self._mat_data_dict = mat_data_dict
    
    def get_from_initial_condition_data_dict(self, key):
        return self._initial_condition_data_dict.get(key, None)
    
    def add_to_mat_data_dict(self, key, value):
        self._mat_data_dict[key] = value

    def get_from_mat_data_dict(self, key):
        return self._mat_data_dict.get(key, None)
    
initial_condition_data = {
    "target_density": 0.5,
    "density_increment": 0.1,
    "density_power": 2.0,
    "initial_youngs_modulus": 2.0e+5,
    "cost_lambda": 5,
    "loop_num": 11,
    "decide_val_threshold": 0.1,
    "start_phase_num": 11,
    "threshold": 0.001,
}

om = OptimizeManager(initial_condition_data)

def calculate_quadrilateral_area(vertices):
    x_coords = [vertex['x'] for vertex in vertices]
    y_coords = [vertex['y'] for vertex in vertices]
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in zip(x_coords, y_coords, x_coords[1:] + x_coords[:1], y_coords[1:] + y_coords[:1])))
    return area

def increment_phase_number(filename):
    match = re.search(r'phase_(\d+)\.dat', filename)
    
    new_filename = ""
    if match:
        original_number = int(match.group(1))
        incremented_number = original_number + 1
        new_filename = filename.replace(f'phase_{original_number}', f'phase_{incremented_number}')
    else:
        print("dat file name is maybe wrong. (increment_phase_number)")
    return new_filename

def get_file_content(file_path):
    encodings = ['utf-8', 'shift_jis', 'iso-8859-1', 'latin1']
    content = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.readlines()
            print(f"File successfully read with encoding: {encoding}")
            break  # 成功した場合、ループを抜ける
        except UnicodeDecodeError:
            print(f"Failed to read file with encoding: {encoding}")
            continue
    else:
        # 全てのエンコーディングで失敗した場合
        raise UnicodeDecodeError("Failed to read the file with provided encodings.")
    return content

def format_field_left(value, length=8):
    return f'{value:<{length}}'

def format_field_right(value, length=8):
    return f'{value:>{length}}'

def format_float(value):
    formatted_value = "{:.4E}".format(value)
    parts = formatted_value.split('E')
    return f"{parts[0]}E{int(parts[1])}"

def process_nastran_file_fixed_length(input_file_path, output_file_path):
    cquad4_cards = []
    pshell_card = None
    mat1_card = None

    content = get_file_content(input_file_path)

    for line in content:
        if line.startswith('CQUAD4'):
            cquad4_cards.append(line.strip())
        elif line.startswith('PSHELL'):
            pshell_card = line.strip()
        elif line.startswith('MAT1'):
            mat1_card = line.strip()

    new_content = []
    card_index = 1

    for cquad4 in cquad4_cards:
        new_pshell = (
            format_field_left('PSHELL', 8) +
            format_field_right(card_index, 8) +
            format_field_right(card_index, 8) +
            pshell_card[24:]
        )
        new_mat1 = (
            format_field_left('MAT1', 8) +
            format_field_right(card_index, 8) +
            mat1_card[16:]
        )
        new_cquad4 = (
            format_field_left('CQUAD4', 8) +
            format_field_right(card_index, 8) +
            format_field_right(card_index, 8) +
            cquad4[24:]
        )

        new_content.append(new_pshell)
        new_content.append(new_mat1)
        new_content.append(new_cquad4)
        card_index += 1

    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in content:
            if not line.startswith(('CQUAD4', 'PSHELL', 'MAT1', 'ENDDATA')):
                file.write(line)
        for line in new_content:
            file.write(line + '\n')
        file.write("ENDDATA\n")

def extract_stress_values(filename):
    lines = get_file_content(filename)

    stress_data = {}
    start_reading = False

    for i in range(len(lines) - 1):
        line = lines[i]
        next_line = lines[i + 1]

        if re.search(r'S T R E S S E S\s+I N\s+Q U A D R I L A T E R A L\s+E L E M E N T S\s+\( Q U A D 4 \)', line):
            start_reading = True
            continue
        
        if re.search(r'MSC NASTRAN.+PAGE.+', line):
            start_reading = False
        
        # if start_reading and re.match(r'^\s*\d+\s+', line):
        if start_reading and re.match(r'^\s*0\d*\s+\d+\s+.*', line):
            element_id = int(line[1:9].strip())
            
            normal_x1 = float(line[26:43].strip())
            normal_y1 = float(line[44:58].strip())
            shear_xy1 = float(line[59:73].strip())

            normal_x2 = float(next_line[26:43].strip())
            normal_y2 = float(next_line[44:58].strip())
            shear_xy2 = float(next_line[59:73].strip())

            normal_x_avg = (normal_x1 + normal_x2) / 2
            normal_y_avg = (normal_y1 + normal_y2) / 2
            shear_xy_avg = (shear_xy1 + shear_xy2) / 2

            stress_data[element_id] = (normal_x_avg, normal_y_avg, shear_xy_avg)

    return stress_data

def check_skip_optimize(youngmodulus, initial_youngmodulus, threshold, phase_num):
    if phase_num == 1:
        return False
    if abs(youngmodulus - initial_youngmodulus) < threshold:
        return True
    return False

def rename_file(original_file_path, new_file_path):
    try:
        os.rename(original_file_path, new_file_path)
        print(f"ファイルが {original_file_path} から {new_file_path} にリネームされました。")
    except FileNotFoundError:
        print(f"ファイル {original_file_path} が見つかりません。")
    except PermissionError:
        print(f"ファイル {original_file_path} に対するアクセスが拒否されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def delete_file(file_path):
    try:
        os.remove(file_path)
        print(f"ファイル {file_path} が削除されました。")
    except FileNotFoundError:
        print(f"ファイル {file_path} が見つかりません。")
    except PermissionError:
        print(f"ファイル {file_path} に対するアクセスが拒否されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

def calculate_density(E_i, E_0, n):
    if E_0 == 0:
        raise ValueError("E_0 must be non-zero.")
    return (E_i / E_0) ** (1 / n)

def main():
    loop_num = om.get_from_initial_condition_data_dict('loop_num')
    start_phase_num = om.get_from_initial_condition_data_dict('start_phase_num') - 1
    for i in range(start_phase_num, loop_num):
        phase_num = i + 1
        result = main2(phase_num)
        if result == -1:
            logging.error(f"{phase_num}/{loop_num}の処理で失敗しました")
        if result == 1:
            logging.info(f"{phase_num}/{loop_num}で、最適化を完了しました")

def main2(phase_num):
    dat_file_path = sys.argv[1]
    f06_file_path = sys.argv[2]
    start_time_1 = time.time()

    target_density = om.get_from_initial_condition_data_dict('target_density')
    density_increment = om.get_from_initial_condition_data_dict('density_increment')
    density_power = om.get_from_initial_condition_data_dict('density_power')
    initial_youngs_modulus = om.get_from_initial_condition_data_dict('initial_youngs_modulus')
    cost_lambda = om.get_from_initial_condition_data_dict('cost_lambda')
    decide_val_threshold = om.get_from_initial_condition_data_dict('decide_val_threshold')
    threshold = om.get_from_initial_condition_data_dict('threshold')

    input_dat_file_name = "{}_phase_{}.dat".format(str(dat_file_path)[:-4], phase_num - 1)
    if phase_num == 1:
        process_nastran_file_fixed_length(dat_file_path, input_dat_file_name)

    logging.info(f"解析に使用した入力ファイル名：{input_dat_file_name}")

    node_dict = {}
    pshell_dict = {}
    mat1_dict = {}
    cquad4_dict = {}
    all_node_id_set = set()
    with open(input_dat_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('$'):
                continue
            if line.startswith('GRID'):
                # フォーマットは8カラムで固定長（8文字ずつ）なので、それに基づいてフィールドを取得
                grid_id = int(line[8:16].strip())
                x = float(line[24:32].strip())
                y = float(line[32:40].strip())
                z = float(line[40:48].strip())
                node_dict[grid_id] = [x, y, z]
                all_node_id_set.add(grid_id)
            if line.startswith('PSHELL'):
                elem_id = int(line[8:16].strip())
                mat_id = int(line[16:24].strip())
                thickness = float(line[24:32].strip())
                pshell_dict[elem_id] = [mat_id, thickness]
            if line.startswith('MAT1'):
                mat_id = int(line[8:16].strip())
                youngmodulus = float(line[16:24].strip())
                if check_skip_optimize(youngmodulus, initial_youngs_modulus, threshold, phase_num):
                    del pshell_dict[mat_id]
                else:
                    poissonratio = float(line[32:40].strip())
                    value = [youngmodulus, poissonratio]
                    mat1_dict[mat_id] = value
            if line.startswith('CQUAD4'):
                elem_id = int(line[8:16].strip())
                if elem_id in pshell_dict:
                    x1 = int(line[24:32].strip())
                    x2 = int(line[32:40].strip())
                    x3 = int(line[40:48].strip())
                    x4 = int(line[48:56].strip())
                    value = [x1, x2, x3, x4]
                    cquad4_dict[elem_id] = value

    input_f06_file_name = "{}_phase_{}.f06".format(str(f06_file_path)[:-4], phase_num - 1)
    if phase_num == 1:
        shutil.copy(f06_file_path, input_f06_file_name)
    stress_dict = extract_stress_values(input_f06_file_name)

    merged_elem_list = []
    for pshell_key, value in pshell_dict.items():
        eid = pshell_key
        merged_dict = {}
        merged_dict['eid'] = eid
        cquad4_value = cquad4_dict[eid]
        merged_dict['nodes'] = cquad4_value
        mat1_value = mat1_dict[value[0]]
        thickness = value[1]
        merged_dict['thickness'] = thickness
        merged_dict['youngsmodulus'] = mat1_value[0]
        merged_dict['poissonratio'] = mat1_value[1]
        stress_value = stress_dict[eid]
        merged_dict['stressxx'] = stress_value[0]
        merged_dict['stressyy'] = stress_value[1]
        merged_dict['stressxy'] = stress_value[2]

        node_data = []
        # points = []
        for nid in cquad4_value:
            node_value = node_dict[nid]
            node_value_dict = {}
            node_value_dict['nid'] = nid
            node_value_dict['x'] = node_value[0]
            node_value_dict['y'] = node_value[1]
            node_value_dict['z'] = node_value[2]
            node_data.append(node_value_dict)
            # points.append((node_value[0], node_value[1]))
        merged_dict['node_data'] = node_data
        area = calculate_quadrilateral_area(node_data)
        merged_dict['area'] = area
        # polygon = Polygon(points)
        # merged_dict['polygon'] = polygon
        merged_dict['volume'] = area * thickness

        merged_elem_list.append(merged_dict)

    # print(merged_elem_list)
    
    elapsed_time_1 = time.time() - start_time_1
    logging.info(f"入力データの読み込みにかかった時間：{str(elapsed_time_1)}")
    # sheet.cell(row=row_start, column=col_start, value="Read Input Data")
    # sheet.cell(row=row_start, column=col_start + 1, value=f"{str(elapsed_time_1)}")
    # row_start += 1
            
    start_time_2 = time.time()

    energy_list_for_scale = []
    volume_list_for_scale = []

    first_density = target_density
    # energy_part_elem_dict = {}
    ising_index_eid_map = {}
    nInternalid = len(merged_elem_list)
    h = defaultdict(int)
    J = defaultdict(int)
    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        stressxx = elem.get('stressxx', 0)
        stressyy = elem.get('stressyy', 0)
        stressxy = elem.get('stressxy', 0)
        poissonratio = float(elem.get('poissonratio', 0))
        youngsmodulus = float(elem.get('youngsmodulus', 0))
        volume = float(elem.get('volume', 0))

        ising_index_eid_map[eid] = index

        density_now = 0
        if phase_num == 1:
            density_now = first_density
        else:
            # eid_row = eid + 20
            # density_now = sheet.cell(row=eid_row, column=col_start - 2).value
            density_now = calculate_density(youngsmodulus, initial_youngs_modulus, density_power)

        density_plus_delta = density_now + density_increment
        density_minus_delta = density_now - density_increment

        alpha_value = pow(density_plus_delta, (1 - density_power))
        beta_value = pow(density_minus_delta, (1 - density_power))

        kappa_i = (pow(stressxx, 2.0) - 2.0 * poissonratio * stressxx * stressyy + pow(stressyy, 2.0) + 2.0 * (1.0 + poissonratio) * pow(stressxy, 2.0)) * volume / initial_youngs_modulus
        
        energy_list_for_scale.append(alpha_value * kappa_i)
        energy_list_for_scale.append(beta_value * kappa_i)

        volume_list_for_scale.append(volume)
        volume_list_for_scale.append(-1.0 * volume)

    std_of_energy_list = np.std(energy_list_for_scale)
    std_of_volume_list = np.std(volume_list_for_scale)

    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        stressxx = elem.get('stressxx', 0)
        stressyy = elem.get('stressyy', 0)
        stressxy = elem.get('stressxy', 0)
        poissonratio = float(elem.get('poissonratio', 0))
        youngsmodulus = float(elem.get('youngsmodulus', 0))
        volume = float(elem.get('volume', 0))

        ising_index_eid_map[eid] = index

        density_now = 0
        if phase_num == 1:
            density_now = first_density
        else:
            # eid_row = eid + 20
            # density_now = sheet.cell(row=eid_row, column=col_start - 2).value
            density_now = calculate_density(youngsmodulus, initial_youngs_modulus, density_power)

        density_plus_delta = density_now + density_increment
        density_minus_delta = density_now - density_increment

        alpha_value = pow(density_plus_delta, (1 - density_power))
        beta_value = pow(density_minus_delta, (1 - density_power))
        k_0 = (alpha_value - beta_value) / 2.0
        kappa_i = (pow(stressxx, 2.0) - 2.0 * poissonratio * stressxx * stressyy + pow(stressyy, 2.0) + 2.0 * (1.0 + poissonratio) * pow(stressxy, 2.0)) * volume / initial_youngs_modulus

        h_first = k_0 * kappa_i / std_of_energy_list / np.sqrt(nInternalid) / 3.0
        h[index] = h_first

        # energy_part_elem_dict[eid] = h[index]

        for j_index in range(index + 1, nInternalid):
            # list_key = list(optimize_elem_dict.keys())
            # volume_j = optimize_elem_dict[list_key[j_index]].get('volume', 0)
            volume_j = merged_elem_list[j_index].get('volume', 0)
            J[(index,j_index)] = 2.0 * cost_lambda * volume * volume_j / pow(std_of_volume_list, 2) / nInternalid / 9.0

    ising_index_dict = {}

    sampler = LeapHybridSampler()
    response = sampler.sample_ising(h, J)

    for sample, E in response.data(fields=['sample','energy']):
        S_minus_1 = [k for k,v in sample.items() if v == -1]
        S_plus_1 = [k for k,v in sample.items() if v == 1]

        for elem in S_minus_1:
            ising_index_dict[elem] = -1

        for elem in S_plus_1:
            ising_index_dict[elem] = 1

        # for key, value in ising_index_dict.items():
        #     print(f"キー：{key}, バリュー：{value}")
        # print(f"イジングモデルの各要素の最適化後の値は: {ising_index_dict} となる")

    # start_time_3 = time.time()
    mat_youngmodulus = {}
    # row_start = 19
    # sheet.cell(row=row_start, column=col_start, value=f"phase_{phase_num}")
    # row_start += 1
    # sheet.cell(row=row_start, column=col_start, value="Element number")
    # sheet.cell(row=row_start, column=col_start + 1, value="Density")
    # sheet.cell(row=row_start, column=col_start + 2, value="Energy part")
    # width = initial_condition_data["width"]
    # height = initial_condition_data["height"]
    # div_x = width * initial_condition_data["divide_num"]
    # div_y = height * initial_condition_data["divide_num"]
    # data = np.full((div_y, div_x), np.nan)

    zero_fix_density_index_list = []

    b_fin_optimize = 0
    if len(merged_elem_list) <= (2800 / 100):
        logging.info(f"最適化が完了していない要素の数が{(2800 / 100)}以下になったため、要素の0/1を決定します")
        b_fin_optimize = 1
    
    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        youngsmodulus = float(elem.get('youngsmodulus', 0))
        # eid = index + 1
        # sheet.cell(row=row_start + index + 1, column=col_start, value=eid)
        dens_value_old = first_density
        if not phase_num == 1:
            dens_value_old = calculate_density(youngsmodulus, initial_youngs_modulus, density_power)
            # dens_value_old = float(sheet.cell(row=row_start + index + 1, column=col_start - 2).value)
        ising_index = ising_index_eid_map[eid]
        ising_value = ising_index_dict[ising_index]
        dens_value = dens_value_old + density_increment * ising_value

        # dens_cell = sheet.cell(row=row_start + index + 1, column=col_start + 1, value=float(dens_value))

        # energy_part = energy_part_elem_dict.get(eid, 0)
        # sheet.cell(row=row_start + index + 1, column=col_start + 2, value=float(energy_part))   
        if b_fin_optimize:
            if dens_value >= (0.5 + threshold):
               dens_value = 1.0
            else:
               dens_value = 0.0
                         
        b_use_youngmodulus = True
        if dens_value >= (1.0 - decide_val_threshold - threshold):
            # fill_red = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
            # dens_cell.fill = fill_red
            dens_value = 1.0
        if dens_value <= (decide_val_threshold + threshold):
            zero_fix_density_index_list.append(int(eid))
            # fill_blue = PatternFill(start_color="CCCCFF", end_color="CCCCFF", fill_type="solid")
            # dens_cell.fill = fill_blue
            b_use_youngmodulus = False

        if b_use_youngmodulus == True:
            mat_youngmodulus[str(eid)] = pow(dens_value, density_power) * initial_youngs_modulus

        # sum_volume += value['volume'] * dens_value

        # range_polygon = next((elem['polygon'] for elem in merged_elem_list if str(elem['eid']) == str(eid)), None)
        # block_width, block_height = width / div_x, height / div_y
        # min_x, min_y, max_x, max_y = range_polygon.bounds
        # start_x = max(int(min_x // block_width), 0)
        # end_x = min(int(np.ceil(max_x / block_width)), div_x)
        # start_y = max(int(min_y // block_height), 0)
        # end_y = min(int(np.ceil(max_y / block_height)), div_y)
        # for i in range(start_x, end_x):
        #     for j in range(start_y, end_y):
        #         block_right_top = Point((i + 1) * block_width, (j + 1) * block_height)
        #         if range_polygon.contains(block_right_top):
        #             data[j, i] = dens_value

    # sheet.cell(row=row_start - 1, column=col_start + 1, value="Sum volume is ->")
    # sheet.cell(row=row_start - 1, column=col_start + 2, value=float(sum_volume))

    # plt.imshow(data, cmap='viridis', origin='lower', extent=[0, width, 0, height], vmin=0, vmax=1.0)
    # plt.colorbar()
    # plt.title("Density distribution of elements")
    # plt.xlabel("x")
    # plt.ylabel("y")
    # temp_image_path = "optimize_cae_density_temp.png"
    # plt.savefig(temp_image_path)
    # plt.close()

    # new_sheet_name = "Image on phase " + str(phase_num)
    # new_sheet = workbook.create_sheet(new_sheet_name)

    # new_sheet['A1'] = '要素の密度分布'
    # img = Image(temp_image_path)
    # new_sheet.add_image(img, 'A3')

    # phase_num += 1
    # sheet.cell(row=10, column=1, value=phase_num)

    # row_start = 15
    # end_time_3 = time.time()
    # elapsed_time_3 = end_time_3 - start_time_3
    # sheet.cell(row=row_start, column=col_start, value="Update excel file")
    # sheet.cell(row=row_start, column=col_start + 1, value=f"{str(elapsed_time_3)}")
    # row_start += 1

    start_time_4 = time.time()

    lines = get_file_content(input_dat_file_name)

    new_dat_file_name = increment_phase_number(input_dat_file_name)
    new_dat_temp_file_name = new_dat_file_name + ".tmp"
    same_pshell_flag = 0
    used_node_id_set = set()
    with open(new_dat_temp_file_name, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.startswith('PSHELL'):
                elem_id = int(line[8:16].strip())
                if elem_id in zero_fix_density_index_list:
                    line = f"${line}"
                    same_pshell_flag = 1
            if line.startswith('MAT1'):
                if same_pshell_flag == 1:
                    same_pshell_flag = 2
                    line = f"${line}"
                else:
                    line_strip = line.strip()
                    mat_id = int(line[8:16].strip())
                    youngmodulus_value = mat_youngmodulus.get(str(mat_id), None)
                    if youngmodulus_value is None:
                        print(f"youngmodulus_value id({mat_id}) has already fixed.")
                    else:
                        youngmodulus_formatted = format_float(youngmodulus_value)
                        line = (
                            format_field_left('MAT1', 8) +
                            format_field_right(mat_id, 8) +
                            format_field_right(youngmodulus_formatted, 8) +
                            line_strip[24:] +
                            '\n'
                        )
            if line.startswith('CQUAD4'):
                if same_pshell_flag == 2:
                    same_pshell_flag = 0
                    line = f"${line}"
                else:
                    x1 = int(line[24:32].strip())
                    x2 = int(line[32:40].strip())
                    x3 = int(line[40:48].strip())
                    x4 = int(line[48:56].strip())
                    used_node_id_set.add(x1)
                    used_node_id_set.add(x2)
                    used_node_id_set.add(x3)
                    used_node_id_set.add(x4)

            file.write(line)

    lines_new = get_file_content(new_dat_temp_file_name)
        
    unused_node_set = all_node_id_set - used_node_id_set
    if unused_node_set:
        with open(new_dat_file_name, 'w', encoding='utf-8') as file:
            for line in lines_new:
                if line.startswith('SPC1'):
                    fix_node = int(line[24:32].strip())
                    if fix_node in unused_node_set:
                        line = f"${line}"
                if line.startswith('GRID'):
                    grid_node = int(line[8:16].strip())
                    if grid_node in unused_node_set:
                        line = f"${line}"
                file.write(line)
        b_erase_temp_file = False
        if b_erase_temp_file:
            delete_file(new_dat_temp_file_name)
    else:
        rename_file(new_dat_temp_file_name, new_dat_file_name)

    logging.info(f"最適化後のdatファイル名：{new_dat_file_name}")

    ### nastran実行

    # end_time_4 = time.time()
    # elapsed_time_4 = end_time_4 - start_time_4
    # sheet.cell(row=row_start, column=col_start, value="Put LISA file and do FEM solver")
    # sheet.cell(row=row_start, column=col_start + 1, value=f"{str(elapsed_time_4)}")

    # workbook.save(sys.argv[3])
    # os.remove(temp_image_path)

    logging.info(f"success optimization on phase {phase_num}")

    return b_fin_optimize

if __name__ == '__main__':
    logging.info("\n\n")
    if len(sys.argv) <= 2:
        logging.info("Usage: python cae_optimize_nastran.py <dat_file_path> <f06_file_path>")
        sys.argv = ["cae_optimize_nastran.py", "C:\\work\\github\\q-annealing-d-wave-test\\test.dat", "C:\\work\\github\\q-annealing-d-wave-test\\test.f06"]
    logging.info("最適化を開始します")
    logging.info(f"引数: {sys.argv}")
    main()
