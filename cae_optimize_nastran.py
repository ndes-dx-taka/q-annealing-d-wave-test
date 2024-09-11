# flag_data_listの説明：[b_do_optimize, b_is_set_sys_argv_on_program, b_erase_temp_file, b_do_nastran, b_for_debug]
# 開発ネットワークでの連続実行では下記。
flag_data_list = [True, False, True, True, True]
# nastranがない場合のデバッグ時は下記
# flag_data_list = [True, True, True, False, True]

import logging
# LOG_LEVELS = {
#     1: logging.DEBUG,      # 1: デバッグレベル
#     2: logging.INFO,       # 2: 情報レベル
#     3: logging.WARNING,    # 3: 警告レベル
#     4: logging.ERROR,      # 4: エラーレベル
#     5: logging.CRITICAL,   # 5: クリティカルレベル
#     6: logging.NOTSET      # 6: ログ出力なし (NOTSET は全てのメッセージを無視)
# }
log_level = logging.DEBUG

from collections import defaultdict
import csv
import copy
# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import numpy as np
import os
import pprint
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMParameters, ADMMOptimizer
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import GroverOptimizer
from qiskit.primitives import Sampler, StatevectorSampler
from qiskit_aer import Aer, AerSimulator
from qiskit_optimization.algorithms import CplexOptimizer, ADMMOptimizer, ScipyMilpOptimizer, SlsqpOptimizer
import random
import re
import sys
import shutil
import subprocess
import time

def setup_logging(logfilepath):
    logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(logfilepath)
                        # logging.StreamHandler()
                    ])

def format_float_youngmodulus(value):
    formatted_value = "{:.4E}".format(value)
    parts = formatted_value.split('E')
    return f"{parts[0]}E{int(parts[1])}"

def format_float_thickness(value):
    return "{:.2E}".format(value)

class OptimizeManager:
    def __init__(self, flag_data_list=None, thickness_youngsmodulus_data_dict=None):
        if flag_data_list is None:
            logging.warning("Please set initial_condition_data_dict")
            flag_data_list = []
        if thickness_youngsmodulus_data_dict is None:
            thickness_youngsmodulus_data_dict = {}
        self._flag_data_list = flag_data_list
        self._thickness_youngsmodulus_data_dict = thickness_youngsmodulus_data_dict
        self._is_solid = False
        self._cost_lambda = -1.0
        self._mat_thickness_youngmodulus_remain = {}
        self._sum_target_volume = -1.0
    
    def get_from_flag_data_list(self, index):
        return self._flag_data_list[index]
    
    def add_to_thickness_youngsmodulus_data_dict(self, key, value):
        self._thickness_youngsmodulus_data_dict[key] = value

    def get_from_thickness_youngsmodulus_data_dict(self, key):
        return self._thickness_youngsmodulus_data_dict.get(key, None)
    
    def write_thickness_youngsmodulus_data_to_csv(self, csvpath):
        with open(csvpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            solid_flag = ("1" if self._is_solid else "0")
            writer.writerow(["Solid_flag", solid_flag])

            for key, value in self._thickness_youngsmodulus_data_dict.items():
                b_thickness = False if str(sys.argv[15]) == "0" else True
                value_format = format_float_thickness(value) if b_thickness else format_float_youngmodulus(value)
                writer.writerow([key, str(value_format)])

    def load_thickness_youngsmodulus_data_from_csv(self, csvpath):
        if len(self._thickness_youngsmodulus_data_dict) == 0:
            with open(csvpath, 'r') as csvfile:
                reader = csv.reader(csvfile)

                for row in reader:
                    if len(row) == 2:
                        key, value = row
                        if str(key) == "Solid_flag":
                            if value == "1":
                                self._is_solid = True
                        elif str(key) == "First_sum_volume":
                            self._sum_target_volume = float(value)
                        else:
                            self._thickness_youngsmodulus_data_dict[int(key)] = float(value)
            logging.info(f"Data loaded from {csvpath} into _thickness_youngsmodulus_data_dict.")
        else:
            logging.info("_thickness_youngsmodulus_data_dict is not empty, skipping load.")

    def update_thickness_youngsmodulus_csv(self, csvpath, rownumber, new_row):
        with open(csvpath, 'r', newline='') as file:
            reader = list(csv.reader(file))
        reader.insert(rownumber, new_row)
        with open(csvpath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(reader)
        self._sum_target_volume = new_row[1]

    def is_solid(self):
        return self._is_solid
    
    def set_to_solid(self):
        self._is_solid = True

    def set_cost_lambda(self, lambda_val):
        self._cost_lambda = lambda_val

    def get_cost_lambda(self):
        return self._cost_lambda

om = OptimizeManager(flag_data_list)

class CustomError(Exception):
    pass

def rename_file(original_file_path, new_file_path):
    try:
        os.rename(original_file_path, new_file_path)
        logging.info(f"ファイルが {original_file_path} から {new_file_path} にリネームされました。")
    except FileNotFoundError:
        logging.warning(f"ファイル {original_file_path} が見つかりません。")
    except PermissionError:
        logging.warning(f"ファイル {original_file_path} に対するアクセスが拒否されました。")
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")

def delete_file(file_path):
    try:
        os.remove(file_path)
        logging.info(f"ファイル {file_path} が削除されました。")
    except FileNotFoundError:
        logging.warning(f"ファイル {file_path} が見つかりません。")
    except PermissionError:
        logging.warning(f"ファイル {file_path} に対するアクセスが拒否されました。")
    except Exception as e:
        logging.error(f"エラーが発生しました: {e}")

def calculate_area(vertices):
    if len(vertices) == 3:
        area = calculate_triangular_area(vertices)
    if len(vertices) == 4:
        area = calculate_quadrilateral_area(vertices)
    return area

def calculate_triangular_area(vertices):
    x1, y1 = vertices[0]['x'], vertices[0]['y']
    x2, y2 = vertices[1]['x'], vertices[1]['y']
    x3, y3 = vertices[2]['x'], vertices[2]['y']
    a = x2 - x1
    b = y2 - y1
    c = x3 - x1
    d = y3 - y1
    area = 0.5 * abs(a * d - b * c)
    return area

def calculate_quadrilateral_area(vertices):
    x_coords = [vertex['x'] for vertex in vertices]
    y_coords = [vertex['y'] for vertex in vertices]
    area = 0.5 * abs(sum(x1 * y2 - x2 * y1 for x1, y1, x2, y2 in zip(x_coords, y_coords, x_coords[1:] + x_coords[:1], y_coords[1:] + y_coords[:1])))
    return area

def calc_tetrahedron_volume(vertices):
    x1, y1, z1 = vertices[0]['x'], vertices[0]['y'], vertices[0]['z']
    x2, y2, z2 = vertices[1]['x'], vertices[1]['y'], vertices[1]['z']
    x3, y3, z3 = vertices[2]['x'], vertices[2]['y'], vertices[2]['z']
    x4, y4, z4 = vertices[3]['x'], vertices[3]['y'], vertices[3]['z']
    
    a = np.array([x2 - x1, y2 - y1, z2 - z1])
    b = np.array([x3 - x1, y3 - y1, z3 - z1])
    c = np.array([x4 - x1, y4 - y1, z4 - z1])
    
    M = np.array([a, b, c])
    
    det_M = np.linalg.det(M)
    
    volume = abs(det_M) / 6.0
    
    return volume

def create_D_inverse(poissonratio, youngmodulus):
    nu = poissonratio
    D_inv = np.array([
        [1,  -nu, -nu,  0,            0,            0],
        [-nu,  1, -nu,  0,            0,            0],
        [-nu, -nu,  1,  0,            0,            0],
        [0,    0,   0,  2*(1+nu),     0,            0],
        [0,    0,   0,  0,        2*(1+nu),         0],
        [0,    0,   0,  0,            0,        2*(1+nu)]
    ])

    D_inv = D_inv / youngmodulus
    
    return D_inv

def increment_phase_number(filename):
    match = re.search(r'phase_(\d+)\.dat', filename)
    
    new_filename = ""
    if match:
        original_number = int(match.group(1))
        incremented_number = original_number + 1
        new_filename = filename.replace(f'phase_{original_number}', f'phase_{incremented_number}')
    else:
        logging.error("datファイルの名前が間違っている可能性があります. (関数：increment_phase_number)")
    return new_filename

def get_file_content(file_path):
    encodings = ['utf-8', 'shift_jis', 'iso-8859-1', 'latin1']
    content = None

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.readlines()
            logging.debug(f"File successfully read with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            logging.debug(f"Failed to read file with encoding: {encoding}")
            continue
    else:
        # 全てのエンコーディングで失敗した場合
        raise UnicodeDecodeError("Failed to read the file with provided encodings.")
    return content

def format_field_left(value, length=8):
    return f'{value:<{length}}'

def format_field_right(value, length=8):
    return f'{value:>{length}}'

def safe_int_conv(sub_value):
    if sub_value:
        return int(sub_value)
    else:
        return ""

def safe_float_conv(sub_value):
    if sub_value:
        pattern = r'(-?\d+\.\d+)-(\d+)'
        if re.search(pattern, sub_value):
            sub_value_conv = re.sub(pattern, r'\1E-\2', sub_value)
            return float(sub_value_conv)
        else:
            return float(sub_value)
    else:
        return ""
    
def group_elements_for_csv(elements):
    elements = sorted(elements)
    grouped_elements = []
    start = elements[0]
    end = elements[0]

    for i in range(1, len(elements)):
        if elements[i] == end + 1:
            end = elements[i]
        else:
            if start == end:
                grouped_elements.append(f"{start}")
            else:
                grouped_elements.append(f"{start}-{end}")
            start = elements[i]
            end = elements[i]
    
    if start == end:
        grouped_elements.append(f"{start}")
    else:
        grouped_elements.append(f"{start}-{end}")

    return " ".join(grouped_elements)

def rename_old_filename(original_file_name):
    if not os.path.exists(original_file_name):
        return

    base_name, extension = os.path.splitext(original_file_name)
    index = 1
    old_filename = original_file_name

    while os.path.exists(old_filename):
        old_filename = f"{base_name}_old_{index}{extension}"
        index += 1

    rename_file(original_file_name, old_filename)
    return

def create_first_phase_zero_nastran_file(input_file_path, output_file_path):
    ctetra_cquad4_ctria3_cards_dict = {}
    psolid_pshell_card_dict = {}
    mat1_card_dict = {}
    ctetra_cquad4_ctria3_psolid_pshell_id_dict = {}
    psolid_pshell_mat1_id_dict = {}
    elem_type_dict = {}  # CQUAD4: 0,  CTRIA3: 1,  CTETRA: 2

    base_name, ext = os.path.splitext(input_file_path)

    content = get_file_content(input_file_path)

    for line in content:
        if line.startswith('CTETRA') or line.startswith('CQUAD4') or line.startswith('CTRIA3'):
            ctetra_cquad4_ctria3_id = int(line[8:16].strip())
            psolid_pshell_id = int(line[16:24].strip())
            ctetra_cquad4_ctria3_cards_dict[ctetra_cquad4_ctria3_id] = line.strip()
            ctetra_cquad4_ctria3_psolid_pshell_id_dict[ctetra_cquad4_ctria3_id] = psolid_pshell_id
            if line.startswith('CQUAD4'):
                elem_type_dict[ctetra_cquad4_ctria3_id] = int(0)
            if line.startswith('CTRIA3'):
                elem_type_dict[ctetra_cquad4_ctria3_id] = int(1)
            if line.startswith('CTETRA'):
                elem_type_dict[ctetra_cquad4_ctria3_id] = int(2)
        elif line.startswith('PSOLID') or line.startswith('PSHELL'):
            psolid_pshell_id = int(line[8:16].strip())
            mat_id = int(line[16:24].strip())
            psolid_pshell_mat1_id_dict[psolid_pshell_id] = mat_id
            psolid_pshell_card_dict[psolid_pshell_id] = line.strip()
            if line.startswith('PSOLID'):
                om.set_to_solid()
        elif line.startswith('MAT1'):
            mat_id = int(line[8:16].strip())
            mat1_card_dict[mat_id] = line.strip()

    new_content = []

    for ctetra_cquad4_ctria3_id, ctetra_cquad4_ctria3_card in ctetra_cquad4_ctria3_cards_dict.items():
        elem_type_id = elem_type_dict[ctetra_cquad4_ctria3_id]
        psolid_pshell_id = ctetra_cquad4_ctria3_psolid_pshell_id_dict[ctetra_cquad4_ctria3_id]
        mat_id = psolid_pshell_mat1_id_dict[psolid_pshell_id]
        psolid_pshell_card = psolid_pshell_card_dict[psolid_pshell_id]
        mat1_card = mat1_card_dict[mat_id]
        new_elem = None
        thickness = None
        if elem_type_id == 0 or elem_type_id == 1:
            thickness = safe_float_conv(psolid_pshell_card[24:32].strip())
            bending_rigidity_str = psolid_pshell_card[40:48].strip()
            bending_rigidity = "" if bending_rigidity_str == "" else safe_float_conv(bending_rigidity_str)
            new_elem = (
                format_field_left('PSHELL', 8) +
                format_field_right(ctetra_cquad4_ctria3_id, 8) +
                format_field_right(ctetra_cquad4_ctria3_id, 8) +
                format_field_right(thickness, 8) +
                format_field_right(ctetra_cquad4_ctria3_id, 8) +
                format_field_right(bending_rigidity, 8) +
                format_field_right(ctetra_cquad4_ctria3_id, 8) +
                psolid_pshell_card[56:]
            )
        if elem_type_id == 2:
            new_elem = (
                format_field_left('PSOLID', 8) +
                format_field_right(ctetra_cquad4_ctria3_id, 8) +
                format_field_right(ctetra_cquad4_ctria3_id, 8) +
                psolid_pshell_card[24:]
            )
        new_mat1 = (
            format_field_left('MAT1', 8) +
            format_field_right(ctetra_cquad4_ctria3_id, 8) +
            mat1_card[16:]
        )
        elem_type_str = None
        if elem_type_id == 0:
            elem_type_str = 'CQUAD4'
        if elem_type_id == 1:
            elem_type_str = 'CTRIA3'
        if elem_type_id == 2:
            elem_type_str = 'CTETRA'
        new_ctetra_cquad4_ctria3 = (
            format_field_left(elem_type_str, 8) +
            format_field_right(ctetra_cquad4_ctria3_id, 8) +
            format_field_right(ctetra_cquad4_ctria3_id, 8) +
            ctetra_cquad4_ctria3_card[24:]
        )

        new_content.append(new_elem)
        new_content.append(new_mat1)
        new_content.append(new_ctetra_cquad4_ctria3)

        youngsmodulus = safe_float_conv(mat1_card[16:24].strip())
        b_use_thickness = False if str(sys.argv[15]) == "0" else True
        om.add_to_thickness_youngsmodulus_data_dict(ctetra_cquad4_ctria3_id, (thickness if b_use_thickness else youngsmodulus))

    reserve_data_csv = base_name + '_reserve_data_for_single_opt.csv'
    rename_old_filename(reserve_data_csv)
    om.write_thickness_youngsmodulus_data_to_csv(reserve_data_csv)

    rename_old_filename(output_file_path)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in content:
            if not line.startswith(('CTETRA', 'CQUAD4', 'CTRIA3', 'PSOLID', 'PSHELL', 'MAT1', 'ENDDATA')):
                file.write(line)
        for line in new_content:
            file.write(line + '\n')
        file.write("ENDDATA\n")

    # マテリアル重複対応
    psolid_pshell_ctetra_cquad4_ctria3_id_dict = {}
    mat1_ctetra_cquad4_ctria3_id_dict = {}
    for ctetra_cquad4_ctria3_id, psolid_pshell_id in ctetra_cquad4_ctria3_psolid_pshell_id_dict.items():
        if psolid_pshell_id in psolid_pshell_ctetra_cquad4_ctria3_id_dict:
            psolid_pshell_ctetra_cquad4_ctria3_id_dict[psolid_pshell_id].append(ctetra_cquad4_ctria3_id)
            mat_id = psolid_pshell_mat1_id_dict[psolid_pshell_id]
            if mat_id in mat1_ctetra_cquad4_ctria3_id_dict:
                mat1_ctetra_cquad4_ctria3_id_dict[mat_id].append(ctetra_cquad4_ctria3_id)
            else:
                new_list = []
                new_list.append(ctetra_cquad4_ctria3_id)
                mat1_ctetra_cquad4_ctria3_id_dict[psolid_pshell_id] = new_list
        else:
            new_list = []
            new_list.append(ctetra_cquad4_ctria3_id)
            psolid_pshell_ctetra_cquad4_ctria3_id_dict[psolid_pshell_id] = new_list
            mat_id = psolid_pshell_mat1_id_dict[psolid_pshell_id]
            if mat_id in mat1_ctetra_cquad4_ctria3_id_dict:
                mat1_ctetra_cquad4_ctria3_id_dict[mat_id].append(ctetra_cquad4_ctria3_id)
            else:
                new_list = []
                new_list.append(ctetra_cquad4_ctria3_id)
                mat1_ctetra_cquad4_ctria3_id_dict[psolid_pshell_id] = new_list

    output_csv = base_name + '.csv'
    rename_old_filename(output_csv)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PSOLID/PSHELL information'])
        for key, value in psolid_pshell_card_dict.items():
            writer.writerow([value[0:8].strip(),
                             key,
                             safe_int_conv(value[16:24].strip()),
                             safe_float_conv(value[24:32].strip()),
                             safe_int_conv(value[32:40].strip())]
                             )
        writer.writerow([])
        writer.writerow(['MAT information'])
        for key, value in mat1_card_dict.items():
            writer.writerow([key,
                             safe_float_conv(value[16:24].strip()),
                             safe_float_conv(value[24:32].strip()),
                             safe_float_conv(value[32:40].strip()),
                             safe_float_conv(value[40:48].strip()),
                             safe_float_conv(value[48:56].strip()),
                             safe_float_conv(value[56:64].strip())]
                             )
        writer.writerow([])
        writer.writerow(['Elements number corresponding to PSOLID/PSHELL'])
        for psolid_pshell_id, elements in psolid_pshell_ctetra_cquad4_ctria3_id_dict.items():
            grouped_elements = 'group : ' + group_elements_for_csv(elements)
            writer.writerow([psolid_pshell_id, grouped_elements])
        writer.writerow([])
        writer.writerow(['Elements number corresponding to MAT1'])
        for mat1_id, elements in mat1_ctetra_cquad4_ctria3_id_dict.items():
            grouped_elements = 'group : ' + group_elements_for_csv(elements)
            writer.writerow([mat1_id, grouped_elements])
        

def extract_stress_values(filename):
    lines = get_file_content(filename)

    stress_data = {}
    start_reading_pshell = False
    start_reading_ctetra = False
    ctetra_element_id = 0
    ctetra_active = False # 「0GRID CS  4 GP」の次の行が要素番号が書いてある

    for i in range(len(lines) - 1):
        line = lines[i]
        next_line = lines[i + 1]

        if (re.search(r'S T R E S S E S\s+I N\s+Q U A D R I L A T E R A L\s+E L E M E N T S\s+\( Q U A D 4 \)', line) or
            re.search(r'S T R E S S E S\s+I N\s+T R I A N G U L A R\s+E L E M E N T S\s+\( T R I A 3 \)', line)):
            start_reading_pshell = True
            continue

        if (re.search(r'S T R E S S E S\s+I N\s+T E T R A H E D R O N   S O L I D\s+E L E M E N T S\s+\( C T E T R A \)', line)):
            start_reading_ctetra = True
            continue
        
        # if re.search(r'MSC NASTRAN.+PAGE.+', line):
        if re.search(r'.+MSC.NASTRAN.+PAGE.+', line):
            start_reading_pshell = False
            start_reading_ctetra = False
        
        # if start_reading and re.match(r'^\s*\d+\s+', line):
        if start_reading_pshell and re.match(r'^\s*0\d*\s+\d+\s+.*', line):
            element_id = int(line[1:9].strip())
            
            normal_x1 = safe_float_conv(line[26:43].strip())
            normal_y1 = safe_float_conv(line[44:58].strip())
            shear_xy1 = safe_float_conv(line[59:73].strip())

            normal_x2 = safe_float_conv(next_line[26:43].strip())
            normal_y2 = safe_float_conv(next_line[44:58].strip())
            shear_xy2 = safe_float_conv(next_line[59:73].strip())

            stress_part_1 = (pow(normal_x1, 2.0) + pow(normal_y1, 2.0) + pow(normal_x2, 2.0) + pow(normal_y2, 2.0)) / 2.0
            stress_part_2 = normal_x1 * normal_y1 + normal_x2 * normal_y2
            stress_part_3 = shear_xy1 + shear_xy2

            von_mises_1 = safe_float_conv(line[117:131].strip())
            von_mises_2 = safe_float_conv(next_line[117:131].strip())

            stress_data[element_id] = (stress_part_1, stress_part_2, stress_part_3, max(von_mises_1, von_mises_2))

        if start_reading_ctetra and re.match(r'.+0GRID\sCS.+GP.*', line):
            ctetra_element_id = int(line[3:11].strip())
            ctetra_active = True
        
        if start_reading_ctetra and ctetra_active and re.match(r'.+CENTER.+X.+XY.+', line):
            ctetra_active = False
            next_next_line = lines[i + 2]
            normal_x = safe_float_conv(line[27:41].strip())
            normal_y = safe_float_conv(next_line[27:41].strip())
            normal_z = safe_float_conv(next_next_line[27:41].strip())
            shear_xy = safe_float_conv(line[46:60].strip())
            shear_yz = safe_float_conv(next_line[46:60].strip())
            shear_zx = safe_float_conv(next_next_line[46:60].strip())
            von_mises = safe_float_conv(line[114:129].strip())
            stress_data[ctetra_element_id] = (normal_x, normal_y, normal_z, von_mises, shear_xy, shear_yz, shear_zx)

    # b_for_debug = om.get_from_flag_data_list(4)
    # if b_for_debug:
    #     logging.debug(f"{filename}における、辞書stress_dataのデータ内容:\n" + pprint.pformat(stress_data, indent=4))

    return stress_data

def check_skip_optimize(thickness_youngmodulus, initial_thickness_youngmodulus, threshold, phase_num):
    if phase_num == 1:
        return False
    if abs(thickness_youngmodulus - initial_thickness_youngmodulus) < threshold:
        return True
    return False

def calculate_thickness_density_percentage(thickness_youngmodulus, initial_thickness_youngsmodulus, density_power, b_use_thickness, threshold):
    if b_use_thickness:
        if initial_thickness_youngsmodulus <= threshold:
            raise ValueError("thickness must be non-zero.")
        return (thickness_youngmodulus / initial_thickness_youngsmodulus)
    else:
        if initial_thickness_youngsmodulus <= threshold:
            raise ValueError("E_0 must be non-zero.")
        return (thickness_youngmodulus / initial_thickness_youngsmodulus) ** (1 / density_power)

def check_nastran_log_for_pattern(log_file_name):
    pattern = re.compile(r'.*MSC\.Nastran finished.+')
    with open(log_file_name, 'r') as file:
        for line in file:
            if pattern.match(line):
                return True
    return False

def check_nastran_f06_for_pattern(f06_file_name):
    flag1 = False
    flag2 = False
    pattern1 = re.compile(r'.*\* \* \* END OF JOB \* \* \*')
    pattern2 = re.compile(r'.*\*\*\* USER FATAL MESSAGE')
    with open(f06_file_name, 'r') as file:
        for line in file:
            if pattern1.match(line):
                flag1 = True
            if pattern2.match(line):
                flag2 = True
    if flag1 == True and flag2 == False:
        return True
    return False

def run_nastran(dat_file_name, nastran_exe_path):
    try:
        result = subprocess.run([nastran_exe_path, dat_file_name], check=True)
        f06_file_name = os.path.splitext(dat_file_name)[0] + ".f06"
        log_file_name = os.path.splitext(dat_file_name)[0] + ".log"
        log_interval = 30
        start_wait_time = time.time()
        last_log_time = start_wait_time
        logging.info(f"最適化後のファイルに対して、Nastranを実行中... 経過時間: 0秒")
        print(f"最適化後のファイルに対して、Nastranを実行中... 経過時間: 0秒")
        while True:
            time.sleep(5)  # 5秒待機
            current_time = time.time()
            if (current_time - last_log_time) >= log_interval:
                elapsed_time = int(current_time - start_wait_time)  # 経過秒数を計算
                logging.info(f"最適化後のファイルに対して、Nastranを実行中... 経過時間: {elapsed_time}秒")
                print(f"最適化後のファイルに対して、Nastranを実行中... 経過時間: {elapsed_time}秒")
                last_log_time = current_time
            if check_nastran_log_for_pattern(log_file_name):
                if check_nastran_f06_for_pattern(f06_file_name) == False:
                    print(f"{f06_file_name}の記載に異常があるようです。")
                    raise CustomError(f"{f06_file_name}の記載に異常があるようです。")
                logging.debug(f"{f06_file_name}と{log_file_name}から終了の記載が見つかったため、Nastranが終了したとして次に進みます。")
                break
        logging.info(f"Nastranの実行がリターンコード{result.returncode}で終了しました。")
        print(f"Nastranの実行がリターンコード{result.returncode}で終了しました。")
    except CustomError as e:
        logging.error(f"Nastranの実行で異常が発生しました。: {e}")
        print(f"Nastranの実行で異常が発生しました。: {e}")
        return 1
    except subprocess.CalledProcessError as e:
        logging.error(f"Nastranの実行に失敗しました。: {e}")
        print(f"Nastranの実行に失敗しました。: {e}")
        return 2
    return 0

def calculate_ising_part(
        elem, 
        initial_thickness_youngsmodulus, 
        volume, 
        first_thickness_density_percentage, 
        density_power_calc, 
        density_increment,
        phase_num, 
        b_use_thickness,
        is_solid_flag,
        threshold,
        return_value):
    thickness = float(elem.get('thickness', 0))
    youngsmodulus = float(elem.get('youngsmodulus', 0))

    thickness_density_percentage_now = 0.0
    if phase_num == 1:
        thickness_density_percentage_now = first_thickness_density_percentage
    else:
        thickness_youngmodulus = thickness if b_use_thickness else youngsmodulus
        thickness_density_percentage_now = calculate_thickness_density_percentage(thickness_youngmodulus, initial_thickness_youngsmodulus, density_power_calc, b_use_thickness, threshold)

    density_plus_delta = thickness_density_percentage_now + density_increment
    density_minus_delta = thickness_density_percentage_now - density_increment

    alpha_value = pow(density_plus_delta, (1 - density_power_calc))
    beta_value = pow(density_minus_delta, (1 - density_power_calc))

    poissonratio = float(elem.get('poissonratio', 0))
    youngmodulus_for_calculate = (youngsmodulus if b_use_thickness else initial_thickness_youngsmodulus)
    energy_part = 0.0
    if is_solid_flag:
        normal_x = elem.get('normal_x', 0)
        normal_y = elem.get('normal_y', 0)
        normal_z = elem.get('normal_z', 0)
        shear_xy = elem.get('shear_xy', 0)
        shear_yz = elem.get('shear_yz', 0)
        shear_zx = elem.get('shear_zx', 0)
        # D{ε}={σ}よりひずみベクトル{ε}を計算し、{ε}^T{σ}を求める
        D_inv_matrix = create_D_inverse(poissonratio, youngmodulus_for_calculate)
        stress_vector = np.array([normal_x, normal_y, normal_z, shear_xy, shear_yz, shear_zx])
        strain_vector = np.dot(D_inv_matrix, stress_vector)
        energy_part = np.dot(strain_vector, stress_vector)
    else:
        stress_part_1 = elem.get('stress_part_1', 0)
        stress_part_2 = elem.get('stress_part_2', 0)
        stress_part_3 = elem.get('stress_part_3', 0)
        energy_part = (stress_part_1 - poissonratio * stress_part_2 + (1.0 + poissonratio) * stress_part_3) / youngmodulus_for_calculate

    kappa_i = energy_part * volume

    # 最適化計算用
    return_value.append(alpha_value)
    return_value.append(beta_value)
    return_value.append(kappa_i)

    # csv出力用
    von_mises = float(elem.get('von_mises', 0))
    return_value.append(von_mises)
    return_value.append(energy_part)

def write_data_to_csv_for_check_ising_val(csv_file_name, return_value_dict, phase_num):
    column_key = 1 
    column_value1 = 4 * phase_num - 2
    column_value2 = 4 * phase_num - 1
    column_value3 = 4 * phase_num
    column_value4 = 4 * phase_num + 1

    try:
        with open(csv_file_name, mode='r', newline='') as csvfile:
            reader = list(csv.reader(csvfile))
    except FileNotFoundError:
        reader = []

    if not reader:
        reader.append([''] * column_value4)
        reader.append([''] * column_value4)
    if len(reader[0]) < column_value4:
        reader[0].extend([''] * (column_value4 - len(reader[0])))
    if len(reader[1]) < column_value4:
        reader[1].extend([''] * (column_value4 - len(reader[1])))
    reader[0][column_value1 - 1] = phase_num
    reader[1][column_value1 - 1] = 'von mises'
    reader[1][column_value2 - 1] = "energy_part"
    reader[1][column_value3 - 1] = 'h_first'
    reader[1][column_value4 - 1] = 'ising value'

    for key in return_value_dict.keys():
        found = False
        if phase_num == 1:
            row = [''] * column_value4
            row[column_key - 1] = key
            row[column_value1 - 1] = return_value_dict[key][3]
            row[column_value2 - 1] = return_value_dict[key][4]
            row[column_value3 - 1] = return_value_dict[key][5]
            row[column_value4 - 1] = return_value_dict[key][6]
            reader.append(row)
        else:
            for row in reader[2:]:  # Skip header
                if len(row) > 0 and row[0] == key:
                    if len(row) < column_value4:
                        row.extend([''] * (column_value4 - len(row)))
                    row[column_value1 - 1] = return_value_dict[key][3]
                    row[column_value2 - 1] = return_value_dict[key][4]
                    row[column_value3 - 1] = return_value_dict[key][5]
                    row[column_value4 - 1] = return_value_dict[key][6]
                    found = True
                    break
            if not found:
                row = [''] * column_value4
                row[0] = key
                row[column_value1 - 1] = return_value_dict[key][3]
                row[column_value2 - 1] = return_value_dict[key][4]
                row[column_value3 - 1] = return_value_dict[key][5]
                row[column_value4 - 1] = return_value_dict[key][6]
                reader.append(row)

    with open(csv_file_name, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(reader)

def write_data_to_csv_youngsmodulus(csv_file_name, mat_thickness_youngmodulus_temp, phase_num):
    file_exists = os.path.isfile(csv_file_name)
    
    if file_exists:
        with open(csv_file_name, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            data = list(reader)  # CSVの内容をリストに読み込む
    else:
        data = []  # 新規の場合は空のリスト

    # 既存のキーを辞書に格納
    existing_keys = {row[0]: index for index, row in enumerate(data)}
    
    # 既存のキーの行に対して、指定された列（phase_num）の初期値を0に設定
    for key, row_index in existing_keys.items():
        if len(data[row_index]) <= phase_num:
            # 必要な列まで拡張し、初期値を0に設定
            data[row_index].extend([0] * (phase_num - len(data[row_index]) + 1))
        # すでに存在する行の指定された列（phase_num）の初期値を0にする
        if key not in mat_thickness_youngmodulus_temp:
            data[row_index][phase_num] = 0

    # mat_thickness_youngmodulus_tempのキーを処理
    keys = list(mat_thickness_youngmodulus_temp.keys())
    for key in keys:
        if key in existing_keys:
            row_index = existing_keys[key]
        else:
            row_index = len(data)
            # 新しい行を追加し、初期値を0に設定
            data.append([key] + [0] * phase_num)
        
        # 指定された列(phase_num)の値を更新
        data[row_index][phase_num] = mat_thickness_youngmodulus_temp[key]

    # CSVファイルにデータを書き込む
    with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def calc_hamiltonian_energy(sigma, h, J):
    energy = 0
    for i in range(len(sigma)):
        energy += h[i] * sigma[i]
    for i in range(len(sigma)):
        for j in range(i + 1, len(sigma)):
            energy += J[i, j] * sigma[i] * sigma[j]
    return energy

def do_simulated_annealing_single_optimize(
        h,
        J,
        initial_temp,
        cooling_rate,
        max_iter,
        S_minus_1,
        S_plus_1
):
    num_spins = len(h)
    sigma = np.random.choice([-1, 1], size=num_spins)
    best_sigma = np.random.choice([-1, 1], size=num_spins)

    current_energy = calc_hamiltonian_energy(sigma, h, J)
    best_energy = current_energy
    temperature = initial_temp

    for iteration in range(max_iter):
        i = np.random.randint(num_spins)
        sigma[i] *= -1
        delta_energy = 2 * sigma[i] * h[i]
        for k in range(len(sigma)):
            if k < i:
                delta_energy += 2 * J[k, i] * sigma[k] * sigma[i]
            if k > i:
                delta_energy += 2 * J[i, k] * sigma[i] * sigma[k]
        # new_energy = calc_hamiltonian_energy(sigma, h, J)
        # delta_energy = new_energy - current_energy
        # delta_energy = 2 * sigma[i] * (h[i] + np.sum(J[i] * sigma))
        if (current_energy + delta_energy) < best_energy:
            best_energy = (current_energy + delta_energy)
            best_sigma = copy.deepcopy(sigma)
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            current_energy += delta_energy
        else:
            sigma[i] *= -1
        temperature *= cooling_rate

    # check_energy = calc_hamiltonian_energy(best_sigma, h, J)

    for i, val in enumerate(best_sigma):
        if val == -1:
            S_minus_1.append(i)
        elif val == 1:
            S_plus_1.append(i)

def do_qiskit_single_optimize(
        h,
        J,
        S_minus_1,
        S_plus_1
):
    qp = QuadraticProgram()
    num_variables = max(max(h.keys()), max(j for i, j in J.keys())) + 1
    for i in range(num_variables):
        qp.binary_var(f'x{i}')
    linear = {}
    quadratic = {}
    # 線形項のQUBO変数からイジング変数への変換: h_i * σ_i -> h_i * (2*x_i - 1)
    for i in h:
        linear[f'x{i}'] = 2 * h[i]
    # 二次項の変換: J_ij * σ_i * σ_j -> J_ij * (2*x_i - 1) * (2*x_j - 1)
    for (i, j), Jij in J.items():
        quadratic[f'x{i}', f'x{j}'] = 4 * Jij
        linear[f'x{i}'] -= 2 * Jij
        linear[f'x{j}'] -= 2 * Jij
    qp.minimize(linear=linear, quadratic=quadratic)

    # 下記のあたりは量子ビット数が増えると使えない
    # optimizer = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    # result = optimizer.solve(qp)

    # grover_optimizer = GroverOptimizer(num_value_qubits=num_variables, sampler=StatevectorSampler())
    # result = grover_optimizer.solve(qp)

    # admm_optimizer = ADMMOptimizer()
    # result = admm_optimizer.solve(qp)

    # slsqp_optimizer = SlsqpOptimizer()
    # result = slsqp_optimizer.solve(qp)

    cplex_optimizer = CplexOptimizer()
    result = cplex_optimizer.solve(qp)

    # scipymilp_optimizer = ScipyMilpOptimizer()
    # result = scipymilp_optimizer.solve(qp)

    # backend = Aer.get_backend('statevector_simulator')
    # logging.debug(f"Qiskitの最適化問題の内容：{qp.prettyprint()}")

    solution = result.x

    for i, val in enumerate(solution):
        if val == 0:
            S_minus_1.append(i)  # 0の場合、S_mにインデックスを追加
        elif val == 1:
            S_plus_1.append(i)  # 1の場合、S_pにインデックスを追加

def do_dwave_single_optimize(
        h,
        J,
        S_minus_1,
        S_plus_1
):
    sampler = LeapHybridSampler()
    response = sampler.sample_ising(h, J)
    logging.debug(f"D-waveの最適化結果の生データ：{response.record}")

    sample = next(response.data(fields=['sample']))
    for k, v in sample.items():
        if v == -1:
            S_minus_1.append(k)
        if v == 1:
            S_plus_1.append(k)

def do_single_optimize(
        s_optimize_rule,
        h,
        J,
        ising_index_dict,
        inverse_ising_index_eid_map,
        merged_elem_list,
        b_use_thickness,
        density_increment,
        first_sum_volume,
        phase_num,
        cost_lambda_calc,
        b_for_debug
):
    S_minus_1 = []
    S_plus_1 = []
    if s_optimize_rule == "d-wave":
        do_dwave_single_optimize(h, J, S_minus_1, S_plus_1)
    elif s_optimize_rule == "qiskit":
        do_qiskit_single_optimize(h, J, S_minus_1, S_plus_1)
    elif s_optimize_rule.startswith("sa-"):
        initial_temp = 10
        max_iter = int(s_optimize_rule.split('-')[1])
        temp_threshold = float(1e-3) # 温度が最終的に0.001度になるようにする
        cooling_rate = float((temp_threshold / initial_temp) ** (1.0 / max_iter))
        do_simulated_annealing_single_optimize(h, J, initial_temp, cooling_rate, max_iter, S_minus_1, S_plus_1)
    minus_vol = 0.0
    plus_vol = 0.0

    for idx in S_minus_1:
        ising_index_dict[idx] = -1
        eid_temp = inverse_ising_index_eid_map[idx]
        elem_temp = next((d for d in merged_elem_list if int(d['eid']) == int(eid_temp)), None)
        if b_use_thickness:
            area = float(elem_temp.get('area', 0))
            thickness_diff = density_increment
            volume_diff = thickness_diff * area
            minus_vol += volume_diff
        else:
            density_diff = density_increment
            volume_pre = float(elem_temp.get('volume', 0))
            volume_reflect_density_diff = volume_pre * density_diff
            minus_vol += volume_reflect_density_diff
            
    for idx in S_plus_1:
        ising_index_dict[idx] = 1
        eid_temp = inverse_ising_index_eid_map[idx]
        elem_temp = next((d for d in merged_elem_list if int(d['eid']) == int(eid_temp)), None)
        if b_use_thickness:
            area = float(elem_temp.get('area', 0))
            thickness_diff = density_increment
            volume_diff = thickness_diff * area
            plus_vol += volume_diff
        else:
            density_diff = density_increment
            volume_pre = float(elem_temp.get('volume', 0))
            volume_reflect_density_diff = volume_pre * density_diff
            plus_vol += volume_reflect_density_diff

    vol_diff = abs(plus_vol - minus_vol)
    diff_percentage = vol_diff * 100.0 / first_sum_volume
    logging.info(f"最適化による体積の増分：{vol_diff}、初期体積に占める変化量の割合：{diff_percentage}(%)（※ペナルティ係数：{cost_lambda_calc}）")
    percentage_threshold = 0.1
    if diff_percentage < percentage_threshold:
        logging.info(f"最適化における初期体積に占める変化量の割合：{diff_percentage}(%)が、{percentage_threshold}(%)を下回ったため、{phase_num}回目の最適化を完了します。")
        print(f"最適化における初期体積に占める変化量の割合：{diff_percentage}(%)が、{percentage_threshold}(%)を下回ったため、{phase_num}回目の最適化を完了します。")
        return True
    else:
        logging.info(f"最適化における初期体積に占める変化量の割合：{diff_percentage}(%)が、{percentage_threshold}(%)を上回ったため、{phase_num}回目の最適化を再度行います。")
        print(f"最適化における初期体積に占める変化量の割合：{diff_percentage}(%)が、{percentage_threshold}(%)を上回ったため、{phase_num}回目の最適化を再度行います。")
    if b_for_debug:
        element_count_minus = len(S_minus_1)
        logging.debug(f"The number of elements minus ising index in the array is: {element_count_minus}, cost lambda is: {cost_lambda_calc}")
        element_count_plus = len(S_plus_1)
        logging.debug(f"The number of elements plus ising index in the array is: {element_count_plus}, cost lambda is: {cost_lambda_calc}")
    return False

def main():
    loop_num = int(sys.argv[9])
    start_phase_num = int(sys.argv[10]) - 1

    # 途中からの実行に対応
    if start_phase_num >= 1:
        base_name, ext = os.path.splitext(sys.argv[1])
        reserve_data_csv = base_name + '_reserve_data_for_single_opt.csv'
        om.load_thickness_youngsmodulus_data_from_csv(reserve_data_csv)

    for i in range(start_phase_num, loop_num):
        phase_num = i + 1
        logging.info(f"最適化処理開始：{phase_num}回目（最大最適化ループ回数：{loop_num}回）")
        print(f"最適化処理開始：{phase_num}回目（最大最適化ループ回数：{loop_num}回）")
        result = main2(phase_num)
        if result == -1:
            logging.error(f"{phase_num}/{loop_num}の処理で失敗しました")
            print(f"{phase_num}/{loop_num}の処理で失敗しました")
        if result == 1:
            logging.info(f"{phase_num}/{loop_num}で、最適化を完了しました")
            print(f"{phase_num}/{loop_num}で、最適化を完了しました")

def main2(phase_num):
    dat_file_path = sys.argv[1]
    f06_file_path = os.path.splitext(dat_file_path)[0] + ".f06"
    start_time_1 = time.time()

    b_do_nastran = om.get_from_flag_data_list(3)
    nastran_exe_path = sys.argv[4]
    if phase_num == 1:
        if b_do_nastran:
            return_code = run_nastran(dat_file_path, nastran_exe_path)
            if return_code != 0:
                return -1

    target_thickness_density_percentage = float(sys.argv[5])
    density_increment = float(sys.argv[6])
    density_power = float(sys.argv[7])
    cost_lambda = float(sys.argv[8])
    decide_val_threshold = float(sys.argv[11])
    threshold = float(sys.argv[12])

    input_dat_file_name = "{}_phase_{}.dat".format(str(dat_file_path)[:-4], phase_num - 1)
    if phase_num == 1:
        create_first_phase_zero_nastran_file(dat_file_path, input_dat_file_name)

    logging.info(f"解析に使用した入力ファイル名：{input_dat_file_name}")
    print(f"解析に使用した入力ファイル名：{input_dat_file_name}")

    b_use_thickness = False if str(sys.argv[15]) == "0" else True
    node_dict = {}
    psolid_pshell_dict = {}
    mat1_dict = {}
    ctetra_cquad4_ctria3_dict = {}
    force_grid_list = []
    all_node_id_set = set()
    with open(input_dat_file_name, 'r', encoding='utf-8') as file:
        for line in file:
            if line.startswith('$'):
                continue
            if line.startswith('FORCE'):
                # 荷重がかかっている節点を含む要素を残す。最適化ファイルは、全てのFORCEカードが必ずPSHELLカードより先に出てくるためこの順序で良い
                force_grid_id = int(line[16:24].strip())
                force_grid_list.append(force_grid_id)
            if line.startswith('GRID'):
                # フォーマットは8カラムで固定長（8文字ずつ）なので、それに基づいてフィールドを取得
                grid_id = int(line[8:16].strip())
                x = safe_float_conv(line[24:32].strip())
                y = safe_float_conv(line[32:40].strip())
                z = safe_float_conv(line[40:48].strip())
                node_dict[grid_id] = [x, y, z]
                all_node_id_set.add(grid_id)
            if line.startswith('PSOLID') or line.startswith('PSHELL'):
                elem_id = int(line[8:16].strip())
                mat_id = int(line[16:24].strip())
                thickness = 0.0
                if line.startswith('PSHELL'):
                    thickness = float(line[24:32].strip())
                b_skip_optimize_psolid_pshell = False
                if b_use_thickness:
                    thickness = safe_float_conv(line[24:32].strip())
                    initial_thickness_temp = om.get_from_thickness_youngsmodulus_data_dict(elem_id)
                    if check_skip_optimize(thickness, initial_thickness_temp, threshold, phase_num):
                        b_skip_optimize_psolid_pshell = True
                if not b_skip_optimize_psolid_pshell:
                    psolid_pshell_dict[elem_id] = [mat_id, thickness]
            if line.startswith('MAT1'):
                mat_id = int(line[8:16].strip())
                youngmodulus = safe_float_conv(line[16:24].strip())
                initial_youngsmodulus_temp = om.get_from_thickness_youngsmodulus_data_dict(mat_id)
                b_skip_optimize_mat1 = False
                if not b_use_thickness:
                    if check_skip_optimize(youngmodulus, initial_youngsmodulus_temp, threshold, phase_num):
                        b_skip_optimize_mat1 = True
                if b_skip_optimize_mat1:
                    del psolid_pshell_dict[mat_id]
                else:
                    poissonratio = safe_float_conv(line[32:40].strip())
                    value = [youngmodulus, poissonratio]
                    mat1_dict[mat_id] = value
            if line.startswith('CTETRA') or line.startswith('CQUAD4') or line.startswith('CTRIA3'):
                elem_id = int(line[8:16].strip())
                if elem_id in psolid_pshell_dict:
                    value = None
                    x1 = int(line[24:32].strip())
                    x2 = int(line[32:40].strip())
                    x3 = int(line[40:48].strip())
                    if line.startswith('CTETRA') or line.startswith('CQUAD4'):
                        x4 = int(line[48:56].strip())
                        value = [x1, x2, x3, x4]
                    if line.startswith('CTRIA3'):
                        value = [x1, x2, x3]
                    b_is_remain_elem_forcely = False
                    for x in value:
                        count = force_grid_list.count(x)
                        if count > 0:
                            logging.debug(f"要素{elem_id}は節点{x}を持ち、この節点はFORCEカードで荷重がかかっているため、最適化の対象から外します（残ることが確定）。")
                            b_is_remain_elem_forcely = True
                            initial_thickness_youngsmodulus_temp = om.get_from_thickness_youngsmodulus_data_dict(elem_id)
                            om._mat_thickness_youngmodulus_remain[str(elem_id)] = initial_thickness_youngsmodulus_temp
                            break
                    if b_is_remain_elem_forcely:
                        del psolid_pshell_dict[elem_id]
                        del mat1_dict[elem_id]
                    else:
                        ctetra_cquad4_ctria3_dict[elem_id] = value

    input_f06_file_name = "{}_phase_{}.f06".format(str(f06_file_path)[:-4], phase_num - 1)
    if phase_num == 1:
        shutil.copy(f06_file_path, input_f06_file_name)
    stress_dict = extract_stress_values(input_f06_file_name)

    merged_elem_list = []
    upper_limit_of_stress = float(sys.argv[14])
    is_solid_flag = om.is_solid()
    sum_volume_pre = 0.0
    for psolid_pshell_key, value in psolid_pshell_dict.items():
        eid = psolid_pshell_key
        stress_value = stress_dict[eid]
        von_mises = stress_value[3]
        thickness = value[1]
        if von_mises > upper_limit_of_stress:
            logging.debug(f"eid={eid}の要素のvon mises応力値{von_mises}が、基準値として指定した{upper_limit_of_stress}を超えたためにこの要素の最適化をスキップします。")
            mat_value = mat1_dict[eid]
            thickness_youngmodulus_temp = thickness if b_use_thickness else mat_value[0]
            om._mat_thickness_youngmodulus_remain[str(eid)] = thickness_youngmodulus_temp
            continue

        if b_use_thickness:
            initial_thickness_check = om.get_from_thickness_youngsmodulus_data_dict(eid)
            if initial_thickness_check >= threshold:
                if (thickness / initial_thickness_check) < (decide_val_threshold + threshold):
                    continue

        merged_dict = {}
        merged_dict['eid'] = eid
    
        if is_solid_flag:
            merged_dict['normal_x'] = stress_value[0]
            merged_dict['normal_y'] = stress_value[1]
            merged_dict['normal_z'] = stress_value[2]
            merged_dict['shear_xy'] = stress_value[4]
            merged_dict['shear_yz'] = stress_value[5]
            merged_dict['shear_zx'] = stress_value[6]
        else:
            merged_dict['stress_part_1'] = stress_value[0]
            merged_dict['stress_part_2'] = stress_value[1]
            merged_dict['stress_part_3'] = stress_value[2]

        merged_dict['von_mises'] = von_mises
        ctetra_cquad4_ctria3_value = ctetra_cquad4_ctria3_dict[eid]
        merged_dict['nodes'] = ctetra_cquad4_ctria3_value
        mat1_value = mat1_dict[value[0]]
        merged_dict['thickness'] = thickness
        youngsmodulus = mat1_value[0]
        merged_dict['youngsmodulus'] = youngsmodulus
        merged_dict['poissonratio'] = mat1_value[1]

        node_data = []
        for nid in ctetra_cquad4_ctria3_value:
            node_value = node_dict[nid]
            node_value_dict = {}
            node_value_dict['nid'] = nid
            node_value_dict['x'] = node_value[0]
            node_value_dict['y'] = node_value[1]
            node_value_dict['z'] = node_value[2]
            node_data.append(node_value_dict)

        merged_dict['node_data'] = node_data
        volume = 0.0
        if is_solid_flag:
            volume = calc_tetrahedron_volume(node_data)
            merged_dict['volume'] = volume
        else:
            area = calculate_area(node_data)
            merged_dict['area'] = area
            volume = area * thickness
            merged_dict['volume'] = volume

        volume_reflect_density = volume
        if not b_use_thickness:
            density_pre = target_thickness_density_percentage
            if phase_num > 1:
                initial_youngsmodulus = om.get_from_thickness_youngsmodulus_data_dict(eid)
                density_pre = calculate_thickness_density_percentage(youngsmodulus, initial_youngsmodulus, density_power, b_use_thickness, threshold)
            volume_reflect_density = volume * density_pre
            merged_dict['density_pre'] = density_pre
        merged_dict['volume_reflect_density'] = volume_reflect_density
        sum_volume_pre += volume_reflect_density

        merged_elem_list.append(merged_dict)

    if phase_num == 1:
        base_name, ext = os.path.splitext(dat_file_path)
        reserve_data_csv = base_name + '_reserve_data_for_single_opt.csv'
        new_row = ['First_sum_volume', sum_volume_pre]
        om.update_thickness_youngsmodulus_csv(reserve_data_csv, 2, new_row)

    logging.debug(f"{phase_num}回目の最適化前の、密度を考慮した体積の合計値：{sum_volume_pre}")
    
    elapsed_time_1 = time.time() - start_time_1
    logging.info(f"入力データの読み込みにかかった時間：{str(elapsed_time_1)} [s]")

    start_time_2 = time.time()

    energy_list_for_scale = []
    volume_list_for_scale = []

    # b_use_thicknessの時は、density_power=0とした相当の動作をするケースのための変数
    density_power_calc = 0 if b_use_thickness else density_power

    first_thickness_density_percentage = target_thickness_density_percentage
    return_value_dict = {}
    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        initial_thickness_youngsmodulus = om.get_from_thickness_youngsmodulus_data_dict(eid)
        volume = float(elem.get('volume', 0))
        return_value = [] # [alpha_value, beta_value, kappa_i, energy_part]
        calculate_ising_part(elem,
                             initial_thickness_youngsmodulus,
                             volume,
                             first_thickness_density_percentage,
                             density_power_calc,
                             density_increment,
                             phase_num,
                             b_use_thickness,
                             is_solid_flag,
                             threshold,
                             return_value)
        
        k_0 = (return_value[0] - return_value[1]) / 2.0
        energy_list_for_scale.append(k_0 * return_value[2] * 1.0)   # K_0 * kappa_i * x_i(1.0)
        energy_list_for_scale.append(k_0 * return_value[2] * -1.0)  # K_0 * kappa_i * x_i(-1.0)

        volume_list_for_scale.append(density_increment * volume * 1.0)   # Δρ * v_i * x_i(1.0)
        volume_list_for_scale.append(density_increment * volume * -1.0)  # Δρ * v_i * x_i(-1.0)

        return_value_dict[eid] = return_value

    std_of_energy_list = np.std(energy_list_for_scale)
    std_of_volume_list = np.std(volume_list_for_scale)

    ising_index_eid_map = {}
    nInternalid = len(merged_elem_list)
    h = defaultdict(int)
    J = defaultdict(int)
    cost_lambda_calc = cost_lambda
    cost_lambda_calc_multiply = (1 + 5 ** 0.5) / 2  # 黄金比
    old_cost_lambda =  om.get_cost_lambda()
    if old_cost_lambda > 0:
        cost_lambda_calc = (old_cost_lambda / cost_lambda_calc_multiply)
    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        ising_index_eid_map[eid] = index

        return_val = return_value_dict[eid]

        k_0 = (return_val[0] - return_val[1]) / 2.0
        h_first = (k_0 * return_val[2]) / (3.0 * np.sqrt(nInternalid) * std_of_energy_list)
        # h_first = (k_0 * return_val[2]) / (2.0 * np.sqrt(nInternalid) * std_of_energy_list)
        h[index] = h_first
        return_val.append(h_first)

        for j_index in range(index + 1, nInternalid):
            volume_j = merged_elem_list[j_index].get('volume', 0)
            J[(index,j_index)] = (2.0 * cost_lambda_calc * pow(density_increment, 2) * volume * volume_j) / (9.0 * nInternalid * pow(std_of_volume_list, 2))
            # J[(index,j_index)] = (cost_lambda_calc * pow(density_increment, 2) * volume * volume_j) / (pow(nInternalid, 2) * pow(std_of_volume_list, 2))

    elapsed_time_2 = time.time() - start_time_2
    logging.info(f"最適化処理の準備にかかった時間：{str(elapsed_time_2)} [s]")

    b_fin_optimize = 0
    finish_elem_num = int(sys.argv[13])
    if len(merged_elem_list) <= finish_elem_num:
        logging.info(f"最適化が完了していない要素の数({len(merged_elem_list)})が{finish_elem_num}以下になったため、要素の0/1を決定します")
        b_fin_optimize = 1

    logging.info(f"{phase_num}回目の最適化を開始します。")
    print(f"{phase_num}回目の最適化を開始します。")
    print("最適化実行中…")

    start_time_3 = time.time()

    ising_index_dict = {}

    b_for_debug = om.get_from_flag_data_list(4)
    b_do_optimize = om.get_from_flag_data_list(0)
    inverse_ising_index_eid_map = {value: key for key, value in ising_index_eid_map.items()}
    if b_do_optimize and not b_fin_optimize:
        b_do_single_optimize = True
        n_optimize_num = 1
        while b_do_single_optimize:
            if n_optimize_num >= 5:
                print("試行回数が5回を超えたため、最適化を進めます。")
                break
            print(f"ペナルティ係数：{cost_lambda_calc}での最適化を開始します。フェーズ数：{phase_num}、試行回数：{n_optimize_num}回目")
            n_optimize_num += 1
            s_optimize_rule = sys.argv[16]
            b_fin_single_optimize = do_single_optimize(
                s_optimize_rule,
                h,
                J,
                ising_index_dict,
                inverse_ising_index_eid_map,
                merged_elem_list,
                b_use_thickness,
                density_increment,
                om._sum_target_volume,
                phase_num,
                cost_lambda_calc,
                b_for_debug
            )
            if b_fin_single_optimize:
                b_do_single_optimize = False
                b_omit_calculate_cost_lambda_each_time = True
                if b_omit_calculate_cost_lambda_each_time:
                    om.set_cost_lambda(cost_lambda_calc)
            else:
                cost_lambda_calc *= cost_lambda_calc_multiply
                for key in J:
                    J[key] *= cost_lambda_calc_multiply
    else:
        # テスト用(最適化のリソース節約のため)
        for index, elem in enumerate(merged_elem_list):
            ising_index_dict[index] = (1 if random.random() < 0.5 else -1)

    if b_for_debug:
        for index, elem in enumerate(merged_elem_list):
            eid = int(elem.get('eid', 0))
            ising_idx = ising_index_eid_map[eid]
            ising_val = ising_index_dict[ising_idx]
            return_val = return_value_dict[eid]
            return_val.append(ising_val)

        base_name, ext = os.path.splitext(sys.argv[1])
        h_first_and_von_mises_csv = base_name + '_for_debug_return_value.csv'
        if phase_num == 1:
            rename_old_filename(h_first_and_von_mises_csv)
        write_data_to_csv_for_check_ising_val(h_first_and_von_mises_csv, return_value_dict, phase_num)

    elapsed_time_3 = time.time() - start_time_3
    logging.info(f"{phase_num}回目の最適化処理の実行と集計にかかった時間：{str(elapsed_time_3)} [s]")
    print(f"{phase_num}回目の最適化が終わりました。")
    print(f"最適化処理の実行と集計にかかった時間：{str(elapsed_time_3)} [s]")

    start_time_4 = time.time()

    mat_thickness_youngmodulus = {}
    zero_fix_density_index_list = []
    
    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        thickness = float(elem.get('thickness', 0))
        youngsmodulus = float(elem.get('youngsmodulus', 0))
        dens_value_old = first_thickness_density_percentage
        initial_thickness_youngsmodulus = om.get_from_thickness_youngsmodulus_data_dict(eid)
        if not phase_num == 1:
            thickness_youngmodulus = thickness if b_use_thickness else youngsmodulus
            dens_value_old = calculate_thickness_density_percentage(thickness_youngmodulus, initial_thickness_youngsmodulus, density_power, b_use_thickness, threshold)
        ising_index = ising_index_eid_map[eid]
        ising_value = ising_index_dict[ising_index]
        dens_value = dens_value_old + density_increment * ising_value

        if b_fin_optimize:
            if dens_value >= (0.5 + threshold):
               dens_value = 1.0
            else:
               dens_value = 0.0
                         
        b_use_thickness_youngmodulus = True
        if dens_value >= (1.0 - decide_val_threshold - threshold):
            dens_value = 1.0
        if dens_value <= (decide_val_threshold + threshold):
            if not b_use_thickness:
                zero_fix_density_index_list.append(int(eid))
                b_use_thickness_youngmodulus = False

        if b_use_thickness_youngmodulus == True:
            if b_use_thickness:
                mat_thickness_youngmodulus[str(eid)] = dens_value * initial_thickness_youngsmodulus
            else:
                mat_thickness_youngmodulus[str(eid)] = pow(dens_value, density_power) * initial_thickness_youngsmodulus

    lines = get_file_content(input_dat_file_name)

    new_dat_file_name = increment_phase_number(input_dat_file_name)
    new_dat_temp_file_name = new_dat_file_name + ".tmp"
    same_psolid_pshell_flag = 0
    used_node_id_set = set()
    with open(new_dat_temp_file_name, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.startswith('PSOLID') or line.startswith('PSHELL'):
                elem_id = int(line[8:16].strip())
                mat_id = int(line[16:24].strip())
                if elem_id in zero_fix_density_index_list:
                    line = f"${line}"
                    same_psolid_pshell_flag = 1
                else:
                    if b_use_thickness and line.startswith('PSHELL'):
                        line_strip = line.strip()
                        thickness_value = mat_thickness_youngmodulus.get(str(elem_id), None)
                        if thickness_value is None:
                            logging.debug(f"thickness_value id({elem_id}) has already fixed.")
                        else:
                            thickness_formatted = format_float_thickness(thickness_value)
                            line = (
                                format_field_left('PSHELL', 8) +
                                format_field_right(elem_id, 8) +
                                format_field_right(mat_id, 8) +
                                format_field_right(thickness_formatted, 8) +
                                line_strip[32:] +
                                '\n'
                            )

            if line.startswith('MAT1'):
                if same_psolid_pshell_flag == 1:
                    same_psolid_pshell_flag = 2
                    line = f"${line}"
                else:
                    line_strip = line.strip()
                    mat_id = int(line[8:16].strip())
                    if not b_use_thickness:
                        youngmodulus_value = mat_thickness_youngmodulus.get(str(mat_id), None)
                        if youngmodulus_value is None:
                            logging.debug(f"youngmodulus_value id({mat_id}) has already fixed.")
                        else:
                            youngmodulus_formatted = format_float_youngmodulus(youngmodulus_value)
                            line = (
                                format_field_left('MAT1', 8) +
                                format_field_right(mat_id, 8) +
                                format_field_right(youngmodulus_formatted, 8) +
                                line_strip[24:] +
                                '\n'
                            )
            if line.startswith('CTETRA') or line.startswith('CQUAD4') or line.startswith('CTRIA3'):
                if same_psolid_pshell_flag == 2:
                    same_psolid_pshell_flag = 0
                    line = f"${line}"
                else:
                    x1 = int(line[24:32].strip())
                    x2 = int(line[32:40].strip())
                    x3 = int(line[40:48].strip())
                    used_node_id_set.add(x1)
                    used_node_id_set.add(x2)
                    used_node_id_set.add(x3)
                    if line.startswith('CTETRA') or line.startswith('CQUAD4'):
                        x4 = int(line[48:56].strip())
                        used_node_id_set.add(x4)

            file.write(line)

    lines_new = get_file_content(new_dat_temp_file_name)
        
    rename_old_filename(new_dat_file_name)
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
        b_erase_temp_file = om.get_from_flag_data_list(2)
        if b_erase_temp_file:
            delete_file(new_dat_temp_file_name)
    else:
        rename_file(new_dat_temp_file_name, new_dat_file_name)

    if b_for_debug:
        base_name, ext = os.path.splitext(sys.argv[1])
        youngmodulus_csv = base_name + '_for_debug_youngsmodulus.csv'
        if phase_num == 1:
            rename_old_filename(youngmodulus_csv)
        merged_mat_thickness_youngmodulus = {**mat_thickness_youngmodulus, **(om._mat_thickness_youngmodulus_remain)}
        sorted_mat_thickness_youngmodulus = dict(sorted(merged_mat_thickness_youngmodulus.items(), key=lambda item: int(item[0])))
        write_data_to_csv_youngsmodulus(youngmodulus_csv, sorted_mat_thickness_youngmodulus, phase_num)

    logging.info(f"最適化後のdatファイル名：{new_dat_file_name}")

    elapsed_time_4 = time.time() - start_time_4
    logging.info(f"最適化処理の結果の解析とdatファイル出力にかかった時間：{str(elapsed_time_4)} [s]")

    start_time_5 = time.time()

    # nastran実行
    if b_do_nastran:
        return_code = run_nastran(new_dat_file_name, nastran_exe_path)
        if return_code != 0:
            return -1

    elapsed_time_5 = time.time() - start_time_5
    logging.info(f"nastran解析にかかった時間：{str(elapsed_time_5)} [s]")

    logging.info(f"{phase_num}回目の最適化処理に成功しました。")
    print(f"{phase_num}回目の最適化処理に成功しました。")
    logging.info("\n")

    return b_fin_optimize

if __name__ == '__main__':
    b_is_set_sys_argv_on_program = om.get_from_flag_data_list(1)
    if b_is_set_sys_argv_on_program:
            sys.argv = [
                "cae_optimize_nastran.py", 
                # "C:\\work\\github\\q-annealing-d-wave-test\\test2-shell2.dat",
                "C:\\work\\github\\q-annealing-d-wave-test\\test-shell1.dat",
                "notuse",
                "C:\\work\\github\\q-annealing-d-wave-test\\cae_opti_vscode_debug.log",
                "C:\\MSC.Software\\MSC_Nastran\\20122\\bin\\nastranw.exe",
                0.5,  ### target_density
                0.1,  ### density_increment
                2.0,  ### density_power
                4,    ### cost_lambda
                1,   ### loop_num
                1,    ### start_phase_num
                0.1,  ### decide_val_threshold
                0.001,  ### threshold
                0,    ### finish_elem_num
                20000,  ### upper_limit_of_stress
                0,  ### use_thickness_flag
                "sa-100000",  ### "d-wave", "qiskit", "sa-{num of loop}"(ex. "sa-1000000")
            ]
    setup_logging(sys.argv[3])
    logging.info("\n\n")
    if len(sys.argv) <= 17:
        logging.info(
            "Usage: python cae_optimize_nastran.py <dat_file_path> <f06_file_path> <log_file_path> <target_density> <density_increment> ... (need 17 arguments)"
            )
        logging.error(
            "Please check arguments!!"
            )
    logging.info("* * * * * * * * * * 最適化プログラムを開始します * * * * * * * * * *")
    print("* * * * * * * * * * 最適化プログラムを開始します * * * * * * * * * *")
    logging.info(f"コマンドライン引数: {sys.argv}")
    print(f"コマンドライン引数: {sys.argv}")
    main()