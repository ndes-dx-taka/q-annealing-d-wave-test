# flag_data_listの説明：[b_do_optimize, b_is_set_sys_argv_on_program, b_erase_temp_file, b_do_nastran, b_use_thickness]
# 開発ネットワークでの連続実行では下記。
flag_data_list = [True, False, True, True, True]
# nastranがない場合のデバッグ時は下記
# flag_data_list = [True, True, False, False]
# flag_data_list = [True, False, False, False]
# 現在使用中
flag_data_list = [False, False, True, False, True]

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
# from dwave.system.samplers import DWaveSampler
# from dwave.system.composites import EmbeddingComposite
from dwave.system import LeapHybridSampler
import numpy as np
import os
import pprint
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
    
    def get_from_flag_data_list(self, index):
        return self._flag_data_list[index]
    
    def add_to_thickness_youngsmodulus_data_dict(self, key, value):
        self._thickness_youngsmodulus_data_dict[key] = value

    def get_from_thickness_youngsmodulus_data_dict(self, key):
        return self._thickness_youngsmodulus_data_dict.get(key, None)
    
    def write_thickness_youngsmodulus_data_to_csv(self, csvpath):
        with open(csvpath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            for key, value in self._thickness_youngsmodulus_data_dict.items():
                b_thickness = self.get_from_flag_data_list(4)
                value_format = format_float_thickness(value) if b_thickness else format_float_youngmodulus(value)
                writer.writerow([key, str(value_format)])

    def load_thickness_youngsmodulus_data_from_csv(self, csvpath):
        if len(self._thickness_youngsmodulus_data_dict) == 0:
            with open(csvpath, 'r') as csvfile:
                reader = csv.reader(csvfile)

                for row in reader:
                    if len(row) == 2:
                        key, value = row
                        self._thickness_youngsmodulus_data_dict[int(key)] = float(value)
            logging.info(f"Data loaded from {csvpath} into _thickness_youngsmodulus_data_dict.")
        else:
            logging.info("_thickness_youngsmodulus_data_dict is not empty, skipping load.")

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
    cquad4_ctria3_cards_dict = {}
    pshell_card_dict = {}
    mat1_card_dict = {}
    cquad4_ctria3_pshell_id_dict = {}
    pshell_mat1_id_dict = {}
    elem_type_dict = {}  # CQUAD4: 0,  CTRIA3: 1

    base_name, ext = os.path.splitext(input_file_path)

    content = get_file_content(input_file_path)

    for line in content:
        if line.startswith('CQUAD4') or line.startswith('CTRIA3'):
            cquad4_ctria3_id = int(line[8:16].strip())
            pshell_id = int(line[16:24].strip())
            cquad4_ctria3_cards_dict[cquad4_ctria3_id] = line.strip()
            cquad4_ctria3_pshell_id_dict[cquad4_ctria3_id] = pshell_id
            if line.startswith('CQUAD4'):
                elem_type_dict[cquad4_ctria3_id] = int(0)
            if line.startswith('CTRIA3'):
                elem_type_dict[cquad4_ctria3_id] = int(1)
        elif line.startswith('PSHELL'):
            pshell_id = int(line[8:16].strip())
            mat_id = int(line[16:24].strip())
            pshell_mat1_id_dict[pshell_id] = mat_id
            pshell_card_dict[pshell_id] = line.strip()
        elif line.startswith('MAT1'):
            mat_id = int(line[8:16].strip())
            mat1_card_dict[mat_id] = line.strip()

    new_content = []

    for cquad4_ctria3_id, cquad4_ctria3_card in cquad4_ctria3_cards_dict.items():
        pshell_id = cquad4_ctria3_pshell_id_dict[cquad4_ctria3_id]
        mat_id = pshell_mat1_id_dict[pshell_id]
        pshell_card = pshell_card_dict[pshell_id]
        mat1_card = mat1_card_dict[mat_id]
        thickness = float(pshell_card[24:32].strip())
        bending_rigidity_str = pshell_card[40:48].strip()
        bending_rigidity = "" if bending_rigidity_str == "" else float(bending_rigidity_str)
        new_pshell = (
            format_field_left('PSHELL', 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            format_field_right(thickness, 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            format_field_right(bending_rigidity, 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            pshell_card[56:]
        )
        new_mat1 = (
            format_field_left('MAT1', 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            mat1_card[16:]
        )
        elem_type_id = elem_type_dict[cquad4_ctria3_id]
        elem_type_str = None
        if elem_type_id == 0:
            elem_type_str = 'CQUAD4'
        if elem_type_id == 1:
            elem_type_str = 'CTRIA3'
        new_cquad4_ctria3 = (
            format_field_left(elem_type_str, 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            format_field_right(cquad4_ctria3_id, 8) +
            cquad4_ctria3_card[24:]
        )

        new_content.append(new_pshell)
        new_content.append(new_mat1)
        new_content.append(new_cquad4_ctria3)

        youngsmodulus = float(mat1_card[16:24].strip())
        b_use_thickness = om.get_from_flag_data_list(4)
        om.add_to_thickness_youngsmodulus_data_dict(cquad4_ctria3_id, (thickness if b_use_thickness else youngsmodulus))

    reserve_data_csv = base_name + '_reserve_data_for_single_opt.csv'
    rename_old_filename(reserve_data_csv)
    om.write_thickness_youngsmodulus_data_to_csv(reserve_data_csv)

    rename_old_filename(output_file_path)
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for line in content:
            if not line.startswith(('CQUAD4', 'CTRIA3', 'PSHELL', 'MAT1', 'ENDDATA')):
                file.write(line)
        for line in new_content:
            file.write(line + '\n')
        file.write("ENDDATA\n")

    pshell_cquad4_ctria3_id_dict = {}
    mat1_cquad4_ctria3_id_dict = {}
    for cquad4_ctria3_id, pshell_id in cquad4_ctria3_pshell_id_dict.items():
        if pshell_id in pshell_cquad4_ctria3_id_dict:
            pshell_cquad4_ctria3_id_dict[pshell_id].append(cquad4_ctria3_id)
            mat_id = pshell_mat1_id_dict[pshell_id]
            if mat_id in mat1_cquad4_ctria3_id_dict:
                mat1_cquad4_ctria3_id_dict[mat_id].append(cquad4_ctria3_id)
            else:
                new_list = []
                new_list.append(cquad4_ctria3_id)
                mat1_cquad4_ctria3_id_dict[pshell_id] = new_list
        else:
            new_list = []
            new_list.append(cquad4_ctria3_id)
            pshell_cquad4_ctria3_id_dict[pshell_id] = new_list
            mat_id = pshell_mat1_id_dict[pshell_id]
            if mat_id in mat1_cquad4_ctria3_id_dict:
                mat1_cquad4_ctria3_id_dict[mat_id].append(cquad4_ctria3_id)
            else:
                new_list = []
                new_list.append(cquad4_ctria3_id)
                mat1_cquad4_ctria3_id_dict[pshell_id] = new_list

    output_csv = base_name + '.csv'
    rename_old_filename(output_csv)
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['PSHELL information'])
        for key, value in pshell_card_dict.items():
            writer.writerow([key,
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
        writer.writerow(['Elements number corresponding to PSHELL'])
        for pshell_id, elements in pshell_cquad4_ctria3_id_dict.items():
            grouped_elements = 'group : ' + group_elements_for_csv(elements)
            writer.writerow([pshell_id, grouped_elements])
        writer.writerow([])
        writer.writerow(['Elements number corresponding to MAT1'])
        for mat1_id, elements in mat1_cquad4_ctria3_id_dict.items():
            grouped_elements = 'group : ' + group_elements_for_csv(elements)
            writer.writerow([mat1_id, grouped_elements])
        

def extract_stress_values(filename):
    lines = get_file_content(filename)

    stress_data = {}
    start_reading = False

    for i in range(len(lines) - 1):
        line = lines[i]
        next_line = lines[i + 1]

        if (re.search(r'S T R E S S E S\s+I N\s+Q U A D R I L A T E R A L\s+E L E M E N T S\s+\( Q U A D 4 \)', line) or
            re.search(r'S T R E S S E S\s+I N\s+T R I A N G U L A R\s+E L E M E N T S\s+\( T R I A 3 \)', line)):
            start_reading = True
            continue
        
        # if re.search(r'MSC NASTRAN.+PAGE.+', line):
        if re.search(r'.+MSC.NASTRAN.+PAGE.+', line):
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

            stress_part_1 = (pow(normal_x1, 2.0) + pow(normal_y1, 2.0) + pow(normal_x2, 2.0) + pow(normal_y2, 2.0)) / 2.0
            stress_part_2 = normal_x1 * normal_y1 + normal_x2 * normal_y2
            stress_part_3 = shear_xy1 + shear_xy2

            von_mises_1 = float(line[117:131].strip())
            von_mises_2 = float(next_line[117:131].strip())

            stress_data[element_id] = (stress_part_1, stress_part_2, stress_part_3, max(von_mises_1, von_mises_2))

    logging.debug(f"{filename}における、辞書stress_dataのデータ内容:\n" + pprint.pformat(stress_data, indent=4))

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

    b_use_thickness = om.get_from_flag_data_list(4)
    node_dict = {}
    pshell_dict = {}
    mat1_dict = {}
    cquad4_ctria3_dict = {}
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
                x = float(line[24:32].strip())
                y = float(line[32:40].strip())
                z = float(line[40:48].strip())
                node_dict[grid_id] = [x, y, z]
                all_node_id_set.add(grid_id)
            if line.startswith('PSHELL'):
                elem_id = int(line[8:16].strip())
                mat_id = int(line[16:24].strip())
                thickness = float(line[24:32].strip())
                initial_thickness_temp = om.get_from_thickness_youngsmodulus_data_dict(elem_id)
                b_skip_optimize_pshell = False
                if b_use_thickness:
                    if check_skip_optimize(thickness, initial_thickness_temp, threshold, phase_num):
                        b_skip_optimize_pshell = True
                if not b_skip_optimize_pshell:
                    pshell_dict[elem_id] = [mat_id, thickness]
            if line.startswith('MAT1'):
                mat_id = int(line[8:16].strip())
                youngmodulus = float(line[16:24].strip())
                initial_youngsmodulus_temp = om.get_from_thickness_youngsmodulus_data_dict(mat_id)
                b_skip_optimize_mat1 = False
                if not b_use_thickness:
                    if check_skip_optimize(youngmodulus, initial_youngsmodulus_temp, threshold, phase_num):
                        b_skip_optimize_mat1 = True
                if b_skip_optimize_mat1:
                    del pshell_dict[mat_id]
                else:
                    poissonratio = float(line[32:40].strip())
                    value = [youngmodulus, poissonratio]
                    mat1_dict[mat_id] = value
            if line.startswith('CQUAD4') or line.startswith('CTRIA3'):
            # if line.startswith('CQUAD4'):
                elem_id = int(line[8:16].strip())
                if elem_id in pshell_dict:
                    value = None
                    x1 = int(line[24:32].strip())
                    x2 = int(line[32:40].strip())
                    x3 = int(line[40:48].strip())
                    if line.startswith('CQUAD4'):
                        x4 = int(line[48:56].strip())
                        value = [x1, x2, x3, x4]
                    if line.startswith('CTRIA3'):
                        value = [x1, x2, x3]
                    b_is_remain_elem_forcely = False
                    for x in value:
                        count = force_grid_list.count(x)
                        if count > 0:
                            logging.debug(f"要素{elem_id}は節点{x}を持ち、この節点はFORCEカードで荷重がかかっているため、最適化の対象から外します（残すことは確定）。")
                            b_is_remain_elem_forcely = True
                            break
                    if b_is_remain_elem_forcely:
                        del pshell_dict[elem_id]
                        del mat1_dict[elem_id]
                    else:
                        cquad4_ctria3_dict[elem_id] = value

    input_f06_file_name = "{}_phase_{}.f06".format(str(f06_file_path)[:-4], phase_num - 1)
    if phase_num == 1:
        shutil.copy(f06_file_path, input_f06_file_name)
    stress_dict = extract_stress_values(input_f06_file_name)

    merged_elem_list = []
    upper_limit_of_stress = float(sys.argv[14])
    for pshell_key, value in pshell_dict.items():
        eid = pshell_key
        stress_value = stress_dict[eid]
        von_mises = stress_value[3]
        if von_mises > upper_limit_of_stress:
            logging.debug(f"eid={eid}の要素のvon mises応力値{von_mises}が、基準値として指定した{upper_limit_of_stress}を超えたためにこの要素の最適化をスキップします。")
            continue

        thickness = value[1]
        if b_use_thickness:
            initial_thickness_check = om.get_from_thickness_youngsmodulus_data_dict(eid)
            if initial_thickness_check >= threshold:
                if (thickness / initial_thickness_check) < (decide_val_threshold + threshold):
                    continue

        merged_dict = {}
        merged_dict['eid'] = eid
    
        merged_dict['stress_part_1'] = stress_value[0]
        merged_dict['stress_part_2'] = stress_value[1]
        merged_dict['stress_part_3'] = stress_value[2]
        merged_dict['von_mises'] = von_mises

        cquad4_ctria3_value = cquad4_ctria3_dict[eid]
        merged_dict['nodes'] = cquad4_ctria3_value
        mat1_value = mat1_dict[value[0]]
        merged_dict['thickness'] = thickness
        merged_dict['youngsmodulus'] = mat1_value[0]
        merged_dict['poissonratio'] = mat1_value[1]

        node_data = []
        for nid in cquad4_ctria3_value:
            node_value = node_dict[nid]
            node_value_dict = {}
            node_value_dict['nid'] = nid
            node_value_dict['x'] = node_value[0]
            node_value_dict['y'] = node_value[1]
            node_value_dict['z'] = node_value[2]
            node_data.append(node_value_dict)
        merged_dict['node_data'] = node_data
        area = calculate_area(node_data)
        merged_dict['area'] = area
        merged_dict['volume'] = area * thickness

        merged_elem_list.append(merged_dict)
    
    elapsed_time_1 = time.time() - start_time_1
    logging.info(f"入力データの読み込みにかかった時間：{str(elapsed_time_1)} [s]")

    start_time_2 = time.time()

    energy_list_for_scale = []
    volume_list_for_scale = []

    # b_use_thicknessの時は、density_power=0とした相当の動作をするケースのための変数
    density_power_calc = 0 if b_use_thickness else density_power

    first_thickness_density_percentage = target_thickness_density_percentage
    ising_index_eid_map = {}
    nInternalid = len(merged_elem_list)
    h = defaultdict(int)
    J = defaultdict(int)
    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        stress_part_1 = elem.get('stress_part_1', 0)
        stress_part_2 = elem.get('stress_part_2', 0)
        stress_part_3 = elem.get('stress_part_3', 0)
        poissonratio = float(elem.get('poissonratio', 0))
        thickness = float(elem.get('thickness', 0))
        youngsmodulus = float(elem.get('youngsmodulus', 0))
        volume = float(elem.get('volume', 0))

        ising_index_eid_map[eid] = index

        thickness_density_percentage_now = 0.0
        initial_thickness_youngsmodulus = om.get_from_thickness_youngsmodulus_data_dict(eid)
        if phase_num == 1:
            thickness_density_percentage_now = first_thickness_density_percentage
        else:
            thickness_youngmodulus = thickness if b_use_thickness else youngsmodulus
            thickness_density_percentage_now = calculate_thickness_density_percentage(thickness_youngmodulus, initial_thickness_youngsmodulus, density_power, b_use_thickness, threshold)

        density_plus_delta = thickness_density_percentage_now + density_increment
        density_minus_delta = thickness_density_percentage_now - density_increment

        alpha_value = pow(density_plus_delta, (1 - density_power_calc))
        beta_value = pow(density_minus_delta, (1 - density_power_calc))

        # kappa_i = (pow(stressxx, 2.0) - 2.0 * poissonratio * stressxx * stressyy + pow(stressyy, 2.0) + 2.0 * (1.0 + poissonratio) * pow(stressxy, 2.0)) * volume / initial_thickness_youngsmodulus
        kappa_i = (stress_part_1 - poissonratio * stress_part_2 + (1.0 + poissonratio) * stress_part_3) * volume / initial_thickness_youngsmodulus
        
        energy_list_for_scale.append(alpha_value * kappa_i)
        energy_list_for_scale.append(beta_value * kappa_i)

        volume_list_for_scale.append(volume)
        volume_list_for_scale.append(-1.0 * volume)

    std_of_energy_list = np.std(energy_list_for_scale)
    std_of_volume_list = np.std(volume_list_for_scale)

    for index, elem in enumerate(merged_elem_list):
        eid = int(elem.get('eid', 0))
        stress_part_1 = elem.get('stress_part_1', 0)
        stress_part_2 = elem.get('stress_part_2', 0)
        stress_part_3 = elem.get('stress_part_3', 0)
        poissonratio = float(elem.get('poissonratio', 0))
        thickness = float(elem.get('thickness', 0))
        youngsmodulus = float(elem.get('youngsmodulus', 0))
        volume = float(elem.get('volume', 0))

        ising_index_eid_map[eid] = index

        thickness_density_percentage_now = 0.0
        initial_thickness_youngsmodulus = om.get_from_thickness_youngsmodulus_data_dict(eid)
        if phase_num == 1:
            thickness_density_percentage_now = first_thickness_density_percentage
        else:
            thickness_youngmodulus = thickness if b_use_thickness else youngsmodulus
            thickness_density_percentage_now = calculate_thickness_density_percentage(thickness_youngmodulus, initial_thickness_youngsmodulus, density_power, b_use_thickness, threshold)

        density_plus_delta = thickness_density_percentage_now + density_increment
        density_minus_delta = thickness_density_percentage_now - density_increment

        alpha_value = pow(density_plus_delta, (1 - density_power_calc))
        beta_value = pow(density_minus_delta, (1 - density_power_calc))
        k_0 = (alpha_value - beta_value) / 2.0
        # kappa_i = (pow(stressxx, 2.0) - 2.0 * poissonratio * stressxx * stressyy + pow(stressyy, 2.0) + 2.0 * (1.0 + poissonratio) * pow(stressxy, 2.0)) * volume / initial_thickness_youngsmodulus
        kappa_i = (stress_part_1 - poissonratio * stress_part_2 + (1.0 + poissonratio) * stress_part_3) * volume / initial_thickness_youngsmodulus

        h_first = k_0 * kappa_i / std_of_energy_list / np.sqrt(nInternalid) / 3.0
        h[index] = h_first

        for j_index in range(index + 1, nInternalid):
            volume_j = merged_elem_list[j_index].get('volume', 0)
            J[(index,j_index)] = 2.0 * cost_lambda * volume * volume_j / pow(std_of_volume_list, 2) / nInternalid / 9.0

    elapsed_time_2 = time.time() - start_time_2
    logging.info(f"最適化処理の準備にかかった時間：{str(elapsed_time_2)} [s]")

    logging.info(f"{phase_num}回目の最適化を開始します。")
    print(f"{phase_num}回目の最適化を開始します。")
    print("最適化実行中…")

    start_time_3 = time.time()

    ising_index_dict = {}

    b_do_optimize = om.get_from_flag_data_list(0)
    if b_do_optimize:
        sampler = LeapHybridSampler()
        response = sampler.sample_ising(h, J)

        for sample, E in response.data(fields=['sample','energy']):
            S_minus_1 = [k for k,v in sample.items() if v == -1]
            S_plus_1 = [k for k,v in sample.items() if v == 1]

            for elem in S_minus_1:
                ising_index_dict[elem] = -1

            for elem in S_plus_1:
                ising_index_dict[elem] = 1
    else:
        # テスト用(最適化のリソース節約のため)
        for index, elem in enumerate(merged_elem_list):
            ising_index_dict[index] = (1 if random.random() < 0.5 else -1)

    elapsed_time_3 = time.time() - start_time_3
    logging.info(f"{phase_num}回目の最適化処理の実行と集計にかかった時間：{str(elapsed_time_3)} [s]")
    print(f"{phase_num}回目の最適化が終わりました。")
    print(f"最適化処理の実行と集計にかかった時間：{str(elapsed_time_3)} [s]")

    start_time_4 = time.time()

    mat_thickness_youngmodulus = {}
    zero_fix_density_index_list = []
    b_fin_optimize = 0
    finish_elem_num = int(sys.argv[13])
    if len(merged_elem_list) <= finish_elem_num:
        logging.info(f"最適化が完了していない要素の数が{finish_elem_num}以下になったため、要素の0/1を決定します")
        b_fin_optimize = 1
    
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
    same_pshell_flag = 0
    used_node_id_set = set()
    with open(new_dat_temp_file_name, 'w', encoding='utf-8') as file:
        for line in lines:
            if line.startswith('PSHELL'):
                elem_id = int(line[8:16].strip())
                mat_id = int(line[16:24].strip())
                if elem_id in zero_fix_density_index_list:
                    line = f"${line}"
                    same_pshell_flag = 1
                else:
                    if b_use_thickness:
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
                if same_pshell_flag == 1:
                    same_pshell_flag = 2
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
            # if line.startswith('CQUAD4'):
            if line.startswith('CQUAD4') or line.startswith('CTRIA3'):
                if same_pshell_flag == 2:
                    same_pshell_flag = 0
                    line = f"${line}"
                else:
                    x1 = int(line[24:32].strip())
                    x2 = int(line[32:40].strip())
                    x3 = int(line[40:48].strip())
                    used_node_id_set.add(x1)
                    used_node_id_set.add(x2)
                    used_node_id_set.add(x3)
                    if line.startswith('CQUAD4'):
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
                "C:\\work\\github\\q-annealing-d-wave-test\\check.dat",
                "C:\\work\\github\\q-annealing-d-wave-test\\check.f06",
                "C:\\work\\github\\q-annealing-d-wave-test\\cae_opti_vscode_debug.log",
                "C:\\MSC.Software\\MSC_Nastran\\20122\\bin\\nastranw.exe",
                0.5,  ### target_density
                0.1,  ### density_increment
                2.0,  ### density_power
                5,    ### cost_lambda
                15,   ### loop_num
                1,    ### start_phase_num
                0.1,  ### decide_val_threshold
                0.001,  ### threshold
                0,    ### finish_elem_num
                20,  ### upper_limit_of_stress
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