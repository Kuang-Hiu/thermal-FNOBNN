import pyvista as pv
import pandas as pd
import re
import vtk
import numpy as np
import os
import shutil


def nhom_file(folder_path: str) -> None:
    """
    Nhóm các file .vtu trong một thư mục vào các thư mục con theo tiền tố tên file.

    Quy ước nhóm
    ------------
    - Mỗi file .vtu có tên dạng: <tien_to>-phần_còn_lại.vtu
      Ví dụ:
          "Case01-step1.vtu"  -> được nhóm vào thư mục "Case01"
          "A_10-test.vtu"     -> được nhóm vào thư mục "A_10"

    Cách hoạt động
    --------------
    - Duyệt qua tất cả file trong `folder_path`.
    - Với mỗi file kết thúc bằng `.vtu`:
        * Lấy phần trước dấu `-` đầu tiên làm tên thư mục nhóm.
        * Tạo thư mục con tương ứng trong `folder_path` nếu chưa tồn tại.
        * Di chuyển file đó vào thư mục con tương ứng.

    Tham số
    -------
    folder_path : str
        Đường dẫn tới thư mục chứa các file .vtu cần nhóm.

    Ghi chú
    -------
    - Hàm chỉ xử lý trực tiếp các file .vtu trong thư mục `folder_path`
      (không duyệt đệ quy các thư mục con).
    - Nếu file đã nằm đúng thư mục rồi, `shutil.move` vẫn xử lý bình thường
      (nếu cần có thể tự kiểm tra để bỏ qua).
    """
    # Lặp qua tất cả các mục trong thư mục gốc
    for filename in os.listdir(folder_path):
        if not filename.endswith(".vtu"):
            continue  # bỏ qua file không phải .vtu

        # Tên thư mục nhóm lấy phần trước dấu '-' đầu tiên
        folder_name = filename.split('-')[0]

        # Tạo thư mục mới (nếu chưa có)
        new_dir = os.path.join(folder_path, folder_name)
        os.makedirs(new_dir, exist_ok=True)

        # Đường dẫn cũ và mới
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(new_dir, filename)

        # Nếu đường dẫn cũ và mới trùng nhau thì bỏ qua (tránh move lỗi)
        if os.path.abspath(old_path) == os.path.abspath(new_path):
            continue

        shutil.move(old_path, new_path)


def read_file(root):
    """
    Đọc toàn bộ các file .vtu trong thư mục và trích xuất dữ liệu phục vụ huấn luyện mô hình.

    Tham số
    --------
    root : str
        Đường dẫn đến thư mục chứa các file .vtu.

    Chức năng
    ---------
    - Duyệt qua tất cả file .vtu trong thư mục.
    - Trích xuất 8 tham số (a0..a7) từ tên file (dạng số thực hoặc số nguyên).
    - Đọc dữ liệu lưới bằng VTK và PyVista.
    - Lấy tọa độ điểm đầu tiên (x, y, z) của lưới.
    - Tách và lưu dữ liệu theo từng bước thời gian (time step) cho:
        * Temperature_@_t=...
        * Pressure_@_t=...
        * Displacement_field,_X-component_@_t=...
        * Displacement_field,_Y-component_@_t=...
        * Stress_tensor,_x-component_@_t=...
        * Stress_tensor,_y-component_@_t=...
    - Chỉ giữ lại những time step mà đầy đủ cả 6 trường trên.
    - Bỏ qua các giá trị NaN.

    Trả về
    --------
    all_data_X : np.ndarray, shape = (N_files, 11)
        Mỗi dòng: [x, y, z, a0, a1, a2, a3, a4, a5, a6, a7]

    all_data_y : np.ndarray, dtype=object
        Mỗi phần tử tương ứng với một file:
            array(T_i, 6) với T_i là số time step hợp lệ,
            cột tương ứng: [Temperature, Pressure, dx, dy, sx, sy]

    Ghi chú
    -------
    - Hàm chỉ lấy dữ liệu tại điểm lưới đầu tiên.
    - Nếu file bị lỗi (thiếu dữ liệu, đọc thất bại, không đủ tham số trong tên file, v.v.),
      hàm sẽ in thông báo và bỏ qua file đó.
    """
    reader = vtk.vtkXMLUnstructuredGridReader()
    all_data_X = []
    all_data_y = []

    # Duyệt qua toàn bộ file trong thư mục
    for filename in os.listdir(root):
        if not filename.endswith(".vtu"):
            continue  # bỏ qua file không phải .vtu

        file_path = os.path.join(root, filename)

        # Lấy số từ tên file (int hoặc float)
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", filename)
        if len(numbers) < 8:
            print(f"Warning: File {filename} không đủ 8 tham số trong tên (chỉ có {len(numbers)}). Bỏ qua.")
            continue

        try:
            # Chuyển a0..a7 sang float
            a_vals = [float(n) for n in numbers[:8]]
            a0, a1, a2, a3, a4, a5, a6, a7 = a_vals

            # Đọc lưới bằng VTK để lấy tọa độ điểm đầu
            reader.SetFileName(file_path)
            reader.Update()
            points = reader.GetOutput().GetPoints()
            if points is None or points.GetNumberOfPoints() == 0:
                print(f"Warning: File {filename} không có điểm lưới. Bỏ qua.")
                continue
            x, y, z = points.GetPoint(0)

            # Đọc bằng PyVista để lấy point_data
            mesh = pv.read(file_path)

            temperature_data = {}
            pressure_data = {}
            displacement_x_data = {}
            displacement_y_data = {}
            stress_x_data = {}
            stress_y_data = {}

            # Duyệt qua tên các trường trong point_data
            for key in mesh.point_data.keys():
                t_match  = re.search(r"Temperature_@_t=(\d+(?:\.\d+)?)", key)
                p_match  = re.search(r"Pressure_@_t=(\d+(?:\.\d+)?)", key)
                dx_match = re.search(r"Displacement_field,_X-component_@_t=(\d+(?:\.\d+)?)", key)
                dy_match = re.search(r"Displacement_field,_Y-component_@_t=(\d+(?:\.\d+)?)", key)
                sx_match = re.search(r"Stress_tensor,_x-component_@_t=(\d+(?:\.\d+)?)", key)
                sy_match = re.search(r"Stress_tensor,_y-component_@_t=(\d+(?:\.\d+)?)", key)

                if t_match:
                    t = float(t_match.group(1))
                    temperature_data[t] = mesh.point_data[key]
                if p_match:
                    t = float(p_match.group(1))
                    pressure_data[t] = mesh.point_data[key]
                if dx_match:
                    t = float(dx_match.group(1))
                    displacement_x_data[t] = mesh.point_data[key]
                if dy_match:
                    t = float(dy_match.group(1))
                    displacement_y_data[t] = mesh.point_data[key]
                if sx_match:
                    t = float(sx_match.group(1))
                    stress_x_data[t] = mesh.point_data[key]
                if sy_match:
                    t = float(sy_match.group(1))
                    stress_y_data[t] = mesh.point_data[key]

            # Lấy giao các time step có đủ cả 6 trường
            common_time_steps = (
                set(temperature_data.keys())
                & set(pressure_data.keys())
                & set(displacement_x_data.keys())
                & set(displacement_y_data.keys())
                & set(stress_x_data.keys())
                & set(stress_y_data.keys())
            )

            if not common_time_steps:
                print(f"Warning: File {filename} không có time step nào đủ 6 trường. Bỏ qua.")
                continue

            time_steps = sorted(common_time_steps)

            # Lưu X: tọa độ + tham số từ tên file
            all_data_X.append([x, y, z, a0, a1, a2, a3, a4, a5, a6, a7])

            # Lưu y: [T, P, dx, dy, sx, sy] theo từng time step
            item_value = []
            for t in time_steps:
                T  = temperature_data[t][0]
                P  = pressure_data[t][0]
                dx = displacement_x_data[t][0]
                dy = displacement_y_data[t][0]
                sx = stress_x_data[t][0]
                sy = stress_y_data[t][0]

                # Bỏ NaN
                if not (pd.isna(T) or pd.isna(P) or pd.isna(dx) or pd.isna(dy) or pd.isna(sx) or pd.isna(sy)):
                    item_value.append([T, P, dx, dy, sx, sy])

            if len(item_value) == 0:
                print(f"Warning: File {filename} toàn bộ dữ liệu bị NaN. Bỏ qua.")
                # X vừa append ở trên, nếu không muốn giữ thì có thể pop() lại:
                all_data_X.pop()
                continue

            all_data_y.append(np.array(item_value))

        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    return np.array(all_data_X), np.array(all_data_y, dtype=object)


def import_vtu_file(outer_forder):
    """
    Input: Đường dẫn tới folder chưa các folder đã phân loại
    Output: X shape: (n_sample, 11) include: (x, y, z, physical_paramerters...)
            y shape: (n_sample, 1915, 6) include: (T, p, u, v, sx, sy)
    """

    #outer_forder = '/content/drive/MyDrive/Linh tinh/tesst'
    all_data_X = []
    all_data_y = []
    for inner_forder in os.listdir(outer_forder):
        X_data, y_data = read_file(os.path.join(outer_forder, inner_forder))
        all_data_X.append(X_data)
        all_data_y.append(y_data)
    return all_data_X, all_data_y