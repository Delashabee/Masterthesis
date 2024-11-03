# Name: Jiaming Yu
# Time:
def scale_coordinates(input_file, output_file, scale_factor):
    coordinates = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        # 去除空行和换行符
        lines = [line.strip() for line in lines if line.strip()]
        # 将字符串转换为浮点数
        numbers = [float(line) for line in lines]
        # 每三个数字组成一个坐标
        coordinates = [numbers[i:i+3] for i in range(0, len(numbers), 3)]
    # 放大坐标
    scaled_coordinates = []
    for coord in coordinates:
        scaled_coord = [x * scale_factor for x in coord]
        scaled_coordinates.append(scaled_coord)
    # 将结果写入输出文件
    with open(output_file, 'w') as f:
        for coord in scaled_coordinates:
            for value in coord:
                f.write(f"{value}\n")

# 使用示例
scale_coordinates('D:/Programfiles/Myscvp/points_on_sphere/pack.3.40.txt', 'D:/Programfiles/Myscvp/points_on_sphere/pack.3.40_r30.txt', 30)
