import xml.etree.ElementTree as ET
import base64
import struct
import csv
import io
import sys
from typing import List, Dict, Any


def parse_particle_tracks_xml(xml_content: str) -> List[Dict[str, Any]]:
    """
    解析粒子轨迹XML数据

    Args:
        xml_content: XML字符串内容

    Returns:
        包含所有轨迹数据的字典列表
    """
    print("开始解析XML文档...")

    try:
        # 解析XML
        root = ET.fromstring(xml_content)
        print("✓ XML解析成功")
    except Exception as e:
        print(f"✗ XML解析失败: {e}")
        return []

    # 解析数据项定义
    print("\n正在解析数据项定义...")
    items = {}
    item_count = 0
    for item in root.find('Items'):
        item_id = item.get('id')
        item_name = item.get('name')
        item_type = item.get('type')
        items[item_id] = {
            'name': item_name,
            'type': item_type,
            'units': item.get('units', ''),
            'component': item.get('component', ''),
            'componentOf': item.get('componentOf', '')
        }
        print(f"  ✓ 数据项 {item_id}: {item_name} ({item_type})")
        item_count += 1
    print(f"总共解析了 {item_count} 个数据项")

    all_tracks = []
    total_points = 0

    # 解析所有Section
    print("\n开始解析轨迹数据部分...")
    sections = root.find('Tracks')
    if sections is None:
        print("✗ 未找到轨迹数据部分")
        return []

    section_count = len(list(sections))
    print(f"找到 {section_count} 个Section")

    section_num = 0
    for section in sections:
        section_id = section.get('id')
        length = int(section.get('length'))
        print(f"\n--- 解析 Section {section_id} ---")
        print(f"Section 长度: {length} 个数据点")
        section_num += 1

        section_data = {}
        data_item_count = 0

        # 解析每个数据项
        print(f"正在解析Section中的 {len(section)} 个数据项...")
        for data in section:
            item_id = data.get('item')
            constant = data.get('constant') == 'true'
            data_format = data.get('dataFormat')
            minimum = data.get('minimum', 'N/A')
            maximum = data.get('maximum', 'N/A')

            item_info = items.get(item_id, {})
            item_name = item_info.get('name', f"Unknown_{item_id}")
            item_type = item_info.get('type', 'UNKNOWN')

            # 获取数据文本
            data_text = data.text.strip() if data.text else ""
            data_length = len(data_text)

            print(f"  ✓ 数据项 {item_id} ({item_name}):")
            print(f"     类型: {item_type}, 格式: {data_format}, 常量: {constant}")
            print(f"     数据长度: {data_length} 字符, 范围: [{minimum}, {maximum}]")

            if data_format == 'ASCII':
                # ASCII格式数据
                values = [float(x) if '.' in x else int(x) for x in data_text.split()]
                print(f"     ASCII数据解析完成，共 {len(values)} 个值")

            elif data_format == 'Base64/LE':
                # Base64编码的小端字节序数据
                try:
                    binary_data = base64.b64decode(data_text)
                    print(f"     Base64解码完成，二进制长度: {len(binary_data)} 字节")

                    if item_type == 'FLOAT':
                        # 32位浮点数
                        values = list(struct.unpack(f'<{len(binary_data) // 4}f', binary_data))
                        print(f"     解析为 {len(values)} 个FLOAT值")
                    elif item_type == 'INTEGER32':
                        # 32位整数
                        values = list(struct.unpack(f'<{len(binary_data) // 4}i', binary_data))
                        print(f"     解析为 {len(values)} 个INTEGER32值")
                    elif item_type == 'OPTION':
                        # 选项（作为整数处理）
                        values = list(struct.unpack(f'<{len(binary_data) // 4}i', binary_data))
                        print(f"     解析为 {len(values)} 个OPTION值")
                    else:
                        # 其他类型，默认按浮点数处理
                        values = list(struct.unpack(f'<{len(binary_data) // 4}f', binary_data))
                        print(f"     默认解析为 {len(values)} 个FLOAT值")
                except Exception as e:
                    print(f"     ✗ Base64解码失败: {e}")
                    values = []
            else:
                # 未知格式
                print(f"     ⚠ 未知数据格式: {data_format}")
                values = []

            # 如果是常量，扩展为数组
            if constant and len(values) == 1 and length > 1:
                print(f"     常量数据，扩展为 {length} 个相同的值: {values[0]}")
                values = values * length
            elif constant:
                print(f"     常量数据: {values}")

            if len(values) != length:
                print(f"     ⚠ 数据长度不匹配: 期望 {length}, 实际 {len(values)}")

            section_data[item_id] = values
            data_item_count += 1

        print(f"Section {section_id} 数据项解析完成，共 {data_item_count} 个数据项")

        # 将数据重组为每行一个数据点
        print(f"正在重组数据为 {length} 行...")
        for i in range(length):
            track_point = {'Section': section_id, 'PointIndex': i}

            for item_id, item_info in items.items():
                if item_id in section_data and i < len(section_data[item_id]):
                    value = section_data[item_id][i]

                    # 根据数据项类型处理
                    if items[item_id]['type'] == 'OPTION':
                        # 对于OPTION类型，可以保留原始值
                        track_point[item_info['name']] = int(value)
                    elif isinstance(value, float):
                        track_point[item_info['name']] = round(value, 6)  # 限制小数位数
                    else:
                        track_point[item_info['name']] = value
                else:
                    track_point[item_info['name']] = None

            all_tracks.append(track_point)

            # 显示进度
            if (i + 1) % 100 == 0:  # 每100个点显示一次进度
                print(f"    已处理 {i + 1}/{length} 个数据点...")

        total_points += length
        print(f"✓ Section {section_id} 处理完成，添加了 {length} 个数据点")

    print(f"\n所有Section解析完成!")
    print(f"总计解析了 {section_count} 个Section，{total_points} 个数据点")
    return all_tracks


def save_to_csv(tracks_data: List[Dict[str, Any]], csv_filename: str):
    """
    将轨迹数据保存到CSV文件

    Args:
        tracks_data: 轨迹数据列表
        csv_filename: CSV文件名
    """
    if not tracks_data:
        print("没有数据可保存")
        return

    print(f"\n开始保存数据到CSV文件: {csv_filename}")

    # 获取所有列名
    fieldnames = list(tracks_data[0].keys())
    print(f"CSV将包含 {len(fieldnames)} 列:")
    for i, col in enumerate(fieldnames[:10]):  # 只显示前10列
        print(f"  {i + 1}. {col}")
    if len(fieldnames) > 10:
        print(f"  ... 以及 {len(fieldnames) - 10} 个其他列")

    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print("CSV文件头已写入")

        total_rows = len(tracks_data)
        for i, track in enumerate(tracks_data):
            writer.writerow(track)

            # 显示进度
            if (i + 1) % 1000 == 0 or i == total_rows - 1:
                percentage = (i + 1) / total_rows * 100
                print(f"  已写入 {i + 1}/{total_rows} 行 ({percentage:.1f}%)")

    print(f"\n✓ 数据已成功保存到 {csv_filename}")
    print(f"总数据点数量: {len(tracks_data)}")


def get_region_mapping(tracks_data: List[Dict[str, Any]]) -> Dict[int, str]:
    """
    从数据中提取区域映射关系
    """
    print("\n正在分析区域映射关系...")
    region_mapping = {}
    region_counts = {}

    for track in tracks_data:
        region_id = track.get('Region')
        if region_id is not None:
            if region_id not in region_mapping:
                region_mapping[int(region_id)] = f"Region_{region_id}"
            region_counts[region_id] = region_counts.get(region_id, 0) + 1

    if region_mapping:
        print(f"检测到 {len(region_mapping)} 个不同区域:")
        for region_id, region_name in sorted(region_mapping.items()):
            count = region_counts.get(region_id, 0)
            print(f"  Region {region_id}: {count} 个数据点")

    return region_mapping


def analyze_data_statistics(tracks_data: List[Dict[str, Any]]):
    """
    分析数据统计信息
    """
    print("\n正在分析数据统计信息...")

    if not tracks_data:
        print("没有数据可分析")
        return

    # 统计Section信息
    sections = {}
    for track in tracks_data:
        section = track['Section']
        sections[section] = sections.get(section, 0) + 1

    print("Section统计:")
    for section, count in sorted(sections.items(), key=lambda x: int(x[0])):
        print(f"  Section {section}: {count} 个数据点")

    # 统计列信息
    if tracks_data:
        sample_track = tracks_data[0]
        print(f"\n数据列信息 (共 {len(sample_track)} 列):")
        for i, (col_name, value) in enumerate(list(sample_track.items())[:15]):  # 只显示前15列
            col_type = type(value).__name__
            print(f"  {i + 1}. {col_name}: {col_type}")

        if len(sample_track) > 15:
            print(f"  ... 以及 {len(sample_track) - 15} 个其他列")

    # 分析数值范围
    print("\n数值范围分析:")
    numeric_columns = ['Particle X Position', 'Particle Y Position', 'Particle Z Position', 'COLORBY']

    for col in numeric_columns:
        if col in sample_track:
            values = [track[col] for track in tracks_data if track[col] is not None]
            if values:
                min_val = min(values)
                max_val = max(values)
                avg_val = sum(values) / len(values)
                print(f"  {col}:")
                print(f"    最小值: {min_val:.6f}")
                print(f"    最大值: {max_val:.6f}")
                print(f"    平均值: {avg_val:.6f}")


def main():
    """
    主函数：从标准输入读取XML，解析并保存为CSV
    """
    print("=" * 60)
    print("粒子轨迹XML数据解析工具")
    print("=" * 60)

    # 从标准输入读取XML内容
    print("\n正在读取XML数据...")

    # 如果从文件读取，可以使用以下代码：
    with open('particle-track.xml', 'r', encoding='utf-8') as f:
        xml_content = f.read()

    # # 这里我们使用传入的XML内容
    # xml_content = sys.stdin.read()

    if not xml_content:
        print("错误：没有读取到XML内容")
        print("请通过标准输入提供XML数据，例如: python script.py < input.xml")
        return

    print("XML数据读取完成，开始解析...")

    try:
        # 解析XML数据
        tracks_data = parse_particle_tracks_xml(xml_content)

        if not tracks_data:
            print("错误：没有解析到数据")
            return

        print(f"\n✓ 解析完成! 共获取 {len(tracks_data)} 个数据点")

        # 分析数据统计信息
        analyze_data_statistics(tracks_data)

        # 提取区域映射
        region_mapping = get_region_mapping(tracks_data)

        # 保存到CSV文件
        csv_filename = "particle_tracks.csv"
        save_to_csv(tracks_data, csv_filename)

        # 显示前几个数据点作为示例
        print(f"\n前3个数据点示例:")
        for i, track in enumerate(tracks_data[:3]):
            print(f"\n数据点 {i} (Section {track['Section']}, 索引 {track['PointIndex']}):")
            # 只显示主要数据列
            main_columns = ['Section', 'PointIndex', 'Injection', 'Particle ID', 'Particle Time',
                            'Region', 'Periodic Side', 'Particle X Position', 'Particle Y Position',
                            'Particle Z Position', 'COLORBY']
            for col in main_columns:
                if col in track:
                    value = track[col]
                    if isinstance(value, float):
                        print(f"  {col}: {value:.6f}")
                    else:
                        print(f"  {col}: {value}")

            # 显示前3个数据列后换行
            if i < 2:
                print("-" * 40)

        print(f"\n前3个数据点已显示，完整数据请查看 {csv_filename}")

        print(f"\n{'=' * 60}")
        print("解析完成!")
        print(f"输出文件: {csv_filename}")
        print(f"数据总量: {len(tracks_data)} 行")
        print("=" * 60)

    except ET.ParseError as e:
        print(f"\n✗ XML解析错误: {e}")
    except Exception as e:
        print(f"\n✗ 处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()