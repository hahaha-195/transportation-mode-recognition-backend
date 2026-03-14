import json
import os
from typing import Dict, Any, List

# --- 配置 ---
ROOT_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = "exp3.geojson"


# --- 结束配置 ---

def extract_osm_properties(element: Dict[str, Any]) -> tuple[str, int]:
    """
    根据用户提供的GeoJSON格式，从 properties 中的 '@id' 键中提取 OSM 的 type 和 id。
    例如：从 '@id': 'node/124205138' 中提取 'node' 和 124205138。
    """
    elem_type = ""
    elem_id = 0

    props = {}
    # 实体类型为 Feature，且包含 properties 字典
    if element.get('type') == 'Feature' and 'properties' in element:
        props = element['properties']

    # 检查核心的关键字段 @id
    at_id_string = props.get('@id')

    if at_id_string and isinstance(at_id_string, str) and '/' in at_id_string:
        try:
            # 格式: "type/id"
            parts = at_id_string.split('/')
            elem_type = parts[0].lower()  # node, way, relation
            elem_id = int(parts[1])  # 唯一数字ID
        except (ValueError, IndexError):
            # ID 无法转换为数字或字符串格式不正确
            return "", 0

    # 最终验证实体类型
    if elem_type in ['node', 'way', 'relation'] and elem_id != 0:
        return elem_type, elem_id

    return "", 0  # 无法识别的实体


def load_and_merge_osm_elements(root_dir: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    递归加载指定目录及其子目录下的所有 GEOJSON/JSON 文件，合并去重，并打印每个文件的贡献。
    """

    # 存储最终合并和去重的实体，结构：{'node': {id: element_data}, 'way': {...}, 'relation': {...}}
    merged_elements = {
        'node': {},
        'way': {},
        'relation': {}
    }

    # 遍历目录及其子目录
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(('.json', '.geojson')):
                file_path = os.path.join(dirpath, filename)

                # 忽略脚本自身和输出文件
                if filename == os.path.basename(__file__) or filename == OUTPUT_FILE:
                    continue

                print(f"\n-> 正在处理文件: {os.path.relpath(file_path, root_dir)}")

                # 初始化当前文件贡献统计
                current_file_stats = {'node': 0, 'way': 0, 'relation': 0, 'total': 0}

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        # 确定实体列表：基于您的文件是 FeatureCollection，列表在 'features' 键下
                        elements_list = data.get('features')

                        if not elements_list:
                            print(f"  警告: 文件 {filename} 缺少 'features' 键或格式异常。贡献记为 0。")
                            continue

                        # 遍历文件中的所有实体
                        for element in elements_list:
                            elem_type, elem_id = extract_osm_properties(element)

                            if elem_type and elem_id != 0:
                                # 记录贡献：如果 ID 在合并字典中是新增的，则计数
                                if elem_id not in merged_elements[elem_type]:
                                    current_file_stats[elem_type] += 1
                                    current_file_stats['total'] += 1

                                # 执行合并/覆盖操作 (字典键值赋值，实现去重)
                                merged_elements[elem_type][elem_id] = element

                    print(
                        f"  贡献新实体数: Node: {current_file_stats['node']}, Way: {current_file_stats['way']}, Relation: {current_file_stats['relation']} (Total: {current_file_stats['total']})")

                except json.JSONDecodeError:
                    print(f"  错误: JSON 解码失败，贡献记为 0。请检查编码或文件完整性。")
                except Exception as e:
                    print(f"  错误: 处理时发生未知错误 ({e})，贡献记为 0。")

    return merged_elements


def compile_final_geojson(merged_elements: Dict[str, Dict[int, Dict[str, Any]]]) -> Dict[str, Any]:
    """
    将合并后的字典结构重新组织成标准的 GEOJSON FeatureCollection 格式。
    """

    final_features_list = []

    # 将 Node, Way, Relation 的字典值合并到最终列表中
    for elem_type in ['node', 'way', 'relation']:
        final_features_list.extend(merged_elements[elem_type].values())

    # 构造最终的 GEOJSON FeatureCollection 结构
    final_data = {
        "type": "FeatureCollection",
        "name": "beijing_osm_full_enhanced",
        "features": final_features_list,
    }

    return final_data


# --- 主执行流程 ---
if __name__ == "__main__":
    print(f"--- 启动 OSM 数据验证与合并脚本 ---")
    print(f"扫描目录: {ROOT_DATA_DIR}")

    # 1. 合并和去重实体
    merged_data_dict = load_and_merge_osm_elements(ROOT_DATA_DIR)

    # 2. 统计最终结果
    total_nodes = len(merged_data_dict['node'])
    total_ways = len(merged_data_dict['way'])
    total_relations = len(merged_data_dict['relation'])
    total_elements = total_nodes + total_ways + total_relations

    print("\n--- 最终合并和去重结果 ---")
    print(f"节点 (Node) 总数: {total_nodes}")
    print(f"路径 (Way) 总数: {total_ways}")
    print(f"关系 (Relation) 总数: {total_relations}")
    print(f"最终 OSM 实体总数: {total_elements} 条")

    # 3. 编译最终 GEOJSON 结构
    final_osm_geojson = compile_final_geojson(merged_data_dict)

    # 4. 写入输出文件
    output_path = os.path.join(ROOT_DATA_DIR, OUTPUT_FILE)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_osm_geojson, f, ensure_ascii=False, indent=2)

    print(f"\n--- 任务完成 ---")
    print(f"验证完成，完整增强数据集已保存到: {output_path}")