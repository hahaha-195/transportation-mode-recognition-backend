import json
from collections import Counter

OSM_PATH = "../data/exp3.geojson"

# OSM 真正的 tag 一般都是字符串 → 字符串
def is_valid_tag_value(v):
    return isinstance(v, (str, int, float))

def load_osm(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_osm(osm):
    features = osm.get("features", [])
    tag_counter = Counter()
    type_counter = Counter()
    maxspeed_count = 0

    for feat in features:
        props = feat.get("properties", {})

        # --- 1. 处理第一层标准标签 ---
        for k, v in props.items():
            if k.startswith("@") and k != "@relations":  # 跳过普通元字段，保留 relations 待处理
                continue
            if is_valid_tag_value(v):
                tag_counter[(k, str(v))] += 1
                if k == "maxspeed":
                    maxspeed_count += 1

        # --- 2. 处理嵌套的 @relations (修复核心) ---
        relations = props.get("@relations", [])
        if isinstance(relations, list):
            for rel in relations:
                reltags = rel.get("reltags", {})
                if isinstance(reltags, dict):
                    for rk, rv in reltags.items():
                        if is_valid_tag_value(rv):
                            # 将线路标签也计入总统计
                            tag_counter[(rk, str(rv))] += 1

        # --- 3. 类型统计 ---
        osm_id = props.get("@id", "")
        if isinstance(osm_id, str) and "/" in osm_id:
            osm_type = osm_id.split("/")[0]
            type_counter[osm_type] += 1

    return {
        "type": type_counter,
        "tags": tag_counter,
        "maxspeed_count": maxspeed_count
    }

def print_key_findings(stats):
    print("\n========== 实验三 OSM 数据关键特征检查 ==========\n")

    def show(tag, value):
        count = stats["tags"].get((tag, value), 0)
        print(f"{tag}={value:<25} : {count}")

    print("【OSM 元素类型】")
    for k, v in stats["type"].items():
        print(f"{k:<10} : {v}")

    print("\n【公共交通站点】")
    show("highway", "bus_stop")
    show("railway", "station")
    show("public_transport", "station")
    show("railway", "subway_entrance")

    print("\n【交通辅助 POI】")
    show("amenity", "parking")
    show("amenity", "bicycle_rental")
    show("amenity", "taxi")

    print("\n【道路类型】")
    for hw in ["footway", "cycleway", "primary", "secondary", "tertiary", "residential"]:
        show("highway", hw)

    print("\n【速度信息】")
    print(f"带 maxspeed 的道路数量: {stats['maxspeed_count']}")

    print("\n【线路 Relation】")
    show("route", "bus")
    show("route", "subway")

    print("\n===============================================\n")

if __name__ == "__main__":
    osm = load_osm(OSM_PATH)
    stats = analyze_osm(osm)
    print_key_findings(stats)
