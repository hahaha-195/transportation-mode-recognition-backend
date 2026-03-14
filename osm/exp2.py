import json
from collections import Counter, defaultdict

OSM_PATH = "../data/exp2.geojson"

def load_osm(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_osm(osm):
    elements = osm.get("elements", osm.get("features", []))

    stats = {
        "node": Counter(),
        "way": Counter(),
        "relation": Counter(),
        "tags": Counter(),
        "maxspeed_count": 0
    }

    for elem in elements:
        # 兼容 geojson / overpass json
        etype = elem.get("type", "feature")
        tags = elem.get("tags", elem.get("properties", {}))

        # 类型统计
        if etype in ["node", "way", "relation"]:
            stats[etype][1] += 1

        # 标签统计
        for k, v in tags.items():
            stats["tags"][(k, v)] += 1

        # maxspeed 专项统计
        if "maxspeed" in tags:
            stats["maxspeed_count"] += 1

    return stats

def print_key_findings(stats):
    print("\n========== OSM 数据关键特征检查 ==========\n")

    def show(tag, value):
        count = stats["tags"].get((tag, value), 0)
        print(f"{tag}={value:<25} : {count}")

    print("【公共交通站点】")
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

    print("\n=========================================\n")

if __name__ == "__main__":
    osm = load_osm(OSM_PATH)
    stats = analyze_osm(osm)
    print_key_findings(stats)
