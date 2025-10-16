from neo4j import GraphDatabase
from neo4j_viz import Node, Relationship, VisualizationGraph
from neo4j_viz.colors import ColorSpace
from palettable.wesanderson import Moonrise1_5

# --- 连接信息 ---
URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "12345678" # 请替换为您的密码

snomed_ct_top_level_key_to_id = {
    'body structure': 123037004,
    'finding': 404684003,
    'environment / location': 308916002,
    'event': 272379006,
    'observable entity': 363787002,
    'organism': 410607006,
    'product': 373873005,
    'physical force': 78621006,
    'physical object': 260787004,
    'procedure': 71388002,
    'qualifier value': 362981000,
    'record artifact': 419891008,
    'situation': 243796009,
    'metadata': 900000000000441003,
    'social context': 48176007,
    'special concept': 370115009,
    'specimen': 123038009,
    'staging scale': 254291000,
    'substance': 105590001
}

def get_all_nodes_and_relationships_simple(driver):
    """
    一次性获取所有节点和关系。仅适用于小型数据库！
    """
    print("正在尝试一次性获取所有数据（不推荐用于大型图）...")
    all_data = {
        "nodes": [],
        "relationships": []
    }
    
    with driver.session(database="neo4j") as session:
        # 获取所有节点
        nodes_result = session.run("MATCH (n) RETURN n").data()
        all_data["nodes"] = [record for record in nodes_result]
        
        # 获取所有关系
        rels_result = session.run("MATCH ()-[r]->() RETURN r").data()
        all_data["relationships"] = [record for record in rels_result]
        
    return all_data

def get_all_nodes_and_relationships_with_id(driver):
    """
    一次性拉取整张图（仅限小型库），
    返回格式：
      all_data["nodes"] = [{id, labels, props}, ...]
      all_data["relationships"] = [{id, type, source, target, props}, ...]
    """
    all_data = {"nodes": [], "relationships": []}

    with driver.session(database="neo4j") as sess:
        # 节点
        node_cypher = """
        MATCH (n)
        RETURN {
            id:      elementId(n),
            labels:  labels(n),
            props:   properties(n)
        } AS node
        """
        for rec in sess.run(node_cypher):
            all_data["nodes"].append(rec["node"])

        # 关系
        rel_cypher = """
        MATCH (a)-[r]->(b)
        RETURN {
            id:      elementId(r),
            type:    type(r),
            source:  elementId(a),
            target:  elementId(b),
            props:   properties(r)
        } AS rel
        """
        for rec in sess.run(rel_cypher):
            all_data["relationships"].append(rec["rel"])

    return all_data

def build_vg(all_data):
    """
    all_data 结构：
      {"nodes": [ {id, labels, props}, ... ],
       "relationships": [ {id, type, source, target, props}, ... ]}
    """
    nodes = [
        Node(
            id=n["id"], # 直接用 elementId() 字符串
            caption=n["props"].get("name", ""), # 图里显示中文/英文名称
            size=15, # 固定半径；可改成按度数/属性缩放
            properties=n["props"], # 鼠标 hover 时可查看详情
        )
        for n in all_data["nodes"]
    ]

    relationships = [
        Relationship(
            id=r["id"],          # 可选；不填会自动生成
            source=r["source"],  # 必须跟 nodes 里 id 完全一致
            target=r["target"],
            caption=r["type"],   # 在连线上显示关系类型
            properties=r["props"]
        )
        for r in all_data["relationships"]
    ]

    vg = VisualizationGraph(nodes=nodes, relationships=relationships)

    palette_19 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#3182bd", "#e6550d", "#31a354", "#756bb1"
    ]
    category_colors = dict(zip(snomed_ct_top_level_key_to_id.keys(), palette_19))

    # vg.color_nodes(
    #     field="caption",
    #     colors=["red", "#7fffd4", '#27A6F5', "hsl(270, 60%, 70%)"],
    #     color_space=ColorSpace.DISCRETE
    # )
    vg.color_nodes(
        property="top_level_category",
        colors=category_colors,
        color_space=ColorSpace.DISCRETE
    )
    return vg


if __name__ == '__main__':
    # 使用示例
    driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    # simple_data = get_all_nodes_and_relationships_simple(driver)
    # print(f"获取到 {len(simple_data['nodes'])} 个节点")
    # print(f"获取到 {len(simple_data['relationships'])} 个关系")
    # print(f'Nodes: {simple_data["nodes"][:5]}\n\n')  # 打印前5个节点
    # print(f"Relations: {simple_data['relationships'][:5]}")  # 打印前5个关系
    # driver.close()

    data = get_all_nodes_and_relationships_with_id(driver)
    vg = build_vg(data)