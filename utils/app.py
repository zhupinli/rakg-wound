from flask import Flask, render_template
from .vis import get_all_nodes_and_relationships_with_id, build_vg
from .vis import URI, USERNAME, PASSWORD
from neo4j import GraphDatabase

app = Flask(__name__)

# --- Neo4j 连接信息 ---
driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

@app.route('/')
def graph_view():


    all_data = get_all_nodes_and_relationships_with_id(driver)
    VG = build_vg(all_data)

    # VG.tooltip_style = {
    #     "backgroundColor": "white",
    #     "color": "#FFFFFF",
    #     "borderRadius": "4px",
    #     "fontSize": "12px",
    #     "padding": "6px 8px"
    # }

    # 3. 从IPython.display.HTML对象中提取出HTML字符串
    # .data 属性包含了完整的HTML内容
    graph_html_string = VG.render(
        width="100%",
        height="100vh"    # 把固定 600px 改成占满视口
    ).data
    
    # 4. 将这段HTML字符串传递给前端模板
    return render_template('index.html', graph_html=graph_html_string)

if __name__ == '__main__':
    app.run(debug=True)