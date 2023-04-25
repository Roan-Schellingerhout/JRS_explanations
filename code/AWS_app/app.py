import os
import json 
import random 

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict 
from itertools import chain, groupby
from flask import Flask, session, render_template, request, redirect, url_for

app = Flask(__name__)
app.secret_key = 'dasjil91283jklsa'

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/view_graph.html", methods=['GET', 'POST'])
def view_graph():
    if request.method == 'POST':
        if request.form['button'] == 'Kandidaat':
            session["direction"] = "candidate"   
            session["control"] = "candidate"

            session["full_graph"], session["simple_graph"] = create_graphs()
            session["simple_graph_company"], session["full_graph_company"] = session["simple_graph"], session["full_graph"]

        elif request.form['button'] == 'Bedrijfsvertegenwoordiger':            
            session["direction"] = "company"
            session["control"] = "company"

            session["full_graph"], session["simple_graph"] = create_graphs()
            session["simple_graph_company"], session["full_graph_company"] = session["simple_graph"], session["full_graph"]

        else:
            session["control"] = "recruiter"
            session["simple_graph"], session["full_graph"], session["simple_graph_company"], session["full_graph_company"] = create_graphs(recruiter=True)

        decision = random.random()

        if decision < 0.333:
            return render_template("view_graph.html", 
                                   graph_simple=session["simple_graph"],
                                   graph_full=session["full_graph"],
                                   graph_simple_company=session["simple_graph_company"],
                                   graph_full_company=session["full_graph_company"],
                                   direction=session["control"])     
        elif 0.333 < decision < 0.666:
            return redirect(url_for('view_bars'))
        else:
            return redirect(url_for('view_texts'))

    else:    
        return render_template("view_graph.html", 
                                graph_simple=session["simple_graph"],
                                graph_full=session["full_graph"],
                                graph_simple_company=session["simple_graph_company"],
                                graph_full_company=session["full_graph_company"],
                                direction=session["control"])
        

def create_graphs(recruiter=False):
    truth_values = defaultdict(list)
    li = []

    session["label_reverse"] = {"heeft subtype" : "is subtype van",
                                "requiresDegree" : "isRequiredDegreeOf",
                                "maxDegree" : "isMaxDegreeOf",
                                "hasDegree" : "isUserDegreeOf",
                                "inCurrentIndustry" : "isCurrentIndustryOf", 
                                "wantsIndustry" : "isWantedIndustryOf",
                                "inCurrentType" : "isCurrentTypeOf", 
                                "wil subtype" : "is gewild subtype van", 
                                "hasSkill" : "isHeldBy", 
                                "heeft vervuld" : "is vervuld door",

                                "is subtype van" : "heeft subtype", 
                                "isRequiredDegreeOf" : "requiresDegree", 
                                "isMaxDegreeOf" : "maxDegree",
                                "isUserDegreeOf" : "hasDegree",
                                "isCurrentIndustryOf" : "inCurrentIndustry",
                                "isWantedIndustryOf" : "wantsIndustry",
                                "isCurrentTypeOf" : "inCurrentType",
                                "isWantedTypeOf" : "wantsType",
                                "isHeldBy" : "hasSkill",
                                "is vervuld door" : "heeft vervuld"}

    for truth in os.listdir("./data/ground_truth"):
        if ".csv" in truth:
            df = pd.read_csv(f"./data/ground_truth/{truth}", header=None)
            li.append(df)
                    
    truths = pd.concat(li, axis=0, ignore_index=True)
    truth_dict = {key1: dict(group[[1, 2]].values) for key1, group in truths.groupby(0)}

    hits = defaultdict(lambda : defaultdict(lambda : defaultdict))
    misses = defaultdict(lambda : defaultdict(lambda : defaultdict))

    for i in os.listdir("./data/hits"):
        if ".json" in i:
            full = json.load(open(f"./data/hits/{i}", encoding="utf-8"))
            for k, v in full.items():
                g = nx.node_link_graph(v)
                
                hits[i.split(".")[0]][k] = g
                
                    
    for i in os.listdir("./data/misses"):
        if ".json" in i:
            full = json.load(open(f"./data/misses/{i}", encoding="utf-8"))
            for k, v in full.items():
                misses[i.split(".")[0]][k] = nx.node_link_graph(v)   

    if recruiter:
        with open("candidate_explanations.json") as f:
            curr_explanations = json.load(f)

        candidate_simple, candidate_full = draw_graph('u4119', 
                                                     'j147542', 
                                                     truth_dict, 
                                                     curr_explanations, 
                                                     "candidate", 
                                                     hits,
                                                     misses)

        with open("company_explanations.json") as f:
            curr_explanations = json.load(f)

        company_simple, company_full = draw_graph('u4119', 
                                                  'j147542', 
                                                   truth_dict, 
                                                   curr_explanations, 
                                                   "company", 
                                                   hits,
                                                   misses)

        return candidate_full, candidate_simple, company_full, company_simple

    else:
        with open(f"{session['direction']}_explanations.json") as f:
            curr_explanations = json.load(f)

        return draw_graph('u4119', 
                          'j147542', 
                          truth_dict, 
                          curr_explanations, 
                          session['direction'], 
                          hits,
                          misses)



@app.route("/view_bars.html", methods=['GET', 'POST'])
def view_bars():
    data = sorted([(i["label"], i["value"], i["color"]) for i in session["full_graph"]["nodes"]], key = lambda x: x[1])
    
    data_simple = [(k, list(v)) for k, v in groupby([(i["node_type"], i["value"], i["color"]) for i in sorted(session["full_graph"]["nodes"], 
                                                                                                              key=lambda x: x["node_type"])], 
                                                    key=lambda x: x[0])]
    
    data_simple = sorted([(k, sum(i[1] for i in v), v[0][2]) for k, v in data_simple], key=lambda x: x[1])
    

    if "full_graph_company" in session:
        data_company = sorted([(i["label"], i["value"], i["color"]) for i in session["full_graph_company"]["nodes"]], key = lambda x: x[1])
        data_company_simple = [(k, list(v)) for k, v in groupby([(i["node_type"], i["value"], i["color"]) for i in sorted(session["full_graph_company"]["nodes"], 
                                                                                                                         key=lambda x: x["node_type"])], 
                                                                key=lambda x: x[0])]
    
        data_company_simple = sorted([(k, sum(i[1] for i in v), v[0][2]) for k, v in data_company_simple], key=lambda x: x[1])
        
    # if session["control"] == "company":
    #     data, data_company = data_company, data

    data = [i for i in data if i[0] not in ("u4119", "j147542")]
    data_simple = [i for i in data_simple if i[0] not in ("u4119", "j147542")]
    data_company = [i for i in data_company if i[0] not in ("u4119", "j147542")]
    data_company_simple = [i for i in data_company_simple if i[0] not in ("u4119", "j147542")] 

    return render_template("/view_bars.html",
                           xdata=[i[0] for i in data],
                           ydata=[np.round(i[1], 2) for i in data],
                           bg_colors = [i[2].upper() for i in data],

                           xdata_simple=[i[0] for i in data_simple],
                           ydata_simple=[i[1] for i in data_simple],
                           bg_colors_simple = [i[2] for i in data_simple],
                           
                           company_xdata=[i[0] for i in data_company],
                           company_ydata=[np.round(i[1], 2) for i in data_company],
                           bg_colors_company = [i[2].upper() for i in data_company],
          
                           company_xdata_simple=[i[0] for i in data_company_simple],
                           company_ydata_simple=[i[1] for i in data_company_simple],
                           bg_colors_company_simple = [i[2] for i in data_company_simple],
            
                           direction=session["control"])

                
@app.route("/view_texts.html", methods=['GET', 'POST'])
def view_texts():
    return render_template("/view_texts.html",
                           direction=session["control"])


_attrs = dict(id='id', source='source', target='target', key='key')
 
# This is stolen from networkx JSON serialization. It basically just changes what certain keys are.
def node_link_data(G, attrs=_attrs):
    
    multigraph = G.is_multigraph()
    id_ = attrs['id']
    source = attrs['source']
    target = attrs['target']
    
    # Allow 'key' to be omitted from attrs if the graph is not a multigraph.
    key = None if not multigraph else attrs['key']
    if len(set([source, target, key])) < 3:
        raise nx.NetworkXError('Attribute names are not unique.')
        
    data = {}
    data['directed'] = G.is_directed()
    data['multigraph'] = multigraph
    data['graph'] = G.graph
    data['nodes'] = [dict(chain(G.nodes[n].items(), [(id_, n), ('label', n)])) for n in G]
    
    if multigraph:
        data['links'] = [
            dict(chain(d.items(),
                       [('from', u), ('to', v), (key, k)]))
            for u, v, k, d in G.edges(keys=True, data=True)]
    
    else:
        data['links'] = [
            dict(chain(d.items(),
                       [('from', u), ('to', v)]))
            for u, v, d in G.edges(data=True)]
                    
    return data

def draw_graph(node_1, node_2, truth_dict, curr_explanations, direction, hits, misses, simple_view = False):
    truth_value = truth_dict[node_1][node_2]

    # Gather information from the explanation
    sample = curr_explanations[node_1][node_2]
    graph = sample["graph"]

    curr_score = np.round(float(sample["full_score"]), 4)
    full_score = np.round(float(sample[f"{direction}_score"]), 4)

    edges = [(i[0], i[1]) for i in graph]
    weights = {(i[0], i[1]) : i[2] for i in graph}

    # Recover the original graph
    G = {**hits, **misses}[node_1][node_2].copy()

    if direction == "company":
        node_1, node_2 = node_2, node_1
        G = G.reverse()

    # Initialize weights
    nx.set_edge_attributes(G, weights, name="weight")
    nx.set_node_attributes(G, {node_1: 10}, "weight")
    nx.set_node_attributes(G, {node: 0 for node in G if node != node_1}, "weight")

    nx.set_node_attributes(G, {'j147542': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"}, 'j65504': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"},
                               'Secretaresse': {"node_type": "Vacature types", "color" : "#fcba03", "shape" : "square"},'u3844': {"node_type": "Kandidaten", "color" : "#f26884"}, 'u3726': {"node_type": "Kandidaten", "color" : "#f26884"}, 
                               'Administratief medewerker': {"node_type": "Vacature types", "color" : "#fcba03", "shape" : "square"}, 'j127155': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"},
                               'u4119': {"node_type": "Kandidaten", "color" : "#f26884"}, '47329ebfff17ff08e8fce96bc0959ea8': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"}, 
                               'j210812': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"}, 'j190070': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"}, 
                               'j180599': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"}, 'Assistent': {"node_type": "Vacature types", "color" : "#fcba03", "shape" : "square"}, 
                               'Klerk': {"node_type": "Vacature types", "color" : "#fcba03", "shape" : "square"}, 'j107073': {"node_type": "Vacatures", "color" : "#2bc253", "shape" : "hexagon"}})
    
    G = nx.relabel_nodes(G, {"47329ebfff17ff08e8fce96bc0959ea8" : "j157652"})
    
    att_weights = nx.get_edge_attributes(G, "weight")
    att_weights = {k : float(v) for k, v in att_weights.items()}
    
    # Find paths
    paths = list(nx.all_simple_paths(G, node_1, node_2))
    bfs_paths = [list(path) for path in map(nx.utils.pairwise, paths)]
    longest_path = max([len(i) for i in bfs_paths])
    bfs_edges = []

    # Convert path to edges sorted from node_1 to node_2
    for i in range(longest_path):
        for path in bfs_paths:
            if i < len(path):
                bfs_edges.append(path[i])

    checked = set()
                
    # Calculate node and edge weights/sizes
    for edge in bfs_edges:
        if edge in checked:
            continue
        else:
            checked.add(edge)
        
        node_weights = nx.get_node_attributes(G, "weight")

        if edge not in att_weights:
            edge = (edge[1], edge[0])
                    
        nx.set_edge_attributes(G, {edge: np.round(att_weights[edge] * float(node_weights[edge[0]]), 3)}, 
                               name="value")
        
        nx.set_node_attributes(G, {edge[1]: float(node_weights[edge[1]]) + float(node_weights[edge[0]]) * att_weights[edge]}, name="weight")

        
    final_weights = nx.get_edge_attributes(G, "value")

    # If company, make sure to reverse the labels
    labels = nx.get_edge_attributes(G, "label")

    if direction == "company":
        labels = {edge: session["label_reverse"][label] for edge, label in labels.items()}
        nx.set_edge_attributes(G, labels, "label")

    nx.set_edge_attributes(G, {edge: f"{labels[edge]}\nWaarde: {final_weights[edge]}" for edge in G.edges()}, name="label")
           
    shortest_paths = nx.all_simple_paths(G, 
                                         node_1, 
                                         node_2)

    # Calculate path weights
    paths = [[(path[i], path[i + 1]) for i in range(len(path) - 1)] for path in list(shortest_paths)]
    path_weights = [(path, sum([att_weights[x, y] for x, y in path])) for path in paths]
    
    # Take the top 3 most important paths
    best_paths = [i[0] for i in sorted(path_weights, key=lambda x: x[1], reverse=True) if i[1] > 0.01][:3]
    G_simple = G.edge_subgraph(set([item for sublist in best_paths for item in sublist]))
    

    # for node in G:
    #     # print(G.nodes[node])
    #     del G.nodes[node]["embedding"]
    #     del G.nodes[node]["experience"]

    nx.set_node_attributes(G, {node: {"value": nx.get_node_attributes(G, "weight")[node]} for node in G})
    nx.set_node_attributes(G, {node: {"id" : i} for i, node in enumerate(G)}) 
    nx.set_node_attributes(G, {edge: {"value": nx.get_edge_attributes(G, "weight")[edge]} for edge in G.edges()})

    for _, _, d in G.edges(data=True):
        for att in ["embedding", "weight"]:
            d.pop(att, None)

    for _, _, d in G_simple.edges(data=True):
        for att in ["embedding", "weight"]:
            d.pop(att, None)

    G = nx.relabel_nodes(G, {node : "Kandidaat " + node[1:] for node in G if node[0] == "u"})
    G = nx.relabel_nodes(G, {node : "Vacature " + node[1:] for node in G if node[0] == "j"})

    G_simple = nx.relabel_nodes(G_simple, {node : "Kandidaat " + node[1:] for node in G_simple if node[0] == "u"})
    G_simple = nx.relabel_nodes(G_simple, {node : "Vacature " + node[1:] for node in G_simple if node[0] == "j"})

    string = node_link_data(G)
    string["directed"] = "true"
    string["multigraph"] = "false"

    string_simple = node_link_data(G_simple)
    string_simple["directed"] = "true"
    string_simple["multigraph"] = "false"

    return string, string_simple
    
if __name__ == '__main__':
    app.run('0.0.0.0', port=8080)