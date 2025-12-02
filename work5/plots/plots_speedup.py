import json
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import sys

objs_VecsumNobr = list()
objs_VecsumBr = list()
objs_VecsumDif = list()


def read_file() -> list:
    file_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    
    with open(file_path, "r") as file:
        data = json.load(file)
        return data["benchmarks"]


def plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[obj['number_elements'] for obj in objs_VecsumNobr],
                             y=[obj['diff'] for obj in objs_VecsumDif], name="difference"))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  title="SpeedUp",
                  xaxis_title="Количество элементов в векторе",
                  yaxis_title="Ускорение",
                  margin=dict(l=0, r=0, t=30, b=0))

    pyo.plot(fig, filename=path_output_file(), auto_open=True)


def path_output_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "benchmark_plot_speedup.html")
    return output_file


def parsing_json(list_obj):
    for obj in list_obj:
        if "VecsumNobrStrategy" in obj["name"]:
            obj_CPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_VecsumNobr.append(obj_CPU)
        elif "VecsumBrStrategy" in obj["name"]:
            obj_GPU = {"name": (obj["name"].split("/"))[0], "number_elements": (obj["name"].split("/"))[1], "real_time": obj["real_time"]}
            objs_VecsumBr.append(obj_GPU)
    for obj in range(len(objs_VecsumNobr)):
      obj_diff = {"diff": objs_VecsumNobr[obj]["real_time"] / objs_VecsumBr[obj]["real_time"]}
      objs_VecsumDif.append(obj_diff)


if __name__ == '__main__':
    list_obj = read_file()
    parsing_json(list_obj)
    plot()
    # print(objs_GPUFull, end="\n")
    # print(objs_GPUCore, end="\n")
    # print(objs_CPU, end="\n")
    # print(objs_GPUCore, end="\n")
    # print(objs_CPUManual, end="\n")