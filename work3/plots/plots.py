import json
import plotly.graph_objs as go
import plotly.offline as pyo
import os
import sys

objs_matMulGPU_cur = list()
objs_matMulGPU_past = list()

def read_file() -> dict:
    file_path = os.path.join(os.path.dirname(__file__), 'benchmark_results.json')
    
    with open(file_path, "r") as file:
        data = json.load(file)
        return {
            "current": data.get("benchmarks", []),
            "past": data.get("benchmarks_past", [])
        }

def plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[obj['number_elements'] for obj in objs_matMulGPU_cur],
                             y=[obj['real_time'] for obj in objs_matMulGPU_cur], name="GPU CUDA Shmem"))
    fig.add_trace(go.Scatter(x=[obj['number_elements'] for obj in objs_matMulGPU_past],
                             y=[obj['real_time'] for obj in objs_matMulGPU_past], name="GPU CUDA Naive"))
    fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  title="Зависимость времени выполнения от количества элементов в матрице",
                  xaxis_title="Количество элементов в матрице",
                  yaxis_title="Время выполнения(ns)",
                  margin=dict(l=0, r=0, t=30, b=0))

    pyo.plot(fig, filename=path_output_file(), auto_open=True)


def path_output_file():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "benchmark_plot.html")
    return output_file


def parsing_json(list_obj, is_cur=True):
    for obj in list_obj:
        if "matMulGPU" in obj["name"]:
            obj_GPU = {
                "name": (obj["name"].split("/"))[0], 
                "number_elements": (obj["name"].split("/"))[1], 
                "real_time": obj["real_time"]
            }
            if is_cur:
                objs_matMulGPU_cur.append(obj_GPU)
            else:
                objs_matMulGPU_past.append(obj_GPU)


if __name__ == '__main__':
    list_obj = read_file()
    current_benchmarks = list_obj["current"]
    past_benchmarks = list_obj["past"]
    parsing_json(current_benchmarks, is_cur=True)
    parsing_json(past_benchmarks, is_cur=False)
    plot()
    # print(objs_GPUFull, end="\n")
    # print(objs_GPUCore, end="\n")
    # print(objs_CPU, end="\n")
    # print(objs_GPUCore, end="\n")
    # print(objs_CPUManual, end="\n")