# dataload.py
import json
from typing import List, Dict, Tuple
import numpy as np

def load_json_data(file_path: str) -> List[Dict]:
    """
    加载 JSON 文件并返回数据列表。
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def parse_task_info(data: List[Dict]) -> List[Dict]:
    """
    解析 JSON 数据，提取任务信息、资源限制和依赖关系图（edge_index）。
    """
    parsed_data = []
    for item in data:
        input_str = item['input']
        output_str = item['output']

        # 解析输入
        input_dict = eval(input_str)  # 将字符串转换为字典
        tasks = input_dict['tasks']
        res_limits = input_dict['res_limits']

        # 解析输出
        output_dict = eval(output_str)
        task_times = output_dict['task_times']
        cmax = output_dict['Cmax']

        # 按照 task['id'] 对 tasks 进行排序
        tasks_sorted = sorted(tasks, key=lambda x: x['id'])
        tasks = tasks_sorted[1:-1]  # 去掉第一个和最后一个任务
        for task in tasks:
            dep = []
            for d in task['dep']:
                if d != 0:
                    dep.append(d - 1)  # 将依赖关系转换为 0-based
            task['dep'] = dep  # 更新任务的依赖关系
            task['id'] -= 1 # 将任务 ID 从 1-based 转换为 0-based

        # 生成 edge_index
        edge_index = []
        for task in tasks:
            task_id = task['id']
            for dep in task['dep']:
                edge_index.append([dep, task_id])  # 依赖关系：dep -> task_id

        # 将 edge_index 转换为 NumPy 数组
        edge_index = np.array(edge_index, dtype=np.int64).T  # 形状: (2, num_edges)

        # 将解析后的数据存入字典
        parsed_data.append({
            'tasks': tasks,
            'res_limits': res_limits,
            'edge_index': edge_index,  # 添加 edge_index
            'cmax': cmax
        })
    return parsed_data

def split_train_test(data: List[Dict], train_ratio: float = 0.8) -> (List[Dict], List[Dict]):
    """
    将数据集划分为训练集和测试集。
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data